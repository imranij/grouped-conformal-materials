"""Run the UC-MatReg material-property benchmark.

The script downloads two public UCI materials datasets, evaluates common
regression baselines, trains the proposed uncertainty-calibrated stacked
ensemble, and writes publication-ready result tables and figures.
"""

from __future__ import annotations

import argparse
import math
import shutil
import ssl
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGB = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42
ALPHA = 0.10

DATASET_SOURCES = {
    "concrete": {
        "url": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
        "zip": "concrete_compressive_strength.zip",
        "target": "Concrete compressive strength",
        "display": "Concrete compressive strength",
        "unit": "MPa",
        "uci_doi": "10.24432/C5PK67",
    },
    "superconductivity": {
        "url": "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "zip": "superconductivity_data.zip",
        "target": "critical_temp",
        "display": "Superconductor critical temperature",
        "unit": "K",
        "uci_doi": "10.24432/C53P47",
    },
}


def ensure_dirs() -> None:
    for path in (RAW_DIR, RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_and_extract() -> None:
    ensure_dirs()
    for name, meta in DATASET_SOURCES.items():
        zip_path = RAW_DIR / meta["zip"]
        extract_dir = RAW_DIR / name
        extract_dir.mkdir(parents=True, exist_ok=True)
        if zip_path.exists() and zip_path.stat().st_size == 0:
            zip_path.unlink()
        if not zip_path.exists():
            print(f"Downloading {name} from UCI...")
            request = Request(meta["url"], headers={"User-Agent": "UC-MatReg/1.0"})
            tmp_path = zip_path.with_suffix(".zip.part")
            try:
                with urlopen(request, timeout=90) as response, open(tmp_path, "wb") as out:
                    shutil.copyfileobj(response, out)
            except (ssl.SSLError, URLError):
                # Some local Python distributions reject the current UCI chain even
                # when the host is reachable in browsers and PowerShell. The URL is
                # fixed and cited; this fallback only bypasses local CA validation.
                context = ssl._create_unverified_context()
                with urlopen(request, timeout=90, context=context) as response, open(tmp_path, "wb") as out:
                    shutil.copyfileobj(response, out)
            tmp_path.replace(zip_path)
        if not any(extract_dir.iterdir()):
            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)


def _standardize_columns(columns: list[str]) -> list[str]:
    cleaned = []
    for col in columns:
        col = str(col).strip()
        col = col.replace("(", "").replace(")", "")
        col = col.replace("-", "_").replace("/", "_per_")
        col = "_".join(col.split())
        cleaned.append(col)
    return cleaned


def load_concrete() -> tuple[pd.DataFrame, pd.Series]:
    files = list((RAW_DIR / "concrete").glob("*.xls*"))
    if not files:
        raise FileNotFoundError("Concrete_Data.xls was not found after extraction.")
    engine = "xlrd" if files[0].suffix.lower() == ".xls" else "openpyxl"
    try:
        df = pd.read_excel(files[0], engine=engine)
    except ImportError as exc:
        raise ImportError(
            f"Reading {files[0].name} requires the '{engine}' package. "
            "Install code/requirements.txt before running the benchmark."
        ) from exc
    df.columns = _standardize_columns(list(df.columns))
    target_col = [c for c in df.columns if "Concrete_compressive_strength" in c][0]
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col]).astype(float)
    return X, y


def load_superconductivity() -> tuple[pd.DataFrame, pd.Series]:
    csv_path = RAW_DIR / "superconductivity" / "train.csv"
    if not csv_path.exists():
        candidates = list((RAW_DIR / "superconductivity").rglob("train.csv"))
        if not candidates:
            raise FileNotFoundError("train.csv was not found after extraction.")
        csv_path = candidates[0]
    df = pd.read_csv(csv_path)
    df.columns = _standardize_columns(list(df.columns))
    y = df["critical_temp"].astype(float)
    X = df.drop(columns=["critical_temp"]).astype(float)
    return X, y


def load_datasets() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    loaders = {
        "concrete": load_concrete,
        "superconductivity": load_superconductivity,
    }
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for name, loader in loaders.items():
        try:
            datasets[name] = loader()
        except Exception as exc:  # keep the benchmark running on available data
            print(f"WARNING: skipping dataset '{name}' (could not load: {exc})")
    if not datasets:
        raise RuntimeError("No datasets could be loaded.")
    return datasets


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def metric_dict(
    dataset: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_seconds: float,
    pred_seconds: float,
) -> dict[str, float | str]:
    return {
        "dataset": dataset,
        "model": model,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "sMAPE_percent": smape(y_true, y_pred),
        "train_seconds": train_seconds,
        "predict_seconds": pred_seconds,
    }


def scaled_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", model),
        ]
    )


def tree_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def make_models(profile: str) -> dict[str, Pipeline]:
    if profile == "fast":
        rf_trees, et_trees, hgb_iter, xgb_trees = 120, 160, 160, 240
        mlp_iter = 150
    else:
        rf_trees, et_trees, hgb_iter, xgb_trees = 220, 260, 260, 420
        mlp_iter = 350

    models: dict[str, Pipeline] = {
        "RidgeCV": scaled_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "ElasticNetCV": scaled_pipeline(
            ElasticNetCV(
                l1_ratio=[0.05, 0.2, 0.5, 0.8],
                alphas=np.logspace(-4, 1, 20),
                cv=5,
                max_iter=10000,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "Random forest": tree_pipeline(
            RandomForestRegressor(
                n_estimators=rf_trees,
                max_features="sqrt",
                min_samples_leaf=2,
                bootstrap=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "Extra trees": tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=et_trees,
                max_features="sqrt",
                min_samples_leaf=1,
                bootstrap=False,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "HistGradientBoosting": tree_pipeline(
            HistGradientBoostingRegressor(
                max_iter=hgb_iter,
                learning_rate=0.055,
                max_leaf_nodes=31,
                l2_regularization=0.03,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
        ),
        # Deep-learning baseline (reviewer request): a multilayer perceptron so the
        # comparison is not limited to linear/tree models.
        "MLP (deep)": scaled_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                early_stopping=True,
                n_iter_no_change=15,
                max_iter=mlp_iter,
                random_state=RANDOM_STATE,
            )
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = tree_pipeline(
            XGBRegressor(
                n_estimators=xgb_trees,
                max_depth=5,
                learning_rate=0.045,
                subsample=0.88,
                colsample_bytree=0.86,
                reg_lambda=1.5,
                reg_alpha=0.01,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
        )

    return models


@dataclass
class PredictionIntervals:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    coverage: float
    mean_width: float
    qhat: float


class UCMatRegressor:
    """Uncertainty-calibrated stacked regression for tabular materials data."""

    def __init__(
        self,
        base_models: dict[str, Pipeline],
        alpha: float = ALPHA,
        n_splits: int = 5,
        residual_estimators: int = 180,
    ):
        self.base_models = base_models
        self.alpha = alpha
        self.n_splits = n_splits
        self.residual_estimators = residual_estimators
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.model_names_: list[str] = []
        self.final_models_: dict[str, Pipeline] = {}
        self.residual_model_: Pipeline | None = None
        self.qhat_: float | None = None
        self.global_qhat_: float | None = None
        self.global_residual_: float | None = None

    @staticmethod
    def _meta_features(models: dict[str, Pipeline], X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([model.predict(X) for model in models.values()])

    def _fit_stack(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model_names_ = list(self.base_models.keys())
        oof = np.zeros((len(X), len(self.base_models)), dtype=float)
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            for train_idx, val_idx in kfold.split(X):
                fold_model = clone(model)
                fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                oof[val_idx, model_idx] = fold_model.predict(X.iloc[val_idx])

        meta_model = LinearRegression(positive=True)
        meta_model.fit(oof, y.to_numpy())
        weights = meta_model.coef_.astype(float)
        if weights.sum() <= 1e-12:
            weights, _ = nnls(oof, y.to_numpy())
            self.intercept_ = 0.0
        else:
            self.intercept_ = float(meta_model.intercept_)
        if weights.sum() <= 1e-12:
            weights = np.ones(len(self.base_models), dtype=float)
        self.weights_ = weights

        self.final_models_ = {}
        for name, model in self.base_models.items():
            final_model = clone(model)
            final_model.fit(X, y)
            self.final_models_[name] = final_model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UCMatRegressor":
        x_fit, x_temp, y_fit, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE
        )
        x_resid, x_cal, y_resid, y_cal = train_test_split(
            x_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
        )

        self._fit_stack(x_fit.reset_index(drop=True), y_fit.reset_index(drop=True))

        resid_pred = self.predict(x_resid)
        abs_resid = np.abs(y_resid.to_numpy() - resid_pred)
        self.global_residual_ = float(np.median(abs_resid) + 1e-6)

        self.residual_model_ = tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=self.residual_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        )
        self.residual_model_.fit(x_resid, abs_resid)

        cal_pred = self.predict(x_cal)
        cal_abs = np.abs(y_cal.to_numpy() - cal_pred)
        sigma = np.maximum(self.residual_model_.predict(x_cal), self.global_residual_ * 0.15)
        scores = cal_abs / sigma
        n_cal = len(scores)
        level = min(1.0, math.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal)
        self.qhat_ = float(np.quantile(scores, level, method="higher"))
        self.global_qhat_ = float(np.quantile(cal_abs, level, method="higher"))

        # Refit the point-prediction ensemble on every available training
        # example after the interval calibration statistics have been fixed.
        self._fit_stack(X.reset_index(drop=True), y.reset_index(drop=True))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None or not self.final_models_:
            raise RuntimeError("The model must be fitted before prediction.")
        meta = self._meta_features(self.final_models_, X)
        return meta @ self.weights_ + self.intercept_

    def predict_interval(self, X: pd.DataFrame, y_true: pd.Series | None = None) -> PredictionIntervals:
        pred = self.predict(X)
        if self.residual_model_ is None or self.qhat_ is None or self.global_residual_ is None:
            raise RuntimeError("The model must be fitted before interval prediction.")
        sigma = np.maximum(self.residual_model_.predict(X), self.global_residual_ * 0.15)
        width = self.qhat_ * sigma
        lower = pred - width
        upper = pred + width
        coverage = np.nan
        if y_true is not None:
            yv = y_true.to_numpy()
            coverage = float(np.mean((yv >= lower) & (yv <= upper)))
        return PredictionIntervals(
            lower=lower,
            upper=upper,
            width=width,
            coverage=coverage,
            mean_width=float(np.mean(2 * width)),
            qhat=float(self.qhat_),
        )

    def predict_global_interval(self, X: pd.DataFrame, y_true: pd.Series | None = None) -> PredictionIntervals:
        """Constant-width split conformal interval for the same point model.

        This is a deliberately simple uncertainty baseline. Comparing it against
        the locally adaptive interval separates the value of residual-scale
        modeling from the point-prediction ensemble itself.
        """
        pred = self.predict(X)
        if self.global_qhat_ is None:
            raise RuntimeError("The model must be fitted before interval prediction.")
        width = np.full(len(pred), self.global_qhat_, dtype=float)
        lower = pred - width
        upper = pred + width
        coverage = np.nan
        if y_true is not None:
            yv = y_true.to_numpy()
            coverage = float(np.mean((yv >= lower) & (yv <= upper)))
        return PredictionIntervals(
            lower=lower,
            upper=upper,
            width=width,
            coverage=coverage,
            mean_width=float(np.mean(2 * width)),
            qhat=float(self.global_qhat_),
        )


def select_uc_base_models(all_models: dict[str, Pipeline]) -> dict[str, Pipeline]:
    preferred = ["RidgeCV", "Random forest", "Extra trees", "HistGradientBoosting", "XGBoost"]
    return {name: all_models[name] for name in preferred if name in all_models}


def evaluate_dataset(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    profile: str,
) -> tuple[
    list[dict[str, float | str]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = make_models(profile)
    rows: list[dict[str, float | str]] = []
    predictions = pd.DataFrame({"y_true": y_test.to_numpy()})

    for model_name, model in models.items():
        print(f"[{name}] Training {model_name}...")
        t0 = time.perf_counter()
        fitted = clone(model).fit(x_train, y_train)
        train_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = fitted.predict(x_test)
        pred_seconds = time.perf_counter() - t0
        rows.append(metric_dict(name, model_name, y_test.to_numpy(), pred, train_seconds, pred_seconds))
        predictions[model_name] = pred

    print(f"[{name}] Training UC-MatReg...")
    uc = UCMatRegressor(select_uc_base_models(models), alpha=ALPHA)
    t0 = time.perf_counter()
    uc.fit(x_train.reset_index(drop=True), y_train.reset_index(drop=True))
    train_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    uc_pred = uc.predict(x_test)
    pred_seconds = time.perf_counter() - t0
    rows.append(metric_dict(name, "UC-MatReg", y_test.to_numpy(), uc_pred, train_seconds, pred_seconds))
    predictions["UC-MatReg"] = uc_pred

    intervals = uc.predict_interval(x_test, y_test)
    global_intervals = uc.predict_global_interval(x_test, y_test)
    interval_df = pd.DataFrame(
        {
            "dataset": name,
            "y_true": y_test.to_numpy(),
            "prediction": uc_pred,
            "lower_90": intervals.lower,
            "upper_90": intervals.upper,
            "interval_width": intervals.upper - intervals.lower,
            "covered": (y_test.to_numpy() >= intervals.lower) & (y_test.to_numpy() <= intervals.upper),
            "global_lower_90": global_intervals.lower,
            "global_upper_90": global_intervals.upper,
            "global_interval_width": global_intervals.upper - global_intervals.lower,
            "global_covered": (y_test.to_numpy() >= global_intervals.lower)
            & (y_test.to_numpy() <= global_intervals.upper),
        }
    )

    uncertainty_baselines = pd.DataFrame(
        [
            {
                "dataset": name,
                "interval_method": "Adaptive",
                "n_test": len(y_test),
                "nominal_coverage": 1.0 - ALPHA,
                "empirical_coverage": intervals.coverage,
                "mean_interval_width": intervals.mean_width,
                "qhat": intervals.qhat,
            },
            {
                "dataset": name,
                "interval_method": "Global",
                "n_test": len(y_test),
                "nominal_coverage": 1.0 - ALPHA,
                "empirical_coverage": global_intervals.coverage,
                "mean_interval_width": global_intervals.mean_width,
                "qhat": global_intervals.qhat,
            },
        ]
    )
    adaptive_width = float(uncertainty_baselines.loc[
        uncertainty_baselines["interval_method"] == "Adaptive", "mean_interval_width"
    ].iloc[0])
    uncertainty_baselines["width_vs_adaptive"] = (
        uncertainty_baselines["mean_interval_width"] / adaptive_width
    )

    rows[-1]["interval_coverage_90"] = intervals.coverage
    rows[-1]["mean_interval_width"] = intervals.mean_width
    rows[-1]["conformal_qhat"] = intervals.qhat
    rows[-1]["ensemble_weights"] = "; ".join(
        f"{n}={w:.3f}" for n, w in zip(uc.model_names_, uc.weights_)
    )

    print(f"[{name}] Computing feature importance and OOD diagnostics...")
    max_perm = min(len(x_test), 1200)
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(x_test), size=max_perm, replace=False)
    x_perm = x_test.iloc[sample_idx]
    y_perm = y_test.iloc[sample_idx]
    perm = permutation_importance(
        uc,
        x_perm,
        y_perm,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "dataset": name,
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    nn = NearestNeighbors(n_neighbors=min(10, len(x_train)), metric="euclidean")
    nn.fit(x_train_scaled)
    distances, _ = nn.kneighbors(x_test_scaled)
    knn_distance = distances.mean(axis=1)
    ood = pd.DataFrame(
        {
            "dataset": name,
            "knn_distance": knn_distance,
            "absolute_error": np.abs(y_test.to_numpy() - uc_pred),
        }
    )
    ood["distance_bin"] = pd.qcut(ood["knn_distance"], q=5, duplicates="drop")

    # Covariate-shift conformal coverage (reviewer request): stratify the test set
    # by applicability-domain distance (kNN) into quintiles and report empirical
    # coverage and interval width within each shift bin. Q1 = in-domain, Q5 = far OOD.
    covered = (y_test.to_numpy() >= intervals.lower) & (y_test.to_numpy() <= intervals.upper)
    width = intervals.upper - intervals.lower
    shift = pd.DataFrame(
        {
            "knn_distance": knn_distance,
            "covered": covered.astype(float),
            "interval_width": width,
            "abs_error": np.abs(y_test.to_numpy() - uc_pred),
        }
    )
    try:
        shift["shift_bin"] = pd.qcut(
            shift["knn_distance"], q=5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop"
        )
    except ValueError:
        shift["shift_bin"] = "Q1"
    shift_summary = (
        shift.groupby("shift_bin", observed=True)
        .agg(
            n=("covered", "size"),
            knn_distance_mean=("knn_distance", "mean"),
            empirical_coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .reset_index()
    )
    shift_summary.insert(0, "dataset", name)
    shift_summary["nominal_coverage"] = 1.0 - ALPHA

    predictions.insert(0, "dataset", name)
    predictions["row_id"] = np.arange(len(predictions))

    return (
        rows,
        predictions,
        interval_df,
        pd.concat([importance_df, ood], axis=0, ignore_index=True),
        shift_summary,
        uncertainty_baselines,
    )


def originpro_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.labelweight": "semibold",
            "axes.edgecolor": "#202020",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#D2D2D2",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.95,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "legend.frameon": True,
            "legend.edgecolor": "#202020",
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    fig.savefig(FIGURES_DIR / f"{stem}.png")
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values(["dataset", "MAE"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=False)
    palette = {
        "UC-MatReg": "#C00000",
        "XGBoost": "#1F77B4",
        "HistGradientBoosting": "#2CA02C",
        "Extra trees": "#9467BD",
        "Random forest": "#FF7F0E",
        "RidgeCV": "#555555",
        "ElasticNetCV": "#8C564B",
        "MLP (deep)": "#17BECF",
    }
    for ax, dataset in zip(axes, sorted(metrics["dataset"].unique())):
        part = ordered[ordered["dataset"] == dataset].copy()
        colors = [palette.get(m, "#4C78A8") for m in part["model"]]
        ax.barh(part["model"], part["MAE"], color=colors, edgecolor="#202020", linewidth=0.55)
        ax.invert_yaxis()
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("MAE")
        ax.set_ylabel("")
    fig.suptitle("Model comparison on held-out material-property test sets", fontweight="semibold")
    save_fig(fig, "fig_model_comparison_mae")


def plot_predicted_vs_observed(predictions: pd.DataFrame, dataset_meta: dict[str, dict[str, str]]) -> None:
    for dataset in sorted(predictions["dataset"].unique()):
        part = predictions[predictions["dataset"] == dataset]
        y_true = part["y_true"].to_numpy()
        y_pred = part["UC-MatReg"].to_numpy()
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.scatter(
            y_true,
            y_pred,
            s=20,
            c="#1F77B4",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
        )
        lim_min = min(y_true.min(), y_pred.min())
        lim_max = max(y_true.max(), y_pred.max())
        pad = 0.04 * (lim_max - lim_min)
        ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], color="#C00000", lw=1.3)
        ax.set_xlim(lim_min - pad, lim_max + pad)
        ax.set_ylim(lim_min - pad, lim_max + pad)
        unit = dataset_meta[dataset]["unit"]
        ax.set_xlabel(f"Observed property ({unit})")
        ax.set_ylabel(f"Predicted property ({unit})")
        ax.set_title(dataset_meta[dataset]["display"])
        save_fig(fig, f"fig_predicted_vs_observed_{dataset}")


def plot_intervals(intervals: pd.DataFrame, dataset_meta: dict[str, dict[str, str]]) -> None:
    for dataset in sorted(intervals["dataset"].unique()):
        part = intervals[intervals["dataset"] == dataset].copy()
        part["abs_error"] = np.abs(part["y_true"] - part["prediction"])
        part = part.sort_values("abs_error", ascending=False).head(70).sort_values("prediction")
        x = np.arange(len(part))
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        ax.vlines(x, part["lower_90"], part["upper_90"], color="#2F5597", lw=1.2, alpha=0.80)
        ax.scatter(x, part["prediction"], color="#C00000", s=20, label="Prediction", zorder=3)
        ax.scatter(x, part["y_true"], color="#222222", s=18, label="Observed", zorder=3)
        ax.set_xlabel("Selected test specimens ordered by prediction")
        ax.set_ylabel(f"Property ({dataset_meta[dataset]['unit']})")
        cov = part["covered"].mean() * 100.0
        ax.set_title(f"UC-MatReg 90% conformal intervals: {dataset.capitalize()} subset")
        ax.legend(loc="best")
        ax.text(
            0.02,
            0.96,
            f"Subset coverage: {cov:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": "#202020", "lw": 0.6},
        )
        save_fig(fig, f"fig_conformal_intervals_{dataset}")


def plot_feature_importance(importance: pd.DataFrame) -> None:
    imp = importance.dropna(subset=["importance_mean"])
    for dataset in sorted(imp["dataset"].unique()):
        part = imp[imp["dataset"] == dataset].sort_values("importance_mean", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.barh(
            part["feature"][::-1],
            part["importance_mean"][::-1],
            xerr=part["importance_std"][::-1],
            color="#2CA02C",
            edgecolor="#202020",
            linewidth=0.55,
            error_kw={"elinewidth": 0.9, "capsize": 2.5, "ecolor": "#202020"},
        )
        ax.set_xlabel("Permutation importance: MAE increase")
        ax.set_ylabel("")
        ax.set_title(f"UC-MatReg descriptor importance: {dataset.capitalize()}")
        save_fig(fig, f"fig_feature_importance_{dataset}")


def plot_ood_diagnostics(importance_and_ood: pd.DataFrame) -> None:
    ood = importance_and_ood.dropna(subset=["knn_distance", "absolute_error"]).copy()
    if ood.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, dataset in zip(axes, sorted(ood["dataset"].unique())):
        part = ood[ood["dataset"] == dataset]
        sns.regplot(
            data=part,
            x="knn_distance",
            y="absolute_error",
            ax=ax,
            scatter_kws={"s": 16, "alpha": 0.45, "color": "#1F77B4", "edgecolor": "none"},
            line_kws={"color": "#C00000", "lw": 1.4},
            lowess=True,
        )
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Mean 10-nearest-neighbor distance")
        ax.set_ylabel("Absolute error")
    fig.suptitle("Applicability-domain diagnostic for UC-MatReg", fontweight="semibold")
    save_fig(fig, "fig_ood_error_distance")


def plot_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.axis("off")
    boxes = [
        ("Open materials data\nUCI concrete + superconductors", 0.04, 0.68, 0.18, 0.18, "#D9EAD3"),
        ("Data validation\nmissingness, units, split", 0.28, 0.68, 0.18, 0.18, "#E2F0D9"),
        ("Descriptor matrix\ncomposition/process features", 0.52, 0.68, 0.18, 0.18, "#DDEBF7"),
        ("Baseline learners\nlinear, forest, boosting", 0.04, 0.35, 0.18, 0.18, "#FFF2CC"),
        ("OOF stacking\nnonnegative weights", 0.28, 0.35, 0.18, 0.18, "#FCE4D6"),
        ("UC-MatReg point model\nweighted ensemble", 0.52, 0.35, 0.18, 0.18, "#F4CCCC"),
        ("Residual model\nlocal error scale", 0.76, 0.50, 0.18, 0.18, "#EADCF8"),
        ("Split conformal layer\n90% prediction intervals", 0.76, 0.25, 0.18, 0.18, "#D9E1F2"),
        ("Outputs\nprediction, uncertainty,\nimportance, OOD flag", 0.52, 0.05, 0.22, 0.18, "#D0E0E3"),
    ]
    for text, x, y, w, h, color in boxes:
        rect = plt.Rectangle((x, y), w, h, fc=color, ec="#202020", lw=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="semibold")

    arrows = [
        ((0.22, 0.77), (0.28, 0.77)),
        ((0.46, 0.77), (0.52, 0.77)),
        ((0.61, 0.68), (0.13, 0.53)),
        ((0.22, 0.44), (0.28, 0.44)),
        ((0.46, 0.44), (0.52, 0.44)),
        ((0.70, 0.44), (0.76, 0.59)),
        ((0.70, 0.44), (0.76, 0.34)),
        ((0.76, 0.34), (0.74, 0.14)),
        ((0.61, 0.35), (0.61, 0.23)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "#202020", "shrinkA": 3, "shrinkB": 3},
        )
    ax.text(0.04, 0.95, "Conceptual system architecture of UC-MatReg", fontsize=14, fontweight="bold")
    save_fig(fig, "fig_uc_matreg_architecture")


def plot_coverage_by_shift(shift: pd.DataFrame) -> None:
    """Empirical conformal coverage across applicability-domain (covariate-shift) bins."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    datasets = sorted(shift["dataset"].unique())
    bins = [f"Q{i}" for i in range(1, 6)]
    width = 0.8 / max(len(datasets), 1)
    colors = {"concrete": "#2CA02C", "superconductivity": "#1F77B4"}
    for di, dataset in enumerate(datasets):
        part = shift[shift["dataset"] == dataset].set_index("shift_bin").reindex(bins)
        x = np.arange(len(bins)) + di * width
        ax.bar(
            x, part["empirical_coverage"].to_numpy(), width=width,
            label=dataset.capitalize(), color=colors.get(dataset, "#888888"),
            edgecolor="#202020", linewidth=0.55,
        )
    ax.axhline(0.90, color="#C00000", ls="--", lw=1.3, label="Nominal 90%")
    ax.set_xticks(np.arange(len(bins)) + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(["Q1\n(in-domain)", "Q2", "Q3", "Q4", "Q5\n(far OOD)"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Empirical 90% interval coverage")
    ax.set_title("UC-MatReg conformal coverage under covariate shift")
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    save_fig(fig, "fig_coverage_by_shift")


def _latex_table(df: pd.DataFrame, columns: list[str], headers: list[str],
                 caption: str, label: str, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v).replace("_", r"\_")
    align = "l" * 1 + "r" * (len(columns) - 1)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\resizebox{\linewidth}{!}{%",
             f"\\begin{{tabular}}{{{align}}}", r"\hline",
             " & ".join(headers) + r" \\", r"\hline"]
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(row[c]) for c in columns) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def write_latex_tables(metrics: pd.DataFrame, coverage_summary: pd.DataFrame,
                       shift: pd.DataFrame, uncertainty_baselines: pd.DataFrame,
                       manuscript_dir: Path) -> None:
    """Emit booktabs-free LaTeX tables the manuscript can \\input{}."""
    targets = [RESULTS_DIR, manuscript_dir] if manuscript_dir.exists() else [RESULTS_DIR]

    # 1) per-dataset conformal coverage (nominal vs empirical)
    cov = coverage_summary.copy()
    cov_tex = _latex_table(
        cov, ["dataset", "n_test", "nominal_coverage", "empirical_coverage", "mean_interval_width"],
        ["Dataset", "$n_{test}$", "Nominal", "Empirical", "Mean width"],
        "UC-MatReg 90\\% split-conformal coverage per dataset (held-out test set).",
        "tab:coverage")

    # 2) coverage by covariate-shift bin
    sh = shift.copy()
    sh_tex = _latex_table(
        sh, ["dataset", "shift_bin", "n", "knn_distance_mean", "empirical_coverage", "mean_interval_width"],
        ["Dataset", "Shift bin", "$n$", "kNN dist.", "Coverage", "Mean width"],
        "UC-MatReg conformal coverage stratified by applicability-domain distance "
        "(Q1 in-domain $\\rightarrow$ Q5 far out-of-domain).", "tab:coverage_shift")

    # 3) model comparison incl. the deep (MLP) baseline
    mm = metrics.copy()
    mcols = ["dataset", "model", "MAE", "RMSE", "R2", "sMAPE_percent"]
    mm = mm[mcols].sort_values(["dataset", "MAE"])
    mm_tex = _latex_table(
        mm, mcols, ["Dataset", "Model", "MAE", "RMSE", "$R^2$", "sMAPE\\%"],
        "Regression accuracy on held-out test sets, including the deep MLP baseline.",
        "tab:models")

    ub = uncertainty_baselines.copy()
    ub_tex = _latex_table(
        ub,
        ["dataset", "interval_method", "n_test", "empirical_coverage", "mean_interval_width", "width_vs_adaptive"],
        ["Dataset", "Interval", "$n_{test}$", "Coverage", "Mean width", "Width/adaptive"],
        "Uncertainty-baseline comparison for the same UC-MatReg point model. "
        "Adaptive uses a learned local residual scale; Global uses a constant split-conformal residual quantile.",
        "tab:uncertainty_baselines")

    for d in targets:
        (d / "generated_coverage_table.tex").write_text(cov_tex, encoding="utf-8")
        (d / "generated_coverage_shift_table.tex").write_text(sh_tex, encoding="utf-8")
        (d / "generated_model_metrics_table.tex").write_text(mm_tex, encoding="utf-8")
        (d / "generated_uncertainty_baseline_table.tex").write_text(ub_tex, encoding="utf-8")


def write_dataset_summary(datasets: dict[str, tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    rows = []
    for name, (X, y) in datasets.items():
        rows.append(
            {
                "dataset": name,
                "instances": len(X),
                "features": X.shape[1],
                "target": DATASET_SOURCES[name]["target"],
                "target_unit": DATASET_SOURCES[name]["unit"],
                "missing_values": int(X.isna().sum().sum() + y.isna().sum()),
                "target_min": float(y.min()),
                "target_mean": float(y.mean()),
                "target_std": float(y.std()),
                "target_max": float(y.max()),
                "source_url": DATASET_SOURCES[name]["url"],
                "uci_doi": DATASET_SOURCES[name]["uci_doi"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["fast", "full"],
        default="full",
        help="Use 'fast' for quicker verification and 'full' for manuscript results.",
    )
    args = parser.parse_args()

    originpro_style()
    download_and_extract()
    datasets = load_datasets()
    write_dataset_summary(datasets)

    all_metrics: list[dict[str, float | str]] = []
    all_predictions: list[pd.DataFrame] = []
    all_intervals: list[pd.DataFrame] = []
    all_importance_ood: list[pd.DataFrame] = []
    all_shift: list[pd.DataFrame] = []
    all_uncertainty_baselines: list[pd.DataFrame] = []

    for name, (X, y) in datasets.items():
        rows, preds, intervals, importance_ood, shift, uncertainty_baselines = evaluate_dataset(
            name, X, y, args.profile
        )
        all_metrics.extend(rows)
        all_predictions.append(preds)
        all_intervals.append(intervals)
        all_importance_ood.append(importance_ood)
        all_shift.append(shift)
        all_uncertainty_baselines.append(uncertainty_baselines)

    metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    importance_ood = pd.concat(all_importance_ood, ignore_index=True)
    shift = pd.concat(all_shift, ignore_index=True)
    uncertainty_baselines = pd.concat(all_uncertainty_baselines, ignore_index=True)

    # Per-dataset overall conformal coverage summary (nominal vs empirical).
    coverage_summary = (
        intervals.groupby("dataset")
        .apply(lambda g: pd.Series({
            "n_test": len(g),
            "nominal_coverage": 1.0 - ALPHA,
            "empirical_coverage": float(g["covered"].mean()),
            "mean_interval_width": float(g["interval_width"].mean()),
        }), include_groups=False)
        .reset_index()
    )

    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    intervals.to_csv(RESULTS_DIR / "conformal_intervals.csv", index=False)
    importance_ood.to_csv(RESULTS_DIR / "feature_importance_and_ood.csv", index=False)
    shift.to_csv(RESULTS_DIR / "conformal_coverage_by_shift.csv", index=False)
    uncertainty_baselines.to_csv(RESULTS_DIR / "conformal_uncertainty_baselines.csv", index=False)
    coverage_summary.to_csv(RESULTS_DIR / "conformal_coverage_summary.csv", index=False)

    manuscript_dir = PROJECT_ROOT / "manuscript"
    write_latex_tables(metrics, coverage_summary, shift, uncertainty_baselines, manuscript_dir)

    plot_model_comparison(metrics)
    plot_predicted_vs_observed(predictions, DATASET_SOURCES)
    plot_intervals(intervals, DATASET_SOURCES)
    plot_feature_importance(importance_ood)
    plot_ood_diagnostics(importance_ood)
    plot_coverage_by_shift(shift)
    plot_architecture_diagram()

    print("\nCompleted benchmark.")
    print(metrics.sort_values(["dataset", "MAE"]).to_string(index=False))
    print(f"\nResults written to: {RESULTS_DIR}")
    print(f"Figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
