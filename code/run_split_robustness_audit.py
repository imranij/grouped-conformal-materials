"""Run a small repeated-split robustness audit for UC-MatReg.

The main manuscript benchmark uses one fixed 80/20 train/test split so that all
figures and prediction files are traceable to a single split. This script adds a
lightweight three-or-more-seed audit over the strongest single learners and
UC-MatReg. It is not a replacement for a full Matbench-style validation suite;
it is an explicit stability check for the two shipped datasets.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from run_uc_matreg_benchmark import (
    ALPHA,
    PROJECT_ROOT,
    RESULTS_DIR,
    UCMatRegressor,
    download_and_extract,
    load_datasets,
    make_models,
    metric_dict,
    select_uc_base_models,
)


def _audit_uc_base_models(models: dict) -> dict:
    """Use a smaller UC base set for repeated-split auditing.

    The full manuscript UC model remains unchanged in `run_uc_matreg_benchmark.py`.
    This audit profile is meant to test split sensitivity without recomputing the
    complete publication pipeline multiple times.
    """
    preferred = ["RidgeCV", "Extra trees", "HistGradientBoosting", "XGBoost"]
    return {name: models[name] for name in preferred if name in models}


def _write_latex_table(summary: pd.DataFrame, manuscript_dir: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        coverage = "-"
        if not pd.isna(row["coverage_mean"]):
            coverage = f"{row['coverage_mean']:.3f}"
        rows.append(
            " & ".join(
                [
                    str(row["dataset"]).replace("_", r"\_"),
                    str(row["model"]).replace("_", r"\_"),
                    str(int(row["n_splits"])),
                    f"{row['MAE_mean']:.3f} $\\pm$ {row['MAE_std']:.3f}",
                    f"{row['RMSE_mean']:.3f} $\\pm$ {row['RMSE_std']:.3f}",
                    f"{row['R2_mean']:.3f} $\\pm$ {row['R2_std']:.3f}",
                    coverage,
                ]
            )
            + r" \\"
        )

    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Repeated-split robustness audit over the strongest tabular baselines and UC-MatReg. Splits use independent 80/20 random partitions; coverage is reported only for UC-MatReg.}",
            r"\label{tab:split_robustness}",
            r"\scriptsize",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{llrllll}",
            r"\hline",
            r"Dataset & Model & Splits & MAE & RMSE & $R^2$ & Cov. \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    for target_dir in [RESULTS_DIR, manuscript_dir]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "generated_split_robustness_table.tex").write_text(table, encoding="utf-8")


def run_audit(profile: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    download_and_extract()
    datasets = load_datasets()
    rows: list[dict[str, float | str]] = []

    for dataset_name, (X, y) in datasets.items():
        for seed in seeds:
            print(f"[{dataset_name}] split seed {seed}")
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=seed
            )
            models = make_models(profile)
            selected = [
                name for name in ["Extra trees", "HistGradientBoosting", "XGBoost"]
                if name in models
            ]
            for model_name in selected:
                print(f"[{dataset_name}] seed {seed}: {model_name}")
                t0 = time.perf_counter()
                fitted = clone(models[model_name]).fit(x_train, y_train)
                train_seconds = time.perf_counter() - t0
                t0 = time.perf_counter()
                pred = fitted.predict(x_test)
                pred_seconds = time.perf_counter() - t0
                row = metric_dict(
                    dataset_name,
                    model_name,
                    y_test.to_numpy(),
                    pred,
                    train_seconds,
                    pred_seconds,
                )
                row["seed"] = seed
                rows.append(row)

            print(f"[{dataset_name}] seed {seed}: UC-MatReg")
            uc = UCMatRegressor(
                _audit_uc_base_models(models),
                alpha=ALPHA,
                n_splits=3,
                residual_estimators=60,
            )
            t0 = time.perf_counter()
            uc.fit(x_train.reset_index(drop=True), y_train.reset_index(drop=True))
            train_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = uc.predict(x_test)
            pred_seconds = time.perf_counter() - t0
            intervals = uc.predict_interval(x_test, y_test)
            row = metric_dict(
                dataset_name,
                "UC-MatReg",
                y_test.to_numpy(),
                pred,
                train_seconds,
                pred_seconds,
            )
            row["seed"] = seed
            row["interval_coverage_90"] = intervals.coverage
            row["mean_interval_width"] = intervals.mean_width
            rows.append(row)

    audit = pd.DataFrame(rows)
    summary = (
        audit.groupby(["dataset", "model"], observed=True)
        .agg(
            n_splits=("seed", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            coverage_mean=("interval_coverage_90", "mean"),
            coverage_std=("interval_coverage_90", "std"),
            width_mean=("mean_interval_width", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "MAE_mean", "model"])
    )
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    audit, summary = run_audit(profile=args.profile, seeds=args.seeds)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(RESULTS_DIR / "split_robustness_audit.csv", index=False)
    summary.to_csv(RESULTS_DIR / "split_robustness_summary.csv", index=False)
    _write_latex_table(summary, PROJECT_ROOT / "manuscript")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
