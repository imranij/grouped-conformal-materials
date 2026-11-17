"""Summarize adaptive-vs-global conformal interval artifacts.

`run_uc_matreg_benchmark.py` writes `conformal_intervals.csv` with both the
locally adaptive UC-MatReg intervals and a constant-width global split-conformal
baseline for the same point model. This script is intentionally lightweight: it
does not refit models, and only regenerates the uncertainty-baseline CSV and
LaTeX table from the saved interval-level evidence.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from run_uc_matreg_benchmark import (
    ALPHA,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    _latex_table,
    load_datasets,
)


REQUIRED_COLUMNS = {
    "dataset",
    "covered",
    "interval_width",
    "global_covered",
    "global_interval_width",
}


def main() -> None:
    interval_path = RESULTS_DIR / "conformal_intervals.csv"
    intervals = pd.read_csv(interval_path)
    missing = REQUIRED_COLUMNS.difference(intervals.columns)
    if missing:
        raise ValueError(
            f"{interval_path} is missing columns {sorted(missing)}. "
            "Run code/run_uc_matreg_benchmark.py after the uncertainty-baseline patch."
        )

    rows: list[dict[str, float | str]] = []
    coverage_rows: list[dict[str, float | str]] = []
    for dataset, group in intervals.groupby("dataset", observed=True):
        adaptive_width = float(group["interval_width"].mean())
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "interval_method": "Adaptive",
                    "n_test": len(group),
                    "nominal_coverage": 1.0 - ALPHA,
                    "empirical_coverage": float(group["covered"].mean()),
                    "mean_interval_width": adaptive_width,
                    "width_vs_adaptive": 1.0,
                },
                {
                    "dataset": dataset,
                    "interval_method": "Global",
                    "n_test": len(group),
                    "nominal_coverage": 1.0 - ALPHA,
                    "empirical_coverage": float(group["global_covered"].mean()),
                    "mean_interval_width": float(group["global_interval_width"].mean()),
                    "width_vs_adaptive": float(group["global_interval_width"].mean()) / adaptive_width,
                },
            ]
        )
        coverage_rows.append(
            {
                "dataset": dataset,
                "n_test": len(group),
                "nominal_coverage": 1.0 - ALPHA,
                "empirical_coverage": float(group["covered"].mean()),
                "mean_interval_width": adaptive_width,
            }
        )

    comparison = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    shift = _recompute_shift_coverage(intervals)
    comparison.to_csv(RESULTS_DIR / "conformal_uncertainty_baselines.csv", index=False)
    coverage.to_csv(RESULTS_DIR / "conformal_coverage_summary.csv", index=False)
    shift.to_csv(RESULTS_DIR / "conformal_coverage_by_shift.csv", index=False)

    baseline_table = _latex_table(
        comparison,
        ["dataset", "interval_method", "n_test", "empirical_coverage", "mean_interval_width", "width_vs_adaptive"],
        ["Dataset", "Interval", "$n_{test}$", "Coverage", "Mean width", "Width/adaptive"],
        "Uncertainty-baseline comparison for the same UC-MatReg point model. "
        "Adaptive uses a learned local residual scale; Global uses a constant split-conformal residual quantile.",
        "tab:uncertainty_baselines",
    )
    coverage_table = _latex_table(
        coverage,
        ["dataset", "n_test", "nominal_coverage", "empirical_coverage", "mean_interval_width"],
        ["Dataset", "$n_{test}$", "Nominal", "Empirical", "Mean width"],
        "UC-MatReg 90\\% split-conformal coverage per dataset (held-out test set).",
        "tab:coverage",
    )
    shift_table = _latex_table(
        shift,
        ["dataset", "shift_bin", "n", "knn_distance_mean", "empirical_coverage", "mean_interval_width"],
        ["Dataset", "Shift bin", "$n$", "kNN dist.", "Coverage", "Mean width"],
        "UC-MatReg conformal coverage stratified by applicability-domain distance "
        "(Q1 in-domain $\\rightarrow$ Q5 far out-of-domain).",
        "tab:coverage_shift",
    )

    for target_dir in [RESULTS_DIR, PROJECT_ROOT / "manuscript"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "generated_uncertainty_baseline_table.tex").write_text(
            baseline_table, encoding="utf-8"
        )
        (target_dir / "generated_coverage_table.tex").write_text(
            coverage_table, encoding="utf-8"
        )
        (target_dir / "generated_coverage_shift_table.tex").write_text(
            shift_table, encoding="utf-8"
        )

    print(comparison.to_string(index=False))


def _recompute_shift_coverage(intervals: pd.DataFrame) -> pd.DataFrame:
    """Recompute kNN-distance coverage bins for the saved interval rows."""
    datasets = load_datasets()
    rows: list[pd.DataFrame] = []
    for dataset, (X, y) in datasets.items():
        x_train, x_test, _, _ = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )
        part = intervals[intervals["dataset"] == dataset].reset_index(drop=True)
        if len(part) != len(x_test):
            raise ValueError(
                f"Interval row count for {dataset} ({len(part)}) does not match "
                f"the deterministic test split ({len(x_test)})."
            )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        nn = NearestNeighbors(n_neighbors=min(10, len(x_train)), metric="euclidean")
        nn.fit(x_train_scaled)
        distances, _ = nn.kneighbors(x_test_scaled)
        shift = pd.DataFrame(
            {
                "knn_distance": distances.mean(axis=1),
                "covered": part["covered"].astype(float),
                "interval_width": part["interval_width"].astype(float),
                "abs_error": (part["y_true"] - part["prediction"]).abs(),
            }
        )
        try:
            shift["shift_bin"] = pd.qcut(
                shift["knn_distance"],
                q=5,
                labels=[f"Q{i}" for i in range(1, 6)],
                duplicates="drop",
            )
        except ValueError:
            shift["shift_bin"] = "Q1"
        summary = (
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
        summary.insert(0, "dataset", dataset)
        summary["nominal_coverage"] = 1.0 - ALPHA
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    main()
