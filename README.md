# Grouped-split evaluation of locally adaptive conformal intervals on tabular materials data

Code and result records for the manuscript *When Do Locally Adaptive Conformal Intervals Help? A
Grouped-Split and Chemistry-Family-Shift Evaluation on Tabular Materials Data* (Imran, King Fahd
University of Petroleum and Minerals), under revision at *Engineering Applications of Artificial
Intelligence*.

Archived copy with intermediate files: https://doi.org/10.5281/zenodo.24311174

## What the study does

Normalized-residual split-conformal intervals around a non-negative stacked regressor (UC-MatReg)
are evaluated on UCI superconducting critical temperature (21,263 rows, 15,309 element
compositions) and UCI concrete compressive strength (1,030 rows, 428 mix designs). Every split,
including calibration and stacking, is made over compositions or mix designs, the calibrated
predictor is the deployed one, and every uncertainty interval is a cluster bootstrap over those
units. A leave-one-chemistry-family-out experiment tests coverage under family shift. The decision
rules were fixed before any model was trained (`docs/PREREGISTRATION_07.md`).

## Layout

| Path | Contents |
|---|---|
| `code/run_uc_matreg_benchmark.py` | Original UC-MatReg benchmark (base learners, non-negative stack, residual-scale model). Imported unchanged by the grouped-split scripts. |
| `code/run_split_robustness_audit.py`, `code/audit_uncertainty_baselines.py` | Audits of the original (row-split) release. |
| `code/analysis_2026-11/next_steps_2026-11-04/fit_stacks_07.py` | Fits the frozen stack on grouped (`grouped5`), row-level (`row5`) or leave-one-family-out (`lofo`) splits. |
| `code/analysis_2026-11/next_steps_2026-11-04/analyse_07.py` | Fits the scale model, calibrates, scores and runs the cluster bootstrap. |
| `code/analysis_2026-11/revision_2026-11-05/` | Scripts that write every table and the figure of the manuscript from the outputs, and the derived numbers quoted in the text. |
| `code/analysis_2026-11/triage_diagnostics_2026-11-03.py` | Re-analysis of the original row-split outputs (corrected comparison, Section 5.7 and Table 6). |
| `results/next_steps_2026-11-04/outputs/` | Per-row outputs (`rows_*.csv`), per-fold summaries (`perunit_*.csv`), pooled summaries with bootstrap intervals (`summary_*.json`), sanity checks and library versions. |
| `results/revision_2026-11-05/` | Printed output of the derived-number and triage scripts. |
| `data/results/` | Outputs of the original September 2026 row-split release, used only for the corrected comparison. |
| `docs/PREREGISTRATION_07.md` | Pre-registration written before the grouped runs. |

The fitted stacks (`*.pkl`), task arrays and per-unit arrays (`*.npz`) that `analyse_07.py`
reads are not in this repository because of their size. They are in the Zenodo archive.

## Data

The two UCI datasets are not redistributed. `run_uc_matreg_benchmark.py` downloads them into
`data/raw/` on first use:

- Concrete compressive strength: https://doi.org/10.24432/C5PK67
- Superconductivity data: https://doi.org/10.24432/C53P47

## Environment

Python 3.10 with the versions in `code/requirements-lock.txt` (scikit-learn 1.6.1, numpy 2.1.3,
pandas 2.2.3, scipy 1.15.3, xgboost 3.1.3). The results depend on these versions. Runs used
`OMP_NUM_THREADS=2` and two worker processes, about one minute per outer fold.

```
pip install -r code/requirements-lock.txt
```

## Reproducing the results

Run from the repository root. `<scheme>` is `grouped5`, `row5` or `lofo`; `<dataset>` is `concrete`
or `superconductivity` (`lofo` is defined for superconductivity only).

```
python code/analysis_2026-11/next_steps_2026-11-04/fit_stacks_07.py . results/next_steps_2026-11-04/outputs <scheme> <dataset> full all
python code/analysis_2026-11/next_steps_2026-11-04/analyse_07.py   . results/next_steps_2026-11-04/outputs results/next_steps_2026-11-04/outputs <scheme> <dataset>
```

Tables, figure and derived numbers from the outputs (no training):

```
python code/analysis_2026-11/revision_2026-11-05/make_tables_07.py results/next_steps_2026-11-04/outputs <tables_dir>
python code/analysis_2026-11/revision_2026-11-05/make_family_figure_07.py results/next_steps_2026-11-04/outputs <figure.pdf>
python code/analysis_2026-11/revision_2026-11-05/derived_numbers_07.py results/next_steps_2026-11-04/outputs
python code/analysis_2026-11/triage_diagnostics_2026-11-03.py .
```

`make_tables_07.py` run on the committed outputs reproduces the four tables of the manuscript
byte for byte. Rerunning the fits overwrites the committed outputs, so write to another folder
when checking.

## Main results

- Superconductivity: the adaptive interval is 16% narrower than constant-width split conformal on
  the same point model (width ratio 0.838, 95% interval 0.822 to 0.853) at coverage of 0.894
  against 0.898.
- Concrete: no width gain once coverage is matched (ratio 0.981).
- Holding one chemistry family out of training and calibration drops coverage to between 0.12 and
  0.72 for both interval rules.
- A family-balanced scale model narrowed the in-distribution family coverage gap by 1.5 points and
  failed its pre-registered target.
- The stack is not more accurate than Extra Trees or XGBoost under grouped splits.

## Citation

Imran. Code, result records and table scripts for "When Do Locally Adaptive Conformal Intervals
Help? A Grouped-Split and Chemistry-Family-Shift Evaluation on Tabular Materials Data". Zenodo,
2026. https://doi.org/10.5281/zenodo.24311174

## Licence

MIT, see `LICENSE`.
