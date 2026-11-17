# Pre-registration, paper 07 (UC-MatReg), next steps of 4 November 2026

Written before any model in this folder was trained. It is not changed after results are seen.
Nominal level 0.90 (alpha = 0.10) throughout. All runs use the frozen arm, so the predictor that
produced the calibration residuals is the deployed predictor.

## Common protocol

- Independence unit. Concrete: the mix design, the seven ingredient amounts with curing age
  excluded (428 units in 1,030 rows). Superconductivity: the element composition from
  `unique_m.csv` (15,309 units in 21,263 rows).
- Grouped pipeline. Every split is made over units, not rows: the outer split, the inner
  fit/residual/calibration split (70/15/15 of units), and the five out-of-fold stacking folds.
  The base learners, the non-negative meta-learner and the residual-scale model g(x) are those of
  `source/code/run_uc_matreg_benchmark.py` (profile "full"), imported unchanged. Only the splitters
  differ.
- Row-level control. The same code is run with row-level splits, so that the effect of grouping is
  measured within one implementation.
- Outer design for H1. Five outer folds over units (seed 42), so each row is tested once and five
  calibration draws are seen. Pooled test rows are resampled by unit (2,000 cluster-bootstrap
  draws). Fold-to-fold spread is reported as the calibration-draw variability.
- Library versions are fixed to `source/code/requirements-lock.txt` (scikit-learn 1.6.1, numpy
  2.1.3, pandas 2.2.3, scipy 1.15.3, xgboost 3.1.3), in a separate virtual environment, and the
  published row-level refit run on concrete (seed 42) is reproduced first as an implementation check.
- One attempt per experiment. No setting is changed after its result is seen. If a run fails for a
  technical reason, the fix and the rerun are reported.

## H1. Grouped splits

Prediction (from the triage memo). Concrete new-mix coverage is about 0.84 for both rules.
Superconductivity adaptive-to-constant width ratio stays below 0.85.

Decision rules.
- Concrete coverage prediction is confirmed if the pooled grouped coverage of both rules lies in
  [0.80, 0.88]; it is refuted if either is at or above 0.88.
- Concrete width claim is retained only if the pooled width ratio has a 95% upper bound below 1,
  the coverage difference (adaptive minus constant) has a lower bound above -0.02, and the
  equal-coverage (hindsight) constant width is wider than the adaptive mean width. Otherwise the
  concrete width claim is dropped.
- The superconductivity win survives grouped splits only if all three hold: (i) pooled width ratio
  below 0.85 with a 95% upper bound below 0.90; (ii) coverage difference, adaptive minus constant,
  with a 95% lower bound above -0.02; (iii) the equal-coverage (hindsight) constant width is at
  least 10% wider than the adaptive mean width. If (i) fails, the positive route of the paper fails.

## H4. Leave-one-chemistry-family-out (superconductivity)

Families as in `source/code/shift_groups.yaml` (Cuprate, Iron-based, MgB2-type, Other). For each
family F: train and calibrate on the other three families (grouped inner splits), test on all of F.

Predictions. (a) Marginal coverage below 0.90 for both rules on every held-out family (8 point
estimates). (b) The adaptive rule falls less where the family's error scale is unusual. Scale
shift is s_F = |log(mean |residual| on F / mean |residual| on the calibration set)|. For the family
with the largest s_F, adaptive coverage minus constant coverage is positive with a 95%
composition-bootstrap interval excluding 0. Spearman correlation over the four families between
s_F and the coverage difference is reported as descriptive only (n = 4).

Rules. (a) is confirmed if all 8 coverages are below 0.90. (b) is confirmed under the interval rule
above.

## H3. Coverage-balanced scale model (methodological contribution)

Six pre-specified variants of g(x), one grid, no further attempts:
- W0.5, W1: residual-scale Extra Trees fitted with sample weights proportional to n_family^-lambda
  (lambda 0.5, 1), normalised to mean 1 (a group-weighted loss).
- F: family one-hot indicators appended to the inputs of g.
- F+W1: family indicators plus lambda = 1 weights.
- CT0.5, CT1: coverage-targeted family rescaling, sigma(x) = g(x) * c_f with
  c_f = (q_f / q)^eta, where q_f and q are the 0.9 quantiles of the normalised scores in family f
  and overall, estimated on a held-out half (by composition) of the data g is trained on
  (eta 0.5, 1). A single marginal conformal quantile is kept, so the marginal guarantee holds.

Tuning on calibration-free development folds. Inside each outer training set, the development pool
is the fit set with its grouped out-of-fold stack residuals; the residual set, the calibration set
and the outer test fold are not touched. Three grouped development folds: g is trained on two, and
the third is split by composition into a development calibration half and a development
evaluation half. Selection: among variants whose pooled development width is at most 1.05 times
the unweighted baseline's and whose development coverage is at least 0.88, choose the smallest
family coverage range (max minus min over the four families); ties go to the narrower variant.

Scoring. The selected variant is trained on the residual set, calibrated on the calibration set and
scored once on the held-out fifth. Primary: outer grouped fold 0. Secondary: all five outer folds
pooled, each with its own nested selection.

Decision rule (primary). PASS if, on fold 0, the family coverage range of the selected variant is
below 4.8 points and below that of the unweighted adaptive interval on the same fold, its mean width
is at most 1.05 times the unweighted adaptive width, and its marginal coverage is at least 0.88.
ROBUST if, in addition, the pooled five-fold composition-bootstrap 95% interval for the range
difference (variant minus unweighted) lies below 0.

## Compute

Two CPUs, n_jobs = 2. The "full" profile is kept unless one superconductivity fold takes more than
eight minutes, in which case the "fast" profile of the same code is used for every run and
declared. This choice is made on timing only, before any coverage or width is seen.
