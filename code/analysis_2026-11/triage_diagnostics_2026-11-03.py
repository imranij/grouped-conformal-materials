"""Triage diagnostics for paper 07 (2026-11-03). Reads existing CSVs only; no training.
Reproduces the seed-42 80/20 split with numpy (sklearn ShuffleSplit logic) to recover the
independence unit: mix design (7 ingredients, age excluded) for concrete, element composition
(unique_m.csv) for superconductivity. Cluster bootstrap over those units, 2000 resamples."""
import math, sys, numpy as np, pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else '.'
R = 2000; rng = np.random.default_rng(0)
tp = pd.read_csv(f'{root}/data/results/test_predictions.csv')
ci = pd.read_csv(f'{root}/data/results/conformal_intervals.csv')
def split(n):
    nt = math.ceil(0.2*n); p = np.random.RandomState(42).permutation(n); return p[:nt], p[nt:]
def units(ds):
    if ds == 'concrete':
        c = pd.read_excel(f'{root}/data/raw/concrete/Concrete_Data.xls'); X = c.iloc[:, :7]
        key = X.round(4).astype(str).agg('|'.join, axis=1).values
    else:
        um = pd.read_csv(f'{root}/data/raw/superconductivity/unique_m.csv')
        key = um.drop(columns=['critical_temp', 'material']).round(6).astype(str).agg('|'.join, axis=1).values
    te, tr = split(len(key)); seen = np.isin(key[te], np.unique(key[tr])); return key[te], seen
def cboot(groups, stat, *arrs):
    ug, inv = np.unique(groups, return_inverse=True); idx = [np.where(inv == g)[0] for g in range(len(ug))]
    out = []
    for _ in range(R):
        s = np.concatenate([idx[g] for g in rng.integers(0, len(ug), len(ug))]); out.append(stat(*[a[s] for a in arrs]))
    return np.percentile(out, [2.5, 97.5])
for ds, rival in [('concrete', 'XGBoost'), ('superconductivity', 'Extra trees')]:
    t = tp[tp.dataset == ds]; c = ci[ci.dataset == ds]; g, seen = units(ds); y = t.y_true.values
    eu = np.abs(t['UC-MatReg'].values - y); er = np.abs(t[rival].values - y)
    print(f'\n== {ds}: n_test={len(y)}, units={len(np.unique(g))}, test rows whose unit is in train={seen.sum()} ({seen.mean():.3f})')
    d = eu - er; lo, hi = cboot(g, np.mean, d)
    print(f'MAE UC-MatReg {eu.mean():.3f} vs {rival} {er.mean():.3f}; paired diff {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]; UC better on {np.sum(d<0)}, worse on {np.sum(d>0)}')
    rm = lambda a, b: np.sqrt(np.mean(a**2)) - np.sqrt(np.mean(b**2)); lo, hi = cboot(g, rm, eu, er)
    print(f'RMSE diff {rm(eu,er):+.3f} [{lo:+.3f},{hi:+.3f}]')
    for lab, m in [('seen', seen), ('unseen', ~seen)]:
        print(f'  {lab} n={m.sum()}: MAE UC {eu[m].mean():.3f} {rival} {er[m].mean():.3f}; cov adaptive {c.covered.values[m].mean():.3f} global {c.global_covered.values[m].mean():.3f}')
    wa = c.interval_width.values; wg = c.global_interval_width.values; ca = c.covered.values.astype(float); cg = c.global_covered.values.astype(float)
    ratio = lambda a, b: a.mean()/b.mean(); lo, hi = cboot(g, ratio, wa, wg)
    print(f'width adaptive/global {ratio(wa,wg):.3f} [{lo:.3f},{hi:.3f}] (test-unit resampling only; calibration draw not resampled)')
    dc = ca - cg; lo, hi = cboot(g, np.mean, dc)
    print(f'coverage adaptive-global {dc.mean():+.4f} [{lo:+.4f},{hi:+.4f}]; adaptive-only covered {np.sum((ca==1)&(cg==0))}, global-only covered {np.sum((ca==0)&(cg==1))}')
    r = np.abs(c.prediction.values - c.y_true.values); half = wa/2
    # hindsight diagnostic: constant width that gives the adaptive arm's test coverage
    k = int(np.ceil(ca.mean()*len(r))); wc = 2*np.sort(r)[k-1]
    print(f'HINDSIGHT equal-coverage constant width {wc:.2f} vs adaptive mean {wa.mean():.2f}: ratio {wa.mean()/wc:.3f}')
    # unseen units coverage exact CI
    from math import comb
    m = ~seen; k2 = int(ca[m].sum()); n2 = int(m.sum())
    print(f'unseen-unit adaptive coverage {k2}/{n2}')
