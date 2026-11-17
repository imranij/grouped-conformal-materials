"""Derived numbers for the paper-07 revision (5 Nov 2026). Reads the per-row outputs of
next_steps_2026-11-04/outputs/rows_*.csv only; no model is trained. Prints:
 - row and unit counts per scheme and dataset,
 - Spearman correlation between adaptive interval width (2*half_a) and absolute error,
 - pooled and per-family mean widths for both rules,
 - per-family row and unit counts.
Usage: python3 derived_numbers_07.py <outputs_dir>"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
out = sys.argv[1]
for scheme, ds in [('grouped5','concrete'),('row5','concrete'),('grouped5','superconductivity'),
                   ('row5','superconductivity'),('lofo','superconductivity')]:
    r = pd.read_csv(f'{out}/rows_{scheme}_{ds}.csv')
    wa, wc = 2*r.half_a, 2*r.half_c
    rho, p = spearmanr(wa, r.absr)
    print(f'== {scheme} {ds}: rows={len(r)} units={r.gid.nunique()} folds={r.unit.nunique()}')
    print(f'  cov_a={r.cov_a.mean():.4f} cov_c={r.cov_c.mean():.4f} width_a={wa.mean():.3f} width_c={wc.mean():.3f} ratio={wa.mean()/wc.mean():.4f}')
    print(f'  Spearman(width_a, |err|) pooled = {rho:.3f} (p={p:.2g})')
    fr = []
    for u, g in r.groupby('unit'):
        fr.append(spearmanr(2*g.half_a, g.absr)[0])
    print(f'  Spearman per fold/unit: min {min(fr):.3f} max {max(fr):.3f}')
    print(f'  MAE stack={r.absr.mean():.3f} rival={np.abs(r.p_rival-r.y).mean():.3f} rival_fitset={np.abs(r.p_rival_fitset-r.y).mean():.3f}')
    if ds == 'superconductivity':
        for f, g in r.groupby('fam'):
            print(f'  fam {f}: rows={len(g)} units={g.gid.nunique()} cov_a={g.cov_a.mean():.3f} cov_c={g.cov_c.mean():.3f} '
                  f'width_a={2*g.half_a.mean():.2f} width_c={2*g.half_c.mean():.2f} MAE={g.absr.mean():.2f} '
                  f'spearman={spearmanr(2*g.half_a, g.absr)[0]:.3f}')
