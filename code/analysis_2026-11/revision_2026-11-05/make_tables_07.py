"""Generate the LaTeX tables of the paper-07 revision (5 Nov 2026) from the outputs of
next_steps_2026-11-04 (summary_*.json, rows_*.csv). No number in these tables is typed by hand.
Usage: python3 make_tables_07.py <outputs_dir> <manuscript_dir>"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
out, man = sys.argv[1], sys.argv[2]
J = lambda s, d: json.load(open(f'{out}/summary_{s}_{d}.json'))
f3 = lambda x: f'{x:.3f}'
ci = lambda v, fmt='{:.3f}': f'{fmt.format(v[0])} [{fmt.format(v[1])}, {fmt.format(v[2])}]'
cis = lambda v: f'{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]'

# ---------------- Table: interval efficiency, grouped vs row-level control ----------------
lines = []
for ds, unit, lab in [('superconductivity', 'K', 'Superconductivity (15,309 compositions, 21,263 rows)'),
                      ('concrete', 'MPa', 'Concrete (428 mix designs, 1,030 rows)')]:
    cols = {}
    for s in ['grouped5', 'row5']:
        d = J(s, ds); p = d['pooled']; pu = d['per_unit']; h = d['pooled_hindsight']
        r = pd.read_csv(f'{out}/rows_{s}_{ds}.csv')
        rho = spearmanr(2 * r.half_a, r.absr)[0]
        ratios = [u['ratio'] for u in pu]; hr = [u['ratio_adaptive_over_equalcov_const'] for u in pu]
        cols[s] = [ci(p['cov_a']), ci(p['cov_c']), cis(p['cov_diff']),
                   ci(p['width_a'], '{:.2f}'), ci(p['width_c'], '{:.2f}'), ci(p['width_ratio']),
                   f'{min(ratios):.3f} to {max(ratios):.3f}',
                   f"{h['equal_cov_const_width']:.2f}", f"{h['ratio_adaptive_over_equalcov_const']:.3f}",
                   f'{min(hr):.3f} to {max(hr):.3f}', f'{rho:.3f}']
    names = ['Adaptive coverage', 'Constant-width coverage', 'Coverage difference (adaptive $-$ constant)',
             f'Adaptive mean width ({unit})', f'Constant mean width ({unit})', 'Width ratio (adaptive / constant)',
             'Width ratio, range over the five folds', f'Equal-coverage constant width ({unit})',
             'Adaptive / equal-coverage constant width', 'Same ratio, range over the five folds',
             'Spearman correlation, width and absolute error']
    lines.append(r'\midrule' + '\n' + r'\multicolumn{3}{@{}l}{\emph{' + lab + r'}} \\')
    for n, a, b in zip(names, cols['grouped5'], cols['row5']):
        lines.append(f'{n} & {a} & {b} \\\\')
tab = r'''\begin{table}[!tbp]
\centering
\caption{Adaptive (normalized) against constant-width split-conformal intervals on the same frozen
stack. \emph{Grouped}: every split, including the calibration split and the stacking folds, is made
over compositions (superconductivity) or mix designs (concrete). \emph{Row-level control}: the same
code with row splits. Five outer folds, pooled; brackets are 95\% cluster-bootstrap intervals over
units (2,000 draws). The equal-coverage constant width is the constant width that, chosen with
hindsight on the test rows, gives the adaptive interval's own coverage. Nominal coverage 0.90.}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{@{}Yll@{}}
\toprule
Measure & Grouped & Row-level control \\
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabularx}
\end{table}
'''
open(f'{man}/generated_main_table.tex', 'w').write(tab)

# ---------------- Table: leave one family out ----------------
L = J('lofo', 'superconductivity'); info = {i['tag']: i for i in L['infos']}
U = sorted(L['per_unit'], key=lambda u: -u['scale_shift'])
fam = lambda u: u['tag'].split('_')[-1].replace('MgB2', r'MgB$_2$')
iv = lambda v, sg=False: (f'[{v[1]:+.3f}, {v[2]:+.3f}]' if sg else f'[{v[1]:.3f}, {v[2]:.3f}]')
R = [('Rows', [f"{u['n']:,}" for u in U]), ('Compositions', [f"{u['units']:,}" for u in U]),
     ('Scale shift $s_F$', [f"{u['scale_shift']:.2f}" for u in U]),
     ('Adaptive coverage', [f"{u['cov_a'][0]:.3f}" for u in U]), (r'\quad 95\% interval', [iv(u['cov_a']) for u in U]),
     ('Constant-width coverage', [f"{u['cov_c'][0]:.3f}" for u in U]), (r'\quad 95\% interval', [iv(u['cov_c']) for u in U]),
     ('Difference (adaptive $-$ constant)', [f"{u['cov_diff'][0]:+.3f}" for u in U]), (r'\quad 95\% interval', [iv(u['cov_diff'], True) for u in U]),
     ('Width ratio (adaptive / constant)', [f"{u['ratio']:.2f}" for u in U]),
     ('MAE on the calibration set (K)', [f"{info[u['tag']]['mean_abs_res_cal']:.1f}" for u in U]),
     ('MAE on the held-out family (K)', [f"{u['mae']:.1f}" for u in U])]
rows = [f"{n} & " + ' & '.join(v) + r' \\' for n, v in R]
tab = r"""\begin{table}[!tbp]
\centering
\caption{Leave one chemistry family out (superconductivity), families ordered by scale shift. The
stack, the scale model and both conformal quantiles are fitted and calibrated on the other three
families with grouped inner splits, and every row of the held-out family is scored.
$s_F=|\log(\overline{|r|}_F/\overline{|r|}_{\mathrm{cal}})|$. Intervals are 95\% cluster-bootstrap
intervals over compositions. MAE is that of the frozen stack.}
\label{tab:lofo}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{@{}Yllll@{}}
\toprule
Held-out family & """ + ' & '.join(fam(u) for u in U) + r""" \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open(f'{man}/generated_lofo_table.tex', 'w').write(tab)

# ---------------- Table: point accuracy ----------------
rows = []
for ds, rival, unit in [('concrete', 'XGBoost', 'MPa'), ('superconductivity', 'Extra Trees', 'K')]:
    for sch, slab in [('grouped5', 'grouped'), ('row5', 'row-level control')]:
        p = J(sch, ds)['pooled']
        rows.append(f"{ds.capitalize()}, {slab} & {rival} & {p['mae_stack'][0]:.3f} & {p['mae_rival'][0]:.3f} & {p['mae_diff'][0]:+.3f} \\\\")
        rows.append(f"\\quad 95\\% interval ({unit}) & & {iv(p['mae_stack'])} & {iv(p['mae_rival'])} & {iv(p['mae_diff'], True)} \\\\")
tab = r"""\begin{table}[!tbp]
\centering
\caption{Point accuracy (MAE) of the frozen stack, which is fitted on the 70\% fitting subset,
against the best single learner refitted on the whole outer training set. Five outer folds pooled;
intervals are 95\% cluster-bootstrap intervals over units. A positive difference means the stack is
worse.}
\label{tab:point}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{@{}Yllll@{}}
\toprule
Dataset and split & Learner & Stack MAE & Learner MAE & Difference \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open(f'{man}/generated_point_table.tex', 'w').write(tab)

# ---------------- Table: coverage-balanced scale model, primary fold ----------------
f0 = [u for u in J('grouped5', 'superconductivity')['per_unit'] if u['tag'].endswith('_f0')][0]
pct = lambda v: ' & '.join(f'{x:.3f}' for x in v)
rng = lambda v: f'{100*(max(v)-min(v)):.1f}'
rows = [f"Constant width & {f0['cov_c']:.3f} & {f0['width_c']:.2f} & {pct(f0['famcov_c'])} & {rng(f0['famcov_c'])} \\\\",
        f"Adaptive, unweighted & {f0['cov_a']:.3f} & {f0['width_a']:.2f} & {pct(f0['famcov_a'])} & {rng(f0['famcov_a'])} \\\\",
        f"Adaptive, {f0['h3_chosen']} (selected) & {f0['cov_h3']:.3f} & {f0['width_h3']:.2f} & {pct(f0['famcov_h3'])} & {rng(f0['famcov_h3'])} \\\\"]
tab = r'''\begin{table}[!tbp]
\centering
\caption{Coverage-balanced scale model, primary test on outer grouped fold 0 of superconductivity
(''' + f"{f0['n']:,}" + r''' rows). Cu: cuprate; Fe: iron-based; MgB$_2$: MgB$_2$-type; Other: all remaining compounds. Width is the mean interval width.
The pre-registered pass rule required a family range below 4.8 points, a width at most 1.05 times
the unweighted width and marginal coverage of at least 0.88. The variant failed.}
\label{tab:h3}
\footnotesize
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & & & \multicolumn{4}{c}{Family coverage} & \\
\cmidrule(lr){4-7}
Rule & Coverage & Width (K) & Cu & Fe & MgB$_2$ & Other & Range (pts) \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}}
\end{table}
'''
open(f'{man}/generated_h3_table.tex', 'w').write(tab)
print('tables written')
