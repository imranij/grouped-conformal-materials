"""Figure for the paper-07 revision: family coverage of the adaptive and constant-width intervals,
(a) under composition-grouped five-fold splits and (b) with each family held out of training and
calibration. Reads summary_grouped5_superconductivity.json and summary_lofo_superconductivity.json
(next_steps_2026-11-04/outputs). Point estimates and 95% cluster-bootstrap intervals over compositions.
Usage: python3 make_family_figure_07.py <outputs_dir> <figure_out.pdf>"""
import sys, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
out, fig_path = sys.argv[1], sys.argv[2]
g = json.load(open(f'{out}/summary_grouped5_superconductivity.json'))['pooled']['family_cov']
lofo = {u['tag'].split('_')[-1]: u for u in json.load(open(f'{out}/summary_lofo_superconductivity.json'))['per_unit']}
fams = ['Cuprate', 'Iron-based', 'MgB2-type', 'Other']; labels = ['Cuprate', 'Iron-based', 'MgB$_2$-type', 'Other']
A, C = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, title, get in [
    (axes[0], '(a) Grouped five-fold splits', lambda f, r: g[f'{r}_{f}']),
    (axes[1], '(b) Family held out of training', lambda f, r: lofo[f]['cov_a' if r == 'a' else 'cov_c'])]:
    for i, f in enumerate(fams):
        for dx, r, col, mk, name in [(-0.12, 'a', A, 'o', 'Adaptive (normalised)'), (0.12, 'c', C, 's', 'Constant width')]:
            est, lo, hi = get(f, r)
            ax.errorbar(i + dx, est, yerr=[[est - lo], [hi - est]], fmt=mk, color=col, ms=6, lw=1.5,
                        capsize=0, label=name if i == 0 else None, mfc=col if r == 'a' else 'white', mew=1.5)
    ax.axhline(0.90, color='#6b6a64', lw=1, ls='--')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_title(title, fontsize=9, loc='left')
    ax.set_ylim(0, 1.02); ax.grid(axis='y', color='#e4e3dd', lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Empirical coverage (nominal 0.90)')
axes[1].text(-0.4, 0.91, 'nominal 0.90', fontsize=8, color='#6b6a64', va='bottom', ha='left')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fig_path); print('wrote', fig_path)
