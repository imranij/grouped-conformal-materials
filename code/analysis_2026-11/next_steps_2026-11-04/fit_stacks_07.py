"""Paper 07 next steps (4 Nov 2026): fit the frozen UC-MatReg stack on grouped or row-level splits.

Stage 1 of 2. Imports the published pipeline (source/code/run_uc_matreg_benchmark.py) unchanged
(base learners, profile, meta-learner) and changes only the splitters:
  outer split, inner fit/residual/calibration split (70/15/15) and the 5 out-of-fold stacking folds
  are made over independence units (mix design / composition) when grouped=True.
Frozen arm: the stack fitted on the fit set is the deployed predictor (no refit).
Saves, per outer fold, the stack predictions on the fit set (out-of-fold), residual set,
calibration set and test set, plus the best single learner refit on the whole outer training set.
Stage 2 (analyse_07.py) fits the residual-scale model g(x), calibrates and scores.

Usage: python fit_stacks_07.py <source_dir> <out_dir> <scheme> <dataset> [profile] [folds|all] [budget_seconds]
  Checkpointed: rerun the same command until it prints DONE.
  scheme: grouped5 | row5 | lofo | sanity
"""
import sys, types, time, json, math, pickle
from pathlib import Path
for m in ("seaborn", "matplotlib", "matplotlib.pyplot"):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
sys.modules["matplotlib.pyplot"].Figure = object

SRC = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC / "code"))
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold, train_test_split, GroupShuffleSplit
from scipy.optimize import nnls
import run_uc_matreg_benchmark as B

B.RAW_DIR = SRC / "data" / "raw"
NJ = 2
SEED = 42


def set_jobs(models):
    for m in models.values():
        p = m.get_params()
        if "model__n_jobs" in p:
            m.set_params(model__n_jobs=NJ)
    return models


def load(ds):
    if ds == "concrete":
        X, y = B.load_concrete()
        raw = pd.read_excel(B.RAW_DIR / "concrete" / "Concrete_Data.xls")
        key = raw.iloc[:, :7].round(4).astype(str).agg("|".join, axis=1).values
        fam = np.array(["all"] * len(y))
    else:
        X, y = B.load_superconductivity()
        um = pd.read_csv(B.RAW_DIR / "superconductivity" / "unique_m.csv")
        assert np.allclose(um.critical_temp.values, y.values)
        key = um.drop(columns=["critical_temp", "material"]).round(6).astype(str).agg("|".join, axis=1).values
        fam = np.where((um.Cu > 0) & (um.O > 0) & (um.Fe == 0), "Cuprate",
              np.where((um.Fe > 0) & ((um.As > 0) | (um.Se > 0) | (um.P > 0) | (um.S > 0)), "Iron-based",
              np.where((um.Mg > 0) & (um.B > 0), "MgB2-type", "Other")))
    _, gid = np.unique(key, return_inverse=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), gid, fam


def split_frac(idx, gid, frac, grouped, seed):
    """Split idx into (a, b) with b ~ frac of units (grouped) or rows."""
    if grouped:
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        a, b = next(gss.split(idx, groups=gid[idx]))
        return idx[a], idx[b]
    a, b = train_test_split(idx, test_size=frac, random_state=seed)
    return np.sort(a), np.sort(b)


def unit_split(X, y, gid, tr_idx, te_idx, grouped, tag):
    fp = OUT / f"unit_{tag}.npz"
    if fp.exists():
        z = np.load(fp, allow_pickle=True); return {k: z[k] for k in z.files}
    fit_i, temp = split_frac(tr_idx, gid, 0.30, grouped, SEED)
    res_i, cal_i = split_frac(temp, gid, 0.50, grouped, SEED)
    if grouped:
        perm = np.random.RandomState(SEED).permutation(gid.max() + 1)
        folds = list(GroupKFold(n_splits=5).split(fit_i, groups=perm[gid[fit_i]]))
    else:
        folds = list(KFold(5, shuffle=True, random_state=SEED).split(fit_i))
    fold_id = np.empty(len(fit_i), int)
    for k, (_, va) in enumerate(folds): fold_id[va] = k
    U = dict(tr=tr_idx, te=te_idx, fit_i=fit_i, res_i=res_i, cal_i=cal_i, fold_id=fold_id)
    np.savez(fp, **U); return U


def run_unit(X, y, gid, fam, tr_idx, te_idx, grouped, profile, ds, tag, budget):
    """Checkpointed: every (learner, fold) fit is one task saved to disk; stops after `budget` seconds."""
    t_start = time.perf_counter()
    if (OUT / f"stack_{tag}.pkl").exists():
        print(f"{tag} already assembled", flush=True); return True
    U = unit_split(X, y, gid, tr_idx, te_idx, grouped, tag)
    models = set_jobs(B.make_models(profile)); base = B.select_uc_base_models(models); names = list(base)
    rival = "XGBoost" if ds == "concrete" else "Extra trees"
    fit_i, fold_id = U["fit_i"], U["fold_id"]
    pred_rows = np.concatenate([U["res_i"], U["cal_i"], U["te"]])
    tdir = OUT / f"tasks_{tag}"; tdir.mkdir(exist_ok=True)
    tasks = [(n, k) for n in names for k in range(5)] + [(n, "final") for n in names] + [("rival", "full")]
    for n, k in tasks:
        fp = tdir / f"{n.replace(' ', '_')}__{k}.npy"
        if fp.exists():
            continue
        if time.perf_counter() - t_start > budget:
            print(f"{tag}: budget reached", flush=True); return False
        t0 = time.perf_counter()
        if n == "rival":
            m = clone(models[rival]).fit(X.iloc[U["tr"]], y.iloc[U["tr"]]); p = m.predict(X.iloc[U["te"]])
        elif k == "final":
            m = clone(base[n]).fit(X.iloc[fit_i], y.iloc[fit_i]); p = m.predict(X.iloc[pred_rows])
        else:
            tr, va = fit_i[fold_id != k], fit_i[fold_id == k]
            m = clone(base[n]).fit(X.iloc[tr], y.iloc[tr]); p = m.predict(X.iloc[va])
        np.save(fp, p)
        print(f"{tag} task {n}/{k} {time.perf_counter()-t0:.1f}s", flush=True)
    # assemble
    yf = y.iloc[fit_i].to_numpy()
    oof = np.zeros((len(fit_i), len(names)))
    for j, n in enumerate(names):
        for k in range(5):
            oof[fold_id == k, j] = np.load(tdir / f"{n.replace(' ', '_')}__{k}.npy")
    meta = LinearRegression(positive=True).fit(oof, yf)
    w = meta.coef_.astype(float); b = float(meta.intercept_)
    if w.sum() <= 1e-12:
        w, _ = nnls(oof, yf); b = 0.0
    M = np.column_stack([np.load(tdir / f"{n.replace(' ', '_')}__final.npy") for n in names])
    P = M @ w + b
    nr, nc = len(U["res_i"]), len(U["cal_i"])
    out = dict(tag=tag, ds=ds, grouped=grouped, profile=profile, fit_i=fit_i, res_i=U["res_i"], cal_i=U["cal_i"],
               te_i=U["te"], oof_pred=oof @ w + b, p_res=P[:nr], p_cal=P[nr:nr + nc], p_te=P[nr + nc:],
               p_rival=np.load(tdir / "rival__full.npy"), rival=rival,
               p_rival_fitset=M[nr + nc:, names.index(rival)], weights=dict(zip(names, w.round(4))), intercept=b)
    with open(OUT / f"stack_{tag}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(json.dumps(dict(tag=tag, n_fit=len(fit_i), n_res=nr, n_cal=nc, n_te=len(U["te"]),
                          weights={k: float(v) for k, v in out["weights"].items()}, intercept=b)), flush=True)
    return True


def outer_folds(gid, n, grouped, k=5):
    if grouped:
        ug = np.random.RandomState(SEED).permutation(gid.max() + 1)
        fold_of_unit = np.empty(gid.max() + 1, int); fold_of_unit[ug] = np.arange(len(ug)) % k
        f = fold_of_unit[gid]
    else:
        f = np.empty(n, int)
        for j, (_, te) in enumerate(KFold(k, shuffle=True, random_state=SEED).split(np.arange(n))): f[te] = j
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def sanity(X, y, ds, profile):
    """Published row-level refit and frozen pipeline on seed 42, for the version check."""
    sys.path.insert(0, str(SRC.parent / "source" / "analysis"))
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    x_tr, y_tr = x_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
    res = []
    for arm in ("refit", "frozen"):
        models = set_jobs(B.make_models(profile))
        uc = B.UCMatRegressor(B.select_uc_base_models(models), alpha=B.ALPHA)
        if arm == "frozen":
            orig = B.UCMatRegressor._fit_stack; calls = {"n": 0}
            def once(this, Xa, ya):
                calls["n"] += 1
                if calls["n"] == 1: orig(this, Xa, ya)
            B.UCMatRegressor._fit_stack = once
            try: uc.fit(x_tr, y_tr)
            finally: B.UCMatRegressor._fit_stack = orig
        else:
            uc.fit(x_tr, y_tr)
        iv = uc.predict_interval(x_te, y_te); gv = uc.predict_global_interval(x_te, y_te); p = uc.predict(x_te)
        r = dict(ds=ds, arm=arm, mae=float(np.mean(np.abs(p - y_te.to_numpy()))), cov_adapt=iv.coverage,
                 width_adapt=iv.mean_width, cov_const=gv.coverage, width_const=gv.mean_width)
        print(json.dumps(r), flush=True); res.append(r)
    pd.DataFrame(res).to_csv(OUT / f"sanity_{ds}.csv", index=False)


def main():
    scheme, ds = sys.argv[3], sys.argv[4]
    profile = sys.argv[5] if len(sys.argv) > 5 else "full"
    which = [int(s) for s in sys.argv[6].split(",")] if len(sys.argv) > 6 and sys.argv[6] != "all" else None
    budget = float(sys.argv[7]) if len(sys.argv) > 7 else 1e9
    X, y, gid, fam = load(ds)
    print(f"{ds}: rows {len(y)} units {gid.max()+1}", flush=True)
    if scheme == "sanity":
        return sanity(X, y, ds, profile)
    if scheme in ("grouped5", "row5"):
        g = scheme == "grouped5"
        for j, (tr, te) in enumerate(outer_folds(gid, len(y), g)):
            if which is None or j in which:
                if not run_unit(X, y, gid, fam, tr, te, g, profile, ds, f"{scheme}_{ds}_f{j}", budget):
                    return print("PARTIAL", flush=True)
    elif scheme == "lofo":
        fams = ["Cuprate", "Iron-based", "MgB2-type", "Other"]
        for j, F in enumerate(fams):
            if which is None or j in which:
                if not run_unit(X, y, gid, fam, np.where(fam != F)[0], np.where(fam == F)[0], True, profile, ds,
                                f"lofo_{ds}_{F}", budget):
                    return print("PARTIAL", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
