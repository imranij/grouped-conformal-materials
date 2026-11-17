"""Paper 07 next steps (4 Nov 2026), stage 2: scale model, conformal calibration, scoring.

Reads the stack pickles written by fit_stacks_07.py. For every outer unit it
  - fits the published residual-scale model g(x) (Extra Trees, 180 trees, sqrt, leaf 2, seed 42) on the
    residual set, floors sigma at 0.15 x median residual, calibrates the normalised score and the
    constant-width score on the calibration set (finite-sample level), and scores the test rows;
  - (superconductivity, grouped5 only) runs H3: nested selection of one of six pre-registered
    coverage-balanced variants of g on calibration-free development folds, then scores it once.
Every interval is resampled by independence unit (2,000 cluster-bootstrap draws).

Usage: python analyse_07.py <source_dir> <stack_dir> <out_dir> <scheme> <dataset> [budget_seconds]
  Checkpointed per outer unit: rerun until it prints DONE.
"""
import sys, types, json, math, pickle, glob
from pathlib import Path
for m in ("seaborn", "matplotlib", "matplotlib.pyplot"):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
sys.modules["matplotlib.pyplot"].Figure = object
SRC, STK, OUT = (Path(a).resolve() for a in sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True)
SCHEME, DS = sys.argv[4], sys.argv[5]
sys.path.insert(0, str(SRC / "code"))
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import run_uc_matreg_benchmark as B
B.RAW_DIR = SRC / "data" / "raw"
ALPHA = 0.10; R = 2000; NJ = 2
FAMS = ["Cuprate", "Iron-based", "MgB2-type", "Other"]
CONFIGS = {"base": dict(lam=0.0, ff=False, ct=None), "W0.5": dict(lam=0.5, ff=False, ct=None),
           "W1": dict(lam=1.0, ff=False, ct=None), "F": dict(lam=0.0, ff=True, ct=None),
           "F+W1": dict(lam=1.0, ff=True, ct=None), "CT0.5": dict(lam=0.0, ff=False, ct=0.5),
           "CT1": dict(lam=0.0, ff=False, ct=1.0)}


def load(ds):
    if ds == "concrete":
        X, y = B.load_concrete()
        raw = pd.read_excel(B.RAW_DIR / "concrete" / "Concrete_Data.xls")
        key = raw.iloc[:, :7].round(4).astype(str).agg("|".join, axis=1).values
        fam = np.array(["all"] * len(y))
    else:
        X, y = B.load_superconductivity()
        um = pd.read_csv(B.RAW_DIR / "superconductivity" / "unique_m.csv")
        key = um.drop(columns=["critical_temp", "material"]).round(6).astype(str).agg("|".join, axis=1).values
        fam = np.where((um.Cu > 0) & (um.O > 0) & (um.Fe == 0), "Cuprate",
              np.where((um.Fe > 0) & ((um.As > 0) | (um.Se > 0) | (um.P > 0) | (um.S > 0)), "Iron-based",
              np.where((um.Mg > 0) & (um.B > 0), "MgB2-type", "Other")))
    _, gid = np.unique(key, return_inverse=True)
    return X.reset_index(drop=True), y.to_numpy(float), gid, fam


def qlevel(scores):
    n = len(scores); lev = min(1.0, math.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(scores, lev, method="higher"))


def et():
    return ExtraTreesRegressor(n_estimators=180, max_features="sqrt", min_samples_leaf=2, n_jobs=NJ, random_state=42)


def feats(Xq, famq, ff):
    A = Xq.to_numpy(float)
    if ff:
        A = np.column_stack([A] + [(famq == f).astype(float) for f in FAMS])
    return A


def fit_sigma(cfg, Xtr, rtr, famtr, gidtr):
    floor = 0.15 * (np.median(rtr) + 1e-6)
    w = None
    if cfg["lam"] > 0:
        cnt = pd.Series(famtr).map(pd.Series(famtr).value_counts()).to_numpy(float)
        w = cnt ** (-cfg["lam"]); w = w / w.mean()
    m = et().fit(feats(Xtr, famtr, cfg["ff"]), rtr, sample_weight=w)
    c = {f: 1.0 for f in FAMS}
    if cfg["ct"] is not None:
        a, b = next(GroupShuffleSplit(1, test_size=0.5, random_state=1).split(rtr, groups=gidtr))
        mA = et().fit(feats(Xtr.iloc[a], famtr[a], False), rtr[a])
        s = rtr[b] / np.maximum(mA.predict(feats(Xtr.iloc[b], famtr[b], False)), floor)
        q = np.quantile(s, 1 - ALPHA)
        for f in FAMS:
            sf = s[famtr[b] == f]
            if len(sf) >= 30:
                c[f] = float((np.quantile(sf, 1 - ALPHA) / q) ** cfg["ct"])
    def sig(Xq, famq):
        base = np.maximum(m.predict(feats(Xq, famq, cfg["ff"])), floor)
        return base * np.array([c.get(f, 1.0) for f in famq])
    return sig, c


def fam_range(cov, famv):
    vals = [cov[famv == f].mean() for f in FAMS if (famv == f).any()]
    return max(vals) - min(vals), vals


def h3_select(X, y, fam, gid, P):
    fi = P["fit_i"]; r = np.abs(y[fi] - P["oof_pred"]); Xf = X.iloc[fi]; ff = fam[fi]; gf = gid[fi]
    perm = np.random.RandomState(7).permutation(gid.max() + 1)
    folds = list(GroupKFold(3).split(fi, groups=perm[gf]))
    res = {}
    for name, cfg in CONFIGS.items():
        C, Wd, Fm = [], [], []
        for tr, ho in folds:
            sig, _ = fit_sigma(cfg, Xf.iloc[tr], r[tr], ff[tr], gf[tr])
            a, b = next(GroupShuffleSplit(1, test_size=0.5, random_state=0).split(ho, groups=gf[ho]))
            dc, de = ho[a], ho[b]
            qh = qlevel(r[dc] / sig(Xf.iloc[dc], ff[dc])); half = qh * sig(Xf.iloc[de], ff[de])
            C.append(r[de] <= half); Wd.append(2 * half); Fm.append(ff[de])
        C, Wd, Fm = np.concatenate(C), np.concatenate(Wd), np.concatenate(Fm)
        rg, vals = fam_range(C.astype(float), Fm)
        res[name] = dict(cov=float(C.mean()), width=float(Wd.mean()), range=float(rg), fam_cov=[float(v) for v in vals])
    bw = res["base"]["width"]
    ok = [k for k in CONFIGS if k != "base" and res[k]["width"] <= 1.05 * bw and res[k]["cov"] >= 0.88]
    chosen = sorted(ok, key=lambda k: (res[k]["range"], res[k]["width"]))[0] if ok else None
    return chosen, res


def score_unit(X, y, fam, gid, P, do_h3):
    ri, ci, ti = P["res_i"], P["cal_i"], P["te_i"]
    rR = np.abs(y[ri] - P["p_res"]); rC = np.abs(y[ci] - P["p_cal"]); rT = np.abs(y[ti] - P["p_te"])
    sig, _ = fit_sigma(CONFIGS["base"], X.iloc[ri], rR, fam[ri], gid[ri])
    qa = qlevel(rC / sig(X.iloc[ci], fam[ci])); qc = qlevel(rC)
    sT = sig(X.iloc[ti], fam[ti])
    d = pd.DataFrame(dict(row=ti, gid=gid[ti], fam=fam[ti], y=y[ti], pred=P["p_te"], absr=rT, half_a=qa * sT,
                          half_c=qc, p_rival=P["p_rival"], p_rival_fitset=P["p_rival_fitset"]))
    d["cov_a"] = (rT <= d.half_a).astype(float); d["cov_c"] = (rT <= qc).astype(float)
    info = dict(tag=P["tag"], n_te=len(ti), n_cal=len(ci), n_res=len(ri), qhat_a=qa, qhat_c=qc,
                mean_abs_res_cal=float(rC.mean()), mean_abs_res_test=float(rT.mean()), weights=P["weights"])
    if do_h3:
        chosen, dev = h3_select(X, y, fam, gid, P)
        info["h3_dev"] = dev; info["h3_chosen"] = chosen
        if chosen is not None:
            s3, cf = fit_sigma(CONFIGS[chosen], X.iloc[ri], rR, fam[ri], gid[ri])
            q3 = qlevel(rC / s3(X.iloc[ci], fam[ci])); d["half_h3"] = q3 * s3(X.iloc[ti], fam[ti])
            d["cov_h3"] = (rT <= d.half_h3).astype(float); info["h3_c"] = cf
    return d, info


class CB:
    """Cluster bootstrap by unit, vectorised over per-unit sums."""
    def __init__(self, d, seed=0):
        self.u, self.inv = np.unique(d.gid.values, return_inverse=True)
        self.d = d; self.rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(self.rng.integers(0, len(self.u), len(self.u)), minlength=len(self.u))
                           for _ in range(R)])
    def sums(self, col):
        return np.bincount(self.inv, weights=col, minlength=len(self.u))
    def ci(self, fn):
        pt = fn(np.ones(len(self.u))[None, :])[0]; bs = fn(self.W)
        lo, hi = np.percentile(bs, [2.5, 97.5]); return [float(pt), float(lo), float(hi)]


def pooled(d, has_h3):
    cb = CB(d); n = cb.sums(np.ones(len(d)))
    S = {c: cb.sums(d[c].values) for c in ["cov_a", "cov_c", "half_a", "half_c", "absr"]}
    S["err_rival"] = cb.sums(np.abs(d.y - d.p_rival).values)
    mean = lambda c: (lambda W: (W @ S[c]) / (W @ n))
    out = dict(n=len(d), units=len(cb.u),
               cov_a=cb.ci(mean("cov_a")), cov_c=cb.ci(mean("cov_c")),
               cov_diff=cb.ci(lambda W: (W @ (S["cov_a"] - S["cov_c"])) / (W @ n)),
               width_a=cb.ci(lambda W: 2 * (W @ S["half_a"]) / (W @ n)),
               width_c=cb.ci(lambda W: 2 * (W @ S["half_c"]) / (W @ n)),
               width_ratio=cb.ci(lambda W: (W @ S["half_a"]) / (W @ S["half_c"])),
               mae_stack=cb.ci(mean("absr")), mae_rival=cb.ci(mean("err_rival")),
               mae_diff=cb.ci(lambda W: (W @ (S["absr"] - S["err_rival"])) / (W @ n)))
    if d.fam.iloc[0] != "all":
        fams = [f for f in FAMS if (d.fam == f).any()]
        fm = {f: cb.sums((d.fam == f).values.astype(float)) for f in fams}
        rules = ["a", "c"] + (["h3"] if has_h3 else [])
        if has_h3:
            S["cov_h3"] = cb.sums(d.cov_h3.values); S["half_h3"] = cb.sums(d.half_h3.values)
            out["cov_h3"] = cb.ci(mean("cov_h3"))
            out["width_h3_over_a"] = cb.ci(lambda W: (W @ S["half_h3"]) / (W @ S["half_a"]))
        fc = {}
        for rl in rules:
            Sf = {f: cb.sums(d[f"cov_{rl}"].values * (d.fam == f).values) for f in fams}
            for f in fams:
                fc[f"{rl}_{f}"] = cb.ci(lambda W, f=f: (W @ Sf[f]) / (W @ fm[f]))
            def rng_fn(W, Sf=Sf):
                M = np.stack([(W @ Sf[f]) / (W @ fm[f]) for f in fams]); return M.max(0) - M.min(0)
            out[f"range_{rl}"] = cb.ci(rng_fn)
            if rl == "h3":
                Sa = {f: cb.sums(d.cov_a.values * (d.fam == f).values) for f in fams}
                def dr(W, Sf=Sf, Sa=Sa):
                    M = np.stack([(W @ Sf[f]) / (W @ fm[f]) for f in fams]); A = np.stack([(W @ Sa[f]) / (W @ fm[f]) for f in fams])
                    return (M.max(0) - M.min(0)) - (A.max(0) - A.min(0))
                out["range_diff_h3_minus_a"] = cb.ci(dr)
        out["family_cov"] = fc
    return out


def hindsight(d):
    r = np.sort(d.absr.values); k = int(np.ceil(d.cov_a.mean() * len(r))); wc = 2 * r[k - 1]
    return dict(equal_cov_const_width=float(wc), adaptive_width=float(2 * d.half_a.mean()),
                ratio_adaptive_over_equalcov_const=float(2 * d.half_a.mean() / wc))


def main():
    X, y, gid, fam = load(DS)
    files = sorted(glob.glob(str(STK / f"stack_{SCHEME}_{DS}_*.pkl")))
    do_h3 = (SCHEME == "grouped5" and DS == "superconductivity")
    rows, infos, per = [], [], []
    import time
    t0 = time.perf_counter(); budget = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9
    cdir = OUT / "scored"; cdir.mkdir(exist_ok=True)
    for fpath in files:
        P = pickle.load(open(fpath, "rb"))
        cp = cdir / f"scored_{P['tag']}.pkl"
        if cp.exists():
            d, info = pickle.load(open(cp, "rb"))
        else:
            if time.perf_counter() - t0 > budget:
                return print("PARTIAL", flush=True)
            d, info = score_unit(X, y, fam, gid, P, do_h3); d["unit"] = P["tag"]
            pickle.dump((d, info), open(cp, "wb"))
        rows.append(d)
        hs = hindsight(d)
        pf = dict(tag=P["tag"], n=len(d), cov_a=d.cov_a.mean(), cov_c=d.cov_c.mean(), width_a=2 * d.half_a.mean(),
                  width_c=2 * d.half_c.mean(), ratio=d.half_a.mean() / d.half_c.mean(), mae=d.absr.mean(),
                  mae_rival=np.abs(d.y - d.p_rival).mean(), mae_rival_fitset=np.abs(d.y - d.p_rival_fitset).mean(),
                  scale_shift=abs(np.log(info["mean_abs_res_test"] / info["mean_abs_res_cal"])), **hs)
        if do_h3 and "cov_h3" in d:
            rg_a, va = fam_range(d.cov_a.values, d.fam.values); rg_3, v3 = fam_range(d.cov_h3.values, d.fam.values)
            rg_c, vc = fam_range(d.cov_c.values, d.fam.values)
            pf.update(h3_chosen=info["h3_chosen"], cov_h3=d.cov_h3.mean(), width_h3=2 * d.half_h3.mean(),
                      width_h3_over_a=d.half_h3.mean() / d.half_a.mean(), range_a=rg_a, range_c=rg_c, range_h3=rg_3,
                      famcov_a=[round(v, 4) for v in va], famcov_h3=[round(v, 4) for v in v3], famcov_c=[round(v, 4) for v in vc])
        if SCHEME == "lofo":
            pf.update(pooled(d, False))
        per.append(pf); infos.append(info)
        print(json.dumps({k: (v if not isinstance(v, float) else round(v, 4)) for k, v in pf.items()}, default=str), flush=True)
    allrows = pd.concat(rows, ignore_index=True)
    allrows.to_csv(OUT / f"rows_{SCHEME}_{DS}.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / f"perunit_{SCHEME}_{DS}.csv", index=False)
    summary = dict(scheme=SCHEME, ds=DS, per_unit=per, infos=infos)
    if SCHEME != "lofo":
        summary["pooled"] = pooled(allrows, do_h3 and "cov_h3" in allrows)
        summary["pooled_hindsight"] = hindsight(allrows)
        print(json.dumps(summary["pooled"], default=str, indent=1), flush=True)
    json.dump(summary, open(OUT / f"summary_{SCHEME}_{DS}.json", "w"), default=str, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
