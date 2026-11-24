"""Analysis of the 2026-11-09 Heisenberg runs (no fits anywhere).

Outputs (../results/):
  local.csv      x=0 quantities per run: chi~ = xi_ss/(2 beta), r, G(tau)/Var, Var, rebin factors, drift z
  local_comb.csv per (L,T): runs with the same (L,T) combined (inverse-variance, errors inflated by rebin>1)
  finite.csv     grid-only split-sample chi_delta per (L,T,eps), plus the lower-confidence-limit (LCL) variant
  mfg.csv        <sigma^z_0>(beta x) per (L,T,beta x) (perturbed MFG states)
  runs_table.md  per-run log table for RUNS.md
  analysis.pkl   everything above as python objects

Definitions
  chi~ = xi_ss/(2 beta), xi_ss = A.(X - beta p p^T).A with A = (-1, 1)  [draft Eq. (9) + case (iii), b=0]
  r = xi_ss/(beta Var), floor chi~ >= G(beta/2)/(2) with G normalised by Var   [note Eq. (4)]
  rebin factor = jackknife error from 8 merged bins / error from 40 bins
  drift z = (mean of bins 1-10 - mean of bins 31-40)/sqrt(err1^2+err2^2)
  chi_delta (draft Eq. 28) evaluated only on pairs of simulated fields, |beta x| <= 2 pi, eps in units of pi/4.
    split-sample: pair chosen on bins 1-20, value evaluated on bins 21-40, and vice versa; average.
    LCL variant: the pair maximising (value - 1.645 sigma) on the selection half (one-sided 95% LCL).
    Errors: bootstrap over bins (300 resamples, within each half) of the WHOLE procedure (selection included).
"""
import os, sys, glob, json, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'sse'))
from runlib import load, TAUS
RES = os.path.join(HERE, '..', '..', '..', 'data', 'heisenberg')
RAW = os.path.join(RES, 'raw')
A = np.array([-1.0, 1.0])
EPSMAX = np.pi / 4
EPSF = [0.01, 0.05, 0.1, 0.2]
BD = 2 * np.pi
Z_LCL = 1.645
NBOOT = 300


def jk(bins, f):
    """delete-one jackknife of f over the first axis of every array in dict bins"""
    nb = len(next(iter(bins.values())))
    full = f({k: v.mean(0) for k, v in bins.items()})
    s = {k: v.sum(0) for k, v in bins.items()}
    js = np.array([f({k: (s[k] - v[i]) / (nb - 1) for k, v in bins.items()}) for i in range(nb)])
    return full, np.sqrt((nb - 1) / nb * ((js - js.mean(0)) ** 2).sum(0))


def merge(bins, k):
    return {key: v[:len(v) // k * k].reshape(len(v) // k, k, *v.shape[1:]).mean(1) for key, v in bins.items()}


def local_f(beta):
    def f(m):
        P, X, G = m['P'], m['X'], m['G']
        mean = P @ A; var = P @ A ** 2 - mean ** 2
        xi = A @ (X - beta * np.outer(P, P)) @ A
        return np.concatenate([[xi / (2 * beta), xi / (beta * var), var, mean], (G - mean ** 2) / var])
    return f
# indices into local_f output
I_CHI, I_R, I_VAR, I_M, I_G0 = 0, 1, 2, 3, 4
I_GH = I_G0 + len(TAUS) - 1   # G(beta/2)/Var


def run_meta():
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, '*.json')))]


def local_one(meta):
    beta = meta['beta']
    r = load(os.path.join(RAW, meta['tag'] + '.out'))
    bins = {'P': r['P'], 'X': r['X'], 'G': r['G']}
    f = local_f(beta)
    v, e = jk(bins, f)
    _, e8 = jk(merge(bins, 5), f)
    nb = len(r['P'])
    a, b = {k: x[:10] for k, x in bins.items()}, {k: x[-10:] for k, x in bins.items()}
    (va, ea), (vb, eb) = jk(a, f), jk(b, f)
    with np.errstate(divide='ignore', invalid='ignore'):
        drift = (va - vb) / np.sqrt(ea ** 2 + eb ** 2)
        rebin = e8 / e
    return dict(v=v, e=e, rebin=rebin, drift=drift, nb=nb, bins=bins)


def bures_y(P):
    """y = -arcsin <sigma^z>; Bures angle between diagonal qubit states = |y1 - y2|/2"""
    m = P[..., 1] - P[..., 0]
    return -np.arcsin(np.clip(m, -1, 1))


def chi_matrix(u, y, eps):
    du = np.abs(u[:, None] - u[None, :]); du[du == 0] = np.inf
    return (np.abs(y[:, None] - y[None, :]) - 4 * eps) / du


def finite_one(u, Pb, eps):
    """u: beta x grid (symmetric, sorted); Pb: (nu, nb, 2) per-bin populations.
    returns dict(split=(val, err), lcl=(val, err), pairs_split, pairs_lcl)"""
    nb = Pb.shape[1]
    idx = np.arange(nb)

    def procedure_idx(keep, mode):
        return procedure(keep, mode)[0]

    def procedure(keep, mode):
        halves = [keep[keep < nb // 2], keep[keep >= nb // 2]]
        vals, pairs = [], []
        for a, b in ((0, 1), (1, 0)):
            sel, ev = halves[a], halves[b]
            C = chi_matrix(u, bures_y(Pb[:, sel].mean(1)), eps)
            if mode == 'lcl':
                n = len(sel); s = Pb[:, sel].sum(1)
                Cj = np.array([chi_matrix(u, bures_y((s - Pb[:, k]) / (n - 1)), eps) for k in sel])  # k repeated is fine
                sig = np.sqrt((n - 1) / n * ((Cj - Cj.mean(0)) ** 2).sum(0))
                C = C - Z_LCL * sig
            i, j = np.unravel_index(np.argmax(C), C.shape)
            vals.append(chi_matrix(u, bures_y(Pb[:, ev].mean(1)), eps)[i, j]); pairs.append((u[i], u[j]))
        return np.mean(vals), pairs

    out = {}
    rng = np.random.default_rng(20261109)
    for mode in ('split', 'lcl'):
        full, pairs = procedure(idx, mode)
        # bootstrap over bins, resampling within each half, whole procedure (incl. selection) redone.
        # (delete-one jackknife is inconsistent for argmax-type statistics, so it is not used here)
        h0, h1 = idx[:nb // 2], idx[nb // 2:]
        bs = []
        for _ in range(NBOOT):
            keep = np.concatenate([rng.choice(h0, len(h0)), rng.choice(h1, len(h1))])
            bs.append(procedure_idx(keep, mode))
        out[mode] = (full, float(np.std(bs, ddof=1))); out['pairs_' + mode] = pairs
        out['boot_' + mode] = np.array(bs)
    return out


def combine(vals, errs):
    w = 1 / np.asarray(errs) ** 2
    m = (np.asarray(vals) * w).sum() / w.sum()
    chi2 = (((np.asarray(vals) - m) / np.asarray(errs)) ** 2).sum()
    return m, w.sum() ** -0.5, chi2 / max(len(vals) - 1, 1)


def main():
    metas = run_meta()
    loc, mfg, cand = {}, {}, {}
    rows_md = ['| tag | L | k_BT/J | βx | ⟨n⟩ ops | wall [s] | sweeps | rebin χ̃ | rebin G(β/2) | drift z χ̃ | drift z G(β/2) |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in metas:
        res = local_one(m)
        key = (m['L'], round(m['T'], 6), round(m['betax'], 4), m['tag'].split('_s')[-1])
        loc[key] = dict(meta=m, **{k: res[k] for k in ('v', 'e', 'rebin', 'drift', 'nb')})
        cand.setdefault((m['L'], round(m['T'], 6), round(m['betax'], 4)), []).append((m['tag'], res['bins']['P']))
        rows_md.append(f"| {m['tag']} | {m['L']} | {m['T']:g} | {m['betax']:.3f} | {m['nops']:.0f} | {m['wall_s']:.0f} | "
                       f"{m['therm']}+{m['nbins']}×{m['spb']} | {res['rebin'][I_CHI]:.2f} | {res['rebin'][I_GH]:.2f} | "
                       f"{res['drift'][I_CHI]:+.2f} | {res['drift'][I_GH]:+.2f} |")
    # field grid for the finite-accuracy bound: one run per (L, T, beta x).
    # Rule: the longer rerun ('...long', made because the first run had rebin > 1.3) if it exists, else seed 0.
    used = {}
    for (L, T, ux), lst in cand.items():
        lst = sorted(lst, key=lambda t: (not t[0].endswith('long'), not t[0].endswith('_s0'), t[0]))
        mfg.setdefault((L, T), {})[ux] = lst[0][1]; used[(L, T, ux)] = lst[0][0]
    # ---- x = 0 table
    hdr = 'L,T,seed,chi,chi_err,r,r_err,GhalfVar,GhalfVar_err,floor,floor_err,var,mz,mz_err,rebin_chi,rebin_G,drift_chi,drift_G,nops,wall_s'
    lines = [hdr]
    comb = {}
    for key in sorted(k for k in loc if k[2] == 0.0):
        d = loc[key]; v, e = d['v'], d['e']
        lines.append(','.join(map(str, [key[0], key[1], key[3], v[I_CHI], e[I_CHI], v[I_R], e[I_R], v[I_GH], e[I_GH],
                                        v[I_GH] / 2, e[I_GH] / 2, v[I_VAR], v[I_M], e[I_M], d['rebin'][I_CHI],
                                        d['rebin'][I_GH], d['drift'][I_CHI], d['drift'][I_GH], d['meta']['nops'], d['meta']['wall_s']])))
        comb.setdefault(key[:2], []).append(d)
    open(os.path.join(RES, 'local.csv'), 'w').write('\n'.join(lines) + '\n')
    lc = ['L,T,nruns,chi,chi_err,GhalfVar,GhalfVar_err,r,r_err,floor,floor_err,chi2dof_chi,' + ','.join(f'G{t:.4g}' for t in TAUS) + ',' + ','.join(f'G{t:.4g}_err' for t in TAUS)]
    combd = {}
    for k in sorted(comb):
        ds = comb[k]
        out = {}
        for nm, i in (('chi', I_CHI), ('G', I_GH), ('r', I_R)):
            # inflate each run's error by its rebinning factor when > 1 (conservative)
            m_, e_, c2 = combine([d['v'][i] for d in ds], [d['e'][i] * max(1.0, d['rebin'][i]) for d in ds])
            out[nm] = (m_, e_ * np.sqrt(max(1.0, c2)), c2)      # PDG scale factor if seeds disagree
        Gt = [(1.0, 0.0)] + [combine([d['v'][I_G0 + j] for d in ds], [d['e'][I_G0 + j] * max(1.0, d['rebin'][I_G0 + j]) for d in ds])[:2] for j in range(1, len(TAUS))]
        combd[k] = dict(n=len(ds), chi=out['chi'], G=out['G'], r=out['r'], Gt=np.array(Gt))
        lc.append(','.join(map(str, [k[0], k[1], len(ds), *out['chi'][:2], *out['G'][:2], *out['r'][:2], out['G'][0] / 2, out['G'][1] / 2,
                                     out['chi'][2]] + [g[0] for g in Gt] + [g[1] for g in Gt])))
    open(os.path.join(RES, 'local_comb.csv'), 'w').write('\n'.join(lc) + '\n')
    # ---- MFG states and finite accuracy
    fin, fl = {}, ['L,T,eps_over_epsmax,chi_split,err_split,pairs_split,chi_lcl,err_lcl,pairs_lcl,ngrid']
    ml = ['L,T,betax,mz,mz_err']
    for (L, T), dd in sorted(mfg.items()):
        up = np.array(sorted(dd))
        for u_ in up:
            m = dd[u_][:, 1] - dd[u_][:, 0]
            ml.append(f'{L},{T},{u_},{m.mean()},{m.std(ddof=1)/np.sqrt(len(m))}')
        if len(up) < 5 or up[0] != 0.0: continue
        nb = min(len(dd[u_]) for u_ in up)
        pb = np.array([dd[u_][:nb] for u_ in up])                       # (nu+, nb, 2)
        u = np.concatenate([-up[::-1][:-1], up])
        Pb = np.concatenate([pb[::-1][:-1][..., ::-1], pb], 0)          # exact spin flip: p(-x) = reversed p(x)
        for ef in EPSF:
            r = finite_one(u, Pb, ef * EPSMAX)
            fin[(L, T, ef)] = r
            fl.append(f"{L},{T},{ef},{r['split'][0]},{r['split'][1]},\"{[tuple(map(float, np.round(p, 3))) for p in r['pairs_split']]}\","
                      f"{r['lcl'][0]},{r['lcl'][1]},\"{[tuple(map(float, np.round(p, 3))) for p in r['pairs_lcl']]}\",{len(u)}")
    open(os.path.join(RES, 'finite.csv'), 'w').write('\n'.join(fl) + '\n')
    open(os.path.join(RES, 'mfg.csv'), 'w').write('\n'.join(ml) + '\n')
    open(os.path.join(RES, 'runs_table.md'), 'w').write('\n'.join(rows_md) + '\n')
    pickle.dump(dict(loc={k: {kk: vv for kk, vv in d.items()} for k, d in loc.items()}, comb=combd, fin=fin,
                     mfg={k: {u_: (dd[u_][:, 1] - dd[u_][:, 0]) for u_ in dd} for k, dd in mfg.items()}, used=used, taus=TAUS),
                open(os.path.join(RES, 'analysis.pkl'), 'wb'))
    # ---- console summary
    print('x=0 runs (chi~, G(b/2)/Var, rebin chi/G, drift chi/G, nops, wall):')
    for key in sorted(k for k in loc if k[2] == 0.0):
        d = loc[key]
        print(f"  L={key[0]:2d} T={key[1]:<5g} s{key[3]}: chi~={d['v'][I_CHI]:.4f}({d['e'][I_CHI]:.4f})  G/V={d['v'][I_GH]:.4f}({d['e'][I_GH]:.4f})"
              f"  rebin {d['rebin'][I_CHI]:.2f}/{d['rebin'][I_GH]:.2f}  drift {d['drift'][I_CHI]:+.2f}/{d['drift'][I_GH]:+.2f}"
              f"  nops {d['meta']['nops']:.0f} wall {d['meta']['wall_s']:.0f}s")
    print('finite accuracy (split | LCL):')
    for k in sorted(fin):
        r = fin[k]
        print(f"  L={k[0]:2d} T={k[1]:<5g} eps={k[2]:<4g}: {r['split'][0]:.4f}({r['split'][1]:.4f}) {r['pairs_split']} | "
              f"{r['lcl'][0]:.4f}({r['lcl'][1]:.4f}) {r['pairs_lcl']}")
    flag = [(k, d['rebin'][I_CHI], d['rebin'][I_GH]) for k, d in loc.items() if max(d['rebin'][I_CHI], d['rebin'][I_GH]) > 1.3]
    print('runs with rebin factor > 1.3:', flag)
    dr = [(k, d['drift'][I_CHI], d['drift'][I_GH]) for k, d in loc.items() if max(abs(d['drift'][I_CHI]), abs(d['drift'][I_GH])) > 3]
    print('runs with |drift z| > 3:', dr)


if __name__ == '__main__':
    main()
