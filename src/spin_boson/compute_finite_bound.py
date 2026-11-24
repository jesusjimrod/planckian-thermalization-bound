"""Finite-error bound tau_Pl chi_delta [Eq. pair_bound] for the spin-boson bias family, from MFG states only.

Task (fixed along the coupling sweep; only alpha varies):
    H_S(x) = (Delta/2) sigma_x + (x/2) sigma_z,  |x| <= delta = 2 pi k_B T,  kappa_hat = sigma_z/2,
    ||H_S(x1) - H_S(x2)|| = |x1 - x2|,  Delta = k_B T = 1,  preparation rho_S(0) = I/2,
    accuracies eps/eps_max in EPS, eps_max = pi/4.
By the pi rotation about x combined with bath parity, b(-x) = (b_x(x), 0, -b_z(x)), so only x >= 0 is
computed (grid XGRID), and b(x) is interpolated by cubic splines on [-delta, delta].
chi_delta = max over pairs in the family of 2 [D(w1, w2) - 2 eps]_+ / |x1 - x2|: dense pair grid, then
Nelder-Mead refinement inside the family.

Also: tau_SS^up = max_x [D(I/2, w_x) - eps]_+ / L(alpha), an upper bound on the Shiraishi-Saito
bound, with L the dynamics-free lower bound on delta E_S from ss_bound.py (h_perp = Delta).

Outputs data/finite_bound.csv and data/finite_states.csv (Richardson-extrapolated Bloch vectors).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import mfg_cache
import sb_core as sc
import ss_bound

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
DELTA_BALL = 2 * np.pi
CFG = dict(wc=10.0, beta=1.0, Delta=1.0, N=(100, 200), epsrel=1e-12, delta=DELTA_BALL, nx=41,
           alphas=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0],
           eps_fractions=[0.01, 0.05, 0.1, 0.15, 0.2])
XGRID = np.linspace(0.0, DELTA_BALL, CFG['nx'])


def member_h(x, c=CFG):
    return np.array([c['Delta'], 0.0, x])


def family_states(res, alpha, c=CFG):
    """Richardson-extrapolated b(x) on XGRID."""
    bs = {}
    for N in c['N']:
        bs[N] = np.array([res[mfg_cache.key(member_h(x), alpha, c['wc'], c['beta'], N, c['epsrel'])]['b'] for x in XGRID])
    lo, hi = c['N']
    return (4 * bs[hi] - bs[lo]) / 3, float(np.abs(bs[hi] - bs[lo]).max())


def splines(b):
    xs = np.concatenate([-XGRID[:0:-1], XGRID])
    bx = np.concatenate([b[:0:-1, 0], b[:, 0]]); bz = np.concatenate([-b[:0:-1, 2], b[:, 2]])
    return CubicSpline(xs, bx), CubicSpline(xs, bz)


def chi_delta(sx, sz, eps, delta=DELTA_BALL, ngrid=161, dist=None):
    """max over pairs |x1|,|x2| <= delta of 2[D - 2 eps]_+/||H(x1) - H(x2)|| (x1 < x2); beta = 1.
    dist(x1, x2) = ||H(x1) - H(x2)||, default x2 - x1 (line family)."""
    dist = dist or (lambda x1, x2: x2 - x1)
    bvec = lambda x: np.array([float(sx(x)), 0.0, float(sz(x))])
    def chi12(x1, x2):
        if x2 - x1 < 1e-9:
            return 0.0
        D = sc.bures_angle_bloch(bvec(x1), bvec(x2))
        return 2 * max(D - 2 * eps, 0.0) / dist(x1, x2)
    xs = np.linspace(-delta, delta, ngrid)
    best = (0.0, None)
    B = np.array([bvec(x) for x in xs])
    for i in range(ngrid):
        for j in range(i + 1, ngrid):
            v = chi12(xs[i], xs[j])
            if v > best[0]:
                best = (v, (xs[i], xs[j]))
    if best[1] is None:
        return 0.0, (np.nan, np.nan)
    def f(v):
        x1, x2 = np.clip(v, -delta, delta)
        return -chi12(min(x1, x2), max(x1, x2))
    r = minimize(f, best[1], method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-12, maxiter=2000))
    x1, x2 = np.sort(np.clip(r.x, -delta, delta))
    val = max(-r.fun, best[0])
    return val, ((x1, x2) if -r.fun >= best[0] else best[1])


def main(nproc=10):
    c = CFG
    keys = [mfg_cache.key(member_h(x), a, c['wc'], c['beta'], N, c['epsrel'])
            for a in c['alphas'] for N in c['N'] for x in XGRID]
    res = mfg_cache.ensure(keys, nproc=nproc)
    rows, srows = [], []
    rho0_b = np.zeros(3)
    for a in c['alphas']:
        b, dN = family_states(res, a)
        for x, bb in zip(XGRID, b):
            srows.append(dict(alpha=a, x=x, bx=bb[0], by=bb[1], bz=bb[2], richardson_max_change=dN))
        sx, sz = splines(b)
        L, p, q = ss_bound.best_lower_bound_dE(a, c['wc'], c['beta'], c['Delta'])
        D0 = np.array([sc.bures_angle_bloch(rho0_b, np.array([sx(x), 0, sz(x)])) for x in np.linspace(-c['delta'], c['delta'], 401)])
        for ef in c['eps_fractions']:
            eps = ef * np.pi / 4
            chi, (x1, x2) = chi_delta(sx, sz, eps)
            Dmax = max(D0.max() - eps, 0.0)
            rows.append(dict(alpha=a, alpha_L=2 * a, eps_fraction=ef, eps=eps, chi=chi, x1=x1, x2=x2,
                             D0_max=D0.max(), L_dE=L, tau_SS_up=(Dmax / L if L > 0 else np.inf),
                             richardson_max_change=dN))
        print(f"alpha={a:5.3f}  chi(eps=0.05)={[r['chi'] for r in rows if r['alpha']==a and r['eps_fraction']==0.05][0]:.4f}  "
              f"L_dE={L:.3f}  tau_SS_up(0.05)={[r['tau_SS_up'] for r in rows if r['alpha']==a and r['eps_fraction']==0.05][0]:.4f}", flush=True)
    for name, data in (('finite_bound.csv', rows), ('finite_states.csv', srows)):
        with open(DATA / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (DATA / 'finite_config.json').write_text(json.dumps(c, indent=1))
    print('wrote data/finite_bound.csv, data/finite_states.csv')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
