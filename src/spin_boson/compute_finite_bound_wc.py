"""Finite-error bound of the bias family at a larger cutoff (wc = 100 Delta), for two couplings, to test whether
our bound stays at or above its bare-Gibbs value when the cutoff grows (the speed-limit comparison of
compute_ss_upper.py assumes this for wc != 10). Same task and code as compute_finite_bound.py; N = 200 -> 400
Richardson (the Trotter scale sqrt(K(0)) ~ sqrt(alpha) wc equals that of wc = 10 at alpha_L = 2-20).
Outputs data/finite_bound_wc100.csv. Usage: compute_finite_bound_wc.py nproc"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, sys
from pathlib import Path
import numpy as np
import mfg_cache, ss_bound
import sb_core as sc
import compute_finite_bound as cfb

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
CFG = dict(cfb.CFG); CFG.update(wc=100.0, N=(200, 400), alphas=[0.01, 0.1])

if __name__ == '__main__':
    c = CFG; nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keys = [mfg_cache.key(cfb.member_h(x), a, c['wc'], c['beta'], N, c['epsrel']) for a in c['alphas'] for N in c['N'] for x in cfb.XGRID]
    res = mfg_cache.ensure(keys, nproc=nproc)
    rows = []
    for a in c['alphas']:
        bs = {N: np.array([res[mfg_cache.key(cfb.member_h(x), a, c['wc'], c['beta'], N, c['epsrel'])]['b'] for x in cfb.XGRID]) for N in c['N']}
        b = (4 * bs[c['N'][1]] - bs[c['N'][0]]) / 3
        sx, sz = cfb.splines(b)
        L, p, q = ss_bound.best_lower_bound_dE(a, c['wc'], c['beta'], c['Delta'])
        D0 = max(sc.bures_angle_bloch(np.zeros(3), np.array([sx(x), 0, sz(x)])) for x in np.linspace(-c['delta'], c['delta'], 401))
        for ef in c['eps_fractions']:
            eps = ef * np.pi / 4
            chi, (x1, x2) = cfb.chi_delta(sx, sz, eps)
            rows.append(dict(wc=c['wc'], alpha=a, alpha_L=2 * a, eps_fraction=ef, chi=chi, x1=x1, x2=x2, D0_max=D0, L_dE=L,
                             tau_SS_up=max(D0 - eps, 0) / L if L > 0 else np.inf, richardson_max_change=float(np.abs(b - bs[c['N'][1]]).max())))
            print(f"wc={c['wc']:g} alpha={a} eps={ef}: chi={chi:.5f}  tau_SS_up={rows[-1]['tau_SS_up']:.4f}  D0={D0:.4f}", flush=True)
    with open(DATA / 'finite_bound_wc100.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote data/finite_bound_wc100.csv')
