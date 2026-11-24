"""Angle-resolved locally exact bounds (rotation and optimal perturbation) at fixed couplings, beta Delta = 1.

Same estimator as compute_local.py (b and J of the MFG, 5-point stencil, J_yy = b_x/h_x, Richardson in N),
on a grid of angles theta in (0, pi/2]; F(pi - theta) = F(theta) by the pi rotation about x with bath parity.
N = 100 -> 200 (cross-checked against the N = 200 -> 400 results of compute_local.py at theta = 0.4, 1, pi/2).
Outputs data/local_theta.csv.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, sys
from pathlib import Path
import numpy as np

import mfg_cache
import compute_local as cl

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
CFG = dict(cl.CFG)
CFG.update(N=(100, 200), alphas=[0.01, 0.1, 0.3, 1.0, 5.0],
           thetas=[0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, np.pi / 2])

if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    c = CFG
    keys = []
    for a in c['alphas']:
        for th in c['thetas']:
            for N in c['N']:
                keys += list(cl.stencil_keys(a, th, N, c)[1].values())
    res = mfg_cache.ensure(keys, nproc=nproc)
    rows = []
    for th in c['thetas']:
        for tag, (b, J) in (('weak', cl.analytic_weak(th, c)), ('usc', cl.analytic_usc(th, c))):
            rows.append(dict(alpha=0.0 if tag == 'weak' else np.inf, theta=th, kind=tag, bx=b[0], bz=b[2], **cl.bounds(b, J, th, c)))
        for a in c['alphas']:
            per = {}
            for N in c['N']:
                hb, ks = cl.stencil_keys(a, th, N, c)
                per[N] = cl.response(res, ks, hb, c)
            (bA, JA, _), (bB, JB, _) = per[c['N'][0]], per[c['N'][1]]
            b = (4 * bB - bA) / 3; J = (4 * JB - JA) / 3
            rows.append(dict(alpha=a, theta=th, kind='tempo', bx=b[0], bz=b[2], asym_xz=abs(J[0, 2] - J[2, 0]),
                             dJ_richardson=float(np.abs(J - JB).max()), **cl.bounds(b, 0.5 * (J + J.T), th, c)))
    keys_out = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('alpha', 'theta', 'kind'), k))
    with open(DATA / 'local_theta.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys_out); w.writeheader(); w.writerows(rows)
    print('wrote data/local_theta.csv')
