"""Finite-error bound tau_Pl chi_delta [Eq. pair_bound] for the ROTATION family of Sec. V (fixed |h| = Delta).

Task: H_S(u) = (Delta/2)(cos u sigma_x + sin u sigma_z), |u| <= Phi = pi/3, i.e. the angle to the coupling axis
theta = pi/2 - u sweeps [pi/6, 5pi/6] around the symmetric point; ||H(u1) - H(u2)|| = 2 Delta |sin((u1-u2)/2)|;
ball radius delta = 2 Delta sin(Phi/2) = Delta; preparation rho_S(0) = I/2; Delta = k_B T = 1, wc = 10 Delta.
Phi < pi/2 keeps the family away from alignment (theta = 0, pi), where sigma_z is conserved and the pointer
populations never relax. By the pi rotation about x with bath parity, b(-u) = (b_x(u), 0, -b_z(u)).
Outputs data/finite_bound_rot.csv and data/finite_states_rot.csv (column x = u).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

import mfg_cache
import sb_core as sc
from compute_finite_bound import chi_delta

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
PHI = np.pi / 3
CFG = dict(wc=10.0, beta=1.0, Delta=1.0, N=(100, 200), epsrel=1e-12, Phi=PHI, nu=41,
           alphas=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0],
           eps_fractions=[0.01, 0.05, 0.1, 0.15, 0.2])
UGRID = np.linspace(0.0, PHI, CFG['nu'])


def member_h(u, c=CFG):
    return c['Delta'] * np.array([np.cos(u), 0.0, np.sin(u)])


def dist(u1, u2, c=CFG):
    return 2 * c['Delta'] * abs(np.sin((u2 - u1) / 2))


def splines(b):
    us = np.concatenate([-UGRID[:0:-1], UGRID])
    return (CubicSpline(us, np.concatenate([b[:0:-1, 0], b[:, 0]])),
            CubicSpline(us, np.concatenate([-b[:0:-1, 2], b[:, 2]])))


if __name__ == '__main__':
    c = CFG; nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keys = [mfg_cache.key(member_h(u), a, c['wc'], c['beta'], N, c['epsrel']) for a in c['alphas'] for N in c['N'] for u in UGRID]
    res = mfg_cache.ensure(keys, nproc=nproc)
    rows, srows = [], []
    for a in c['alphas']:
        bs = {N: np.array([res[mfg_cache.key(member_h(u), a, c['wc'], c['beta'], N, c['epsrel'])]['b'] for u in UGRID]) for N in c['N']}
        b = (4 * bs[c['N'][1]] - bs[c['N'][0]]) / 3
        for u, bb in zip(UGRID, b):
            srows.append(dict(alpha=a, x=u, bx=bb[0], by=bb[1], bz=bb[2]))
        sx, sz = splines(b)
        D0 = max(sc.bures_angle_bloch(np.zeros(3), np.array([sx(u), 0, sz(u)])) for u in np.linspace(-PHI, PHI, 201))
        for ef in c['eps_fractions']:
            eps = ef * np.pi / 4
            chi, (u1, u2) = chi_delta(sx, sz, eps, delta=PHI, dist=dist)
            rows.append(dict(alpha=a, alpha_L=2 * a, eps_fraction=ef, eps=eps, chi=chi, x1=u1, x2=u2, D0_max=D0,
                             richardson_max_change=float(np.abs(b - bs[c['N'][1]]).max())))
        print(f"alpha_L={2*a:g}: chi(eps=0.01,0.05,0.2) = " + ", ".join(f"{r['chi']:.4f}" for r in rows[-5:] if r['eps_fraction'] in (0.01, 0.05, 0.2)), flush=True)
    for name, data in (('finite_bound_rot.csv', rows), ('finite_states_rot.csv', srows)):
        with open(DATA / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (DATA / 'finite_rot_config.json').write_text(json.dumps(c, indent=1))
    print('wrote data/finite_bound_rot.csv')
