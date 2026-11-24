"""Finite-error bound for a rotation family closer to alignment (bound only, no dynamics):
H_S(theta) = (Delta/2)(sin theta sigma_x + cos theta sigma_z), theta in [theta0 - Phi, theta0 + Phi] with
theta0 = pi/4, Phi = pi/6 (15-75 degrees from the coupling axis); ||H1 - H2|| = 2 Delta |sin((theta1-theta2)/2)|.
No x -> -x symmetry here, so the whole range is computed. Same estimator as compute_finite_rotation.py.
Outputs data/finite_bound_rot_tilted.csv."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import mfg_cache
from compute_finite_bound import chi_delta

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
TH0, PHI = np.pi / 4, np.pi / 6
CFG = dict(wc=10.0, beta=1.0, Delta=1.0, N=(100, 200), epsrel=1e-12, nu=41,
           alphas=[0.002, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0], eps_fractions=[0.01, 0.05, 0.1, 0.15, 0.2])
UGRID = np.linspace(-PHI, PHI, CFG['nu'])
member_h = lambda u: CFG['Delta'] * np.array([np.sin(TH0 + u), 0.0, np.cos(TH0 + u)])
dist = lambda u1, u2: 2 * CFG['Delta'] * abs(np.sin((u2 - u1) / 2))

if __name__ == '__main__':
    c = CFG; nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keys = [mfg_cache.key(member_h(u), a, c['wc'], c['beta'], N, c['epsrel']) for a in c['alphas'] for N in c['N'] for u in UGRID]
    res = mfg_cache.ensure(keys, nproc=nproc)
    rows = []
    # analytic limits with the same optimizer
    lim = {'bare': lambda u: -np.tanh(0.5) * np.array([np.sin(TH0 + u), 0 * u, np.cos(TH0 + u)]),
           'pinned': lambda u: np.array([0 * u, 0 * u, -np.tanh(0.5 * np.cos(TH0 + u))])}
    for name, f in lim.items():
        sx = lambda u, f=f: f(u)[0]; sz = lambda u, f=f: f(u)[2]
        for ef in c['eps_fractions']:
            chi, (u1, u2) = chi_delta(sx, sz, ef * np.pi / 4, delta=PHI, dist=dist)
            rows.append(dict(alpha=0.0 if name == 'bare' else np.inf, alpha_L=0.0 if name == 'bare' else np.inf, kind=name,
                             eps_fraction=ef, chi=chi, theta1=TH0 + u1, theta2=TH0 + u2))
    for a in c['alphas']:
        bs = {N: np.array([res[mfg_cache.key(member_h(u), a, c['wc'], c['beta'], N, c['epsrel'])]['b'] for u in UGRID]) for N in c['N']}
        b = (4 * bs[c['N'][1]] - bs[c['N'][0]]) / 3
        sx, sz = CubicSpline(UGRID, b[:, 0]), CubicSpline(UGRID, b[:, 2])
        for ef in c['eps_fractions']:
            chi, (u1, u2) = chi_delta(sx, sz, ef * np.pi / 4, delta=PHI, dist=dist)
            rows.append(dict(alpha=a, alpha_L=2 * a, kind='tempo', eps_fraction=ef, chi=chi, theta1=TH0 + u1, theta2=TH0 + u2))
        print(f"alpha_L={2*a:g}: chi(eps=0.01,0.05,0.2) = " + ", ".join(f"{r['chi']:.4f}" for r in rows[-5:] if r['eps_fraction'] in (0.01, 0.05, 0.2)), flush=True)
    for r in rows[:10]:
        if r['eps_fraction'] in (0.01, 0.05, 0.2):
            print(f"{r['kind']}: eps={r['eps_fraction']}: chi={r['chi']:.4f}")
    with open(DATA / 'finite_bound_rot_tilted.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote data/finite_bound_rot_tilted.csv')
