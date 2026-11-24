"""Locally exact bound (Eq. qubit_chi) for the spin-boson MFG at all couplings.

For each coupling alpha (CSK convention, J = alpha nu e^{-nu/wc}; Leggett alpha_L = 2 alpha) and
angle theta: Bloch vector b and response matrix J_ij = d b_i / d h_j of the MFG at
h = Delta (sin th, 0, cos th), Delta = 1. The xz block of J comes from a 5-point stencil
(step s) along h_x and h_z; J_yy = b_x/h_x and J_xy = J_yz = 0 by the U(1) symmetry about z
(checked directly by check_local_symmetry.py). Both b and J are Richardson-extrapolated in
the number of imaginary-time slices, X_inf = (4 X_{2N} - X_N)/3 (Trotter error O(1/N^2)).

Outputs data/local.csv (one row per alpha, theta).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, json, sys
from pathlib import Path
import numpy as np

import mfg_cache
import sb_core as sc

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
CFG = dict(wc=10.0, beta=1.0, Delta=1.0, N=(200, 400), epsrel=1e-12, step=0.04,
           alphas=[0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
           thetas=[np.pi / 2, 1.0, 0.4])


def stencil_keys(alpha, th, N, c=CFG):
    hb = np.array([np.sin(th), 0.0, np.cos(th)]) * c['Delta']
    ks = {'0': mfg_cache.key(hb, alpha, c['wc'], c['beta'], N, c['epsrel'])}
    for d, e in (('x', np.array([1., 0, 0])), ('z', np.array([0, 0, 1.]))):
        for m in (-2, -1, 1, 2):
            ks[f'{d}{m:+d}'] = mfg_cache.key(hb + m * c['step'] * e, alpha, c['wc'], c['beta'], N, c['epsrel'])
    return hb, ks


def response(res, ks, hb, c=CFG):
    s = c['step']; b0 = res[ks['0']]['b']
    J5 = np.zeros((3, 3)); J3 = np.zeros((3, 3))
    for j, d in ((0, 'x'), (2, 'z')):
        bm2, bm1, bp1, bp2 = (res[ks[f'{d}{m:+d}']]['b'] for m in (-2, -1, 1, 2))
        J5[:, j] = (bm2 - 8 * bm1 + 8 * bp1 - bp2) / (12 * s)
        J3[:, j] = (bp1 - bm1) / (2 * s)
    for J in (J5, J3):
        J[1, 1] = b0[0] / hb[0]
    return b0, J5, J3


def analytic_weak(th, c=CFG):
    x = c['beta'] * c['Delta'] / 2; hh = np.array([np.sin(th), 0, np.cos(th)])
    b = -np.tanh(x) * hh
    J = -(c['beta'] / 2) / np.cosh(x) ** 2 * np.outer(hh, hh) - np.tanh(x) / c['Delta'] * (np.eye(3) - np.outer(hh, hh))
    return b, J


def analytic_usc(th, c=CFG):
    x = c['beta'] * c['Delta'] * np.cos(th) / 2; z = np.array([0, 0, 1.])
    return -np.tanh(x) * z, -(c['beta'] / 2) / np.cosh(x) ** 2 * np.outer(z, z)


def bounds(b, J, th, c=CFG):
    beta = c['beta']
    chi, K, nstar = sc.chi_tilde_qubit(b, J, beta)
    nth = np.array([np.cos(th), 0, -np.sin(th)])
    out = dict(chi_opt=chi, chi_rot=np.sqrt(max(nth @ K @ nth, 0)) / beta,
               chi_bias=np.sqrt(max(K[2, 2], 0)) / beta, chi_tunnel=np.sqrt(max(K[0, 0], 0)) / beta,
               chi_y=abs(J[1, 1]) / beta,
               nstar_angle_z=float(np.degrees(np.arccos(min(1.0, abs(nstar[2]))))),
               nstar_x=float(nstar[0] * np.sign(nstar[2] if abs(nstar[2]) > 1e-14 else 1)),
               nstar_z=float(abs(nstar[2])))
    # fraction of the QFI recovered by measuring sigma_z (pointer populations)
    for name, n in (('opt', nstar), ('rot', nth)):
        db = J @ n; FQ = n @ K @ n
        out[f'frac_sz_{name}'] = (db[2] ** 2 / (1 - b[2] ** 2)) / FQ if FQ > 0 else np.nan
        hh = np.array([np.sin(th), 0, np.cos(th)])
        out[f'frac_energy_{name}'] = ((hh @ db) ** 2 / (1 - (hh @ b) ** 2)) / FQ if FQ > 0 else np.nan
    return out


def main(nproc=10):
    c = CFG
    allkeys = []
    for a in c['alphas']:
        for th in c['thetas']:
            for N in c['N']:
                allkeys += list(stencil_keys(a, th, N)[1].values())
    res = mfg_cache.ensure(allkeys, nproc=nproc)
    rows = []
    for th in c['thetas']:
        for tag, (b, J) in (('weak', analytic_weak(th)), ('usc', analytic_usc(th))):
            r = dict(alpha=0.0 if tag == 'weak' else np.inf, theta=th, kind=tag, **bounds(b, J, th))
            r.update(bx=b[0], bz=b[2], Jxx=J[0, 0], Jxz=J[0, 2], Jzx=J[2, 0], Jyy=J[1, 1], Jzz=J[2, 2])
            rows.append(r)
        for a in c['alphas']:
            per = {}
            for N in c['N']:
                hb, ks = stencil_keys(a, th, N)
                per[N] = response(res, ks, hb)
            (bA, J5A, J3A), (bB, J5B, J3B) = per[c['N'][0]], per[c['N'][1]]
            b = (4 * bB - bA) / 3; J = (4 * J5B - J5A) / 3
            Jsym = 0.5 * (J + J.T)
            r = dict(alpha=a, theta=th, kind='tempo', **bounds(b, Jsym, th))
            r.update(bx=b[0], bz=b[2], Jxx=J[0, 0], Jxz=J[0, 2], Jzx=J[2, 0], Jyy=J[1, 1], Jzz=J[2, 2],
                     asym_xz=abs(J[0, 2] - J[2, 0]),
                     dJ_richardson=float(np.abs(J - J5B).max()), dJ_stencil=float(np.abs(J5B - J3B).max()),
                     db_richardson=float(np.abs(b - bB).max()),
                     chi_opt_N400=sc.chi_tilde_qubit(bB, 0.5 * (J5B + J5B.T), c['beta'])[0])
            rows.append(r)
    out = DATA / 'local.csv'
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('alpha', 'theta', 'kind'), k))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (DATA / 'local_config.json').write_text(json.dumps({k: (list(v) if isinstance(v, (list, tuple)) else v)
                                                                 for k, v in c.items()}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
