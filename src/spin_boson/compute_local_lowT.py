"""Symmetric point theta = pi/2 at several temperatures: tau >= hbar chi_zz [Eq. sb_symmetric], chi_zz = |J_zz|.

At theta = pi/2, b_z is odd in h_z, so the 5-point stencil reduces to J_zz = (8 b_z(s) - b_z(2s))/(6 s).
Richardson in N at fixed dtau = beta/N ratio 2. Weak limit: tanh(beta Delta/2)/Delta; ultrastrong (Curie): beta/2.
The bound in units of tau_Pl is chi_zz/beta. Outputs data/local_lowT.csv.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, sys
from pathlib import Path
import numpy as np
import mfg_cache

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
CFG = dict(wc=10.0, betas=[4.0], step=0.04, epsrel=1e-12,
           alphas=[0.001, 0.03, 0.1, 0.3, 1.0, 5.0])
def Npair(beta): return (int(200 * beta), int(400 * beta)) if beta <= 1 else (int(100 * beta), int(200 * beta))


def main(nproc=10):
    c = CFG; s = c['step']; keys = []
    for beta in c['betas']:
        for a in c['alphas']:
            for N in Npair(beta):
                for m in (1, 2):
                    keys.append(mfg_cache.key((1.0, 0.0, m * s), a, c['wc'], beta, N, c['epsrel']))
    res = mfg_cache.ensure(keys, nproc=nproc)
    rows = []
    for beta in c['betas']:
        for a in c['alphas']:
            J = {}
            for N in Npair(beta):
                b1, b2 = (res[mfg_cache.key((1.0, 0.0, m * s), a, c['wc'], beta, N, c['epsrel'])]['b'][2] for m in (1, 2))
                J[N] = (8 * b1 - b2) / (6 * s)
            lo, hi = Npair(beta)
            Jzz = (4 * J[hi] - J[lo]) / 3
            rows.append(dict(beta=beta, alpha=a, alpha_L=2 * a, Jzz=Jzz, chi=abs(Jzz) / beta, richardson=abs(Jzz - J[hi]),
                             weak=np.tanh(beta / 2) / beta, curie=0.5, N_lo=lo, N_hi=hi))
            print(f"beta={beta} alpha={a}: chi_zz/beta={abs(Jzz)/beta:.5f} (weak {np.tanh(beta/2)/beta:.5f}), Richardson change {abs(Jzz-J[hi]):.1e}", flush=True)
    with open(DATA / 'local_lowT.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
