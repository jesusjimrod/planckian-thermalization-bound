"""Production runs, homogeneous 2D Heisenberg AFM, system = site 0 (2026-11-09).

Sets (all with G(tau) on the 10-point tau grid of runlib.TAUS):
  A  L=16, x=0, all 13 temperatures                                  (main curve)
  B  L in {8,32,48}, x=0, kT/J in {1,0.5,0.2,0.1,0.07}                (finite-size spot checks)
  C  L=32, kT/J=0.1, x=0, three extra independent seeds               (resolves CONTEXT 5.2)
  D  L=16, beta*x in FIELDS (x>0), all 13 temperatures                (finite-accuracy bound)
  E  L=32, beta*x in FIELDS (x>0), kT/J in {1,0.5,0.2}                (L-dependence of chi_delta)
Statistics: THERM equilibration sweeps, NBINS bins of SPB sweeps.
Raw output: ../results/raw/<tag>.{in,out,json}. Finished runs are skipped, so the script can be restarted.

Usage:  python run_production.py [ncores] [--dry]
"""
import os, sys, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlib import run_one

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'heisenberg', 'raw')
TEMPS = [3, 2, 1.5, 1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07]
FIELDS = [0.25 * k for k in range(1, 17)] + [4.5, 5.0, 2 * np.pi]          # beta*x > 0
NBINS, SPB, THERM = 40, 10000, 20000
SEC_PER_OP_SWEEP = 7.0e-8                                                    # measured on this machine (L=48, T=0.1)


def tag(L, T, bx, s):
    return f"heis_L{L}_T{T:g}_bx{bx:.4f}_s{s}"


def jobs():
    J = []
    for T in TEMPS: J.append(('A', 16, T, 0.0, 0))
    for L in (8, 32, 48):
        for T in (1, 0.5, 0.2, 0.1, 0.07): J.append(('B', L, T, 0.0, 0))
    for s in (1, 2, 3): J.append(('C', 32, 0.1, 0.0, s))
    for T in TEMPS:
        for bx in FIELDS: J.append(('D', 16, T, bx, 0))
    for T in (1, 0.5, 0.2):
        for bx in FIELDS: J.append(('E', 32, T, bx, 0))
    return J


def est_ops(L, T): return 31540 * (L / 48) ** 2 * (0.1 / T)


def est_sec(j): return SEC_PER_OP_SWEEP * est_ops(j[1], j[2]) * (THERM + NBINS * SPB)


def do(j):
    _, L, T, bx, s = j
    m = run_one(RAW, tag(L, T, bx, s), L, 1 / T, bx * T, NBINS, SPB, THERM)
    return j, m


if __name__ == '__main__':
    nc = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else os.cpu_count()
    J = sorted(jobs(), key=lambda j: -est_sec(j))                  # heaviest first
    tot = sum(map(est_sec, J))
    print(f"{len(J)} jobs, estimated {tot/3600:.1f} core-h, longest {est_sec(J[0])/60:.0f} min "
          f"({J[0]}); ~{tot/3600/min(nc, 7):.1f} h wall on {nc} cores (4P+6E ~ 7 P-equivalents)", flush=True)
    for s in 'ABCDE':
        print(f"  set {s}: {sum(1 for j in J if j[0]==s)} jobs, {sum(est_sec(j) for j in J if j[0]==s)/3600:.2f} core-h")
    if '--dry' in sys.argv: sys.exit()
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    with Pool(nc) as p:
        for k, (j, m) in enumerate(p.imap_unordered(do, J)):
            print(f"[{k+1}/{len(J)} {(time.time()-t0)/60:6.1f} min] set {j[0]} {m['tag']}: "
                  f"nops {m['nops']:.0f}, wall {m['wall_s']:.0f} s", flush=True)
    print('done', flush=True)
