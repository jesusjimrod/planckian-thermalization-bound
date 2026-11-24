"""ED validation of sse_t on 4x2 periodic clusters, with and without local field.

Compares energy, p_k, connected xi_kl (all four entries) and connected G(tau)/Var at all taus.
Usage: python validate_ed.py   -> writes ../results/validation/validation.txt
"""
import os, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlib import run_one, load, TAUS, AOBS, jack
from lib import heis, star_bonds, ed_g
from libt import ed_G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'heisenberg', 'validation')
CASES = [('4x2_b1_x0', 1.0, 0.0), ('4x2_b2_x1.5', 2.0, 1.5), ('4x2_b4_x-2', 4.0, -2.0)]
LX, LY = 4, 2


def run_case(c):
    tag, beta, x = c
    # run_one builds an L x L lattice; the 4x2 cluster needs its own bonds -> write input here
    from libt import write_input_t
    import subprocess, time, json
    from runlib import seed_of, BIN
    os.makedirs(OUT, exist_ok=True)
    fi, fo = os.path.join(OUT, tag + '.in'), os.path.join(OUT, tag + '.out')
    N = LX * LY
    write_input_t(fi, star_bonds(LX, LY, 1.0), heis(N, x=x), beta, seed_of('val_' + tag), 40, 50000, 5000, AOBS, taus=TAUS)
    t0 = time.time()
    with open(fo, 'w') as o: subprocess.run([BIN, fi], stdout=o, check=True)
    return tag, time.time() - t0


def compare(tag, beta, x):
    r = load(os.path.join(OUT, tag + '.out'))
    N = LX * LY; bonds = star_bonds(LX, LY, 1.0); model = heis(N, x=x)
    E, p, xi = ed_g(bonds, model, beta)
    Ged = ed_G(bonds, model, beta, AOBS, TAUS)
    A = np.array(AOBS)

    def f(m):
        En = -m['n'] / beta + r['Csum']
        P = m['P']; xq = m['X'] - beta * np.outer(P, P)
        mean = P @ A; var = P @ A ** 2 - mean ** 2
        return np.concatenate([[En], P, xq.ravel(), (m['G'] - mean ** 2) / var])
    v, e = jack({'n': r['n'], 'P': r['P'], 'X': r['X'], 'G': r['G']}, f)
    ref = np.concatenate([[E], p, xi.ravel(), Ged])
    names = ['E', 'p_dn', 'p_up', 'xi_dd', 'xi_du', 'xi_ud', 'xi_uu'] + [f'G({t:.4g}b)/Var' for t in TAUS]
    pulls = (v - ref) / e
    ok = np.isfinite(pulls) & (e > 0)
    lines = [f'== {tag}: beta={beta}, x={x}, 40 bins x 50000 sweeps']
    for nm, a, b, s, pl in zip(names, v, ref, e, pulls):
        lines.append(f'   {nm:14s} qmc {a: .6f} +- {s:.6f}   ed {b: .6f}   pull {pl: .2f}')
    lines.append(f'   max|pull| = {np.abs(pulls[ok]).max():.2f}, rms = {np.sqrt((pulls[ok]**2).mean()):.2f}, n = {ok.sum()}')
    return lines, pulls[ok]


if __name__ == '__main__':
    with Pool(len(CASES)) as pool:
        walls = dict(pool.map(run_case, CASES))
    allp, txt = [], []
    for c in CASES:
        lines, pl = compare(*c); txt += lines + [f'   wall {walls[c[0]]:.1f} s']; allp.append(pl)
    allp = np.concatenate(allp)
    txt.append(f'ALL: max|pull| = {np.abs(allp).max():.2f}, rms = {np.sqrt((allp**2).mean()):.2f}, n = {len(allp)}')
    open(os.path.join(OUT, 'validation.txt'), 'w').write('\n'.join(txt) + '\n')
    print('\n'.join(txt))
