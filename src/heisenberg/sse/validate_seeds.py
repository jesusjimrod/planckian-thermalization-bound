"""Repeat the beta=1, x=0 4x2 ED check with independent seeds (correlated 2-sigma shift in run 1)."""
import os, sys, subprocess, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlib import load, TAUS, AOBS, jack, seed_of, BIN
from lib import heis, star_bonds, ed_g
from libt import write_input_t
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'heisenberg', 'validation')
def go(s):
    tag = f'4x2_b1_x0_seed{s}'; fi, fo = (os.path.join(OUT, tag + e) for e in ('.in', '.out'))
    write_input_t(fi, star_bonds(4, 2, 1.0), heis(8), 1.0, seed_of('val_' + tag), 40, 50000, 5000, AOBS, taus=TAUS)
    subprocess.run([BIN, fi], stdout=open(fo, 'w'), check=True); return tag
if __name__ == '__main__':
  with Pool(6) as p: tags = p.map(go, range(2, 8))
  E, pe, xe = ed_g(star_bonds(4, 2, 1.0), heis(8), 1.0)
  vals = []
  for t in ['4x2_b1_x0'] + tags:
      r = load(os.path.join(OUT, t + '.out'))
      v, e = jack({'P': r['P'], 'X': r['X']}, lambda m: np.array([m['X'][1, 1] - 1.0 * m['P'][1] ** 2]))
      vals.append((v[0], e[0])); print(f'{t:22s} xi_uu {v[0]:.6f} +- {e[0]:.6f}  pull {(v[0]-xe[1,1])/e[0]: .2f}')
  v = np.array(vals); w = 1 / v[:, 1] ** 2; m = (v[:, 0] * w).sum() / w.sum(); s = w.sum() ** -.5
  chi2 = (((v[:, 0] - m) / v[:, 1]) ** 2).sum()
  print(f'combined {m:.6f} +- {s:.6f}  ed {xe[1,1]:.6f}  pull {(m-xe[1,1])/s:.2f}   chi2/dof between seeds {chi2/(len(v)-1):.2f}')
