"""Check 7: acceptance tests on data/local.csv (compute_local.py).

 (i)   weak limit: chi_opt at the smallest alpha is within 2 lambda/Delta = 2 alpha wc of the bare-Gibbs value
       (the second-order MFG correction is O(lambda/Delta)), and the rotation is optimal there (ratio > 0.99);
 (ii)  ultrastrong trend: at the largest alpha, chi_opt and chi_rot are within 5% of the pinned values
       1/2 sech(x_theta) and 1/2 |sin theta| sech(x_theta);
 (iii) Maxwell symmetry of the Hessian: |J_xz - J_zx| < 1e-4;
 (iv)  numerical budgets: Richardson change of J < 2e-3, stencil (5- vs 3-point) < 1e-3;
 (v)   ceilings: chi_opt <= 1/2 and |J_zz| <= beta/2 (Eq. r_bracket; violated by sign/normalization errors);
 (vi)  theta = pi/2: chi_opt = |J_zz|/beta (the bias channel wins at every coupling; Eq. sb_symmetric).
Exit 1 on failure."""
import csv, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
rows = list(csv.DictReader(open(DATA / 'local.csv')))
f = lambda r, k: float(r[k])
fails = 0
for th in sorted({f(r, 'theta') for r in rows}, reverse=True):
    T = sorted([r for r in rows if f(r, 'theta') == th and r['kind'] == 'tempo'], key=lambda r: f(r, 'alpha'))
    wk = [r for r in rows if f(r, 'theta') == th and r['kind'] == 'weak'][0]
    us = [r for r in rows if f(r, 'theta') == th and r['kind'] == 'usc'][0]
    lo, hi = T[0], T[-1]
    lam = f(lo, 'alpha') * 10.0
    c1 = abs(f(lo, 'chi_opt') - f(wk, 'chi_opt')) < 2 * lam and f(lo, 'chi_rot') / f(lo, 'chi_opt') > 0.99
    c2 = abs(f(hi, 'chi_opt') / f(us, 'chi_opt') - 1) < 0.05 and abs(f(hi, 'chi_rot') / f(us, 'chi_rot') - 1) < 0.05
    c3 = max(f(r, 'asym_xz') for r in T) < 1e-4
    c4 = max(f(r, 'dJ_richardson') for r in T) < 2e-3 and max(f(r, 'dJ_stencil') for r in T) < 1e-3
    c5 = max(f(r, 'chi_opt') for r in T) <= 0.5 + 1e-6 and max(abs(f(r, 'Jzz')) for r in T) <= 0.5 + 1e-6
    c6 = True if abs(th - np.pi / 2) > 1e-9 else max(abs(f(r, 'chi_opt') - abs(f(r, 'Jzz'))) for r in T) < 1e-6
    ok = all([c1, c2, c3, c4, c5, c6]); fails += not ok
    print(f"theta={th:.4f}: (i) {f(lo,'chi_opt'):.5f} vs weak {f(wk,'chi_opt'):.5f} [{c1}]  (ii) {f(hi,'chi_opt'):.5f}/{f(us,'chi_opt'):.5f}, "
          f"rot {f(hi,'chi_rot'):.5f}/{f(us,'chi_rot'):.5f} [{c2}]  (iii) max asym {max(f(r,'asym_xz') for r in T):.1e} [{c3}]  "
          f"(iv) rich {max(f(r,'dJ_richardson') for r in T):.1e}, stencil {max(f(r,'dJ_stencil') for r in T):.1e} [{c4}]  "
          f"(v) max chi {max(f(r,'chi_opt') for r in T):.5f} [{c5}]  (vi) [{c6}]")
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
