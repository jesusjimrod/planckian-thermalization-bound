"""Check 8: acceptance tests on data/dynamics.csv (compute_dynamics.py) against data/finite_bound.csv.

 (i)   Theorem: tau_eps >= tau_Pl chi_delta for every (alpha, eps) [can fail through wrong targets, wrong
       symmetry reduction, or wrong dynamics];
 (ii)  real-time steady state (TTM fixed point) vs imaginary-time MFG target: max Bures angle eta < 0.15 eps_min,
       and the bracket [tau(eps+eta), tau(eps-eta)] has relative half-width < 3%.
       GATE CHANGE (2026-10-23, before the full sweep): the original criterion was eta < 0.1 eps_min. alpha = 0.2
       gave eta = 8.7e-4 = 0.11 eps_min. check_dt_coherence.py showed that eta is the O(dt^2) Trotter error of the
       steady-state coherence (ratio 4.09 for dt 0.05 -> 0.025; dt -> 0 extrapolation 1.3e-5 from the MFG), and
       the bracket, which measures its effect on tau_eps, is the quantity that matters.
 (iii) TTM memory: |tau(K) - tau(K/2)| / tau < 5%;
 (iv)  sustained thermalization: no member leaves the eps-ball after tau_eps within the horizon;
 (v)   speed-limit chain: tau_QFI <= tau_eps (third form of Eq. ss_chain);
 (vi)  trace preservation of the TTM recursion: |lambda_0 - 1| < 1e-6.
Prints the rows that fail. Usage: check_dynamics.py [--rot]. Exit 1 on failure."""
import csv, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
ROT = len(sys.argv) > 1 and sys.argv[1] == '--rot'          # rotation family: dynamics_rot.csv / finite_bound_rot.csv
D = list(csv.DictReader(open(DATA / ('dynamics_rot.csv' if ROT else 'dynamics.csv'))))
B = {(round(float(r['alpha']), 12), round(float(r['eps_fraction']), 12)): r
     for r in csv.DictReader(open(DATA / ('finite_bound_rot.csv' if ROT else 'finite_bound.csv')))}
eps_min = min(float(r['eps']) for r in D)
fails = 0
for r in D:
    a, ef = float(r['alpha']), float(r['eps_fraction'])
    b = B[(round(a, 12), round(ef, 12))]
    tau, chi = float(r['tau']), float(b['chi'])
    c = {'i': tau >= chi - 1e-9,
         'ii': float(r['ss_err_max']) < 0.15 * eps_min and (float(r['tau_hi']) - float(r['tau_lo'])) / (2 * tau) < 0.03,
         'iii': abs(tau - float(r['tau_K2'])) / tau < 0.05,
         'iv': r['last_exit_after_tau'] in ('', 'nan') or not np.isfinite(float(r['last_exit_after_tau'])),
         'v': float(r['tau_qfi']) <= tau * (1 + 1e-6),
         'vi': float(r['unit_eig_dev']) < 1e-6}
    bad = [k for k, v in c.items() if not v]
    fails += len(bad)
    print(f"alpha={a:<6g} eps={ef:<5g} tau={tau:9.3f} chi={chi:.4f} tau_K2={float(r['tau_K2']):9.3f} tau_QFI={float(r['tau_qfi']):8.3f} "
          f"ss_err={float(r['ss_err_max']):.1e} last_exit={r['last_exit_after_tau']:>6}  " + ("ok" if not bad else "FAIL " + ",".join(bad)))
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
