"""Check 11: origin of the real-time steady-state coherence offset (TTM fixed point vs imaginary-time MFG).
At alpha = 0.2, member x = 0 (b = (b_x, 0, 0)), compare the TTM steady state at equal memory t_K = 5 for
dt = 0.05 (K = 100) and dt = 0.025 (K = 200). An O(dt^2) Trotter error must shrink by ~4 (accepted: 2.5-6).
Reports the dt -> 0 extrapolation of b_x and its distance to the MFG. Exit 1 if the scaling is not O(dt^2)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import rt_maps, ttm, sb_core as sc
from compute_dynamics import targets

if __name__ == '__main__':
    a, h = 0.2, (1.0, 0.0, 0.0)
    bt = targets(a)(0.0)
    M = {}
    for dt, te in ((0.05, 10.0), (0.025, 5.0)):
        M[dt] = rt_maps.maps([(a, h)], dt, te, 1e-9, nproc=3, verbose=False)[(a, h)][1]
    db = {}
    for dt in (0.05, 0.025):
        K = int(round(5.0 / dt))
        ss = ttm.steady_state(ttm.transfer_tensors(M[dt], K)); b = np.array([np.trace(ss @ s).real for s in sc.PAULI]) / np.trace(ss).real
        db[dt] = b[0] - bt[0]
        print(f"dt={dt}: K={K} (t_K=5)  b_x={b[0]:+.6f}  MFG b_x={bt[0]:+.6f}  db_x={db[dt]:+.3e}  Bures={sc.bures_angle_bloch(b, bt):.2e}")
    ratio = db[0.05] / db[0.025]
    ext = db[0.025] - (db[0.05] - db[0.025]) / 3
    print(f"ratio db(0.05)/db(0.025) = {ratio:.2f};  dt->0 extrapolated db_x = {ext:+.2e}")
    ok = 2.5 < ratio < 6
    print("RESULT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
