"""Check 12: Richardson extrapolation in dt of the real-time steady state, at strong coupling (alpha = 0.5).
For members x = 0 and x = 2 pi, compare the TTM fixed points at dt = 0.05 (K = 200) and dt = 0.1 (K = 100),
both with memory t_K = 10, and their O(dt^2) extrapolation (4 b_0.05 - b_0.1)/3 with the imaginary-time MFG.
Pass, for both members: (a) the dt = 0.1 and 0.05 mismatches are in the O(dt^2) regime, ratio in [3, 6];
(b) the extrapolated Bloch vector is physical, |b| <= 1 (otherwise the Bures formula clips and a meaningless
zero distance would pass); (c) the extrapolated mismatch is < 1/4 of the dt = 0.05 mismatch.
Correction (2026-10-23): the first version tested only (c) and passed for x = 2 pi through the clipping in (b).
Exit 1 on failure."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import rt_maps, ttm, sb_core as sc
from compute_dynamics import targets

if __name__ == '__main__':
    a = 0.5; tg = targets(a); fails = 0
    xs = [0.0, float(round(2 * np.pi, 10))]
    M = {dt: rt_maps.maps([(a, (1.0, 0.0, x)) for x in xs], dt, 10.0, 1e-9, nproc=4, verbose=False) for dt in (0.1, 0.05)}
    for x in xs:
        bt = tg(x); b = {}
        for dt in (0.1, 0.05):
            E = M[dt][(a, (1.0, 0.0, x))][1]
            ss = ttm.steady_state(ttm.transfer_tensors(E, int(round(10.0 / dt))))
            b[dt] = np.array([np.trace(ss @ s).real for s in sc.PAULI]) / np.trace(ss).real
        bext = (4 * b[0.05] - b[0.1]) / 3
        d05, d10, dex = (sc.bures_angle_bloch(v, bt) for v in (b[0.05], b[0.1], bext))
        ok = {'a': 3 <= d10 / d05 <= 6, 'b': np.linalg.norm(bext) <= 1, 'c': dex < d05 / 4}
        fails += not all(ok.values())
        print(f"x={x:.3f}: Bures to MFG  dt=0.1: {d10:.2e}  dt=0.05: {d05:.2e}  extrapolated: {dex:.2e}  ratio={d10/d05:.2f}  |b_ext|={np.linalg.norm(bext):.6f}  {ok}")
    print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
