"""Check 4: U(1) structure of the MFG response for the continuum Ohmic bath (GibbsTempo).

Claims (main.tex Sec. sb_response): b_y = 0, J_xy = J_zy = 0, J_yy = b_x/h_x, J symmetric (J_xz = J_zx).
J_yy is obtained here by a direct 5-point central difference along h_y (complex H_S), which also
exercises correction 2 (OQuPy returns rho~^T). Failure control: the same difference with raw
OQuPy states (no transpose fix) must give J_yy with the wrong sign. Exit 1 on failure."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import oqupy
import sb_core as sc

N, eps, s = 200, 1e-12, 0.04
fails = 0
for alpha, th in [(0.2, 1.0), (2.0, 1.0), (0.05, 0.4)]:
    hb = np.array([np.sin(th), 0.0, np.cos(th)])
    b0 = sc.bloch(sc.mfg_state(hb, alpha, 10.0, 1.0, N, eps))
    bb = {m: sc.bloch(sc.mfg_state(hb + [0, m * s, 0], alpha, 10.0, 1.0, N, eps)) for m in (-2, -1, 1, 2)}
    Jy = (bb[-2] - 8 * bb[-1] + 8 * bb[1] - bb[2]) / (12 * s)   # 5-point stencil; column y of J
    pred = b0[0] / hb[0]
    # raw OQuPy (transposed) for the control
    bath = oqupy.Bath(sc.SZ, sc.MatsubaraSD(alpha, 10.0, 1.0))
    raw = lambda h: sc.bloch(oqupy.gibbs_tempo_compute(oqupy.System(sc.hamiltonian(h)), bath,
                                                     oqupy.GibbsParameters(n_steps=N, epsrel=eps), progress_type='silent'))
    Jy_raw = (raw(hb + [0, s, 0]) - raw(hb - [0, s, 0])) / (2 * s)
    ok = abs(b0[1]) < 1e-9 and abs(Jy[0]) < 1e-6 and abs(Jy[2]) < 1e-6 and abs(Jy[1] - pred) < 2e-5 * max(1, abs(pred))
    ctrl = abs(Jy_raw[1] - pred) > 0.5 * abs(pred)
    fails += (not ok) + (not ctrl)
    print(f"alpha={alpha} th={th}: b_y={b0[1]:+.1e}  J_xy={Jy[0]:+.1e} J_zy={Jy[2]:+.1e}  J_yy(fd)={Jy[1]:+.7f}  b_x/h_x={pred:+.7f}  ok={ok}"
          f"  | control raw J_yy={Jy_raw[1]:+.5f} -> {'rejected (good)' if ctrl else 'NOT rejected'}", flush=True)
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
