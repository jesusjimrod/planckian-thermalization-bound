"""Check 9: the 3-run map scheme (|0><0|, |1><1|, non-Hermitian |0><1|) is equivalent to the 4-run scheme
(|+>, |+i>) within TEMPO's own precision, for the physics that enters tau_eps.

Intrinsic precision: SVD truncation makes TEMPO only approximately linear and trace preserving; its size is
measured by the trace error of each scheme. Criteria:
 (a) max |E4 - E3| <= 3 x max(trace error of E3, of E4);
 (b) TTM steady states (K = 200) of the two schemes differ by < 1e-4 in Bures angle;
 (c) slowest TTM relaxation rates differ by < 1% (relative);
 (d) the 3-scheme map preserves Hermiticity to < 1e-5.
Failure control: dropping the dagger in E(|1><0|) = E(|0><1|)^dagger must break (d) by > 1e-3.
Exit 1 on failure."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import rt_maps, ttm, sb_core as sc

if __name__ == '__main__':
    a, h, dt, te, ep = 0.2, (1.0, 0.0, 0.0), 0.05, 10.0, 1e-9
    E4 = rt_maps.assemble({i: rt_maps.run_one(a, h, i, dt, te, ep)[1] for i in rt_maps.SCHEME4})
    E3 = rt_maps.assemble({i: rt_maps.run_one(a, h, i, dt, te, ep)[1] for i in rt_maps.SCHEME3})
    X = np.array([[0.3, 0.2 - 0.5j], [0.2 + 0.5j, 0.7]])
    img = lambda E: (E @ X.reshape(4)).reshape(-1, 2, 2)
    herm = lambda E: np.abs(img(E) - np.conj(np.transpose(img(E), (0, 2, 1)))).max()
    trerr = lambda E: np.abs(np.trace(img(E), axis1=1, axis2=2) - np.trace(X)).max()
    d = np.abs(E4 - E3).max(); tr3, tr4 = trerr(E3), trerr(E4)
    T3, T4 = ttm.transfer_tensors(E3, 200), ttm.transfer_tensors(E4, 200)
    bloch = lambda r: np.array([np.trace(r @ s).real for s in sc.PAULI]) / np.trace(r).real
    dss = sc.bures_angle_bloch(bloch(ttm.steady_state(T3)), bloch(ttm.steady_state(T4)))
    r3, r4 = ttm.companion_rates(T3, dt)[1][1], ttm.companion_rates(T4, dt)[1][1]
    Ebad = E3.copy(); Ebad[:, :, 2] = E3[:, :, 1]
    ok = {'a': d <= 3 * max(tr3, tr4), 'b': dss < 1e-4, 'c': abs(r3 - r4) / r4 < 0.01, 'd': herm(E3) < 1e-5}
    ctrl = herm(Ebad) > 1e-3
    print(f"(a) |E4-E3|={d:.1e} vs trace errors {tr3:.1e} (3), {tr4:.1e} (4) [{ok['a']}]  (b) steady-state Bures diff {dss:.1e} [{ok['b']}]  "
          f"(c) slow rates {r3:.6f} vs {r4:.6f} [{ok['c']}]  (d) Hermiticity err {herm(E3):.1e} [{ok['d']}]  "
          f"control (no dagger): {herm(Ebad):.1e} -> {'rejected (good)' if ctrl else 'NOT rejected'}")
    fails = sum(int(not v) for v in ok.values()) + int(not ctrl)
    print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
