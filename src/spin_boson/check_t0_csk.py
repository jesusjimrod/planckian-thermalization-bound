"""Check 10: real-time pipeline (TEMPO maps + TTM) at theta = 1 (CSK Figs. 3-4) against
 (a) the imaginary-time MFG (steady state of the TTM recursion, K = 200; Bures angle), and
 (b) the population decay time t0 = 1/(G+ + G-) of the polaron master equation (CSK Eqs. 10-11), which CSK
     report to agree with TEMPO for alpha >= 0.2 at theta = 1 (their Fig. 4b). Agreement is required to 15%
     for alpha >= 0.2 (polaron theory is not exact there: omega_q sin(theta) << alpha wc only marginally);
     alpha = 0.1 is reported, not tested.
The slowest TTM rate is the companion-matrix eigenvalue closest to (but below) 1. Exit 1 on failure."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv, sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import rt_maps, ttm, sb_core as sc, mfg_cache

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'


def t0_polaron(a, th, wc=10.0, beta=1.0):
    def C(s):
        re = quad(lambda n: a * np.exp(-n / wc) / n * 2 / np.tanh(beta * n / 2) * np.sin(n * s / 2) ** 2, 0, 80 * wc, limit=2000)[0]
        im = quad(lambda n: a * np.exp(-n / wc) / n * np.sin(n * s), 0, 80 * wc, limit=2000)[0]
        return re + 1j * im
    ss = np.linspace(0, 40, 8001); Cs = np.array([C(s) for s in ss])
    Dp, eps = np.sin(th), np.cos(th)
    G = lambda sgn: (Dp / 2) ** 2 * 2 * np.trapz(np.exp(sgn * 1j * eps * ss - 4 * Cs).real, ss)
    return 1 / (G(-1) + G(+1))


if __name__ == '__main__':
    th = 1.0; h = (np.sin(th), 0.0, np.cos(th)); alphas = [0.1, 0.2, 0.3, 0.5]
    M = rt_maps.maps([(a, h) for a in alphas], 0.05, 10.0, 1e-9, nproc=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    fails = 0; rows = []
    for a in alphas:
        t, E, sec = M[(a, tuple(map(float, h)))]
        T = ttm.transfer_tensors(E, 200)
        lam, rates = ttm.companion_rates(T, 0.05)
        ss = ttm.steady_state(T); b_ss = np.array([np.trace(ss @ s).real for s in sc.PAULI]) / np.trace(ss).real
        res = mfg_cache.ensure([mfg_cache.key(h, a, 10.0, 1.0, N, 1e-12) for N in (200, 400)], verbose=False)
        bA, bB = (res[mfg_cache.key(h, a, 10.0, 1.0, N, 1e-12)]['b'] for N in (200, 400))
        b_mfg = (4 * bB - bA) / 3
        dss = sc.bures_angle_bloch(b_ss, b_mfg)
        # slowest real (population) rate: rates[1] may be part of a complex pair (coherence); take the slowest real one
        real_idx = [i for i in range(1, 12) if abs(np.angle(lam[i])) < 1e-6]
        t0_ttm = 1 / rates[real_idx[0]] if real_idx else np.nan
        t0_pol = t0_polaron(a, th)
        ok = dss < 1e-3 and (a < 0.2 or abs(t0_ttm / t0_pol - 1) < 0.15)
        fails += not ok
        rows.append(dict(alpha=a, t0_ttm=t0_ttm, t0_polaron=t0_pol, ss_bures_vs_mfg=dss, bz_ss=b_ss[2], bz_mfg=b_mfg[2], tempo_seconds=sec))
        print(f"alpha={a}: t0 TTM={t0_ttm:.2f}  polaron={t0_pol:.2f}  ratio={t0_ttm/t0_pol:.3f}  |  <tz> ss={b_ss[2]:+.6f} MFG={b_mfg[2]:+.6f}, "
              f"Bures {dss:.1e}  ok={ok}", flush=True)
    with open(DATA / 't0_csk.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
