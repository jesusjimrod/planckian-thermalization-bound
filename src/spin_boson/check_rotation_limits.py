"""Check 13: limits of the rotation-family bound (compute_finite_rotation.py), same pair optimizer.
Bare Gibbs: b(u) = -tanh(beta Delta/2) (cos u, 0, sin u). Pinned (Eq. sb_usc): b(u) = -tanh(beta Delta sin u/2) z_hat.
TEMPO rows at the smallest/largest coupling must approach these (tolerances 2e-3 and 1e-4).
Failure control: using the line distance |u1-u2| instead of the chord 2 sin(|u1-u2|/2) must miss the bare value by > 5e-3.
Exit 1 on failure."""
import csv, sys
import numpy as np
from compute_finite_bound import chi_delta
from compute_finite_rotation import PHI, dist, CFG

bare = (lambda u: -np.tanh(0.5) * np.cos(u), lambda u: -np.tanh(0.5) * np.sin(u))
pinned = (lambda u: 0.0 * u, lambda u: -np.tanh(0.5 * np.sin(u)))
rows = list(csv.DictReader(open('data/finite_bound_rot.csv')))
al = sorted({float(r['alpha']) for r in rows}); fails = 0
for ef in CFG['eps_fractions']:
    eps = ef * np.pi / 4
    cb, pb = chi_delta(*bare, eps, delta=PHI, dist=dist); cp, pp = chi_delta(*pinned, eps, delta=PHI, dist=dist)
    cbad, _ = chi_delta(*bare, eps, delta=PHI)
    lo = [r for r in rows if float(r['alpha']) == al[0] and abs(float(r['eps_fraction']) - ef) < 1e-12][0]
    hi = [r for r in rows if float(r['alpha']) == al[-1] and abs(float(r['eps_fraction']) - ef) < 1e-12][0]
    ok = abs(float(lo['chi']) - cb) < 2e-3 and abs(float(hi['chi']) - cp) < 1e-4; ctrl = abs(cbad - cb) > 5e-3
    fails += (not ok) + (not ctrl)
    print(f"eps={ef:.2f}: bare {cb:.5f} (TEMPO a_L={2*al[0]:g}: {float(lo['chi']):.5f})  pinned {cp:.5f} (TEMPO a_L={2*al[-1]:g}: {float(hi['chi']):.5f})  "
          f"pair u* bare {abs(pb[1]):.3f} / pinned {abs(pp[1]):.3f} rad  ok={ok}  control {cbad:.5f} {'rejected' if ctrl else 'NOT rejected'}")
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
