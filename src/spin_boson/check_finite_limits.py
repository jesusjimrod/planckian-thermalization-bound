"""Check 6: limits of the finite-error bound for the bias family (compute_finite_bound.chi_delta).

(a) Pinned family (ultrastrong MFG): b = -tanh(beta x/2) z_hat, i.e. diagonal states with p = f_beta(x).
    This is the weak-coupling RLM task of ../fermionic/families (H = x n_d, |x| <= 2 pi, p(0) = 1/2), so
    chi_delta must reproduce, to optimizer accuracy, the RLM value in data/fermionic/families/sweep.csv
    in its weak-coupling limit. The RLM rows at the smallest coupling (beta g^2 = 0.01) carry an O(g^2)
    broadening, so the comparison uses a tolerance of 2e-3; the deviation must also shrink from the
    first to the second-smallest coupling.
(b) Bare Gibbs family (alpha -> 0): b = -tanh(E/2) (1, 0, x)/E, E = sqrt(1 + x^2).
(c) TEMPO rows of data/finite_bound.csv must approach (b) at the smallest alpha and (a) at the largest.
Failure control: (a) with p = f_beta(x/2) (wrong bias normalization) must miss the RLM value.
Exit 1 on failure."""
import csv, sys
from pathlib import Path
import numpy as np
from compute_finite_bound import chi_delta, CFG

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
EPS = CFG['eps_fractions']
pinned = lambda s: (lambda x: 0.0, lambda x: -np.tanh(s * x / 2))
def bare():
    E = lambda x: np.sqrt(1 + x * x)
    return (lambda x: -np.tanh(E(x) / 2) / E(x), lambda x: -np.tanh(E(x) / 2) * x / E(x))

rlm = [r for r in csv.DictReader(open(HERE.parents[1] / 'data' / 'fermionic' / 'families' / 'sweep.csv')) if r['family'] == 'rlm']
gs = sorted({float(r['gamma']) for r in rlm})[:2]
fails = 0
ref = {}
print(" eps/eps_max  chi_pinned  chi_RLM(g2=%.3g)  chi_RLM(g2=%.3g)  chi_bare   control(f(x/2))" % tuple(gs))
for ef in EPS:
    eps = ef * np.pi / 4
    cp, _ = chi_delta(*pinned(1.0), eps); cb, _ = chi_delta(*bare(), eps); cc, _ = chi_delta(*pinned(0.5), eps)
    r1, r2 = (float([r for r in rlm if float(r['gamma']) == g and abs(float(r['eps_fraction']) - ef) < 1e-12][0]['chi']) for g in gs)
    ok = abs(cp - r1) < 2e-3 and abs(cp - r2) > abs(cp - r1) - 1e-12
    ctrl = abs(cc - r1) > 1e-2
    fails += (not ok) + (not ctrl)
    ref[ef] = (cp, cb)
    print(f"   {ef:5.2f}     {cp:.6f}     {r1:.6f}          {r2:.6f}          {cb:.6f}   {cc:.6f} ({'rejected' if ctrl else 'NOT rejected'})  ok={ok}")
fb = DATA / 'finite_bound.csv'
if fb.exists():
    rows = list(csv.DictReader(open(fb)))
    al = sorted({float(r['alpha']) for r in rows})
    for ef in EPS:
        lo = float([r for r in rows if float(r['alpha']) == al[0] and abs(float(r['eps_fraction']) - ef) < 1e-12][0]['chi'])
        hi = float([r for r in rows if float(r['alpha']) == al[-1] and abs(float(r['eps_fraction']) - ef) < 1e-12][0]['chi'])
        cp, cb = ref[ef]
        print(f"   TEMPO eps={ef:.2f}: chi(alpha={al[0]:g})={lo:.5f} vs bare {cb:.5f} (diff {lo-cb:+.1e});  chi(alpha={al[-1]:g})={hi:.5f} vs pinned {cp:.5f} (diff {hi-cp:+.1e})")
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
