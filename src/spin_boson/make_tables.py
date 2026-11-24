"""Print the README result tables directly from the data files (no hand-copied numbers)."""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
D = list(csv.DictReader(open(DATA / 'dynamics.csv')))
B = {(round(float(r['alpha']), 12), round(float(r['eps_fraction']), 12)): r for r in csv.DictReader(open(DATA / 'finite_bound.csv'))}
f = lambda r, k: float(r[k])

print('| α_L | t₀ slowest member | τ_ε (0.01) | τ_ε (0.05) [bracket] | τ_ε (0.2) | τ_Pl χ_δ (0.05) | τ_SS^up (0.05) | τ_QFI/τ_ε (0.05) | steady-state η |')
print('|---|---|---|---|---|---|---|---|---|')
for a in sorted({f(r, 'alpha') for r in D}):
    rr = {round(f(r, 'eps_fraction'), 12): r for r in D if f(r, 'alpha') == a}
    r1, r5, r20 = rr[0.01], rr[0.05], rr[0.2]
    b5 = B[(round(a, 12), 0.05)]
    print(f"| {2*a:g} | {f(r5,'t0_max'):.3g} | {f(r1,'tau'):.4g} | {f(r5,'tau'):.4g} [{f(r5,'tau_lo'):.4g}, {f(r5,'tau_hi'):.4g}] | {f(r20,'tau'):.4g} "
          f"| {f(b5,'chi'):.4f} | {f(b5,'tau_SS_up'):.3g} | {f(r5,'tau_qfi')/f(r5,'tau'):.3f} | {f(r5,'ss_err_max'):.1e} |")

lt = DATA / 'local_lowT.csv'
if lt.exists():
    print('\n| α_L | χ_zz/β at βΔ = 4 | weak limit | Richardson change |')
    print('|---|---|---|---|')
    for r in csv.DictReader(open(lt)):
        print(f"| {f(r,'alpha_L'):g} | {f(r,'chi'):.5f} | {f(r,'weak'):.5f} | {f(r,'richardson'):.1e} |")
