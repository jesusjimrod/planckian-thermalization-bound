"""Dynamics-free upper bound on the Shiraishi-Saito bound for the bias family, for several cutoffs.

tau_SS <= tau_SS^up = (D0_max - eps) / L(alpha, wc, T), with L = max_w [W/(p0 + 2 X_w) - Delta/2] (ss_bound.py)
and D0_max = max_x D(I/2, w_x). For wc = 10 the MFG family gives D0_max in [0.7367, 0.7436] over the coupling range
(finite_bound.csv). For the other cutoffs D0_max = 0.744 is ASSUMED (MFG families not computed there); the
trivial cap D0_max <= pi/2 would multiply tau_SS^up by at most 2.1 and leaves the 1/wc scaling unchanged.
Outputs data/ss_upper.csv.
"""
import csv
from pathlib import Path
import numpy as np
import ss_bound

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
ALPHAS = np.logspace(-3, np.log10(5.0), 31)
WCS = [10.0, 30.0, 100.0, 300.0, 1000.0]
EPS = [0.01, 0.05, 0.1, 0.15, 0.2]
D0 = 0.744

if __name__ == '__main__':
    rows = []
    for wc in WCS:
        for a in ALPHAS:
            L, p, q = ss_bound.best_lower_bound_dE(a, wc, 1.0, 1.0)
            for ef in EPS:
                eps = ef * np.pi / 4
                rows.append(dict(wc=wc, alpha=a, alpha_L=2 * a, eps_fraction=ef, L_dE=L, p=p, q=q,
                                 tau_SS_up=((D0 - eps) / L if L > 0 else np.inf)))
        print(f"wc={wc:g}: L(alpha=0.1)={[r['L_dE'] for r in rows if r['wc']==wc and abs(r['alpha']-ALPHAS[np.argmin(abs(ALPHAS-0.1))])<1e-12][0]:.2f}", flush=True)
    with open(DATA / 'ss_upper.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote data/ss_upper.csv')
