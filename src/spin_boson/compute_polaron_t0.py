"""Polaron-master-equation population decay time t0 = 1/(G+ + G-) (CSK Eqs. 10-11) for the unbiased member
(x = 0: Delta_perp = Delta, no bias), which is the slowest member at strong coupling. Theory line for the
relaxation panel beyond the real-time window; it agrees with the TTM t0 to 1.6% (alpha_L = 1) and 0.5%
(alpha_L = 1.4). Outputs data/polaron_t0.csv."""
import csv
from pathlib import Path
import numpy as np
from check_t0_csk import t0_polaron

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
if __name__ == '__main__':
    rows = []
    for a in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        t0 = t0_polaron(a, np.pi / 2)
        rows.append(dict(alpha=a, alpha_L=2 * a, t0_polaron_x0=t0)); print(f"alpha_L={2*a:g}: t0={t0:.4g}", flush=True)
    with open(DATA / 'polaron_t0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
