"""Report which couplings have complete real-time map sets (fixed members + optimal-pair members)."""
import sys
import numpy as np
import rt_maps
from compute_dynamics import CFG, MEMBERS, optimal_pair_members
alphas = [float(a) for a in sys.argv[1:]] or [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
for a in alphas:
    mem = sorted(set(np.round(MEMBERS + optimal_pair_members(a), 10)))
    missing = 0
    for x in mem:
        h = (1.0, 0.0, float(x))
        ok4 = all(rt_maps._cached(a, h, i, CFG['dt'], CFG['t_maps'], CFG['epsrel']) for i in rt_maps.SCHEME4)
        ok3 = all(rt_maps._cached(a, h, i, CFG['dt'], CFG['t_maps'], CFG['epsrel']) for i in rt_maps.SCHEME3)
        missing += not (ok3 or ok4)
    print(f"alpha={a:g}: {len(mem)-missing}/{len(mem)} members complete")
