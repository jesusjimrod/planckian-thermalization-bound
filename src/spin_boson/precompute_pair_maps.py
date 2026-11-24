"""Precompute real-time maps for the optimal-pair members |x1|, |x2| of compute_finite_bound.py (needs
data/finite_bound.csv). Usage: precompute_pair_maps.py nproc alpha1 [alpha2 ...]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import sys
import numpy as np
import rt_maps
from compute_dynamics import CFG, MEMBERS, optimal_pair_members
if __name__ == '__main__':
    nproc = int(sys.argv[1]); alphas = [float(a) for a in sys.argv[2:]]
    pairs = []
    for a in alphas:
        xs = sorted(set(np.round(optimal_pair_members(a), 10)) - set(np.round(MEMBERS, 10)))
        pairs += [(a, (1.0, 0.0, float(x))) for x in xs]
        print(a, 'pair members:', [round(float(x), 4) for x in xs], flush=True)
    rt_maps.maps(pairs, CFG['dt'], CFG['t_maps'], CFG['epsrel'], nproc=nproc)
    print('done', alphas, flush=True)
