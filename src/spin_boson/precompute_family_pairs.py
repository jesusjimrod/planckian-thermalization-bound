"""Precompute real-time maps for the optimal-pair members only (not in the fixed member list) of a family.
Usage: precompute_family_pairs.py {bias|rot} nproc alpha1 [alpha2 ...]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import sys
import numpy as np
import rt_maps
from compute_dynamics import CFG, FAMILIES, optimal_pair_members
if __name__ == '__main__':
    fam, nproc, alphas = sys.argv[1], int(sys.argv[2]), [float(a) for a in sys.argv[3:]]
    fixed = set(np.round(FAMILIES[fam]['members'], 10)); jobs = []
    for a in alphas:
        xs = sorted(x for x in set(np.round(optimal_pair_members(a, fam), 10)) if min(abs(x - f) for f in fixed) > 1e-6)
        print(a, 'pair members:', [round(float(x), 4) for x in xs], flush=True)
        jobs += [(a, FAMILIES[fam]['h'](x)) for x in xs]
    rt_maps.maps(jobs, CFG['dt'], CFG['t_maps'], CFG['epsrel'], nproc=nproc)
    print('done pairs', fam, alphas, flush=True)
