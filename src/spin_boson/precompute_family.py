"""Precompute (cache) real-time TEMPO maps for all members (fixed + optimal pairs) of a family.
Usage: precompute_family.py {bias|rot} nproc alpha1 [alpha2 ...]  (pairs need the family's finite_bound csv)"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import sys
import rt_maps
from compute_dynamics import CFG, FAMILIES, family_members
if __name__ == '__main__':
    fam, nproc, alphas = sys.argv[1], int(sys.argv[2]), [float(a) for a in sys.argv[3:]]
    def members(a):
        try:
            return family_members(a, fam)
        except FileNotFoundError:                     # bound not computed yet: fixed members only
            return list(FAMILIES[fam]['members'])
    pairs = [(a, FAMILIES[fam]['h'](x)) for a in alphas for x in members(a)]
    rt_maps.maps(pairs, CFG['dt'], CFG['t_maps'], CFG['epsrel'], nproc=nproc)
    print('done', fam, alphas, flush=True)
