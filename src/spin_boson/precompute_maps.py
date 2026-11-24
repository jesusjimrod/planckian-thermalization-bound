"""Precompute (cache) real-time TEMPO maps for the fixed members of the bias family (rt_maps.py).
Usage: precompute_maps.py nproc alpha1 [alpha2 ...]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import sys
import rt_maps
from compute_dynamics import CFG, MEMBERS
if __name__ == '__main__':
    nproc = int(sys.argv[1]); alphas = [float(a) for a in sys.argv[2:]]
    rt_maps.maps([(a, (1.0, 0.0, float(round(x, 10)))) for a in alphas for x in MEMBERS], CFG['dt'], CFG['t_maps'], CFG['epsrel'], nproc=nproc)
    print('done', alphas, flush=True)
