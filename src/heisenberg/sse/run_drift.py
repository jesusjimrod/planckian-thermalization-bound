"""Reruns (new seed, 40 x 25000 sweeps, tag suffix 'long') of runs whose drift test |z| > 3
(first 10 vs last 10 bins) for chi~ or G(beta/2)/Var. List frozen in ../results/drift_flagged.txt."""
import os, sys, pickle
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_rebin import do, RES
FLAG = os.path.join(RES, 'drift_flagged.txt')
if __name__ == '__main__':
    if not os.path.exists(FLAG):
        D = pickle.load(open(os.path.join(RES, 'analysis.pkl'), 'rb')); rows = []
        for k, d in D['loc'].items():
            if not k[3].endswith('long') and max(abs(d['drift'][0]), abs(d['drift'][-1])) > 3:
                m = d['meta']; rows.append([str(m['L']), repr(m['T']), repr(m['betax']), k[3], f"{d['drift'][0]:.2f}", f"{d['drift'][-1]:.2f}", str(m['wall_s'])])
        open(FLAG, 'w').write('\n'.join(' '.join(r) for r in rows) + '\n')
    J = [l.split() for l in open(FLAG) if l.strip()]
    print(len(J), 'drift reruns', flush=True)
    with Pool(3) as p:
        for m in p.imap_unordered(do, J): print(m['tag'], m['wall_s'], flush=True)
    print('done', flush=True)
