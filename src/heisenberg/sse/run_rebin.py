"""Reruns of every run whose re-binning factor (40 -> 8 bins) exceeds 1.3 for chi~ or G(beta/2)/Var
(rule in PROMPT.md). New independent seed, bins 2.5x longer (40 x 25000 sweeps). Tag suffix 'long'.
The flagged list is read from ../results/analysis.pkl (first analysis pass), and frozen in ../results/rebin_flagged.txt.
Usage: python run_rebin.py [ncores]
"""
import os, sys, pickle, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlib import run_one
from run_production import RAW, THERM, NBINS, tag as base_tag

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'heisenberg')
SPB_LONG = 25000
FLAG = os.path.join(RES, 'rebin_flagged.txt')


def flagged():
    if os.path.exists(FLAG):
        return [l.split() for l in open(FLAG) if l.strip()]
    D = pickle.load(open(os.path.join(RES, 'analysis.pkl'), 'rb'))
    out = []
    for k, d in D['loc'].items():
        if k[3].endswith('long'): continue
        if max(d['rebin'][0], d['rebin'][-1]) > 1.3:
            m = d['meta']
            out.append([str(m['L']), repr(m['T']), repr(m['betax']), k[3], f"{d['rebin'][0]:.3f}", f"{d['rebin'][-1]:.3f}", str(m['wall_s'])])
    out.sort(key=lambda r: -float(r[-1]))
    open(FLAG, 'w').write('\n'.join(' '.join(r) for r in out) + '\n')
    return out


def do(r):
    L, T, bx, s = int(r[0]), float(r[1]), float(r[2]), r[3]
    t = base_tag(L, T, bx, s + 'long')
    return run_one(RAW, t, L, 1 / T, bx * T, NBINS, SPB_LONG, THERM)


if __name__ == '__main__':
    J = flagged(); nc = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    print(len(J), 'reruns', flush=True); t0 = time.time()
    with Pool(nc) as p:
        for k, m in enumerate(p.imap_unordered(do, J)):
            print(f"[{k+1}/{len(J)} {(time.time()-t0)/60:6.1f} min] {m['tag']}: nops {m['nops']:.0f}, wall {m['wall_s']:.0f} s", flush=True)
    print('done', flush=True)
