"""Disk cache and parallel runner for MFG states (GibbsTempo via sb_core.mfg_state).

Each computed state is appended as one JSON line to data/mfg_cache.jsonl, keyed by
(h, alpha, wc, beta, N, epsrel) rounded to 12 digits. Re-running a sweep only computes
missing states. ProcessPoolExecutor raises BrokenProcessPool if a worker dies
(multiprocessing.Pool would hang).
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
CACHE = DATA / 'mfg_cache.jsonl'


def key(h, alpha, wc, beta, N, epsrel):
    r = lambda v: float(np.round(float(v), 12))
    return (r(h[0]), r(h[1]), r(h[2]), r(alpha), r(wc), r(beta), int(N), float(epsrel))


def load():
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                rho = np.array(d['rho_re']) + 1j * np.array(d['rho_im'])
                out[tuple(d['key'])] = dict(rho=rho, b=np.array(d['b']), seconds=d['seconds'])
    return out


def _work(k):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import sb_core as sc
    h = k[:3]; alpha, wc, beta, N, eps = k[3:]
    t = time.time()
    rho = sc.mfg_state(h, alpha, wc, beta, N, epsrel=eps)
    return k, rho, time.time() - t


def ensure(keys, nproc=10, verbose=True):
    """Compute every missing key (in parallel) and return {key: entry} for all keys."""
    have = load()
    todo = sorted({tuple(k) for k in keys} - set(have), key=lambda k: -k[6])   # large N first
    if todo and verbose:
        print(f"[mfg_cache] {len(todo)} states to compute ({len(have)} cached)", flush=True)
    CACHE.parent.mkdir(exist_ok=True)
    if todo:
        with ProcessPoolExecutor(max_workers=nproc) as ex, open(CACHE, 'a') as f:
            futs = [ex.submit(_work, k) for k in todo]
            for i, fu in enumerate(as_completed(futs)):
                k, rho, sec = fu.result()
                b = [float(np.trace(rho @ s).real) for s in _paulis()]
                f.write(json.dumps(dict(key=list(k), b=b, rho_re=rho.real.tolist(),
                                        rho_im=rho.imag.tolist(), seconds=sec)) + '\n'); f.flush()
                have[k] = dict(rho=rho, b=np.array(b), seconds=sec)
                if verbose and (i + 1) % 20 == 0:
                    print(f"[mfg_cache] {i+1}/{len(todo)} done", flush=True)
    return {tuple(k): have[tuple(k)] for k in keys}


def _paulis():
    import sb_core as sc
    return sc.PAULI
