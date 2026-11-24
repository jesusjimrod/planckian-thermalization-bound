"""Dynamical maps of the reduced spin dynamics from OQuPy real-time TEMPO (no memory truncation
within the map window), cached on disk, for transfer-tensor propagation (ttm.py).

One TEMPO run per (alpha, h, initial state, dt, t_end, epsrel); four initial states give the map:
|0><0|, |1><1|, |+><+|, |+i><+i|. Factorized initial state rho_S(0) (x) omega_B, omega_B thermal at T.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import hashlib, json, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
MAPDIR = DATA / 'maps'
INIT = {'00': np.array([[1, 0], [0, 0]], complex), '11': np.array([[0, 0], [0, 1]], complex),
        'p': 0.5 * np.array([[1, 1], [1, 1]], complex), 'pi': 0.5 * np.array([[1, -1j], [1j, 1]], complex),
        '01': np.array([[0, 1], [0, 0]], complex)}
# Scheme '3' (default): |0><0|, |1><1| and the non-Hermitian |0><1| (TEMPO is linear in the initial
# matrix), with E(|1><0|) = E(|0><1|)^dagger. Scheme '4' (|+>, |+i> instead of |0><1|) is used when
# such runs are already cached. check_maps_scheme.py compares the two.
SCHEME3 = ('00', '11', '01'); SCHEME4 = ('00', '11', 'p', 'pi')


def _tag(alpha, h, init, dt, t_end, epsrel, wc, T):
    s = json.dumps([round(alpha, 12), [round(v, 12) for v in h], init, dt, t_end, epsrel, wc, T])
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _robust_svd():
    """OQuPy's real-time backend calls np.linalg.svd (LAPACK gesdd) through tensornetwork without a fallback;
    gesdd occasionally fails to converge ('SVD did not converge', seen at alpha = 0.3-0.5). Fall back to gesvd,
    as OQuPy's imaginary-time backend already does. Patched in the worker process only."""
    import numpy.linalg as nla, scipy.linalg as sla
    if getattr(nla.svd, '_robust', False):
        return
    orig = nla.svd
    def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
        try:
            return orig(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
        except nla.LinAlgError:
            return sla.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver='gesvd')
    svd._robust = True
    nla.svd = svd; np.linalg.svd = svd


_INDEX = None


def _find_cached(alpha, h, init, dt, t_end, epsrel, wc=10.0, T=1.0):
    """Path of a cached run for these parameters: exact tag first, otherwise any cached run with the same
    settings and |h - h'| < 1e-9 (the tag rounds h to 12 digits, so Hamiltonians that differ only by float
    rounding of the member parameter would otherwise be recomputed)."""
    global _INDEX
    path = MAPDIR / f"{_tag(alpha, h, init, dt, t_end, epsrel, wc, T)}.npz"
    if path.exists():
        return path
    if _INDEX is None:
        _INDEX = []
        for f in MAPDIR.glob('*.npz'):
            try:
                m = json.loads(str(np.load(f)['meta']))
                _INDEX.append((m, f))
            except Exception:
                pass
    for m, f in _INDEX:
        if (m['init'] == init and abs(m['alpha'] - alpha) < 1e-12 and m['dt'] == dt and m['t_end'] == t_end
                and m['epsrel'] == epsrel and m['wc'] == wc and m['T'] == T
                and max(abs(a - b) for a, b in zip(m['h'], h)) < 1e-9):
            return f
    return None


def run_one(alpha, h, init, dt, t_end, epsrel, wc=10.0, T=1.0):
    warnings.filterwarnings('ignore')
    _robust_svd()
    import oqupy, sb_core as sc
    path = MAPDIR / f"{_tag(alpha, h, init, dt, t_end, epsrel, wc, T)}.npz"
    found = _find_cached(alpha, h, init, dt, t_end, epsrel, wc, T)
    if found is not None:
        d = np.load(found); return d['t'], d['states'], float(d['seconds'])
    bath = oqupy.Bath(sc.SZ, sc.MatsubaraSD(alpha, wc, T))
    pars = oqupy.TempoParameters(dt=dt, epsrel=epsrel)          # no memory cutoff
    t0 = time.time()
    dyn = oqupy.Tempo(oqupy.System(sc.hamiltonian(h)), bath, pars, INIT[init], 0.0).compute(end_time=t_end, progress_type='silent')
    sec = time.time() - t0
    t, st = np.array(dyn.times), np.array(dyn.states)
    MAPDIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, t=t, states=st, seconds=sec, meta=json.dumps(dict(alpha=alpha, h=list(h), init=init, dt=dt, t_end=t_end, epsrel=epsrel, wc=wc, T=T)))
    return t, st, sec


def assemble(runs):
    """runs: dict init -> (N+1,2,2) states. Returns E (N+1,4,4), row-major vectorization."""
    if '01' in runs:
        img = {(0, 0): runs['00'], (1, 1): runs['11'], (0, 1): runs['01'],
               (1, 0): np.conj(np.transpose(runs['01'], (0, 2, 1)))}
    else:
        I = runs['00'] + runs['11']; P = 2 * runs['p'] - I; Q = 2 * runs['pi'] - I
        img = {(0, 0): runs['00'], (1, 1): runs['11'], (0, 1): 0.5 * (P + 1j * Q), (1, 0): 0.5 * (P - 1j * Q)}
    n = len(runs["00"]); E = np.zeros((n, 4, 4), complex)
    for (a, b), im in img.items():
        E[:, :, 2 * a + b] = im.reshape(n, 4)
    return E


def _job(args):
    return args, run_one(*args)


def _cached(a, h, init, dt, t_end, epsrel, wc=10.0, T=1.0):
    return _find_cached(a, h, init, dt, t_end, epsrel, wc, T) is not None


def maps(alpha_h_list, dt, t_end, epsrel, nproc=10, verbose=True, scheme=None):
    """alpha_h_list: list of (alpha, h). Returns {(alpha, tuple(h)): (t, E, seconds_total)}."""
    jobs = []
    for a, h in alpha_h_list:
        h = tuple(map(float, h))
        sch = scheme or (SCHEME4 if all(_cached(a, h, i, dt, t_end, epsrel) for i in SCHEME4) else SCHEME3)
        jobs += [(a, h, init, dt, t_end, epsrel) for init in sch]
    out = {}
    if verbose:
        print(f"[rt_maps] {len(jobs)} TEMPO runs (dt={dt}, t_end={t_end}, epsrel={epsrel:.0e})", flush=True)
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs)):
            (a, h, init, *_), (t, st, sec) = fu.result()
            out.setdefault((a, h), {})[init] = (t, st, sec)
            if verbose and (i + 1) % 10 == 0:
                print(f"[rt_maps] {i+1}/{len(jobs)} done", flush=True)
    res = {}
    for k, d in out.items():
        t = d['00'][0]
        res[k] = (t, assemble({i: v[1] for i, v in d.items()}), sum(v[2] for v in d.values()))
    return res
