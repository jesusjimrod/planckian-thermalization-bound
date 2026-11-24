import numpy as np, subprocess, os, itertools
from scipy.linalg import eigh
from lib import write_input_g, jack
HERE = os.path.dirname(os.path.abspath(__file__))
TAUS = [0.0, 1/16, 1/8, 1/4, 3/8, 1/2]
def write_input_t(fn, bonds_g, model, beta, seed, nbins, spb, therm, Aobs, taus=TAUS, K=64, cpad=0.1):
    write_input_g(fn, bonds_g, model, beta, seed, nbins, spb, therm, cpad)
    with open(fn, 'a') as f:
        f.write(" ".join(repr(float(a)) for a in Aobs) + "\n")
        f.write(f"{len(taus)} " + " ".join(repr(float(t)) for t in taus) + f"\n{K}\n")
def run_t(fi, fo):
    with open(fo, 'w') as o: subprocess.run([os.path.join(HERE, 'sse_t'), fi], stdout=o, check=True)
def load_t(fn, beta, d, ntau=len(TAUS)):
    rows = [list(map(float, l.split())) for l in open(fn) if not l.startswith('#')]
    a = np.array(rows)
    return dict(n=a[:, 0], P=a[:, 1:1+d], X=a[:, 1+d:1+d+d*d].reshape(-1, d, d), G=a[:, 1+d+d*d:1+d+d*d+ntau], beta=beta, d=d)
def gobs(r, Aobs):
    """connected G(tau)/Var, r=xi_AA/(beta Var), Var; jackknife"""
    A = np.array(Aobs); b = r['beta']
    def f(m):
        mean = m['P'] @ A; var = m['P'] @ A**2 - mean**2
        xi = A @ (m['X'] - b*np.outer(m['P'], m['P'])) @ A
        return np.concatenate([(m['G'] - mean**2)/var, [xi/(b*var), var]])
    return jack({'P': r['P'], 'X': r['X'], 'G': r['G']}, f)
def ed_G(bonds_g, model, beta, Aobs, taus=TAUS):
    d, eps, Vint, Aw = model; N = eps.shape[0]; A = np.array(Aobs)
    states = np.array(list(itertools.product(range(d), repeat=N)))[:, ::-1]; tot = states.sum(1)
    blocks = []
    for Nt in np.unique(tot):
        idx = np.where(tot == Nt)[0]; S = states[idx]; index = {tuple(s): k for k, s in enumerate(S)}
        H = np.zeros((len(idx),)*2)
        for k, s in enumerate(S):
            H[k, k] += sum(eps[i, s[i]] for i in range(N))
            for (i, j, g) in bonds_g:
                H[k, k] += g*Vint[s[i], s[j]]
                for (a, bb) in [(i, j), (j, i)]:
                    if s[a]+1 < d and s[bb] >= 1 and Aw[s[a], s[bb]] != 0:
                        s2 = s.copy(); s2[a] += 1; s2[bb] -= 1; H[index[tuple(s2)], k] -= g*Aw[s[a], s[bb]]
        ev, V = eigh(H); blocks.append((ev, V, S))
    Emin = min(b[0].min() for b in blocks); Z = sum(np.exp(-beta*(b[0]-Emin)).sum() for b in blocks)
    G = np.zeros(len(taus)); mean = 0; sq = 0
    for ev, V, S in blocks:
        q = np.exp(-beta*(ev-Emin))/Z; Aop = V.T @ (A[S[:, 0]][:, None]*V)
        mean += (q*np.diag(Aop)).sum(); sq += (q*np.diag(Aop @ Aop)).sum()
        for k, t in enumerate(taus):
            tau = t*beta  # <A(tau)A> = sum_mn q_m |A_mn|^2 exp(-tau (E_n - E_m))
            G[k] += np.sum(q[:, None]*Aop**2*np.exp(-tau*(ev[None, :]-ev[:, None])))
    var = sq - mean**2
    return (G - mean**2)/var
