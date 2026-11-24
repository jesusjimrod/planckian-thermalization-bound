import numpy as np, subprocess, os, itertools
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__))

def square_bonds(Lx, Ly):
    b = []
    for y in range(Ly):
        for x in range(Lx):
            i = x + Lx*y
            b.append((i, (x+1) % Lx + Lx*y))
            b.append((i, x + Lx*((y+1) % Ly)))
    return b

# ---- model definitions: return (d, eps[N,d], Vint[d,d], Aw[d,d]) ----
def heis(N, J=1.0, x=0.0):
    """H = J sum S_i.S_j + (x/2) sigma^z_0 ; occupation n=1 is spin up. Sublattice-rotated."""
    d = 2
    eps = np.zeros((N, d)); eps[0] = [-x/2, x/2]
    sz = np.array([-0.5, 0.5])
    Vint = J*np.outer(sz, sz)
    Aw = np.zeros((d, d)); Aw[0, 1] = J/2
    return d, eps, Vint, Aw

def bh(N, U, mu, nmax, t=1.0, x=0.0, I=(0, 1)):
    """H = -t sum (b_i^+ b_j + h.c.) + U/2 n(n-1) - mu n + x P_I(site 0)"""
    d = nmax+1
    n = np.arange(d)
    e = U/2*n*(n-1) - mu*n
    eps = np.tile(e, (N, 1)).astype(float)
    for k in I: eps[0, k] += x
    Vint = np.zeros((d, d))
    Aw = np.zeros((d, d))
    for a in range(d):
        for b in range(d):
            if a+1 < d and b >= 1: Aw[a, b] = t*np.sqrt((a+1)*b)
    return d, eps, Vint, Aw

def write_input(fn, bonds, model, beta, seed, nbins, spb, therm, cpad=0.1):
    d, eps, Vint, Aw = model
    N = eps.shape[0]
    with open(fn, 'w') as f:
        f.write(f"{N} {len(bonds)} {d} {beta!r} {seed} {nbins} {spb} {therm} {cpad!r}\n")
        for i, j in bonds: f.write(f"{i} {j}\n")
        for r in eps: f.write(" ".join(repr(float(v)) for v in r) + "\n")
        for r in Vint: f.write(" ".join(repr(float(v)) for v in r) + "\n")
        for r in Aw: f.write(" ".join(repr(float(v)) for v in r) + "\n")


def load(fn_out, beta):
    Csum = None
    rows = []
    for line in open(fn_out):
        if line.startswith('#'):
            Csum = float(line.split()[2]); continue
        rows.append([float(v) for v in line.split()])
    a = np.array(rows)
    d = int(round((-1 + np.sqrt(1 + 4*(a.shape[1]-1)))/2))
    return dict(n=a[:, 0], P=a[:, 1:1+d], X=a[:, 1+d:].reshape(-1, d, d), Csum=Csum, beta=beta, d=d)

def jack(data, f):
    """data: dict of per-bin arrays (first axis bins); f(dict of means) -> array. returns mean, err"""
    keys = list(data.keys()); nb = len(data[keys[0]])
    full = f({k: data[k].mean(0) for k in keys})
    sums = {k: data[k].sum(0) for k in keys}
    js = np.array([f({k: (sums[k]-data[k][i])/(nb-1) for k in keys}) for i in range(nb)])
    mj = js.mean(0)
    err = np.sqrt((nb-1)/nb*((js-mj)**2).sum(0))
    return full, err

def obs(r):
    """returns E, p, xi(connected)"""
    beta = r['beta']; C = r['Csum']
    def f(m):
        E = -m['n']/beta + C
        p = m['P']
        xi = m['X'] - beta*np.outer(p, p)
        return np.concatenate([[E], p, xi.ravel()])
    v, e = jack({'n': r['n'], 'P': r['P'], 'X': r['X']}, f)
    d = r['d']
    return dict(E=(v[0], e[0]), p=(v[1:1+d], e[1:1+d]), xi=(v[1+d:].reshape(d, d), e[1+d:].reshape(d, d)))

# ---------------- exact diagonalization (statics) -----------------
def ed(bonds, model, beta):
    d, eps, Vint, Aw = model
    N = eps.shape[0]
    deg = np.zeros(N, int)
    for i, j in bonds: deg[i] += 1; deg[j] += 1
    states = np.array(list(itertools.product(range(d), repeat=N)))[:, ::-1]  # states[:,i] = n_i
    tot = states.sum(1)
    res_E, res_w, res_p, res_xi = [], [], [], []
    blocks = []
    for Ntot in np.unique(tot):
        idx = np.where(tot == Ntot)[0]; S = states[idx]
        index = {tuple(s): k for k, s in enumerate(S)}
        D = len(idx); H = np.zeros((D, D))
        for k, s in enumerate(S):
            H[k, k] += sum(eps[i, s[i]] for i in range(N))
            for (i, j) in bonds:
                H[k, k] += Vint[s[i], s[j]]
                for (a, b) in [(i, j), (j, i)]:
                    if s[a]+1 < d and s[b] >= 1 and Aw[s[a], s[b]] != 0:
                        s2 = s.copy(); s2[a] += 1; s2[b] -= 1
                        H[index[tuple(s2)], k] -= Aw[s[a], s[b]]
        ev, V = eigh(H)
        blocks.append((ev, V, S))
    Emin = min(b[0].min() for b in blocks)
    Z = sum(np.exp(-beta*(b[0]-Emin)).sum() for b in blocks)
    E = 0; p = np.zeros(d); xi = np.zeros((d, d))
    for ev, V, S in blocks:
        q = np.exp(-beta*(ev-Emin))/Z
        E += (q*ev).sum()
        # projectors on site 0 in eigenbasis
        Pk = [V.T @ ((S[:, 0] == k)[:, None]*V) for k in range(d)]
        # Kubo weights mu_mn
        Em, En = np.meshgrid(ev, ev, indexing='ij')
        qm, qn = np.meshgrid(q, q, indexing='ij')
        dE = En-Em
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = np.where(np.abs(dE) > 1e-10, (qm-qn)/(beta*dE), qm)
        for k in range(d):
            p[k] += (q*np.diag(Pk[k])).sum()
            for l in range(d):
                xi[k, l] += beta*np.sum(mu*Pk[k]*Pk[l].T)
    xi -= beta*np.outer(p, p)
    return E, p, xi

# ---- variant with per-bond coupling multipliers (bond = (i, j, g)) ----
def write_input_g(fn, bonds_g, model, beta, seed, nbins, spb, therm, cpad=0.1):
    d, eps, Vint, Aw = model
    N = eps.shape[0]
    with open(fn, 'w') as f:
        f.write(f"{N} {len(bonds_g)} {d} {beta!r} {seed} {nbins} {spb} {therm} {cpad!r}\n")
        for i, j, g in bonds_g: f.write(f"{i} {j} {g!r}\n")
        for r in eps: f.write(" ".join(repr(float(v)) for v in r) + "\n")
        for r in Vint: f.write(" ".join(repr(float(v)) for v in r) + "\n")
        for r in Aw: f.write(" ".join(repr(float(v)) for v in r) + "\n")

def star_bonds(Lx, Ly, g):
    """square lattice; bonds touching site 0 get multiplier g, others 1"""
    return [(i, j, g if 0 in (i, j) else 1.0) for i, j in square_bonds(Lx, Ly)]
def ed_g(bonds_g, model, beta):
    """ED with per-bond multipliers: scale Vint and Aw per bond"""
    d, eps, Vint, Aw = model
    import itertools
    N = eps.shape[0]
    states = np.array(list(itertools.product(range(d), repeat=N)))[:, ::-1]
    tot = states.sum(1); blocks = []
    for Ntot in np.unique(tot):
        idx = np.where(tot == Ntot)[0]; S = states[idx]
        index = {tuple(s): k for k, s in enumerate(S)}
        D = len(idx); H = np.zeros((D, D))
        for k, s in enumerate(S):
            H[k, k] += sum(eps[i, s[i]] for i in range(N))
            for (i, j, g) in bonds_g:
                H[k, k] += g*Vint[s[i], s[j]]
                for (a, b) in [(i, j), (j, i)]:
                    if s[a]+1 < d and s[b] >= 1 and Aw[s[a], s[b]] != 0:
                        s2 = s.copy(); s2[a] += 1; s2[b] -= 1
                        H[index[tuple(s2)], k] -= g*Aw[s[a], s[b]]
        ev, V = eigh(H); blocks.append((ev, V, S))
    Emin = min(b[0].min() for b in blocks)
    Z = sum(np.exp(-beta*(b[0]-Emin)).sum() for b in blocks)
    E = 0; p = np.zeros(d); xi = np.zeros((d, d))
    for ev, V, S in blocks:
        q = np.exp(-beta*(ev-Emin))/Z; E += (q*ev).sum()
        Pk = [V.T @ ((S[:, 0] == k)[:, None]*V) for k in range(d)]
        Em, En = np.meshgrid(ev, ev, indexing='ij'); qm, qn = np.meshgrid(q, q, indexing='ij'); dE = En-Em
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = np.where(np.abs(dE) > 1e-10, (qm-qn)/(beta*dE), qm)
        for k in range(d):
            p[k] += (q*np.diag(Pk[k])).sum()
            for l in range(d): xi[k, l] += beta*np.sum(mu*Pk[k]*Pk[l].T)
    xi -= beta*np.outer(p, p)
    return E, p, xi
