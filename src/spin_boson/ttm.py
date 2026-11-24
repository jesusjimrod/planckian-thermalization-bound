"""Dynamical maps from an OQuPy process tensor, and transfer-tensor propagation (Cerrillo & Cao,
PRL 112, 110401 (2014)) to long times.

Vectorization: row-major, vec(rho)[2a+b] = rho[a, b]; a map E acts as vec(E(rho)) = E @ vec(rho).
"""
import numpy as np
import oqupy

E00 = np.array([[1, 0], [0, 0]], complex); E11 = np.array([[0, 0], [0, 1]], complex)
PLUS = 0.5 * np.array([[1, 1], [1, 1]], complex); PLUSI = 0.5 * np.array([[1, -1j], [1j, 1]], complex)


def maps_from_pt(H, pt, progress_type='silent'):
    """E_n (n = 0..N) for system Hamiltonian H, from four physical initial states."""
    runs = {}
    for name, r0 in (('00', E00), ('11', E11), ('p', PLUS), ('pi', PLUSI)):
        dyn = oqupy.compute_dynamics(oqupy.System(H), r0, process_tensor=pt, progress_type=progress_type)
        runs[name] = np.array(dyn.states)                   # (N+1, 2, 2)
    t = np.array(dyn.times)
    I = runs['00'] + runs['11']
    P = 2 * runs['p'] - I                                    # image of |0><1| + |1><0|
    Q = 2 * runs['pi'] - I                                   # image of -i|0><1| + i|1><0|
    img = {(0, 0): runs['00'], (1, 1): runs['11'], (0, 1): 0.5 * (P + 1j * Q), (1, 0): 0.5 * (P - 1j * Q)}
    E = np.zeros((len(t), 4, 4), complex)
    for (a, b), im in img.items():
        E[:, :, 2 * a + b] = im.reshape(len(t), 4)
    return t, E


def transfer_tensors(E, K):
    """T_1..T_K from E_0..E_K (E_0 = identity)."""
    T = np.zeros((K + 1, 4, 4), complex)
    for n in range(1, K + 1):
        acc = E[n].copy()
        for m in range(1, n):
            acc -= T[m] @ E[n - m]
        T[n] = acc
    return T


def propagate(E, K, nmax):
    """E_n for n = 0..nmax: exact for n <= K (input), transfer-tensor recursion beyond."""
    T = transfer_tensors(E, K)
    out = np.zeros((nmax + 1, 4, 4), complex)
    out[:K + 1] = E[:K + 1]
    Tm = T[1:]                                               # T_1..T_K
    for n in range(K + 1, nmax + 1):
        out[n] = np.einsum('mij,mjk->ik', Tm, out[n - K:n][::-1])   # sum_m T_m E_{n-m}
    return out, T


def apply(Es, rho0):
    v = Es @ rho0.reshape(4)
    return v.reshape(-1, 2, 2)


def propagate_state(E, K, rho0, nmax):
    """rho_n (n = 0..nmax) for one initial state: exact maps for n <= K, transfer tensors beyond."""
    T = transfer_tensors(E, K)
    v = np.zeros((nmax + 1, 4), complex)
    v[:K + 1] = E[:K + 1] @ rho0.reshape(4)
    Tm = T[1:]
    for n in range(K + 1, nmax + 1):
        v[n] = np.einsum('mij,mj->i', Tm, v[n - K:n][::-1])
    return v.reshape(-1, 2, 2), T


def companion_rates(T, dt):
    """Eigenvalues of the companion matrix of the recursion E_n = sum_m T_m E_{n-m}: the unit eigenvalue
    (trace preservation) and the decay rates -ln|lambda|/dt of all other modes, sorted slowest first."""
    K = len(T) - 1
    C = np.zeros((4 * K, 4 * K), complex)
    for m in range(1, K + 1):
        C[:4, 4 * (m - 1):4 * m] = T[m]
    C[4:, :-4] = np.eye(4 * (K - 1))
    lam = np.linalg.eigvals(C)
    lam = lam[np.argsort(-abs(lam))]
    rates = -np.log(np.abs(lam)) / dt
    return lam, rates


def steady_state(T):
    """Fixed point of the recursion: null vector of 1 - sum_m T_m, normalized to unit trace."""
    M = np.eye(4) - T[1:].sum(0)
    w, V = np.linalg.eig(M)
    v = V[:, np.argmin(abs(w))].reshape(2, 2)
    return v / np.trace(v)
