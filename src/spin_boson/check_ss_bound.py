"""Check 5: the dynamics-free lower bound on delta E_S (ss_bound.py) on an exactly solvable
finite bath (qubit + 2 bosonic modes, exact diagonalization, factorized thermal initial state).

Tested, at every time on a grid:
 (I1) <P_w(t)> = -2 int_0^t chi_w(t-s) <sigma_z(s)> ds   (exact bath response; sharp identity)
 (I2) <[H_S+V, sigma_z P_w]> = i (2W + <(h_y s_x - h_x s_y) P_w>)   (commutator identity)
 (B1) <P_w(t)^2>^(1/2) <= p0 + 2 X_w(t)
 (B2) delta E_S(t) >= W/(p0 + 2 X_w(t)) - h_perp/2
Failure controls, each through the same comparison as the claim it targets:
 'sin' : chi_w with sin instead of cos           -> must break (I1)
 'half': factor 1 instead of 2 in the response   -> must break (I1)
 'noHS': drop the [H_S, sigma_z] P_w term        -> must break (I2)
Exit 1 if a claim fails or a control is not rejected."""
import sys
import numpy as np

beta = 1.0
nus = np.array([1.7, 4.3]); gs = np.array([0.9, 1.6]); nmax = 28
hx, hz = 1.0, 0.6
w = nus ** 0.5                                   # weight w(nu) = nu^0.5 (any real weight is allowed)
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1., -1]).astype(complex)

a = np.diag(np.sqrt(np.arange(1, nmax)), 1).astype(complex); I_n = np.eye(nmax)
def mode_op(op, k):
    ops = [I_n] * len(nus); ops[k] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out
dimB = nmax ** len(nus); IB = np.eye(dimB); I2 = np.eye(2)
bk = [mode_op(a, k) for k in range(len(nus))]
HB = sum(nus[k] * bk[k].conj().T @ bk[k] for k in range(len(nus)))
B = sum(gs[k] * (bk[k] + bk[k].conj().T) for k in range(len(nus)))
Pw = sum(1j * w[k] * gs[k] * (bk[k].conj().T - bk[k]) for k in range(len(nus)))
HS = 0.5 * (hx * sx + hz * sz)
H = np.kron(HS, IB) + np.kron(I2, HB) + np.kron(sz, B)
A = np.kron(HS, IB) + np.kron(sz, B)                     # H_S + V
Y = np.kron(sz, Pw)
SZ = np.kron(sz, IB); PW = np.kron(I2, Pw); OPERP = np.kron(-hx * sy, IB)   # h_y = 0

eB, UB = np.linalg.eigh(HB); pB = np.exp(-beta * (eB - eB.min())); pB /= pB.sum()
rhoB = (UB * pB) @ UB.conj().T
rhoS = np.diag([0.5, 0.5]).astype(complex)
rho0 = np.kron(rhoS, rhoB)
top = sum(np.trace(rhoB @ mode_op(np.diag((np.arange(nmax) == nmax - 1).astype(float)), k)).real for k in range(len(nus)))

E, U = np.linalg.eigh(H)
r0 = U.conj().T @ rho0 @ U
ts = np.linspace(0, 12, 241)
def state(t):
    ph = np.exp(-1j * E * t)
    return U @ (ph[:, None] * r0 * ph.conj()[None, :]) @ U.conj().T

W = float(np.sum(gs ** 2 * w))
p0 = float(np.sqrt(np.sum(gs ** 2 * w ** 2 / np.tanh(beta * nus / 2))))
def chi_w(s, kind='cos'):
    f = np.cos if kind == 'cos' else np.sin
    return np.sum(gs[:, None] ** 2 * w[:, None] * f(np.outer(nus, s)), axis=0)

A2 = A @ A; PW2 = PW @ PW; COMM = A @ Y - Y @ A; PERPP = OPERP @ PW
rows = []
for t in ts:
    r = state(t)
    ex = lambda O: np.sum(r.T * O)          # Tr(r O)
    rows.append(dict(t=t, sz=ex(SZ).real, P=ex(PW).real, P2=ex(PW2).real,
                     dE=np.sqrt(max(ex(A2).real - ex(A).real ** 2, 0)),
                     comm=ex(COMM), perpP=ex(PERPP).real))
szs = np.array([r_['sz'] for r_ in rows])

def response(t_idx, kind='cos', fac=2.0):
    s = ts[:t_idx + 1]
    if len(s) < 2:
        return 0.0
    integrand = chi_w(ts[t_idx] - s, kind) * szs[:t_idx + 1]
    return -fac * np.trapz(integrand, s)

def X_w(t):
    s = np.linspace(0, t, max(2, int(400 * t) + 2))
    return np.trapz(np.abs(chi_w(s)), s)

fails = 0; ctrl = {'sin': 0.0, 'half': 0.0, 'noHS': 0.0}
err_I1 = err_I2 = 0.0; slack_B1 = slack_B2 = np.inf; nontriv = 0
for i, row in enumerate(rows):
    if i > 0:
        err_I1 = max(err_I1, abs(row['P'] - response(i)))
        ctrl['sin'] = max(ctrl['sin'], abs(row['P'] - response(i, 'sin')))
        ctrl['half'] = max(ctrl['half'], abs(row['P'] - response(i, fac=1.0)))
    pred = 1j * (2 * W + row['perpP'])
    err_I2 = max(err_I2, abs(row['comm'] - pred))
    ctrl['noHS'] = max(ctrl['noHS'], abs(row['comm'] - 1j * 2 * W))
    X = X_w(row['t'])
    slack_B1 = min(slack_B1, (p0 + 2 * X) - np.sqrt(row['P2']))
    L = W / (p0 + 2 * X) - abs(hx) / 2
    nontriv += L > 0
    slack_B2 = min(slack_B2, row['dE'] - L)

tolI1 = 2e-3      # trapezoid error of the response integral on the time grid (dt = 0.05)
print(f"top Fock population (truncation): {top:.1e}")
print(f"(I1) max |<P_w> - response| = {err_I1:.2e}  (tol {tolI1:.0e})"); fails += err_I1 > tolI1
print(f"(I2) max |<[A,Y]> - i(2W+<perp P>)| = {err_I2:.2e}  (tol 1e-8)"); fails += err_I2 > 1e-8
print(f"(B1) min slack (p0+2X) - <P^2>^1/2 = {slack_B1:+.3e}  (must be >= 0)"); fails += slack_B1 < -1e-9
print(f"(B2) min slack dE - L = {slack_B2:+.3e}  (must be >= 0); times with L > 0: {nontriv}/{len(rows)}")
fails += slack_B2 < -1e-9
for k, v in ctrl.items():
    tol = tolI1 if k in ('sin', 'half') else 1e-8
    rej = v > 10 * tol
    print(f"failure control '{k}': deviation {v:.2e} -> {'rejected (good)' if rej else 'NOT rejected'}")
    fails += not rej
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
