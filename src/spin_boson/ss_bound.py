"""Dynamics-free lower bound on the Shiraishi-Saito energy spread for the spin-boson model.

Setting: H = H_S + H_B + V, H_S = (1/2) h.sigma, V = sigma_z (x) B, B = sum_k g_k (b_k + b_k^dag),
J(nu) = sum_k g_k^2 delta(nu - nu_k), factorized initial state rho_S(0) (x) omega_B.
delta E_S(t)^2 = Var_{rho_SB(t)}(H_S + V).

Claim (derivation in README, Sec. "Speed limits"): for any real weight w(nu) and every t >= 0,
    delta E_S(t) >= W / (p0 + 2 X_w(t)) - h_perp/2,
with W = int J w, p0^2 = int J w^2 coth(beta nu/2), X_w(t) = int_0^t |chi_w(s)| ds,
chi_w(s) = int J w cos(nu s), h_perp = (h_x^2 + h_y^2)^(1/2).
Steps: Robertson for A = H_S + V and Y = sigma_z P_w, P_w = i sum_k w_k g_k (b_k^dag - b_k);
[A, Y] = [H_S, sigma_z] P_w + 2 i W; exact bath solution P_w(t) = P_w^free(t) - 2 int chi_w sigma_z;
<P_w^free(t)^2> = p0^2 in the thermal bath state; ||sigma_z(s)|| = 1.

For J = alpha nu exp(-nu/wc) and w = nu^p exp(-q nu), chi_w has the closed form
alpha Gamma(p+2) Re[(s - i tau)^(-(p+2))], s = 1/wc + q, so X_w(inf) is a 1D integral.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma


def _I(p):
    """int_0^inf |Re (1 - i u)^(-(p+2))| du (finite for p > -1)."""
    f = lambda u: abs(np.real((1 - 1j * u) ** (-(p + 2))))
    # sign changes of cos((p+2) arctan u): split at u_k = tan(k pi/(2(p+2)))
    nz = [np.tan((2 * k + 1) * np.pi / (2 * (p + 2))) for k in range(int(p + 2) + 1)
          if (2 * k + 1) * np.pi / (2 * (p + 2)) < np.pi / 2]
    edges = [0.0] + sorted(nz) + [np.inf]
    return sum(quad(f, a, b, limit=400, epsabs=1e-13, epsrel=1e-11)[0] for a, b in zip(edges[:-1], edges[1:]))


def lower_bound_dE(alpha, wc, beta, h_perp, p, q):
    """W/(p0 + 2 X_w(inf)) - h_perp/2 for w = nu^p exp(-q nu); X_w(inf) bounds X_w(t) for all t."""
    s = 1.0 / wc + q
    if s <= 0 or 1.0 / wc + 2 * q <= 0 or p < 0:
        return -np.inf
    W = alpha * gamma(p + 2) * s ** (-(p + 2))
    f = lambda nu: alpha * nu ** (2 * p + 1) * np.exp(-nu * (1 / wc + 2 * q)) / np.tanh(beta * nu / 2)
    up = 80.0 / (1 / wc + 2 * q)
    p0 = np.sqrt(quad(f, 0, up, points=[1 / beta, 1 / (1 / wc + 2 * q)], limit=400)[0])
    X = alpha * gamma(p + 2) * s ** (-(p + 1)) * _I(p)
    return W / (p0 + 2 * X) - h_perp / 2


def best_lower_bound_dE(alpha, wc, beta, h_perp):
    """Maximize the lower bound over (p, q); returns (L, p, q)."""
    best = (-np.inf, None, None)
    for p0 in [0.0, 0.5, 1.0, 2.0, 3.0]:
        for q0 in [0.0, -0.3 / wc, 0.5 / wc]:
            fun = lambda v: -lower_bound_dE(alpha, wc, beta, h_perp, max(v[0], 0.0), v[1])
            r = minimize(fun, [p0, q0], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-10, maxiter=400))
            L = -r.fun
            if np.isfinite(L) and L > best[0]:
                best = (L, max(r.x[0], 0.0), r.x[1])
    return best


def dE0_exact(alpha, wc, beta, var_HS):
    """delta E_S(0) for a factorized initial state: Var(H_S) + K(0), K(0) = int J coth(beta nu/2)."""
    f = lambda nu: alpha * nu * np.exp(-nu / wc) / np.tanh(beta * nu / 2)
    return np.sqrt(var_HS + quad(f, 0, 80 * wc, points=[1 / beta, wc], limit=400)[0])


if __name__ == "__main__":
    for a in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        L, p, q = best_lower_bound_dE(a, 10.0, 1.0, 1.0)
        L1 = lower_bound_dE(a, 10.0, 1.0, 1.0, 0.0, 0.0)
        print(f"alpha={a:5.3f}  L_opt={L:7.3f} (p={p:.2f}, q*wc={q*10:+.2f})  L_w=1={L1:7.3f}  dE(0)[rho=I/2]={dE0_exact(a,10.,1.,0.25+0.0):.3f}")
