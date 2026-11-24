"""Check 1: OQuPy GibbsTempo vs an independent brute-force path sum (CSK Eq. 3).

The path sum is written from CSK Eqs. (3)-(6) only, with bath coefficients from
an independent quadrature of K(u); it does not import OQuPy's integrals.
It decides which Trotter ordering OQuPy uses and whether the result is rho~ or
its transpose (tested with a complex H_S, h_y != 0).

Model (CSK): H_S = (1/2) h.sigma, X = sigma_z, J(nu) = alpha nu exp(-nu/wc).
Pass: max|rho_core - rho_sym| < 1e-9 for every case, where rho_core is
sb_core.mfg_state (OQuPy GibbsTempo with the MatsubaraSD fix and the transpose
fix). Failure controls, through the same comparison: the one-sided splitting,
the transposed state, and raw OQuPy (CustomSD, no fixes) must each deviate by
more than 1e-6 in at least one case. Exit 1 on failure."""
import itertools, sys
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
import oqupy
import sb_core

sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1., -1.]).astype(complex)
Xv = np.array([1., -1.])

def G(u, alpha, wc, beta):
    """G(0)=G'(0)=0, G''=K;  K(u)=int J(nu) cosh(nu(beta/2-u))/sinh(beta nu/2).
    Stable form: [cosh(nu(b/2-u))-cosh(nu b/2)]/sinh(nu b/2)
               = [expm1(-nu u) + e^{-b nu} expm1(nu u)] / (-expm1(-b nu))."""
    def f(nu):
        br = (np.expm1(-nu * u) + np.exp(-beta * nu) * np.expm1(nu * u)) / (-np.expm1(-beta * nu)) + u * nu
        return alpha * np.exp(-nu / wc) / nu * br
    pts = [1e-3, 1.0, wc, 5 * wc]
    return quad(f, 1e-12, 60 * wc, points=pts, limit=800, epsabs=1e-15, epsrel=1e-13)[0]

def etas(alpha, wc, beta, N):
    D = beta / N
    Gm = [G(m * D, alpha, wc, beta) for m in range(N + 1)]
    eta = np.zeros((N, N))
    for k in range(N):
        eta[k, k] = Gm[1]                       # half box: int_0^D dt int_0^t K = G(D)
        for kp in range(k):
            m = k - kp
            eta[k, kp] = Gm[m + 1] - 2 * Gm[m] + Gm[m - 1]
    return eta

def path_sum(E, eta, N):
    P = np.array(list(itertools.product(range(2), repeat=N + 1)))
    s = Xv[P[:, :N]]
    w = np.exp(np.einsum('pk,kl,pl->p', s, np.tril(eta), s)).astype(complex)
    for k in range(N):
        w = w * E[P[:, k], P[:, k + 1]]
    out = np.zeros((2, 2), complex); np.add.at(out, (P[:, 0], P[:, N]), w)
    return out                                    # (WE)^N ordering, bath on j_0..j_{N-1}

def variants(h, alpha, wc, beta, N):
    HS = 0.5 * (h[0] * sx + h[1] * sy + h[2] * sz); D = beta / N
    r1 = path_sum(expm(-D * HS), etas(alpha, wc, beta, N), N)
    Eh = expm(-D * HS / 2); rs = Eh @ r1 @ np.linalg.inv(Eh)     # (E^1/2 W E^1/2)^N
    nrm = lambda r: r / np.trace(r)
    return {'sym': nrm(rs), 'one_sided': nrm(r1), 'sym_transposed': nrm(rs).T}

def oqupy_state(h, alpha, wc, beta, N):
    HS = 0.5 * (h[0] * sx + h[1] * sy + h[2] * sz)
    sd = oqupy.CustomSD(j_function=lambda w: alpha * w, cutoff=wc, cutoff_type='exponential', temperature=1 / beta)
    return oqupy.gibbs_tempo_compute(oqupy.System(HS), oqupy.Bath(sz, sd),
                                     oqupy.GibbsParameters(n_steps=N, epsrel=1e-15), progress_type='silent')

def diff(a, b): return np.abs(a - b).max()

if __name__ == "__main__":
    cases = [((np.sin(1.), 0., np.cos(1.)), 0.2, 10., 1., 8),
             ((np.sin(1.), 0., np.cos(1.)), 0.5, 10., 1., 10),
             ((0.7, 0.4, 0.3), 0.3, 10., 1., 9),          # complex H_S: tests transpose convention
             ((1.0, 0., 0.0), 1.0, 5., 2., 10),           # theta = pi/2, beta = 2
             ((0.3, 0., 0.9), 2.0, 10., 0.5, 8)]
    fails = 0; ctrl = {'one_sided': 0., 'sym_transposed': 0., 'raw_oqupy': 0.}
    print(" case                                  N  |core-sym|  |core-one_sided|  |core-sym^T|  |raw_oqupy-sym|")
    for h, a, wc, beta, N in cases:
        r = sb_core.mfg_state(h, a, wc, beta, N, epsrel=1e-15)
        raw = oqupy_state(h, a, wc, beta, N)
        v = variants(h, a, wc, beta, N)
        d = {k: diff(r, x) for k, x in v.items()}
        d['raw_oqupy'] = diff(raw, v['sym'])
        ok = d['sym'] < 1e-9; fails += not ok
        for k in ctrl: ctrl[k] = max(ctrl[k], d[k])
        print(f" h={tuple(round(x,2) for x in h)} a={a} wc={wc} b={beta} N={N:2d}  {d['sym']:.2e}   {d['one_sided']:.2e}         {d['sym_transposed']:.2e}     {d['raw_oqupy']:.2e}  ok={ok}")
    for k, v in ctrl.items():
        rejected = v > 1e-6
        print(f"failure control '{k}': max deviation {v:.2e} -> {'rejected (good)' if rejected else 'NOT rejected (control failed)'}")
        fails += not rejected
    print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
