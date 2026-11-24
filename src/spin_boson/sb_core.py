"""Core of the spin-boson TEMPO numerics (imaginary time: MFG states and their response).

Model, in the notation of Chiu, Strathearn and Keeling, PRA 106, 012204 (2022) [CSK]:
    H_S = (1/2) h . sigma,    X = sigma_z (pointer basis),
    H_SR = sum_i [ nu_i b_i^dag b_i + g_i X (b_i + b_i^dag) ],
    J(nu) = sum_i g_i^2 delta(nu - nu_i) = alpha nu exp(-nu/wc).
In the manuscript notation (Eq. sb_hamiltonian), Delta = |h|, h = Delta (sin th, 0, cos th),
and g^2 calJ(nu)/pi = J(nu). The reorganization (polaron) energy is
lambda = int J/nu = alpha wc. The Leggett coupling is alpha_L = 2 alpha.
Units: hbar = k_B = 1; energies in units of Delta = omega_q.

Corrections to OQuPy 0.5.0 (1 and 2 found by check_gibbs_pathsum.py, 3 by check_eta_realtime.py):
  1. CustomSD.eta_function(matsubara=True) drops exp(-(beta-tau) w) for w > 36 T
     (overflow guard). That term is O(1) for tau -> beta, so the coefficients that
     couple the first and last imaginary-time slices are wrong (1.5% at N = 8,
     alpha = 0.2, wc = 10). MatsubaraSD below replaces it by a stable quadrature.
  2. GibbsTempo returns the transpose of rho~ (= its complex conjugate). This only
     matters if h_y != 0; mfg_state() transposes back.
  3. Real-time eta(t) is computed by OQuPy with scipy quad of an integrand that oscillates
     ~wc t times; for t >~ 5 the integrals warn and lose accuracy, and the late-time TEMPO
     state then depends erratically on dt. MatsubaraSD uses the closed form instead.
"""
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.special import loggamma
import oqupy

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1., -1.]).astype(complex)
PAULI = (SX, SY, SZ)


class MatsubaraSD(oqupy.CustomSD):
    """Ohmic J(nu) = alpha nu exp(-nu/wc) with an accurate imaginary-time eta function.

    eta(u) = -G(u), with G(0) = G'(0) = 0 and G'' = K, where
    K(u) = int J(nu) cosh(nu(beta/2-u))/sinh(beta nu/2) (CSK Eq. 6). Real-time
    (matsubara=False) calls are passed to OQuPy unchanged.
    """

    def __init__(self, alpha, wc, temperature):
        self.alpha_csk = float(alpha)
        super().__init__(j_function=lambda w: alpha * w, cutoff=wc,
                         cutoff_type='exponential', temperature=temperature)

    @lru_cache(maxsize=2 ** 14)
    def _G(self, u):
        a, wc, beta = self.alpha_csk, self.cutoff, 1.0 / self.temperature
        if u == 0.0:
            return 0.0

        def f(nu):
            # e^{-beta nu} expm1(nu u) written as e^{-nu(beta-u)} - e^{-beta nu} (u <= beta): no overflow
            # for beta nu > 709 (the product form gave 0 * inf = nan at low temperature)
            br = (np.expm1(-nu * u) + np.exp(-nu * (beta - u)) - np.exp(-beta * nu)) \
                / (-np.expm1(-beta * nu)) + u * nu
            return a * np.exp(-nu / wc) / nu * br
        pts = [1e-3, 1.0 / beta, 1.0, wc, 5 * wc]
        pts = sorted(p for p in set(pts) if 1e-12 < p < 60 * wc)
        return quad(f, 1e-12, 60 * wc, points=pts, limit=1000,
                    epsabs=1e-15, epsrel=1e-13)[0]

    def eta_real_time(self, t):
        """Closed form of eta(t) = int_0^t ds int_0^s du C(u) for J = alpha nu e^{-nu/wc} at temperature T:
        eta = alpha [ln(1 + i wc t) - i wc t] + 2 alpha [ln Gamma(1+z) - Re ln Gamma(1+z+i t/beta)],
        z = 1/(beta wc). Derived from C(t) = (alpha/beta^2)[psi'((1/wc + i t)/beta) + psi'(1 + (1/wc - i t)/beta)]
        (geometric expansion of the Bose factors); replaces OQuPy's oscillatory quadrature."""
        a, wc, beta = self.alpha_csk, self.cutoff, 1.0 / self.temperature
        z = 1.0 / (beta * wc)
        return a * (np.log(1 + 1j * wc * t) - 1j * wc * t) \
            + 2 * a * (loggamma(1 + z).real - loggamma(1 + z + 1j * t / beta).real)

    def eta_function(self, tau, epsrel=None, subdiv_limit=None, matsubara=False):
        if not matsubara:
            return complex(self.eta_real_time(float(tau)))
        return -self._G(float(np.round(tau, 15)))

    def K(self, u):
        """Imaginary-time kernel K(u), u in [0, beta] (for checks)."""
        a, wc, beta = self.alpha_csk, self.cutoff, 1.0 / self.temperature
        f = lambda nu: a * nu * np.exp(-nu / wc) * (np.exp(-nu * u) + np.exp(-nu * (beta - u))) \
            / (-np.expm1(-beta * nu))
        return quad(f, 0, 60 * wc, limit=1000, epsabs=1e-14, epsrel=1e-12)[0]


@lru_cache(maxsize=64)
def _bath(alpha, wc, beta):
    return oqupy.Bath(SZ, MatsubaraSD(alpha, wc, 1.0 / beta))


def hamiltonian(h):
    return 0.5 * (h[0] * SX + h[1] * SY + h[2] * SZ)


def mfg_state(h, alpha, wc=10.0, beta=1.0, N=200, epsrel=1e-12):
    """MFG state Tr_R e^{-beta H}/Z from GibbsTempo (symmetric splitting, N slices).

    alpha = 0 returns the bare Gibbs state. The returned matrix is Hermitized
    (the antihermitian part is SVD-truncation noise, ~1e-9 at N = 200).
    """
    H = hamiltonian(h)
    if alpha == 0.0:
        e, U = np.linalg.eigh(H)
        p = np.exp(-beta * (e - e.min())); p /= p.sum()
        return (U * p) @ U.conj().T
    r = oqupy.gibbs_tempo_compute(oqupy.System(H), _bath(float(alpha), float(wc), float(beta)),
                                  oqupy.GibbsParameters(n_steps=int(N), epsrel=float(epsrel)),
                                  progress_type='silent')
    r = r.T                                   # correction 2: OQuPy returns rho~^T
    r = 0.5 * (r + r.conj().T)
    return r / np.trace(r).real


def bloch(rho):
    return np.array([np.trace(rho @ s).real for s in PAULI])


def state_from_bloch(b):
    return 0.5 * (np.eye(2) + b[0] * SX + b[1] * SY + b[2] * SZ)


def fidelity(r1, r2):
    s = sqrtm(r1)
    return float(np.real(np.trace(sqrtm(s @ r2 @ s))) ** 2)


def bures_angle_bloch(b1, b2):
    """Bures angle between two qubit states given by Bloch vectors (closed form)."""
    b1, b2 = np.asarray(b1, float), np.asarray(b2, float)
    F = 0.5 * (1 + b1 @ b2 + np.sqrt(max(0.0, 1 - b1 @ b1) * max(0.0, 1 - b2 @ b2)))
    return float(np.arccos(np.sqrt(min(1.0, max(0.0, F)))))


def chi_tilde_qubit(b, J, beta):
    """Eq. (qubit_chi): chi~ = sqrt(lambda_max[J (I + b b^T/(1-|b|^2)) J]) / beta.
    Returns chi~, the matrix K, and the optimal unit direction n*."""
    G_inv = np.eye(3) + np.outer(b, b) / (1 - b @ b)
    K = J @ G_inv @ J
    K = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(K)
    return np.sqrt(max(w[-1], 0.0)) / beta, K, V[:, -1]
