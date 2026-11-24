"""Definitions of the Planckian thermalization bound, shared by the three notebooks.

Everything in this module is a transcription of the definitions in the paper. Nothing here
performs a simulation: the three model families are solved by the scripts in ``src/``, and these
functions are what turns their output into the bound, and what the notebooks use to re-derive
stored quantities from more elementary ones as a check.

Conventions (used throughout the paper, the data files and the notebooks)
------------------------------------------------------------------------
``hbar = k_B = 1`` and ``beta = 1``. Times are therefore in units of the Planckian time

    tau_Pl = hbar * beta = hbar / (k_B T),

and energies in units of ``k_B T``. The distance between states is the Bures angle

    D(rho, sigma) = arccos( Tr | sqrt(rho) sqrt(sigma) | ),

which takes values in ``[0, pi/2]``. Perturbations are normalized to unit spectral seminorm,
``||kappa|| = 1``, so that no norm appears in the denominators below. An accuracy ``eps`` is
quoted as a fraction of

    eps_max = pi / 4,

the largest accuracy at which the bound can still be nontrivial on a ball of radius
``beta*delta = 2 pi`` (see ``EPS_MAX`` below).

Equation numbers refer to the manuscript.
"""
import csv
import numpy as np

EPS_MAX = np.pi / 4          # largest useful accuracy, in Bures angle, at beta*delta = 2 pi
BETA_DELTA = 2 * np.pi       # radius of the Hamiltonian ball used in every model of the paper


# ----------------------------------------------------------------------------------------------
# distances between equilibrium targets
# ----------------------------------------------------------------------------------------------
def bures_angle(rho, sigma):
    """Bures angle D(rho, sigma) = arccos(F^(1/2)) between two density matrices.

    The fidelity is F^(1/2) = Tr|sqrt(rho) sqrt(sigma)|, computed from the singular values of
    sqrt(rho) sqrt(sigma) so that no matrix square root of a product is needed.
    """
    def _sqrtm(a):
        w, v = np.linalg.eigh(a)
        return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T
    s = np.linalg.svd(_sqrtm(rho) @ _sqrtm(sigma), compute_uv=False)
    return np.arccos(np.clip(s.sum(), -1.0, 1.0))


def bures_angle_classical(p, q):
    """Bures angle between two states that are diagonal in a common basis.

    Then F^(1/2) = sum_i sqrt(p_i q_i), the Bhattacharyya coefficient. This is the case whenever a
    symmetry keeps the whole MFG family diagonal in one basis: the U(1) rotation about z of the
    Heisenberg model, or particle number in the fermionic models.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.arccos(np.clip(np.sqrt(p * q).sum(), -1.0, 1.0))


def bures_angle_qubit(b1, b2):
    """Bures angle between two qubit states given by their Bloch vectors.

    With rho = (I + b.sigma)/2, the root fidelity is

        F^(1/2) = sqrt( [1 + b1.b2 + sqrt((1-|b1|^2)(1-|b2|^2))] / 2 ).
    """
    b1, b2 = np.asarray(b1, float), np.asarray(b2, float)
    r1, r2 = b1 @ b1, b2 @ b2
    f = 0.5 * (1.0 + b1 @ b2 + np.sqrt(max(0.0, (1 - r1) * (1 - r2))))
    return np.arccos(np.clip(np.sqrt(max(f, 0.0)), -1.0, 1.0))


# ----------------------------------------------------------------------------------------------
# the bound at finite accuracy:  tau_eps >= tau_Pl * chi_delta
# ----------------------------------------------------------------------------------------------
def chi_pair(distance, delta_x, eps):
    """Contribution of one pair of family members to chi_delta [Theorem 1, Eq. (8)]:

        chi(x1, x2) = [ ( 2 D(w_x1, w_x2) - 4 eps ) / ( beta |x1 - x2| ) ]_+ ,

    with ``beta = 1``. ``distance`` is the Bures angle between the two mean-force Gibbs targets and
    ``delta_x = x1 - x2`` their separation along the family direction ``kappa_hat``, which has unit
    seminorm, so that ||H(x1) - H(x2)|| = |x1 - x2|.

    The positive part is applied in ``chi_delta``, which maximizes over pairs; this function
    returns the raw maximand so that the maximization sees its sign.
    """
    return (2.0 * np.asarray(distance) - 4.0 * eps) / np.abs(delta_x)


def chi_delta(xs, targets, eps, distance=bures_angle_classical, positive=True):
    """chi_delta maximized over all pairs of the sampled members of a one-parameter family
    H_S(x) = Hbar_S + x kappa_hat, ||kappa_hat|| = 1, |x| <= delta [Eqs. (8) and (30)]:

        chi_delta = max_{x1 != x2}  [ ( 2 D(w_x1, w_x2) - 4 eps ) / |x1 - x2| ]_+ .

    ``xs`` are the family parameters and ``targets[i]`` the equilibrium target at ``xs[i]``, in
    whatever representation ``distance`` accepts (populations, Bloch vectors, density matrices).

    Maximizing over a finite set of pairs can only *lower* the maximum, so the value returned is
    still a valid lower bound on tau_eps / tau_Pl. This is what the Heisenberg analysis does, where
    only the simulated fields are available; the fermionic and spin-boson pipelines refine the
    discrete maximum with a constrained Nelder-Mead search over the continuous ball.

    Returns ``(chi, (x1, x2))``: the value and the pair that attains it.
    """
    xs = np.asarray(xs, float)
    best, arg = -np.inf, (None, None)
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            v = chi_pair(distance(targets[i], targets[j]), xs[i] - xs[j], eps)
            if v > best:
                best, arg = v, (xs[i], xs[j])
    if positive:
        best = max(best, 0.0)
    return best, arg


# ----------------------------------------------------------------------------------------------
# the locally exact bound:  tau >= tau_Pl * chi_tilde,  chi_tilde = max_{||kappa||=1} sqrt(F_kappa)/beta
# ----------------------------------------------------------------------------------------------
def chi_tilde_two_level(p, dp_dx):
    """Locally exact bound when the MFG family stays diagonal and the perturbation is a projector
    onto one of two levels [Eq. (23) of the paper]:

        chi_tilde = |dp/dx| / sqrt( p (1 - p) ),      p = Tr[w P].

    The first inequality of the QFI hierarchy is tight here, so this is the exact value and not
    merely a lower bound. It applies to the occupation of a single fermionic level, P = d^dag d, as
    in the resonant-level model.
    """
    p = np.asarray(p, float)
    return np.abs(dp_dx) / np.sqrt(p * (1.0 - p))


def response_matrix_K(b, J):
    """The matrix K of the qubit case [Eq. (24)]:

        K = J ( I + b b^T / (1 - |b|^2) ) J,

    where ``b`` is the Bloch vector of the MFG state and ``J`` its static response matrix,
    d b / d delta = J n for a perturbation kappa = (1/2) n.sigma. The QFI of that perturbation is
    F_kappa = n^T K n.
    """
    b = np.asarray(b, float).reshape(3)
    J = np.asarray(J, float).reshape(3, 3)
    return J @ (np.eye(3) + np.outer(b, b) / (1.0 - b @ b)) @ J


def chi_tilde_qubit(b, J):
    """Locally exact bound for a qubit, optimized over all perturbations [Eq. (24)]:

        chi_tilde = sqrt( lambda_max(K) ) / beta.

    The maximizing direction is the leading eigenvector of K.
    """
    return float(np.sqrt(max(np.linalg.eigvalsh(response_matrix_K(b, J))[-1], 0.0)))


def chi_tilde_direction(b, J, n):
    """Locally exact bound for one fixed perturbation kappa = (1/2) n.sigma:

        sqrt( n^T K n ) / beta,       |n| = 1.
    """
    n = np.asarray(n, float).reshape(3)
    n = n / np.linalg.norm(n)
    return float(np.sqrt(max(n @ response_matrix_K(b, J) @ n, 0.0)))


# ----------------------------------------------------------------------------------------------
# the single-observable bound and the Planckian criterion
# ----------------------------------------------------------------------------------------------
def r_kappa(xi_kk, var_kk):
    """The correlation factor of Eq. (19),

        r_kappa = xi_kappa,kappa / ( beta Var(kappa) )
                = (1/beta) int_0^beta dlambda G_kappa,kappa(lambda) / G_kappa,kappa(0)
                in (0, 1],

    i.e. the normalized autocorrelation of kappa averaged over the Planckian window w_beta or,
    equivalently, over imaginary time. It is close to one when the equilibrium fluctuations of
    kappa stay correlated over hbar*beta, and small when they decorrelate much faster. This is
    the quantity that decides whether the bound is Planckian.
    """
    return np.asarray(xi_kk, float) / np.asarray(var_kk, float)


def r_floor(G_half, var_kk):
    """The lower bound on r_kappa from a single point of the imaginary-time correlation [Eq. (20)]:

        r_kappa >= G_kappa,kappa(beta/2) / Var(kappa).

    G_kappa,kappa(lambda) is convex and symmetric about lambda = beta/2, so its mean over
    [0, beta] -- which is r_kappa, by Eq. (19) -- is at least its value there. In the spectral
    representation G_kappa,kappa(beta/2) = int dnu S_kappa,kappa(nu) sech(beta hbar nu / 2), so it
    retains only the fluctuations slower than about k_B T / hbar.
    """
    return np.asarray(G_half, float) / np.asarray(var_kk, float)


def self_response_bound(xi_kk, var_kk):
    """The simplest valid bound, from one static susceptibility and one equilibrium variance
    [Eq. (18); ||kappa|| = 1]:

        tau / tau_Pl >= xi_kappa,kappa / sqrt(Var(kappa)) = r_kappa sqrt(Var(kappa)) .
    """
    return np.asarray(xi_kk, float) / np.sqrt(np.asarray(var_kk, float))


def qfi_bracket(xi_kk, var_kk):
    """The bracket of Eq. (21) on the optimal bound for a *given* perturbation kappa:

        r sqrt(Var)   <=   sqrt(F_kappa) / beta   <=   sqrt(r) sqrt(Var)   <=   1/2.

    Returns ``(lower, upper)``. No choice of readout observable A can improve the self-response
    bound by more than a factor r^(-1/2); when the upper end is small, no observable gives a strong
    bound for this kappa. The ceiling 1/2 is never attained: it would need Var(kappa) -> 1/4 and
    r_kappa -> 1, and r_kappa = 1 requires a conserved kappa, which cannot be thermalized.
    """
    r = r_kappa(xi_kk, var_kk)
    s = np.sqrt(np.asarray(var_kk, float))
    return r * s, np.sqrt(r) * s


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def read_csv(path):
    """Read a CSV into a list of dicts, converting numbers and booleans and leaving the rest as
    strings. Empty fields become ``None``. The data files of this repository are small enough that
    this is all the machinery the notebooks need."""
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if v is None or v == '':
                    r[k] = None
                elif v in ('True', 'False'):
                    r[k] = (v == 'True')
                else:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def select(rows, **conditions):
    """Rows matching all ``column=value`` conditions; floats are compared with a tolerance."""
    def ok(r):
        for k, v in conditions.items():
            if isinstance(v, float):
                if r[k] is None or abs(r[k] - v) > 1e-12:
                    return False
            elif r[k] != v:
                return False
        return True
    return [r for r in rows if ok(r)]


def column(rows, *names):
    """Columns of a list of rows as float arrays, in the given order."""
    out = [np.array([np.nan if r[n] is None else r[n] for r in rows], float) for n in names]
    return out[0] if len(out) == 1 else out


def sort_by(rows, key):
    return sorted(rows, key=lambda r: r[key])


# ----------------------------------------------------------------------------------------------
def _selftest():
    """Identities that every definition above must satisfy. Run ``python planckian.py``."""
    rng = np.random.default_rng(0)

    # 1. the three Bures angles agree where their domains overlap
    p, q = np.array([0.3, 0.7]), np.array([0.6, 0.4])
    rho, sig = np.diag(p), np.diag(q)
    b1, b2 = np.array([0., 0., p[0] - p[1]]), np.array([0., 0., q[0] - q[1]])
    d = [bures_angle(rho, sig), bures_angle_classical(p, q), bures_angle_qubit(b1, b2)]
    assert max(d) - min(d) < 1e-12, d

    # 2. Bures angle of two pure orthogonal states is pi/2, of a state with itself 0
    assert abs(bures_angle_classical([1, 0], [0, 1]) - np.pi / 2) < 1e-12
    assert abs(bures_angle_qubit([0, 0, 1], [0, 0, 1])) < 1e-8

    # 3. eps_max is the accuracy at which chi_delta vanishes for the most distinguishable pair
    #    allowed on the ball: 2 D - 4 eps = 0 with D = pi/2 and |x1 - x2| = 2 delta.
    assert abs(chi_pair(np.pi / 2, BETA_DELTA, EPS_MAX)) < 1e-12

    # 3b. the floor of Eq. (20) never exceeds r_kappa itself
    assert r_floor(0.3, 1.0) <= r_kappa(0.9, 1.0)

    # 4. qubit chi_tilde: for a Gibbs state of (Delta/2) sigma_z the two channels of Eq. (25) are
    #    sech(x)/2 (transverse) and tanh(x)/(2x) (longitudinal), the transverse one being larger.
    for x in (0.3, 1.0, 3.0):
        bz = -np.tanh(x)
        # d<sigma>/d delta for kappa = n.sigma/2, from the exact qubit Gibbs state at beta = 1
        J = np.diag([-np.tanh(x) / (2 * x), -np.tanh(x) / (2 * x), -(1 - np.tanh(x) ** 2) / 2])
        b = np.array([0.0, 0.0, bz])
        long_, trans = np.abs(J[2, 2]) / np.sqrt(1 - bz ** 2), np.abs(J[0, 0])
        assert abs(long_ - 0.5 / np.cosh(x)) < 1e-12, (x, long_)
        assert abs(trans - np.tanh(x) / (2 * x)) < 1e-12, (x, trans)
        assert abs(chi_tilde_qubit(b, J) - max(long_, trans)) < 1e-12
        assert abs(chi_tilde_direction(b, J, [1, 0, 0]) - trans) < 1e-12

    # 5. two-level formula against a finite difference of the Fermi function
    eps0, h = 0.7, 1e-5
    f = lambda e: 1 / (1 + np.exp(e))
    assert abs(chi_tilde_two_level(f(eps0), (f(eps0 + h) - f(eps0 - h)) / (2 * h))
               - np.sqrt(f(eps0) * (1 - f(eps0)))) < 1e-8

    # 6. chi_delta of a two-level family reduces to the pairwise formula, and the grid maximum
    #    never exceeds the ceiling 1/2 - 2 eps/(beta delta) of Theorem 1
    xs = np.linspace(-BETA_DELTA, BETA_DELTA, 41)
    tg = [np.array([f(x), 1 - f(x)]) for x in xs]
    for epsf in (0.01, 0.05, 0.2):
        c, _ = chi_delta(xs, tg, epsf * EPS_MAX)
        assert c <= 0.5 - 2 * epsf * EPS_MAX / BETA_DELTA + 1e-12, (epsf, c)

    # 7. the bracket of Eq. (21) is ordered, and r in (0, 1]
    var = rng.uniform(0.05, 0.25, 50)
    xi = var * rng.uniform(0.05, 1.0, 50)
    lo, hi = qfi_bracket(xi, var)
    assert np.all(lo <= hi + 1e-15) and np.all(hi <= 0.5 + 1e-12)
    assert np.all((r_kappa(xi, var) > 0) & (r_kappa(xi, var) <= 1))

    print('planckian.py: all self-tests passed')


if __name__ == '__main__':
    _selftest()
