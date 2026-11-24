"""Exact wide-band fermionic thermalization benchmarks (hbar=k_B=1).

The dynamics are computed afresh; no input CSV is required. Gamma=g**2.
Initial state: rho_S(0) tensor omega_B(beta), p0=1/2 (RLM) or |10> (DQD).
Targets: exact wide-band mean-force Gibbs states. No bath discretization/GGE.

TQFI is the REDUCED-state, time-parameter SLD Fisher information. The primary
QSL is the standard endpoint/average-speed expression evaluated at the common
thermalization time. Its trajectory dependence is explicit in the output.
SS and the global-state time-QFI bound are zero in this flat unbounded-band limit.

Run: python benchmarking.py --points 31 --gamma-max 100
Interactive interface: benchmarking.ipynb. Details: benchmarking_notes.md.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.linalg import expm, eigh
from scipy.optimize import brentq
from scipy.special import digamma

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Parameters:
    beta: float = 1.0
    rlm_energy_1: float = 0.0
    rlm_energy_2: float = 1.0
    dqd_mean_1: float = 1.1
    dqd_mean_2: float = 2.35
    dqd_delta: float = 0.2
    rtol: float = 2e-10
    atol: float = 2e-12
    memory_beta: float = 12.0


def parameters(preset="thesis", **changes):
    if preset == "thesis":
        return Parameters(**changes)
    if preset == "csv":
        return Parameters(**dict({"rlm_energy_2": 3.0, "dqd_delta": 1.2}, **changes))
    raise ValueError("preset must be 'thesis' or 'csv'")


def hermitian(a):
    return (a + a.conj().T)/2


def _trigamma(z):
    """Complex trigamma: recurrence followed by its asymptotic expansion."""
    total = 0j
    while abs(z) < 20:
        total += 1/z**2
        z += 1
    total += 1/z + 1/(2*z**2) + 1/(6*z**3) - 1/(30*z**5)
    total += 1/(42*z**7) - 1/(30*z**9) + 5/(66*z**11)
    return total


def matrix_digamma(b):
    """Analytic 1x1/2x2 matrix function, including the exceptional point."""
    if b.shape == (1, 1):
        return np.array([[digamma(b[0, 0])]])
    c = np.trace(b)/2
    m = b-c*np.eye(2)
    q = np.sqrt(complex(np.trace(m @ m)/2))
    if abs(q) < 1e-6:
        # Exact at q=0. The omitted O(q^2) term is below ~1e-12 here.
        return digamma(c)*np.eye(2) + _trigamma(c)*m
    fp, fm = digamma(c+q), digamma(c-q)
    return (fp+fm)/2*np.eye(2) + (fp-fm)/(2*q)*m


def density(n):
    """N_ij=<d_j^dagger d_i>; basis |00>,|10>,|01>,|11> for DQD."""
    n = hermitian(n)
    eigenvalues = eigh(n, eigvals_only=True)
    if eigenvalues.min() < -2e-8 or eigenvalues.max() > 1+2e-8:
        raise ArithmeticError(f"Nonphysical covariance eigenvalues {eigenvalues}")
    if len(n) == 1:
        p = n[0, 0].real
        return np.diag([1-p, p])
    p11 = np.linalg.det(n).real
    rho = np.zeros((4, 4), complex)
    rho[0, 0] = 1-np.trace(n).real+p11
    rho[1:3, 1:3] = n-p11*np.eye(2)
    rho[3, 3] = p11
    return hermitian(rho)


def bures(rho, sigma):
    if np.max(np.abs(rho-sigma)) < 1e-14:
        return 0.0
    ev, u = eigh(hermitian(rho))
    if ev.min() < -2e-8:
        raise ArithmeticError("Density matrix is not positive")
    root = (u*np.sqrt(np.maximum(ev, 0))) @ u.conj().T
    w = eigh(hermitian(root @ sigma @ root), eigvals_only=True)
    fidelity_root = np.sum(np.sqrt(np.maximum(w, 0)))
    if fidelity_root > 1+2e-8:
        raise ArithmeticError("Fidelity exceeds one")
    return float(np.arccos(np.clip(fidelity_root, 0, 1)))


def gaussian_time_qfi(n, derivative):
    """SLD QFI of a gauge-invariant Gaussian state with parameter t."""
    p, u = eigh(hermitian(n))
    if p.min() < -2e-8 or p.max() > 1+2e-8:
        raise ArithmeticError("Unphysical state in time-QFI")
    p = np.clip(p, 0, 1)  # Only removes roundoff after the explicit check.
    dn = u.conj().T @ derivative @ u
    den = p[:, None]+p[None, :]-2*p[:, None]*p[None, :]
    good = den > 1e-15
    if np.any(np.abs(dn[~good]) > 1e-9):
        return float("inf")
    return float(2*np.sum(np.abs(dn[good])**2/den[good]))


class WideBandDynamics:
    """Exact continuum evolution from the Fourier transform of the Fermi sea.

    Ndot=A N+N A†+Q(t), A=-i h-Gamma J/2.
    Q(0+)=Gamma J/2;
    Qdot=Gamma [exp(At)J-J exp(A†t)]/[2 i beta sinh(pi t/beta)].
    The integrand has a finite t->0 limit. Only a short thermal-memory interval
    is integrated; the exponentially small remainder is monitored. Thereafter
    N=N_MFG+U(N_at_switch-N_MFG)U†. This retains the full slow dark-mode decay.
    """
    def __init__(self, energies, gamma, n0, config=Parameters()):
        if gamma <= 0 or config.beta <= 0:
            raise ValueError("gamma and beta must be positive")
        self.gamma, self.config = float(gamma), config
        self.n0 = np.asarray(n0, complex)
        self.n = len(energies)
        self.h = np.diag(np.asarray(energies, float))
        self.j = np.ones((self.n, self.n))
        self.a = -1j*self.h-gamma*self.j/2
        self.a_dag = self.a.conj().T
        b = .5*np.eye(self.n)-config.beta*self.a/(2*np.pi)
        psi = matrix_digamma(b)
        self.n_mfg = hermitian(.5*np.eye(self.n)-(psi-psi.conj().T)/(2j*np.pi))
        self.q_inf = hermitian(-self.a @ self.n_mfg-self.n_mfg @ self.a_dag)
        self.rho0, self.target = density(self.n0), density(self.n_mfg)
        self.stationary = self.n == 1 and energies[0] == 0 and n0[0, 0] == .5
        self.t_switch = config.memory_beta*config.beta
        decay = -np.max(np.linalg.eigvals(self.a).real)
        if decay <= 0:
            raise ValueError("An undamped mode prevents generic thermalization")
        self.slow_time = 1/decay
        if self.stationary:
            self.switch_error = 0.
            return
        size = self.n**2

        def rhs(t, y):
            n = y[:size].reshape(self.n, self.n)
            q = y[size:].reshape(self.n, self.n)
            if t < 1e-7/max(1., gamma, np.max(np.abs(energies))):
                dq = -gamma/(2*np.pi)*(self.h @ self.j+self.j @ self.h)
            else:
                u = expm(self.a*t)
                dq = gamma/(2j*config.beta*np.sinh(np.pi*t/config.beta))*(u @ self.j-self.j @ u.conj().T)
            dn = self.a @ n+n @ self.a_dag+q
            return np.r_[dn.ravel(), dq.ravel()]

        y0 = np.r_[self.n0.ravel(), (gamma*self.j/2).ravel()]
        self.solution = solve_ivp(rhs, (0., self.t_switch), y0, method="BDF",
                                  rtol=config.rtol, atol=config.atol, dense_output=True)
        if not self.solution.success:
            raise RuntimeError(self.solution.message)
        end = self.solution.y[:, -1]
        self.n_switch = hermitian(end[:size].reshape(self.n, self.n))
        self.switch_error = float(np.max(abs(end[size:].reshape(self.n,self.n)-self.q_inf)))
        if self.switch_error > 2e-7*max(1., gamma):
            raise ArithmeticError(f"Thermal-memory mismatch: {self.switch_error}")

    def at(self, t):
        if t < 0:
            raise ValueError("Time must be nonnegative")
        if self.stationary:
            return self.n0.copy(), np.zeros_like(self.n0)
        if t == 0:
            return self.n0.copy(), hermitian(self.a @ self.n0+self.n0 @ self.a_dag+self.gamma*self.j/2)
        if t <= self.t_switch:
            y = self.solution.sol(t)
            n = hermitian(y[:self.n**2].reshape(self.n, self.n))
            q = hermitian(y[self.n**2:].reshape(self.n, self.n))
            dn = hermitian(self.a @ n+n @ self.a_dag+q)
        else:
            u = expm(self.a*(t-self.t_switch))
            diff = hermitian(u @ (self.n_switch-self.n_mfg) @ u.conj().T)
            n = self.n_mfg+diff
            dn = hermitian(self.a @ diff+diff @ self.a_dag)
        return n, dn

    def error(self, t):
        return bures(density(self.at(t)[0]), self.target)

    def time_qfi(self, t):
        if self.stationary:
            return 0.
        if self.n == 2 and self.gamma*t < 1e-9:
            return self.gamma/t if t > 0 else float("inf")
        return gaussian_time_qfi(*self.at(t))

    def path_length(self, end, epsabs=2e-8):
        """Integrate sqrt(F_t)/2 using t=u² to resolve the pure-state cusp."""
        if end == 0 or self.stationary:
            return 0., 0.
        def integrand(u):
            if u == 0:
                return np.sqrt(self.gamma) if self.n == 2 else 0.
            return u*np.sqrt(self.time_qfi(u*u))
        landmarks = [0., 1e-3/self.gamma, .1/self.gamma, 1/self.gamma,
                     self.config.beta, self.t_switch, self.slow_time, end]
        bounds = np.sqrt(sorted(set(t for t in landmarks if 0 <= t <= end)))
        length = error = 0.
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            val, err = quad(integrand, lo, hi, epsabs=epsabs/max(1,len(bounds)-1),
                            epsrel=2e-7, limit=150)
            length += val
            error += err
        return length, error


def pair(model, gamma, config):
    if model == "rlm":
        energy_lists = [[config.rlm_energy_1], [config.rlm_energy_2]]
        initial = np.array([[.5]])
        range_difference = abs(config.rlm_energy_2-config.rlm_energy_1)
    elif model == "dqd":
        energy_lists = [[mean+config.dqd_delta/2, mean-config.dqd_delta/2]
                        for mean in [config.dqd_mean_1, config.dqd_mean_2]]
        initial = np.diag([1., 0.])
        range_difference = 2*abs(config.dqd_mean_2-config.dqd_mean_1)
    else:
        raise ValueError("model must be rlm or dqd")
    if range_difference == 0:
        raise ValueError("The target Hamiltonians must differ")
    return [WideBandDynamics(e, gamma, initial, config) for e in energy_lists], range_difference


def first_passage(branches, epsilon, samples=480):
    """First sampled common crossing, refined with the exact continuum solution."""
    def error(t):
        return max(branch.error(t) for branch in branches)
    if error(0.) <= epsilon:
        return 0., (0., 0.)
    gamma = branches[0].gamma
    stop = 30*max(b.slow_time for b in branches)
    times = np.r_[0., np.geomspace(1e-7/max(1., gamma), stop, samples)]
    previous_t = 0.
    for t in times[1:]:
        if error(t) <= epsilon:
            root = brentq(lambda s:error(s)-epsilon, previous_t, t,
                          xtol=1e-9, rtol=2e-11)
            return float(root), (float(previous_t), float(t))
        previous_t = t
    raise RuntimeError(f"No common crossing before t={stop}; refine/extend the grid")


def benchmark_point(model, gamma, eps_fractions, config=Parameters(), samples=480):
    branches, range_diff = pair(model, gamma, config)
    target_distance = bures(branches[0].target, branches[1].target)
    out = []
    for fraction in eps_fractions:
        epsilon = fraction*np.pi/4
        tau, bracket = first_passage(branches, epsilon, samples)
        bound = max(0., 2*target_distance-4*epsilon)/range_diff
        lengths, quadrature_errors, endpoint_bounds, target_bounds = [], [], [], []
        for branch in branches:
            length, err = branch.path_length(tau)
            endpoint = bures(branch.rho0, density(branch.at(tau)[0]))
            target_goal = max(0., bures(branch.rho0, branch.target)-epsilon)
            if length > 1e-12:
                if length+max(2e-7,5*err) < endpoint:
                    raise ArithmeticError("Integrated QFI path shorter than endpoint distance")
                endpoint_bounds.append(tau*endpoint/length)
                target_bounds.append(tau*target_goal/length)
            else:
                endpoint_bounds.append(0.)
                target_bounds.append(0.)
            lengths.append(length)
            quadrature_errors.append(err)
        qfi_bound, qfi_target = max(endpoint_bounds), max(target_bounds)
        if max(bound, qfi_bound, qfi_target) > tau+max(2e-5*tau,1e-8):
            raise ArithmeticError("A computed lower bound exceeds the common passage time")
        out.append(dict(model=model, gamma=gamma, g=np.sqrt(gamma), beta=config.beta,
                        eps_fraction=fraction, epsilon=epsilon,
                        target_distance=target_distance, hamiltonian_range=range_diff,
                        tau=tau, pair_bound=bound, tqfi_endpoint=qfi_bound,
                        tqfi_target=qfi_target, ss_wideband=0., global_tqfi_wideband=0.,
                        path_1=lengths[0], path_2=lengths[1],
                        path_error_1=quadrature_errors[0], path_error_2=quadrature_errors[1],
                        qfi_endpoint_1=endpoint_bounds[0], qfi_endpoint_2=endpoint_bounds[1],
                        error_1_at_tau=branches[0].error(tau), error_2_at_tau=branches[1].error(tau),
                        bracket_left=bracket[0], bracket_right=bracket[1],
                        memory_residual=max(b.switch_error for b in branches)))
    return out


def run_benchmarks(config=Parameters(), gamma_values=None, eps_fractions=None, samples=480):
    if gamma_values is None:
        gamma_values = coupling_grid()
    all_rows = []
    for model in ['rlm', 'dqd']:
        fractions = eps_fractions or ([.001,.05,.1,.15,.2] if model=='rlm' else [.01,.05,.1,.15,.2])
        for gamma in gamma_values:
            start = time.monotonic()
            rows = benchmark_point(model, float(gamma), fractions, config, samples)
            all_rows.extend(rows)
            middle = rows[min(1,len(rows)-1)]
            print(f"{model.upper()} g²={gamma:.5g}: tau={middle['tau']:.6g}, "
                  f"pair={middle['pair_bound']:.6g}, TQFI={middle['tqfi_endpoint']:.6g} "
                  f"({time.monotonic()-start:.1f}s)", flush=True)
    return all_rows


def save_results(rows, config, output="benchmarking_results"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'coupling_benchmark.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(parameters=asdict(config), bath="exact flat infinite wide-band continuum",
                    preparations={"rlm":"p0=1/2, factorized thermal bath",
                                  "dqd":"|10>, factorized thermal bath"},
                    common_time="first t with max(D1(t),D2(t)) <= epsilon",
                    tqfi="max_i tau*D(rho_i(0),rho_i(tau))/(integral_0^tau sqrt(F_i(t))/2 dt)",
                    units="hbar=k_B=1; Gamma=g^2; plots use time/beta and sqrt(beta)*g",
                    versions={"numpy":np.__version__,"scipy":__import__('scipy').__version__})
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    plot_results(rows, output)
    return output


def plot_results(rows, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'cm'})
    output = Path(output)
    for model in ['rlm', 'dqd']:
        subset = [r for r in rows if r['model']==model]
        fractions = sorted(set(r['eps_fraction'] for r in subset))
        fig, ax = plt.subplots(figsize=(7.4,5.1), constrained_layout=True)
        colors = plt.cm.viridis(np.linspace(.08,.9,len(fractions)))
        for color, frac in zip(colors, fractions):
            rr = sorted([r for r in subset if r['eps_fraction']==frac], key=lambda r:r['gamma'])
            x = [np.sqrt(r['beta'])*r['g'] for r in rr]
            for key, style, marker in [('tau','-',None),('pair_bound','--',None),('tqfi_endpoint',':','o')]:
                ax.plot(x,[r[key]/r['beta'] for r in rr], linestyle=style, color=color,
                        lw=1.8 if key=='tau' else 1.4, marker=marker, markersize=3,
                        markevery=max(1,len(rr)//8), markerfacecolor='none')
        ax.axhline(0., color='.35', linestyle='-.', lw=1.2)
        ax.set_xscale('log')
        ax.set_yscale('symlog',linthresh=.001,linscale=.35)
        ax.set_ylim(bottom=-.00015)
        ax.set(xlabel=r'$g\sqrt{\beta}$', ylabel=r'Time / $\tau_{\rm Pl}$',
               title=f'{model.upper()}: factorized preparation, wide-band MFG targets')
        top = ax.secondary_xaxis('top', functions=(lambda x:x*x, lambda x:np.sqrt(x)))
        top.set_xlabel(r'$\beta g^2$')
        style_handles = [Line2D([],[],color='k',ls='-',label=r'Common $\tau_\varepsilon$'),
                         Line2D([],[],color='k',ls='--',label='Equilibrium pair bound'),
                         Line2D([],[],color='k',ls=':',marker='o',mfc='none',ms=3,label='Reduced time-QFI (trajectory input)'),
                         Line2D([],[],color='.35',ls='-.',label='Shiraishi–Saito = 0 (wide band)')]
        first = ax.legend(handles=style_handles, loc='upper left',fontsize=8)
        ax.add_artist(first)
        ax.legend(handles=[Line2D([],[],color=c,label=rf'$\varepsilon={f}\pi/4$')
                           for c,f in zip(colors,fractions)],loc='lower left',fontsize=8)
        ax.grid(alpha=.18,which='major')
        for suffix in ['pdf','png']:
            fig.savefig(output/f'{model}_coupling_benchmark.{suffix}',dpi=190)
        plt.close(fig)


def coupling_grid(points=31, low=.01, high=100.):
    """Log grid including the requested decade landmarks exactly."""
    landmarks = [x for x in [.01,.1,1.,10.,100.] if low <= x <= high]
    return np.unique(np.r_[np.geomspace(low,high,points),landmarks])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=31)
    parser.add_argument('--gamma-min',type=float,default=.01)
    parser.add_argument('--gamma-max',type=float,default=100.)
    parser.add_argument('--preset',choices=['thesis','csv'],default='thesis')
    parser.add_argument('--output',default='benchmarking_results')
    parser.add_argument('--samples',type=int,default=480)
    args = parser.parse_args()
    config = parameters(args.preset)
    rows = run_benchmarks(config,coupling_grid(args.points,args.gamma_min,args.gamma_max),samples=args.samples)
    save_results(rows, config, args.output)


if __name__ == '__main__':
    main()
