"""Real-time thermalization times tau_eps [Eq. tau_eps] for the spin-boson bias family.

Per coupling alpha and member x: four OQuPy real-time TEMPO runs without memory truncation up to t_maps
(rt_maps.py) give the exact dynamical maps E_n, n <= t_maps/dt. Transfer tensors with memory K = t_maps/dt
propagate rho_S(t) from rho_S(0) = I/2 to long times (ttm.py); a second memory K/2 measures the
TTM-memory sensitivity. D_x(t) = D(rho_S(t,x), w_x) uses the imaginary-time MFG targets w_x
(Richardson-extrapolated, data/finite_states.csv, spline in x).

Task as in compute_finite_bound.py: H_S(x) = (sigma_x + x sigma_z)/2, |x| <= 2 pi, rho_S(0) = I/2. Members:
x in MEMBERS (x >= 0; x -> -x is a symmetry of the task: pi rotation about x plus bath parity, which
leaves I/2 (x) omega_B invariant) plus |x1|, |x2| of the optimal pairs of compute_finite_bound.py.

Families: 'bias' (above; default) and 'rot', the rotation family of Sec. V at fixed |h| = Delta,
H_S(u) = (Delta/2)(cos u sigma_x + sin u sigma_z), |u| <= pi/3 (compute_finite_rotation.py), same symmetry.

Usage: compute_dynamics.py [--rot] nproc alpha1 [alpha2 ...]   (dt, t_maps, epsrel in CFG)
Outputs data/dynamics.csv (rot: dynamics_rot.csv), replacing rows of these alphas, and data/dyn_a<alpha>.npz
(rot: dynrot_a<alpha>.npz).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import csv, sys, time
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

import rt_maps
import sb_core as sc
import ttm

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data' / 'spin_boson'
CFG = dict(dt=0.05, t_maps=10.0, epsrel=1e-9, eps_fractions=[0.01, 0.05, 0.1, 0.15, 0.2])
MEMBERS = list(np.linspace(0.0, 2 * np.pi, 11))
FAMILIES = {
    'bias': dict(h=lambda x: (1.0, 0.0, float(x)), members=MEMBERS, states='finite_states.csv',
                 bound='finite_bound.csv', out='dynamics.csv', npz='dyn_a{a:g}.npz'),
    'rot': dict(h=lambda u: (float(np.cos(u)), 0.0, float(np.sin(u))), members=list(np.linspace(0.0, np.pi / 3, 11)),
                states='finite_states_rot.csv', bound='finite_bound_rot.csv', out='dynamics_rot.csv', npz='dynrot_a{a:g}.npz'),
}


def targets(alpha, fam='bias'):
    rows = [r for r in csv.DictReader(open(DATA / FAMILIES[fam]['states'])) if abs(float(r['alpha']) - alpha) < 1e-12]
    x = np.array([float(r['x']) for r in rows]); bx = np.array([float(r['bx']) for r in rows]); bz = np.array([float(r['bz']) for r in rows])
    xs = np.concatenate([-x[:0:-1], x])
    sx = CubicSpline(xs, np.concatenate([bx[:0:-1], bx])); sz = CubicSpline(xs, np.concatenate([-bz[:0:-1], bz]))
    return lambda xx: np.array([float(sx(xx)), 0.0, float(sz(xx))])


def optimal_pair_members(alpha, fam='bias'):
    """|x*| of the optimal pair (-x*, x*) for each accuracy. The task is symmetric under x -> -x, so the optimal
    pair is symmetric; the optimizer returns |x1| and |x2| equal to ~1e-7, and x* is their mean (rounded to 1e-8),
    which changes chi_12 only at second order in the asymmetry."""
    out = []
    for r in csv.DictReader(open(DATA / FAMILIES[fam]['bound'])):
        if abs(float(r['alpha']) - alpha) < 1e-12 and float(r['chi']) > 0:
            out.append(round(0.5 * (abs(float(r['x1'])) + abs(float(r['x2']))), 8))
    return out


def bloch_series(states):
    """Bloch vectors of rho/Tr(rho): TEMPO maps preserve the trace only to ~1e-5 (SVD truncation), and the
    TTM recursion would otherwise turn that into a slow drift of the trace."""
    tr = np.einsum('nii->n', states).real
    return np.stack([np.einsum('nij,ji->n', states, s).real for s in sc.PAULI], axis=1) / tr[:, None]


def D_series(states, bt):
    return np.array([sc.bures_angle_bloch(b, bt) for b in bloch_series(states)])


def first_entry(t, Dm, eps):
    inside = Dm <= eps
    if not inside.any():
        return np.nan, np.nan
    i = int(np.argmax(inside))
    tau = t[0] if i == 0 else t[i - 1] + (eps - Dm[i - 1]) * (t[i] - t[i - 1]) / (Dm[i] - Dm[i - 1])
    out = np.where(~inside[i:])[0]
    return float(tau), (float(t[i + out[-1]]) if len(out) else np.nan)


def family_members(a, fam='bias'):
    """Fixed members plus optimal-pair members; a pair member within 1e-6 of a fixed one (e.g. a pair clipped
    to the family edge) is the same Hamiltonian and is not added again."""
    fixed = [float(x) for x in np.round(FAMILIES[fam]['members'], 10)]
    extra = [x for x in np.round(optimal_pair_members(a, fam), 10) if min(abs(x - f) for f in fixed) > 1e-6]
    return sorted(set(fixed + [float(x) for x in extra]))


def main(nproc, alphas, fam='bias'):
    c = CFG; dt = c['dt']; K = int(round(c['t_maps'] / dt)); K2 = K // 2; F = FAMILIES[fam]
    mem = {a: family_members(a, fam) for a in alphas}
    M = rt_maps.maps([(a, F['h'](x)) for a in alphas for x in mem[a]], dt, c['t_maps'], c['epsrel'], nproc=nproc)
    rho0 = 0.5 * np.eye(2, dtype=complex)
    rows = []
    for a in alphas:
        tgt = targets(a, fam); per = []
        for x in mem[a]:
            t_m, E, sec = M[(a, tuple(map(float, F['h'](x))))]
            T = ttm.transfer_tensors(E, K)
            lam, rates = ttm.companion_rates(T, dt)
            slow = float(rates[1])                                  # rates[0] ~ 0: steady state
            ss = ttm.steady_state(T); b_ss = np.array([np.trace(ss @ s).real for s in sc.PAULI]) / np.trace(ss).real
            ss_err = sc.bures_angle_bloch(b_ss, tgt(x))
            per.append(dict(x=x, E=E, T=T, slow=slow, ss_err=ss_err, seconds=sec, unit=float(abs(lam[0]))))
        tmax = min(2e5, max(60.0, max((np.log(1.0 / (0.1 * c['eps_fractions'][0] * np.pi / 4)) + 2.0) / max(p['slow'], 1e-12) for p in per)))
        n = int(round(tmax / dt)); t = np.arange(n + 1) * dt
        D_main, D_alt, speed = [], [], []
        for p in per:
            st, _ = ttm.propagate_state(p['E'], K, rho0, n)
            b = bloch_series(st); D_main.append(np.array([sc.bures_angle_bloch(bb, tgt(p['x'])) for bb in b]))
            db = np.gradient(b, dt, axis=0); bb2 = np.minimum(np.einsum('ni,ni->n', b, b), 1 - 1e-14)
            FS = np.einsum('ni,ni->n', db, db) + np.einsum('ni,ni->n', b, db) ** 2 / (1 - bb2)
            speed.append(0.5 * np.sqrt(np.maximum(FS, 0)))          # Bures speed dD/dt = sqrt(F^S)/2
            st2, _ = ttm.propagate_state(p['E'], K2, rho0, n); D_alt.append(D_series(st2, tgt(p['x'])))
        D_main, D_alt, speed = np.array(D_main), np.array(D_alt), np.array(speed)
        D_init = np.array([sc.bures_angle_bloch(np.zeros(3), tgt(p['x'])) for p in per])
        Dm, Da = D_main.max(0), D_alt.max(0)
        for ef in c['eps_fractions']:
            eps = ef * np.pi / 4
            tau, last_exit = first_entry(t, Dm, eps); tau2, _ = first_entry(t, Da, eps)
            # bracket for the real-time vs imaginary-time steady-state mismatch eta (Bures angle):
            # tau(eps + eta) <= tau <= tau(eps - eta) if the trajectory error stays below eta
            eta = max(p['ss_err'] for p in per)
            tau_lo, _ = first_entry(t, Dm, eps + eta); tau_hi, _ = first_entry(t, Dm, max(eps - eta, 1e-12))
            i_tau = int(min(n, np.nan_to_num(tau) / dt))
            # reduced-trajectory QFI speed limit (third form of Eq. ss_chain): 2 D_x / <sqrt F^S>_tau = D_x tau / L_x(tau)
            Lpath = np.trapz(speed[:, :i_tau + 1], dx=dt, axis=1) if i_tau > 0 else np.full(len(per), np.nan)
            tau_qfi = float(np.nanmax(np.maximum(D_init - eps, 0) * tau / Lpath)) if i_tau > 0 else np.nan
            last_member = float(per[int(np.argmax(D_main[:, max(i_tau - 1, 0)]))]['x'])
            rows.append(dict(alpha=a, eps_fraction=ef, eps=eps, tau=tau, tau_K2=tau2, tau_lo=tau_lo, tau_hi=tau_hi, tau_qfi=tau_qfi, last_exit_after_tau=last_exit,
                             horizon=tmax, ss_err_max=max(p['ss_err'] for p in per),
                             slowest_rate_min=min(p['slow'] for p in per), t0_max=1 / min(p['slow'] for p in per),
                             D0_max=float(Dm[0]), n_members=len(per), last_member=last_member, dt=dt, t_maps=c['t_maps'], epsrel=c['epsrel'],
                             tempo_seconds=sum(p['seconds'] for p in per), unit_eig_dev=max(abs(p['unit'] - 1) for p in per)))
        stride = max(1, n // 4000)
        np.savez_compressed(DATA / F['npz'].format(a=a), t=t[::stride], D=D_main[:, ::stride], D_K2=D_alt[:, ::stride],
                            x=np.array([p['x'] for p in per]), slow=np.array([p['slow'] for p in per]),
                            ss_err=np.array([p['ss_err'] for p in per]))
        r = [q for q in rows if q['alpha'] == a and q['eps_fraction'] == 0.05][0]
        print(f"alpha={a:g}: tau(0.05)={r['tau']:.2f} (K/2: {r['tau_K2']:.2f}), t0_max={r['t0_max']:.2f}, ss_err_max={r['ss_err_max']:.1e}, "
              f"horizon={tmax:.0f}, TEMPO cpu={r['tempo_seconds']:.0f}s", flush=True)
    path = DATA / F['out']
    old = [q for q in (list(csv.DictReader(open(path))) if path.exists() else []) if float(q['alpha']) not in alphas]
    allrows = old + rows
    allrows.sort(key=lambda q: (float(q['alpha']), float(q['eps_fraction'])))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(allrows)


if __name__ == '__main__':
    args = sys.argv[1:]
    fam = 'rot' if args and args[0] == '--rot' else 'bias'
    if fam == 'rot':
        args = args[1:]
    main(int(args[0]), [float(v) for v in args[1:]], fam)
