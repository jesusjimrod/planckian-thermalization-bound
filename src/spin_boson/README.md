# Spin–boson model: mean-force Gibbs states and thermalization times with TEMPO

Production code for Sec. V of the paper. Output: `data/spin_boson/`; tutorial:
`spin_boson.ipynb`.

A spin coupled through $\sigma_z$ to an Ohmic bath with exponential cutoff, treated with
time-evolving matrix product operators (TEMPO) in imaginary time for the mean-force Gibbs (MFG)
states and in real time for the dynamics. The implementation is OQuPy 0.5.0 with the corrections
listed below; the reference for the method is Chiu, Strathearn and Keeling, *Phys. Rev. A* **106**,
012204 (2022), whose conventions the code follows.

## Conventions

$$H_S=\tfrac12\,\bm h\cdot\bm\sigma,\qquad V=\sigma_z\otimes\sum_k g_k(b_k+b_k^\dagger),
  \qquad J(\nu)=\sum_k g_k^2\,\delta(\nu-\nu_k)=\alpha\,\nu\,e^{-\nu/\omega_c}.$$

$\hbar=k_B=1$ and $\Delta=|\bm h|=1$ is the energy unit; $\beta\Delta=1$ and $\omega_c=10\Delta$
unless stated. The manuscript writes the spectral density as
$g^2\mathcal J(\Omega)=\tfrac\pi2\alpha\,\Omega\,e^{-\Omega/\Omega_c}$ with $\alpha$ the
**Leggett coupling**, so that $\alpha_{\rm CSK}=\alpha/2$: the figures and the paper use
$\alpha$, while the code and the CSV files store $\alpha_{\rm CSK}$ (and `finite_bound.csv`
carries both, as `alpha` and `alpha_L`). The reorganization energy is
$\lambda=\alpha_{\rm CSK}\omega_c$. Relative to OQuPy's `PowerLawSD`,
$\alpha_{\rm CSK}=2\alpha_{\rm OQ}$ for a $\sigma_z$ coupling operator; the code uses a
`CustomSD`-derived `MatsubaraSD` directly.

The perturbations are normalized to unit seminorm, as in Eq. (38) of the manuscript:
$\kappa_\theta=\tfrac1\Delta\partial_\theta H_S=\tfrac12\bm n_\theta\cdot\bm\sigma$ with
$\|\kappa_\theta\|=1$. Since the code works at $\Delta=1$, this is also what it computes.

## Environment

OQuPy 0.5.0 pins NumPy < 2, so it needs its own environment, separate from the one the notebooks
use:

```
python3.11 -m venv venv_tempo
venv_tempo/bin/pip install oqupy==0.5.0 nbformat ipykernel
venv_tempo/bin/python src/spin_boson/compute_local_theta.py
```

## Corrections to OQuPy 0.5.0

All three are in `sb_core.py`; the released package is otherwise used unchanged.

1. **Imaginary-time bath coefficients.** An overflow guard in `CustomSD.eta_function` drops the
   term $e^{-(\beta-\tau)\omega}$ for $\omega\gtrsim36\,T$. That term is $O(1)$ as $\tau\to\beta$,
   so the coefficients coupling the first and last slices are wrong, and the resulting bias does
   **not** decrease with the number of slices $N$: it is 1.5% in the last coefficient at $N=8$ and
   biases $\langle\tau^x\rangle$ of the MFG state by $+4\times10^{-4}$. `MatsubaraSD` replaces
   them by a stable quadrature of the twice-integrated kernel. After the fix, the reference value
   of Chiu *et al.* is reproduced more closely than by the released code.
2. **Transposition.** `GibbsTempo` returns $\tilde\rho^{\mathsf T}$, which equals
   $\tilde\rho^{*}$ for a Hermitian state and is therefore invisible unless $h_y\neq0$, where it
   flips the sign of $\langle\sigma_y\rangle$. `mfg_state()` transposes back.
3. **Real-time $\eta(t)$** is replaced by its closed form for this spectral density,
   $\eta(t)=\alpha[\ln(1+i\omega_ct)-i\omega_ct]+2\alpha[\ln\Gamma(1+z)-\mathrm{Re}\ln\Gamma(1+z+it/\beta)]$
   with $z=1/(\beta\omega_c)$. The released quadrature agrees with it to $2\times10^{-8}$ up to
   $t=20$, so this is a robustness fix, not a correction of an error.

Two further robustness changes were needed during the runs: OQuPy's real-time backend calls
`np.linalg.svd` without a fallback and failed once, so `rt_maps._robust_svd` falls back to
`gesvd` as the imaginary-time backend already does; and the stable form of the kernel integral
was changed for $\beta\nu>709$, where an intermediate product overflowed. The $\beta\Delta=1$
results are unaffected by the latter.

## Pipeline

| step | script | output | cost |
|---|---|---|---|
| MFG states | `sb_core.mfg_state`, `mfg_cache.py` | `data/spin_boson/mfg_cache.jsonl` (not distributed) | 3–50 s per state, $N=100$–400 |
| locally exact bound | `compute_local.py` | `local.csv` | ~860 states |
| angle scan | `compute_local_theta.py` | `local_theta.csv` (Fig. 3) | ~60 states |
| low temperature | `compute_local_lowT.py` | `local_lowT.csv` | 24 states, $N=400\to800$ |
| finite-error bound | `compute_finite_bound.py` | `finite_bound.csv`, `finite_states.csv` | ~1150 states |
| larger cutoff | `compute_finite_bound_wc.py` | `finite_bound_wc100.csv` | |
| speed-limit upper bound | `ss_bound.py`, `compute_ss_upper.py` | `ss_upper.csv` | quadratures |
| real-time maps | `rt_maps.py`, `precompute_maps.py` | `data/spin_boson/maps/` (not distributed) | 3–15 CPU-min per member |
| dynamics | `ttm.py`, `compute_dynamics.py` | `dynamics.csv` (Fig. 4) | seconds, once the maps exist |
| tables | `make_tables.py` | `tables.md` | |

The response is taken with a 5-point stencil of step $0.04\Delta$ along $h_x$ and $h_z$, with
$J_{yy}=b_x/h_x$ from the symmetry, and Richardson-extrapolated in $N$.

### Why transfer tensors

Plain TEMPO at the SVD precision needed here costs about 0.6 s per step and $\tau_\varepsilon$ can
be $10^5$ steps away; process-tensor TEMPO was slower still in a pilot. Instead, the maps are
computed exactly to $t=K\,dt$ with untruncated memory (three runs per member) and propagated by
transfer tensors with $K=200$. Inside that window the maps are the full-memory maps. The
extrapolation error is monitored by comparing $K$ with $K/2$ and by $\eta$, the Bures-angle
mismatch between the real-time steady state and the imaginary-time MFG target; it falls as
$1/K^2$, and its $1/K^2$ extrapolation reproduces the imaginary-time value to $4\times10^{-6}$ in
a pilot.

A note on precision: at SVD precision $10^{-7}$ the late-time real-time state drifts by about
$10^{-3}$ and depends erratically on $dt$, from accumulated truncation error. Production uses
$10^{-9}$ or better, where the approach to the MFG state is monotonic.

## Validation

Fourteen check scripts, each exiting non-zero on failure; their output is in `logs/`. Most are
tested against deliberately mutated versions of themselves, which they must reject.

| script | claim | result |
|---|---|---|
| `check_gibbs_pathsum.py` | the corrected `GibbsTempo` equals a brute-force path sum, 5 cases including complex $H_S$ | PASS, $\leq6\times10^{-15}$; one-sided splitting, transposed state and raw OQuPy all rejected |
| `check_eta_realtime.py` | the closed-form $\eta$ against its defining double integral | PASS |
| `check_realtime_orientation.py` | real-time states and maps are not transposed | PASS, $10^{-8}$ |
| `check_local_symmetry.py` | $b_y=0$, $J_{xy}=J_{zy}=0$, $J_{yy}=b_x/h_x$ | PASS to 7 digits; raw OQuPy rejected on the sign of $J_{yy}$ |
| `check_local_results.py` | weak and ultrastrong limits, Maxwell symmetry $J_{xz}=J_{zx}$, stencil and Richardson budgets, $\widetilde\chi\leq1/2$ | PASS (asymmetry $\leq2.5\times10^{-5}$, Richardson $\leq10^{-4}$) |
| `check_ss_bound.py` | every step of the $\delta E_S$ lemma, on a qubit plus two modes by exact diagonalization | PASS; three mutations rejected |
| `check_finite_limits.py` | the pinned family equals the weak-coupling resonant-level task of `../fermionic/families`, and the bare-Gibbs limit | PASS, $\chi_\delta$ to $1.3\times10^{-3}$ |
| `check_maps_scheme.py` | the 3-run and 4-run map schemes agree within TEMPO precision | PASS |
| `check_t0_csk.py` | transfer-tensor decay time against polaron theory | $t_0$ PASS; steady state fails at $\alpha_L\geq0.6$, by the $O(dt^2)$ coherence error below |
| `check_dt_coherence.py` | that offset is $O(dt^2)$ Trotter error | PASS, ratio 4.09; $dt\to0$ within $1.3\times10^{-5}$ of the MFG state |
| `check_dt_richardson.py` | Richardson in $dt$ at $\alpha_L=1$ | **FAIL** (non-asymptotic, unphysical $|b|>1$): not used anywhere |
| `check_dynamics.py` | five gates per real-time row, the Theorem among them | 30 of 50 rows pass, all with $\alpha_L\leq0.4$; see below |
| `check_rotation_limits.py`, `check_local_results.py` | limits of the rotation channel | PASS |

`check_dynamics.py` is the one that does not pass everywhere, and the figure reflects that
rather than hiding it. Its gates are (i) the Theorem, (ii) $\eta<0.15\,\varepsilon_{\min}$ with a
bracket below 3%, (iii) transfer-tensor memory $|\tau(K)-\tau(K/2)|/\tau<5\%$, (iv) sustained
thermalization, and (v) $\tau_{\rm QFI}\leq\tau_\varepsilon$. Fourteen rows ($\alpha_L=0.6$ to
1.4) fail (ii) alone; those points appear in Fig. 4 as open markers carrying the bracket as an
error bar. Six rows ($\alpha_L=2$) have no $\tau_\varepsilon$ within the horizon and are dropped.
The count on the last line of the log is of failed gate instances, $14\times1+6\times4=38$, not
of rows.

The imaginary-time side — every bound in the paper — is unaffected by this; it is a limitation of
the real-time numerics only.

## Scope note on the speed-limit scripts

`ss_bound.py`, `compute_ss_upper.py` and `check_ss_bound.py` evaluate a dynamics-free *upper*
bound on the Shiraishi–Saito time: a variational lower bound on $\delta E_S(t)$ that needs only
the mean-force targets and the spectral density, and therefore holds for the exact dynamics
whatever method computes it. It was used during this work to confirm that no evaluation of the
Shiraishi–Saito bound could exceed ours for this task. The final manuscript reaches the same
conclusion directly — the trajectory form needs the energy fluctuations of $H_S+V$ in the state
of the compound, which methods for the reduced dynamics do not provide, and the static form
vanishes because $\|V\|$ is unbounded for a bosonic bath — so the lemma is not part of the
paper. The scripts, the data and the validation log are kept here as supporting material.
