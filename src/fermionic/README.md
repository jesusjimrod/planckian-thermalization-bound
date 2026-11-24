# Fermionic models: exact dynamics and exact mean-force Gibbs states

Production code for Sec. IV of the paper. Output: `data/fermionic/`; tutorial:
`fermionic.ipynb`.

$N$ fermionic levels coupled to one common reservoir with a Lorentzian spectral density of width
$\Lambda$. The total Hamiltonian is quadratic and conserves particle number, so for a Gaussian,
number-conserving preparation both the mean-force Gibbs (MFG) state and the reduced dynamics are
known in closed form at arbitrary coupling. Units: $\hbar=\beta=1$; $\gamma=g^2$ is the
hybridization at the Fermi level; energies in $k_BT$, times in $\tau_{\rm Pl}$. The seminorm is
the Fock-space spectral seminorm, which for a quadratic $\kappa=\sum_{mn}k_{mn}d^\dagger_md_n$ is
the trace norm of $k$.

## Layout

| folder | role |
|---|---|
| `engine/` | the solver. `continuum.py` is the finite-$\Lambda$ Lorentzian model; `wideband.py` and `wideband_reference.py` the wide-band limit; `benchmark.py` the thermalization-time search; `model_parameters.py` the double-dot parameters. Used unchanged by everything below. |
| `local_sensitivity/` | the locally exact bound $\widetilde\chi$ (Fig. 1) and the bandwidth table |
| `families/` | the finite-accuracy bound and $\tau_\varepsilon$ for the three fixed Hamiltonian families, in the wide band |
| `speedlimits/` | the same at six reservoir bandwidths, against the Shiraishi–Saito bound |

Run each script from the repository root; each writes into its own folder under
`data/fermionic/`.

```
python src/fermionic/local_sensitivity/local_sensitivity.py     # ~1 min
python src/fermionic/local_sensitivity/ss_bandwidth_table.py    # ~1 min
python src/fermionic/local_sensitivity/make_table.py            # the bandwidth table, LaTeX
python src/fermionic/families/compute.py                        # ~10 min, 8 processes
python src/fermionic/speedlimits/compute.py                     # ~10 min, 8 processes
```

`speedlimits/compute.py` also tries to reproduce an earlier fixed-pair run as a regression test.
That raw run is not distributed here, so the check is skipped unless
`src/fermionic/speedlimits/reference/coupling_benchmark.csv` is present.

## What is computed

**Families** $H_S(x)=\bar H_S+x\hat\kappa$ with $\|\hat\kappa\|=1$ and $|\beta x|\leq\delta=2\pi$,
fixed for a whole sweep so that only $g^2$ varies:

| name | $\hat\kappa$ | $H_S(x)$ | preparation |
|---|---|---|---|
| `rlm` | $n_d$ | $x\,n_d$ | $p(0)=1/2$ |
| `dqd_common` | $N_S/2$ | $\mathrm{diag}(1.2,1.0)+x\,N_S/2$, $\beta\Delta=0.2$ | $(n_1,n_2)=(1,0)$ |
| `dqd_dark` | $n_-$ | $\mathrm{diag}(1.2,1.0)+x\,n_-$, $\beta\Delta=0.2$ | $(n_1,n_2)=(1,0)$ |

The value $\beta\delta=2\pi$ is the one the manuscript fixes: for a weakly coupled level at
resonance the thermal occupation then runs from nearly 1 at $x=-\delta$ to nearly 0 at
$x=\delta$, so the extreme targets are nearly orthogonal. (The comment `delta = 8 eps_max` in
`families/compute.py` records an earlier justification of the same number, from the superseded
condition $\varepsilon\leq\beta\delta/8$; the current condition is $\varepsilon<\beta\delta/4$.)

**Bound.** $\chi_\delta=\max_{x_1\neq x_2}\bigl[2D(\tilde\omega_1,\tilde\omega_2)-4\varepsilon\bigr]_+/|x_1-x_2|$,
evaluated on a dense pair grid and refined by constrained Nelder–Mead, so the optimal pair is
free to move inside the fixed ball as the coupling changes.

**Time.** $\tau_\varepsilon=\min\{t:\max_x D(\rho_S(t,x),\tilde\omega_x)\leq\varepsilon\}$, with
$x$ running over 21 equally spaced members plus the maximizing pair, so
$\tau_\varepsilon\geq\tau_{\rm Pl}\chi_\delta$ holds by construction. Each member is evaluated on
its own time grid, resolving its own oscillation frequencies; the intervals in which it is within
$\varepsilon$ are refined with `brentq` and intersected over members.

## Validation

Each script writes a `checks.json` beside its data. The entries recorded are, in summary:

- $\tau_{\rm Pl}\chi_\delta\leq\tau_\varepsilon$ and $\tau_{\rm SS}\leq\tau_\varepsilon$: no
  violations on any row.
- No member leaves the $\varepsilon$-ball after $\tau_\varepsilon$, so $\tau_\varepsilon$ is also
  the sustained time.
- Member sampling: going from 21 to 41 members changes $\tau_\varepsilon$ by at most $4\times10^{-11}$
  relative; the extreme members $x=\pm\delta$ set the time.
- The vectorized error used for the grid search agrees with the engine's scalar error to
  $10^{-7}$, and every crossing is refined with the scalar one.
- Resonant level: $\tau_\varepsilon=0$ exactly where $\chi_\delta=0$, at every accuracy. With
  $p(0)=1/2$ the preparation is the geodesic midpoint of the targets at $\pm\delta$, so this is
  an identity rather than a numerical coincidence.
- Strong-coupling limits of $\widetilde\chi$ agree with $\sqrt{f(1-f)}$ and $\sqrt{f(1-f)}/2$ to
  $10^{-5}$.
- Finite $\Lambda$ against the wide band, inside the wide-band regime: the double dot agrees to
  $5\times10^{-4}$ at $\Lambda=10^4k_BT$ and $5\times10^{-6}$ at $10^6k_BT$; the resonant level to
  0.3% away from its threshold, with the largest deviations next to it where the threshold itself
  shifts slightly.
- The double-dot solver for non-diagonal $h$ reproduces the engine to $4\times10^{-11}$ wherever
  $h$ is diagonal and the two overlap.

## Scope

- The finite-$\Lambda$ engine handles diagonal $h$ only, so the dark-mode family, which needs
  tunnelling, is wide-band only.
- $\Lambda=10^2k_BT$ lies outside the wide-band regime at every coupling shown. Its rows are kept
  in `benchmark.csv` and flagged by `wb_regime`, but they describe a narrow-band model rather than
  an approximation of the wide-band one.
- The double-dot families are sub-families of the full ball of Theorem 1, not the whole ball: for
  $\delta\geq0.2$ the ball contains $\Delta=0$ Hamiltonians with real tunnelling, whose exact dark
  mode never thermalizes.

## Relation to the analytics of the manuscript

The expressions in the code were checked line by line against Sec. IV and Appendix B:

- `dqd_solver.mfg` builds $\Gamma=-ih-\tfrac{g^2}{2}\bm u\bm u^{\rm T}$ and
  $C^{\rm th}=\tfrac12\mathbb I-\tfrac{1}{2\pi i}[\psi(\tfrac12\mathbb I-\tfrac{\Gamma}{2\pi})-{\rm h.c.}]$,
  i.e. Eqs. (28) and (29) at $\hbar=\beta=1$, in the convention
  $C_{mn}=\mathrm{Tr}[\rho_S d_n^\dagger d_m]$.
- `compute.p_th` is Eq. (32); `theta = arcsin(sqrt(p_th))` gives the diagonal Bures angle
  $D=|\theta_x-\theta_y|$, and `rlm_thresholds` solves
  $D(\tilde\omega_{-\delta},\tilde\omega_\delta)=2\varepsilon$ exactly as the text does.
- The eigenvalues of $\Gamma$ reduce to
  $-i\bar\epsilon-\tfrac{g^2}{2}\pm\tfrac{\sqrt{g^4-\Delta^2}}{2}$ for `HBAR = diag(1.2, 1.0)`.
- `span` is the Fock-space seminorm $\sum_i|\lambda_i(k)|$; `ss_static` and `ss_variance` are the
  first and second forms of the chain of Eq. (11).

- The `slowest_rate` column, plotted as the dotted curve in Fig. S1(b), is
  $\min_{|x|\leq\delta}2|\mathrm{Re}\,\lambda_+(x)|$: the inverse of the longest
  **occupation** relaxation time, $2\hbar g^2/\Delta^2$ at strong coupling. Since
  $\Gamma_{\rm slow}=\Delta^2/(4\hbar g^2)$ of Sec. IV C is an *amplitude* rate, that curve is
  $1/(2\Gamma_{\rm slow})$, as the figure and its caption label it.
