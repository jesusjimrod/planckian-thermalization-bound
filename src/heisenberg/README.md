# Two-dimensional Heisenberg antiferromagnet: stochastic series expansion

Production code for Sec. VI of the paper. Output: `data/heisenberg/`; tutorial:
`heisenberg.ipynb`.

The model is the spin-1/2 Heisenberg antiferromagnet on an $L\times L$ square lattice with
periodic boundaries, $J>0$, in units $J=1$. The system is one spin, site 0; the bath is the rest
of the lattice; the interaction is the four bonds of that site. The perturbation is a local
field, $H_S(x)=\tfrac{x}{2}\sigma^z_0$, with $\|\sigma^z_0/2\|=1$. Because the lattice is
homogeneous, $V$ and $H_B$ have the same scale: there is no separate coupling constant, and
increasing $J$ at fixed temperature is the same as lowering $k_BT/J$.

Two symmetries of $H_{\rm tot}$ are used, and they are exact rather than assumptions. SU(2) makes
the zero-field mean-force Gibbs state of one site exactly $\mathbb I/2$ at every $T$, $J$ and
$L$, and the response isotropic, so $\widetilde\chi=\xi_{\sigma\sigma}/(2\beta)=r_\kappa/2$ is the
optimum over *all* single-spin perturbations [Eq. (42)]. U(1) about $z$ keeps every
$\tilde\omega_x$ diagonal in $\sigma^z_0$, so the Bures angle between targets is the classical
one.

## Method

Stochastic series expansion with heat-bath directed loops (`sse/sse_t.cpp`), a generic solver for
any sign-free, number-conserving bond Hamiltonian with per-bond couplings. A sublattice rotation
makes the antiferromagnet sign-free; there is no Trotter error and the expansion order is not
truncated. The local field is split equally over the four bonds of site 0.

With $n$ the expansion order and $C_k$ the number of propagated states having $n_0=k$ at site 0,

$$p_k=\Bigl\langle\frac{C_k}{n}\Bigr\rangle,\qquad
  \xi_{kl}=\beta\Bigl\langle\frac{C_kC_l+\delta_{kl}C_k}{n(n+1)}\Bigr\rangle-\beta p_kp_l,
  \qquad \xi_{\sigma\sigma}=4\xi_{\uparrow\uparrow},$$

and the imaginary-time correlation $G(\lambda)=\langle\sigma^z_0(\lambda)\sigma^z_0\rangle$ is
sampled without bias by drawing $m\sim\mathrm{Binomial}(n,\lambda/\beta)$ and a uniform position,
64 samples per $\lambda$ per sweep, on the ten-point grid
$\lambda/\beta\in\{0,\tfrac1{32},\tfrac1{16},\tfrac18,\tfrac3{16},
\tfrac14,\tfrac5{16},\tfrac38,\tfrac7{16},\tfrac12\}$.

Statistics: 20 000 equilibration sweeps, then 40 bins of 10 000 sweeps, jackknife over bins. Seeds
are `zlib.crc32` of the run tag, so runs are reproducible across processes and machines.

$\widetilde\chi$ is taken directly from $\xi$ at $x=0$; no fit is involved anywhere.
$\chi_\delta$ is maximized only over pairs of *simulated* fields, which can only lower a maximum
and therefore leaves it a valid bound. The upward bias of maximizing over noisy estimates is
removed by a split-sample procedure — select the pair on bins 1–20 and evaluate it on bins 21–40,
and vice versa — and the figure uses the stricter variant that selects by a one-sided 95% lower
confidence limit. Errors are bootstrap over bins, 300 resamples, of the whole procedure including
the selection.

## Running

```
g++ -O3 -march=native -o src/heisenberg/sse/sse_t src/heisenberg/sse/sse_t.cpp
python src/heisenberg/sse/run_production.py [ncores]     # writes data/heisenberg/raw/
python src/heisenberg/sse/run_rebin.py                   # bin-independence check
python src/heisenberg/sse/run_drift.py                   # equilibration check
python src/heisenberg/analysis/analyse.py                # the CSV files the notebook reads
```

`run_production.py` skips finished runs, so it can be interrupted and restarted. Cost scales
roughly as $\beta L^2$: one $L=48$, $k_BT=0.1J$ run is about 31 500 operators and took 312 s on
one core.

| file | role |
|---|---|
| `sse/sse_t.cpp` | the SSE solver |
| `sse/lib.py`, `sse/libt.py` | lattice bonds, model tables, input files, exact diagonalization, jackknife |
| `sse/runlib.py` | run driver, deterministic seeds, the $\tau$ grid |
| `sse/run_production.py` | the five production sets: the $L=16$ temperature scan, finite-size spot checks at $L=8,32,48$, extra seeds at $L=32$, and the two field scans |
| `sse/run_rebin.py`, `sse/run_drift.py` | bin independence (errors from 40 bins against 8) and drift (first ten bins against the last ten) |
| `sse/validate_ed.py`, `sse/validate_seeds.py` | validation against exact diagonalization, and seed reproducibility |
| `analysis/analyse.py` | estimators, combination over runs, split-sample and LCL $\chi_\delta$, bootstrap errors |

Only `sse_t.cpp` is distributed; it supersedes the two earlier variants of the solver, which
lacked per-bond couplings and the correlation sampling.

## Validation and limits

- **Exact diagonalization.** All estimators are checked against ED on $4\times2$ clusters
  (`validate_ed.py`); all pulls are within $2.5\sigma$ with rms $\approx1$.
- **SU(2).** The measured $\langle\sigma^z_0\rangle$ at $x=0$ is an independent test of
  equilibration, since the symmetry fixes it to zero. Over the 35 production runs the rms pull is
  1.3$\sigma$, the largest 3.3$\sigma$ at $L=48$, $k_BT=0.07J$ — the run with the longest
  autocorrelation times.
- **Rigorous floor.** $G(\lambda)$ is convex and symmetric about $\beta/2$, so its mean over
  $[0,\beta]$ — which is $r_\kappa$, by Eq. (19) — is at least $G(\beta/2)$, giving
  $\widetilde\chi\geq G(\beta/2)/2$ [Eqs. (20) and (43)] from a single point of the correlation
  function. It is satisfied at every $(L,T)$, as it must be, and it is what makes the
  low-temperature statement safe.
- **Two independent estimators.** $r_\kappa$ comes from the integrated susceptibility
  $\xi_{\sigma\sigma}$ and the `G<lambda>` columns from the sampled correlation function. A
  trapezoid of the latter over $[0,\beta]$ reproduces the former to 0.01–3%, always from above,
  as a 10-point trapezoid of a convex function must.
- **Autocorrelation.** Some low-temperature, large-$L$ runs have bins correlated by up to a
  factor 1.8; `run_rebin.py` flags them and the combination inflates their errors accordingly.
- **Finite size.** $L=16$, 32 and 48 agree within 3%, so the bound has converged in the lattice
  size; only $L=8$ falls below them at the lowest temperature, because the finite-size gap starts
  to matter. This is physics, not a convergence failure, and it is why the strong-coupling
  statement requires $L\to\infty$ before $T\to0$.
- **Field grid.** The finite-accuracy bound is limited by the grid spacing in $\beta x$ and by
  the statistics at low temperature; $\varepsilon=0.01\,\varepsilon_{\max}$ is not usable at the
  lowest temperatures on this grid, and points whose bootstrap error exceeds 20% of their value
  are not shown.
- The $L$ dependence of the finite-accuracy bound is only partly mapped: a full field scan exists
  for $L=16$, and a shorter one for $L=32$.
