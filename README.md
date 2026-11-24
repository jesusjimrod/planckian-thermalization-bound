# Planckian thermalization times in strongly coupled open quantum systems

Code and data for the paper

> J. Jiménez-Rodríguez, J. Calsamiglia and M. Perarnau-Llobet,
> *Planckian thermalization times in strongly coupled open quantum systems*.
> Física Teòrica: Informació i Fenòmens Quàntics, Departament de Física,
> Universitat Autònoma de Barcelona, 08193 Bellaterra (Barcelona), Spain.

The paper gives a lower bound on the time a system needs to thermalize when it is strongly
coupled to its environment,

$$\tau \;\geq\; \tau_{\rm Pl}\,\chi,\qquad \tau_{\rm Pl}=\hbar\beta=\hbar/k_BT,$$

in which the factor $\chi$ depends only on how distinguishable the mean-force Gibbs states of a
family of nearby system Hamiltonians are — a static susceptibility and an equilibrium variance of
a system observable. No knowledge of the dynamics enters. This repository contains the numerics
that characterize that bound in three models, and the data they produced.

## Code overview

Three tutorial notebooks, one per model family. Each explains the model, the quantity being
bounded and the numerical method, re-derives some of the stored quantities from more elementary
ones as a check, and rebuilds that section's figures from the data in `data/`. They read data
only — nothing in them reruns a simulation — so each runs in a few seconds.

| notebook | section of the paper | figures |
|---|---|---|
| `fermionic.ipynb` | IV — exactly solvable fermionic models (resonant level, double quantum dot) | 1, 2, and S1, S2 of Appendix B |
| `spin_boson.ipynb` | V — spin–boson model from weak to ultrastrong coupling, via TEMPO | 3, 4 |
| `heisenberg.ipynb` | VI — a spin in a 2D Heisenberg antiferromagnet, via SSE quantum Monte Carlo | 5 |

Three supporting modules:

- **`planckian.py`** — the definitions of the bound: Bures angles, $\chi_\delta$ maximized over
  pairs, the locally exact $\widetilde\chi$ for a two-level and for a qubit target, the
  correlation factor $r_\kappa$ and the bracket it implies. Running `python planckian.py`
  executes its self-tests.
- **`plotstyle.py`** — the figure style of the paper: geometry, colour code, typography.
- **`schematic.py`** — the lattice drawing of Fig. 5(a).

## Repository layout

```
planckian.py  plotstyle.py  schematic.py     shared definitions and figure style
fermionic.ipynb  spin_boson.ipynb  heisenberg.ipynb
figures/                                     the figures of the paper, as the notebooks write them
data/
  fermionic/{local_sensitivity,families,speedlimits}/
  spin_boson/
  heisenberg/
src/
  fermionic/{engine,local_sensitivity,families,speedlimits}/
  spin_boson/        + logs/ from the validation scripts
  heisenberg/{sse,analysis}/
```

`src/` holds the production code that generated everything in `data/`. It is not needed to run
the notebooks. Each subfolder has its own `README.md` with the method, the scripts, their cost
and how they were validated.

## Installation

```
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/jupyter lab
```

Python 3.11 or newer. The notebooks need only NumPy, SciPy and Matplotlib; the pinned versions
are the ones the published figures were produced with, and nothing depends on them.

Figure text is typeset with LaTeX when a `latex` executable is on the `PATH`, and with
Matplotlib's own mathtext otherwise; `plotstyle.setup()` decides this automatically. Without
LaTeX the figures are correct but the fonts differ slightly from the published ones.

Rerunning the production code needs more: the spin–boson pipeline needs OQuPy 0.5.0 in a separate
environment (it pins NumPy < 2), and the Heisenberg pipeline needs a C++ compiler. Both are
described in the corresponding `src/` README and in the last section of each notebook.

## Data

Everything the figures and tables of the paper rest on is in `data/`, as CSV with a JSON
configuration and, where the pipeline produced one, a JSON record of the checks it ran.

| file | contents |
|---|---|
| `fermionic/local_sensitivity/local_sensitivity.csv` | $\widetilde\chi$ and readout fractions for the resonant-level and double-dot perturbations, against coupling |
| `fermionic/local_sensitivity/ss_bandwidth_table.csv` | $\tau_\varepsilon$, the bound and the Shiraishi–Saito bounds against bandwidth, in table form |
| `fermionic/families/sweep.csv` | $\tau_\varepsilon$ and $\chi_\delta$ for the three fixed Hamiltonian families, five accuracies |
| `fermionic/speedlimits/benchmark.csv` | the same at six reservoir bandwidths, with the Shiraishi–Saito bound |
| `spin_boson/local_theta.csv`, `local.csv`, `local_lowT.csv` | mean-force Gibbs response $(\bm b, J)$ and the locally exact bounds against angle and coupling |
| `spin_boson/finite_bound.csv`, `finite_states.csv` | $\chi_\delta$ of the bias family and the targets it was computed from |
| `spin_boson/dynamics.csv` | $\tau_\varepsilon$ from real-time TEMPO with transfer tensors, with its convergence diagnostics |
| `spin_boson/ss_upper.csv` | a dynamics-free upper bound on the Shiraishi–Saito bound, derived during this work; the final manuscript makes its point without it, so it is supporting material rather than a result |
| `heisenberg/local.csv`, `local_comb.csv` | per-run and combined $\widetilde\chi$, $r_\kappa$ and the imaginary-time correlation of $\sigma^z_0$ |
| `heisenberg/finite.csv` | $\chi_\delta$ on pairs of simulated fields, split-sample and lower-confidence-limit variants |
| `heisenberg/mfg.csv` | the measured polarization of the system spin against the applied field |

Large intermediates are not distributed: the real-time maps and mean-force state cache of the
spin–boson pipeline, and the 1089 raw per-bin Monte Carlo outputs of the Heisenberg runs. The
scripts that produce them are in `src/`, and the files above are the complete input to the
notebooks and to every figure and table of the paper.

## Conventions

$\hbar=k_B=1$ and $\beta=1$ throughout, so energies are in units of $k_BT$ and times in units of
$\tau_{\rm Pl}$. Distances between states are Bures angles, in $[0,\pi/2]$. Perturbations are
normalized to unit spectral seminorm, $\|\kappa\|=1$. Accuracies are quoted as fractions of
$\varepsilon_{\max}=\pi/4$, the largest accuracy at which the bound can be nontrivial, and every
model uses the same ball $\beta\delta=2\pi$. These are collected, with the formulas that use
them, at the top of `planckian.py`.

Equation, figure and section numbers in the notebooks refer to the manuscript.

## License

MIT. See `LICENSE`.
