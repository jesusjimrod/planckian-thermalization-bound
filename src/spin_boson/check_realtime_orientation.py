"""Check 3: orientation of OQuPy real-time states (rho vs rho^T) and of the dynamical maps (ttm.py).
With a negligible bath (alpha = 1e-9), TEMPO and PT-TEMPO must reproduce unitary evolution
rho(t) = e^{-iHt} rho0 e^{iHt}, whose <sigma_y> is odd under rho -> rho^T. The maps from
ttm.maps_from_pt must reproduce the same states for a fifth, generic initial state.
Failure control: comparing with the transposed exact state must fail. Exit 1 on failure."""
import sys, warnings
import numpy as np
from scipy.linalg import expm
import oqupy
import sb_core as sc, ttm
warnings.filterwarnings('ignore')
H = sc.hamiltonian((1.0, 0.0, 0.7)); dt = 0.05; tend = 3.0
rho0 = np.array([[0.8, 0.3 - 0.2j], [0.3 + 0.2j, 0.2]])
bath = oqupy.Bath(sc.SZ, sc.MatsubaraSD(1e-9, 10.0, 1.0))
pars = oqupy.TempoParameters(dt=dt, epsrel=1e-10, tcut=1.0)
dyn = oqupy.Tempo(oqupy.System(H), bath, pars, rho0, 0.0).compute(end_time=tend, progress_type='silent')
ts = np.array(dyn.times); st = np.array(dyn.states)
ex = np.array([expm(-1j * H * t) @ rho0 @ expm(1j * H * t) for t in ts])
e_tempo = np.abs(st - ex).max(); e_T = np.abs(st - np.transpose(ex, (0, 2, 1))).max()
pt = oqupy.pt_tempo_compute(bath, 0.0, tend, pars, progress_type='silent')
t2, E = ttm.maps_from_pt(H, pt)
st2 = ttm.apply(E, rho0); ex2 = np.array([expm(-1j * H * t) @ rho0 @ expm(1j * H * t) for t in t2])
e_maps = np.abs(st2 - ex2).max()
by = np.array([np.trace(s @ sc.SY).real for s in st])
print(f"TEMPO vs exact: {e_tempo:.1e}   TEMPO vs exact^T (control): {e_T:.1e}   maps(PT) vs exact: {e_maps:.1e}   max|<sy>|={abs(by).max():.2f}")
fails = (e_tempo > 1e-6) + (e_maps > 1e-6) + (e_T < 1e-2)
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
