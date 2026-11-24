"""Shared DQD Hamiltonian family and preparation validation."""
import numpy as np

def dqd_energies(delta=.2):
    delta=float(delta)
    if not np.isfinite(delta) or delta==0:
        raise ValueError('dqd_delta must be finite and nonzero. Delta=0 has an exact dark mode; this relaxing-to-MFG sweep does not handle that case.')
    # Preserve the original default floating-point energies exactly.
    if delta==.2:return [[1.2,1.],[2.45,2.25]]
    return [[mean+delta/2,mean-delta/2] for mean in [1.1,2.35]]

def dqd_preparation(occupations=(1.,0.)):
    n=np.asarray(occupations,dtype=float)
    if n.shape!=(2,) or not np.all(np.isfinite(n)) or np.any((n<0)|(n>1)):
        raise ValueError('dqd_occupations must contain two finite occupations in [0,1].')
    return n.tolist()
