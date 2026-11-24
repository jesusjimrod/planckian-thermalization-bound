"""Thesis wide-band dynamics, reusing the established source solver.

Only long-time exponentiation and slow-pole evaluation are changed, to avoid
cancellation as g^2 increases. The vendored original source is wideband_reference.py.
"""
from pathlib import Path
import importlib.util
import numpy as np
from continuum import hermitian,density,bures
p=Path(__file__).resolve().parent/'wideband_reference.py'
spec=importlib.util.spec_from_file_location('thesis_wideband',p)
original=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=original;spec.loader.exec_module(original)

class WideBand(original.WideBandDynamics):
    def __init__(self,energies,gamma,initial,numerics=None):
        kwargs={} if numerics is None else dict(rtol=numerics.rtol,atol=numerics.atol,memory_beta=numerics.memory_beta)
        super().__init__(energies,gamma,initial,original.Parameters(**kwargs))
        self.energies=np.array(energies);self.beta=1.;self.m=self.n
        if self.n==1:self.poles=np.array([-gamma/2-1j*energies[0]])
        else:
            mean=np.mean(energies);d=energies[0]-energies[1]
            root=np.sqrt(complex(gamma*gamma-d*d))
            slow=-d*d/(2*(gamma+root))-1j*mean
            fast=-(gamma+root)/2-1j*mean
            self.poles=np.array([slow,fast])
        self.slow_time=1/min(-self.poles.real)
        self.equilibrium=self.n_mfg
        self.initial=self.n0
    def propagator(self,t):
        if self.n==1:return np.array([[np.exp(self.poles[0]*t)]])
        a,b=self.poles
        if abs(a-b)<1e-8*max(1,self.gamma):return original.expm(self.a*t)
        return (np.exp(a*t)*(self.a-b*np.eye(2))-np.exp(b*t)*(self.a-a*np.eye(2)))/(a-b)
    def at(self,t):
        if self.stationary or t<=self.t_switch:return super().at(t)
        u=self.propagator(t-self.t_switch)
        diff=hermitian(u@(self.n_switch-self.n_mfg)@u.conj().T)
        return self.n_mfg+diff,hermitian(self.a@diff+diff@self.a_dag)
    def error(self,t):return bures(density(self.at(t)[0]),self.target)
