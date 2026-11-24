"""Wide-band DQD dynamics for a general Hermitian 2x2 single-particle h (gate voltages + tunnelling).

Copied from ../pair_optimization_2026-10-15/general_h.py (same logic), so this folder depends only on the
validated engine ../engine (unmodified). The engine ODE
  Ndot = A N + N A^dag + Q,  Qdot = Gamma[e^{At}J - J e^{A^dag t}]/[2i beta sinh(pi t/beta)],
  Q(0+) = Gamma J/2,  dQ/dt(0+) = -Gamma(hJ+Jh)/(2pi),  A = -ih - Gamma J/2,
depends on h only through A, so it holds for non-diagonal h unchanged. Only the stored h and the late-time
propagator are generalized (slow pole = det(A)/fast pole, no cancellation at Gamma >> |Delta|).
validate() compares with the engine for diagonal h.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'engine'))
from wideband import WideBand,original          # noqa: E402
from continuum import bures,density,hermitian   # noqa: E402

def mfg(h,gamma):
    """Wide-band MFG covariance N_ij=<d_j^dag d_i> (Eq. (mfg_correlation) of main.tex)."""
    n=len(h);a=-1j*h-gamma*np.ones((n,n))/2
    psi=original.matrix_digamma(.5*np.eye(n)-a/(2*np.pi))
    return hermitian(.5*np.eye(n)-(psi-psi.conj().T)/(2j*np.pi))

class GeneralH(original.WideBandDynamics):
    def __init__(self,h,gamma,n0,cfg=original.Parameters()):
        h=hermitian(np.asarray(h,complex));self.gamma=float(gamma);self.config=cfg
        self.n0=np.asarray(n0,complex);self.n=len(h);self.h=h
        self.j=np.ones((self.n,self.n));self.a=-1j*h-gamma*self.j/2;self.a_dag=self.a.conj().T
        self.n_mfg=mfg(h,gamma);self.equilibrium=self.n_mfg
        self.q_inf=hermitian(-self.a@self.n_mfg-self.n_mfg@self.a_dag)
        self.rho0,self.target=density(self.n0),density(self.n_mfg)
        self.stationary=False;self.t_switch=cfg.memory_beta*cfg.beta
        tr=np.trace(self.a);det=np.linalg.det(self.a)
        r=np.sqrt(tr*tr/4-det);c=[tr/2+r,tr/2-r];fast=min(c,key=lambda z:z.real)
        slow=det/fast;self.poles=np.array([slow,fast])
        if slow.real>=0:raise ValueError('undamped mode: h has an exact dark mode')
        self.slow_time=1/(-slow.real)
        size=self.n**2;beta=cfg.beta;scale=max(1.,gamma,np.max(np.abs(np.linalg.eigvalsh(h))))
        def rhs(t,y):
            n=y[:size].reshape(2,2);q=y[size:].reshape(2,2)
            if t<1e-7/scale:dq=-gamma/(2*np.pi)*(h@self.j+self.j@h)
            else:
                u=expm(self.a*t);dq=gamma/(2j*beta*np.sinh(np.pi*t/beta))*(u@self.j-self.j@u.conj().T)
            return np.r_[(self.a@n+n@self.a_dag+q).ravel(),dq.ravel()]
        y0=np.r_[self.n0.ravel(),(gamma*self.j/2).ravel()].astype(complex)
        self.solution=solve_ivp(rhs,(0.,self.t_switch),y0,method='BDF',rtol=cfg.rtol,atol=cfg.atol,dense_output=True)
        if not self.solution.success:raise RuntimeError(self.solution.message)
        end=self.solution.y[:,-1];self.n_switch=hermitian(end[:size].reshape(2,2))
        self.switch_error=float(np.max(abs(end[size:].reshape(2,2)-self.q_inf)))
        if self.switch_error>2e-7*max(1.,gamma):raise ArithmeticError(f'memory mismatch {self.switch_error}')
    def propagator(self,t):
        a,b=self.poles
        if abs(a-b)<1e-8*max(1,self.gamma):return expm(self.a*t)
        return (np.exp(a*t)*(self.a-b*np.eye(2))-np.exp(b*t)*(self.a-a*np.eye(2)))/(a-b)
    def at(self,t):
        if t<=self.t_switch:return super().at(t)
        u=self.propagator(t-self.t_switch);diff=hermitian(u@(self.n_switch-self.n_mfg)@u.conj().T)
        return self.n_mfg+diff,hermitian(self.a@diff+diff@self.a_dag)
    def error(self,t):return bures(density(self.at(t)[0]),self.target)

def validate():
    out={}
    for g2 in [.03,1.,30.,1000.]:
        E=[1.2,1.0];ref=WideBand(E,g2,np.diag([1.,0]));new=GeneralH(np.diag(E),g2,np.diag([1.,0]))
        ts=np.r_[np.geomspace(1e-3,new.t_switch,15),np.geomspace(new.t_switch*1.01,20*new.slow_time,15)]
        out[f'max|N_engine-N_general| g2={g2:g}']=float(max(np.max(abs(ref.at(t)[0]-new.at(t)[0])) for t in ts))
    g=GeneralH(np.array([[1.2,-.4],[-.4,1.0]]),10.,np.diag([1.,0]))
    out['nondiagonal_error_at_40_slow_times']=g.error(40*g.slow_time)
    return out

if __name__=='__main__':
    import json;print(json.dumps(validate(),indent=1,default=float))
