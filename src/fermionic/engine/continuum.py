"""Lorentzian CONTINUUM reservoir; no bath-mode discretization in dynamics.

J(w)=J0 W^2/(w^2+W^2), Gamma_g(w)=g^2 J(w), Lambda0^2=J0 W/2.
W is a characteristic width, NOT a compact-support cutoff. hbar=k_B=1.
Derived from ../finite_bandwidth_benchmark/continuum.py, with stable large-width
kernel and pole evaluation. See README.md for numerical and physical assumptions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eig, eigh, expm
from scipy.integrate import solve_ivp, quad_vec, quad
from scipy.special import digamma, expit, exp1, roots_laguerre


_GL_NODES,_GL_WEIGHTS=roots_laguerre(48)

def thermal_kernel(t,width,beta=1.,order=48):
    if t<=0:return np.inf
    x=2*np.pi*t/beta
    if x>1500:return 0.
    a=.5+beta*width/(2*np.pi)
    nodes,weights=(_GL_NODES,_GL_WEIGHTS) if order==48 else roots_laguerre(order)
    z=x+nodes/a
    # Subtract the singular 1/z term analytically, then integrate a smooth
    # bounded remainder with Gauss-Laguerre. No Matsubara cutoff is introduced.
    h=np.empty_like(z);small=z<.01
    zz=z[small]
    h[small]=.5+zz/12-zz**3/720+zz**5/30240
    h[~small]=-1/np.expm1(-z[~small])-1/z[~small]
    ax=a*x
    if ax<50:
        scaled=np.exp(ax)*exp1(ax)
    else:
        term=1/ax;scaled=term
        for k in range(1,24):
            term*=-k/ax;scaled+=term
            if abs(term)<abs(scaled)*2e-16:break
    return float(width/np.pi*np.exp(-x/2)*(scaled+np.dot(weights,h)/a))

def stable_modes(a,energies,lam,width):
    """Refine poles in shifted coordinates, avoiding cancellation in slow DQD decay."""
    poles,vec=eig(a)
    n=len(energies)
    if lam==0:return poles,vec
    mean=np.mean(energies); d=(energies[0]-energies[-1])/2
    for j,root in enumerate(poles):
        s=np.clongdouble(root+1j*mean)
        q=np.clongdouble(width-1j*mean);l=np.longdouble(lam)**2
        for _ in range(15):
            if n==1:
                f=s*(s+q)+l;df=2*s+q
            else:
                f=(s*s+d*d)*(s+q)+2*l*s
                df=2*s*(s+q)+s*s+d*d+2*l
            step=f/df
            s-=step
            if abs(step)<1e-19*max(abs(s),1e-20):break
        poles[j]=complex(s)-1j*mean
        if n==1:
            vector=np.array([1.,1j*complex(s)/lam],complex)
        elif d!=0:
            bright=1j*complex(s)/d
            c=-1j*np.sqrt(2)*lam*bright/complex(s+q)
            vector=np.array([(bright+1)/np.sqrt(2),(bright-1)/np.sqrt(2),c])
        else:continue
        vec[:,j]=vector/np.linalg.norm(vector)
    return poles,vec


def hermitian(x):
    return (x+x.conj().T)/2


def physical(x, tolerance=2e-7):
    eigenvalues=eigh(hermitian(x),eigvals_only=True)
    if eigenvalues.min() < -tolerance or eigenvalues.max() > 1+tolerance:
        raise ArithmeticError(f'Unphysical covariance: {eigenvalues}')
    return eigenvalues


def density(x):
    x=hermitian(x); physical(x)
    if len(x)==1:
        return np.diag([1-x[0,0].real,x[0,0].real])
    p11=np.linalg.det(x).real
    rho=np.zeros((4,4),complex)
    rho[0,0]=1-np.trace(x).real+p11
    rho[1:3,1:3]=x-p11*np.eye(2)
    rho[3,3]=p11
    return hermitian(rho)


def bures(rho,sigma):
    if np.max(abs(rho-sigma))<1e-14:
        return 0.
    p,u=eigh(hermitian(rho))
    if p.min() < -2e-7: raise ArithmeticError('Nonpositive state')
    root=(u*np.sqrt(np.maximum(p,0)))@u.conj().T
    q=eigh(hermitian(root@sigma@root),eigvals_only=True)
    fidelity_root=np.sqrt(np.maximum(q,0)).sum()
    if fidelity_root > 1+2e-7: raise ArithmeticError('Fidelity exceeds one')
    return float(np.arccos(np.clip(fidelity_root,0,1)))


@dataclass(frozen=True)
class Numerics:
    rtol: float = 2e-10
    atol: float = 2e-12
    memory_beta: float = 12.
    spectral_tolerance: float = 2e-9


class LorentzianContinuum:
    """Exact microscopic dots+c covariance with factorized ORIGINAL preparation.

    c=sum(t_k* b_k)/Lambda0. The c--remaining-bath correlations are kept in
    the time-dependent noise source Q(t); this is not a local Lindblad model.
    Qdot=-lambda*k_beta(t)[exp(A t)v e_c^dagger+h.c.], with k_beta(t) given
    by a convergent resummation of all fermionic Matsubara frequencies.
    """
    def __init__(self, energies, g, width=10., beta=1., j0=1., initial=None,
                 numerics=Numerics()):
        if g<0 or min(width,beta,j0)<=0: raise ValueError('Invalid physical parameters')
        self.energies=np.asarray(energies,float); self.n=len(energies); self.m=self.n+1
        if self.n not in (1,2): raise ValueError('RLM or DQD required')
        self.g,self.width,self.beta,self.j0=float(g),float(width),float(beta),float(j0)
        self.gamma=g*g*j0; self.lambda0_squared=j0*width/2
        self.lam=g*np.sqrt(self.lambda0_squared); self.numerics=numerics
        self.initial=(np.diag([.5] if self.n==1 else [1.,0.]).astype(complex)
                      if initial is None else np.asarray(initial,complex))
        physical(self.initial)
        self.k=np.diag(np.r_[self.energies,0.]).astype(complex)
        self.k[:self.n,-1]=self.lam; self.k[-1,:self.n]=self.lam
        self.k2=self.k@self.k
        self.ec=np.eye(self.m,dtype=complex)[:,-1]
        self.v=np.r_[np.ones(self.n),0.].astype(complex)
        self.pc=np.outer(self.ec,self.ec)
        self.a=-1j*self.k-width*self.pc
        self.poles,self.vectors=stable_modes(self.a,self.energies,self.lam,width)
        self.inverse_vectors=np.linalg.inv(self.vectors)
        self.condition_number=float(np.linalg.cond(self.vectors))
        # Structural test: a narrow resonance is NOT a bound state. A tolerance
        # proportional to bandwidth would misclassify the strong-coupling DQD.
        self.bound_mask=np.zeros(self.m,dtype=bool)
        if g==0:
            self.bound_mask=np.abs(self.poles.real)<1e-14
        elif self.n==2 and self.energies[0]==self.energies[1]:
            self.bound_mask[np.argmin(abs(self.poles+1j*self.energies[0]))]=True
        self.bound_weights=[]
        for index in np.flatnonzero(self.bound_mask):
            vec=self.vectors[:,index]; vec=vec/np.linalg.norm(vec)
            self.bound_weights.append((-self.poles[index].imag,np.outer(vec,vec.conj())))
        projected=np.eye(self.m,dtype=complex)-sum((z for _,z in self.bound_weights),
                                                  np.zeros((self.m,self.m),complex))
        pe,pv=eigh(hermitian(projected))
        self.coupled_basis=pv[:,pe>.5]
        self.relaxes=not bool(self.bound_weights)
        stable=-self.poles.real[~self.bound_mask]
        self.slow_time=1/min(stable) if len(stable) else np.inf
        self.span=float(np.sum(np.abs(eigh(self.k,eigvals_only=True))))
        if self.n==1:
            assert abs(self.span-np.sqrt(self.energies[0]**2+4*self.lam**2))<1e-10*max(1,self.span)

        # Analytic evaluation of the EQUILIBRIUM spectral integral, independent
        # of the time-domain solver. Imaginary-axis poles contribute f(E_b)Z_b.
        z=.5-self.beta*self.poles/(2*np.pi)
        if self.condition_number<1e7:
            psi=(self.vectors*digamma(z))@self.inverse_vectors
            self.equilibrium=hermitian(.5*np.eye(self.m)-(psi-psi.conj().T)/(2j*np.pi))
        else:
            # Defective retarded poles are not physical bound states. Direct
            # real-frequency integration avoids diagonalization at such points.
            self.equilibrium=self.spectral_integral()[0]
        physical(self.equilibrium)
        self.target=density(self.equilibrium[:self.n,:self.n])
        self.rho0=density(self.initial)
        self.q_inf=hermitian(-self.a@self.equilibrium-self.equilibrium@self.a.conj().T)
        self.x0=np.zeros((self.m,self.m),complex)
        self.x0[:self.n,:self.n]=self.initial; self.x0[-1,-1]=.5
        self.t_switch=numerics.memory_beta*beta
        nu=np.pi/beta
        self.memory_tail_bound=float(4*self.lam*np.sqrt(self.n)*width
              /(beta*(width+nu)*nu**2)*np.exp(-nu*self.t_switch)
              /(1-np.exp(-2*nu*self.t_switch)))
        size=self.m*self.m
        def rhs(s,y):
            x=y[:size].reshape(self.m,self.m)
            q=self.width*y[size:].reshape(self.m,self.m)
            if s==0: return np.zeros_like(y)
            t=s*s
            w=self.propagator(t)@self.v
            dq=-self.lam*self.thermal_kernel(t)*(np.outer(w,self.ec)+np.outer(self.ec,w.conj()))
            dx=self.a@x+x@self.a.conj().T+q
            return 2*s*np.r_[dx.ravel(),(dq/self.width).ravel()]
        y0=np.r_[self.x0.ravel(),self.pc.ravel()]
        self.solution=solve_ivp(rhs,(0.,np.sqrt(self.t_switch)),y0,method='BDF',
                                rtol=numerics.rtol,atol=numerics.atol,dense_output=True)
        if not self.solution.success: raise RuntimeError(self.solution.message)
        last=self.solution.y[:,-1]
        self.x_switch=hermitian(last[:size].reshape(self.m,self.m))
        self.source_residual=float(np.max(abs(width*last[size:].reshape(self.m,self.m)-self.q_inf)))
        if self.source_residual>2e-7*max(1.,width,self.lam):
            raise ArithmeticError(f'Thermal-source equilibrium mismatch {self.source_residual}')
        self.initial_variance=self.variance(0.)
        expected=float(np.trace(np.diag(self.energies**2)@self.initial
                     -np.diag(self.energies)@self.initial@np.diag(self.energies)@self.initial).real
                     +self.n*self.lam**2/2)
        if abs(expected-self.initial_variance)>1e-10*max(1.,expected):
            raise ArithmeticError('Initial variance normalization failed')

    def spectral_density(self,w):
        return self.j0*self.width**2/(np.asarray(w)**2+self.width**2)

    def self_energy(self,w):
        return self.lam**2/(w+1j*self.width)*np.ones((self.n,self.n))

    def propagator(self,t):
        if self.condition_number<1e7:
            return (self.vectors*np.exp(self.poles*t))@self.inverse_vectors
        return expm(self.a*t)

    def thermal_kernel(self,t):
        """All Matsubara terms: stable integral representation at arbitrarily large W."""
        return thermal_kernel(t,self.width,self.beta)

    def at(self,t):
        if t<0: raise ValueError('t must be nonnegative')
        if t==0:
            x=self.x0.copy(); q=self.width*self.pc
            return x,hermitian(self.a@x+x@self.a.conj().T+q)
        if t<=self.t_switch:
            y=self.solution.sol(np.sqrt(t)); size=self.m*self.m
            x=hermitian(y[:size].reshape(self.m,self.m))
            q=self.width*hermitian(y[size:].reshape(self.m,self.m))
            return x,hermitian(self.a@x+x@self.a.conj().T+q)
        u=self.propagator(t-self.t_switch)
        difference=hermitian(u@(self.x_switch-self.equilibrium)@u.conj().T)
        return self.equilibrium+difference,hermitian(self.a@difference+difference@self.a.conj().T)

    def error(self,t):
        return bures(density(self.at(t)[0][:self.n,:self.n]),self.target)

    def variance(self,t):
        x=self.at(t)[0]
        variance=float(np.trace(self.k2@x-self.k@x@self.k@x).real)
        if variance < -1e-8*max(1.,self.span**2): raise ArithmeticError('Negative variance')
        return max(variance,0.)

    def integrated_speed(self,end):
        if end==0: return 0.,0.
        # Dense resolution during the thermal transient; after that only the
        # finitely many continuum resonance poles occur, without recurrences.
        points=[0.,min(end,1/max(1.,self.width,self.lam)),min(end,1/max(1.,self.gamma)),min(end,self.beta),
                min(end,self.t_switch),end]
        for t in np.geomspace(max(1e-5,self.t_switch),max(self.t_switch,end),24):
            if t<end: points.append(float(t))
        total=error=0.
        points=sorted(set(points))
        for lo,hi in zip(points[:-1],points[1:]):
            value,err=quad(lambda t:np.sqrt(self.variance(t)),lo,hi,
                           epsabs=2e-8,epsrel=2e-8,limit=150)
            total+=value;error+=err
        return float(total),float(error)

    def spectral_integral(self,thermal=True,tolerance=None):
        """Independent real-frequency continuum integral + explicit bound weights.

        Pole-aware interval breaks resolve the very narrow DQD dark resonance.
        The integration covers the entire real line, including both tails.
        """
        tol=self.numerics.spectral_tolerance if tolerance is None else tolerance
        b=self.coupled_basis
        ar=b.conj().T@self.a@b
        er=b.conj().T@self.ec
        def integrand(w):
            r=b@np.linalg.solve(-ar-1j*w*np.eye(len(ar)),er)
            value=self.width/np.pi*np.outer(r,r.conj())
            return value*(expit(-self.beta*w) if thermal else 1.)
        # A single interval spanning many decades can silently miss the Fermi
        # transition and low-energy tails when W is huge. Resolve all scales.
        small=min(1/self.beta,self.gamma,min(-self.poles.real[~self.bound_mask]))
        scales=np.geomspace(max(small,1e-12),max(self.width,self.lam,1/self.beta)*40,100)
        points=[0.]+list(scales)+list(-scales)
        for pole,bound in zip(self.poles,self.bound_mask):
            if bound:continue
            center=-pole.imag;width=-pole.real
            points += [center+width*k for k in [-30,-3,-1,0,1,3,30]]
        points=[-np.inf]+sorted(set(points))+[np.inf]
        value=np.zeros((self.m,self.m),complex);error=0.
        for lo,hi in zip(points[:-1],points[1:]):
            v,e=quad_vec(integrand,lo,hi,epsabs=tol/len(points),epsrel=tol,limit=400)
            value+=v;error+=e
        for energy,weight in self.bound_weights:
            value+=weight*(expit(-self.beta*energy) if thermal else 1.)
        return hermitian(value),float(error)


def model_pair(model,g,width=10.,beta=1.,j0=1.,delta=.2,numerics=Numerics()):
    energies=([np.array([0.]),np.array([1./beta])] if model=='rlm' else
              [np.array([mean+delta/2,mean-delta/2]) for mean in [1.1,2.35]])
    return [LorentzianContinuum(e,g,width,beta,j0,numerics=numerics) for e in energies]
