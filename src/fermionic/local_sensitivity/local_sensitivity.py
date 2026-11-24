"""Local (locally exact) Planckian bounds for the RLM and DQD in the wide-band limit.

Implements the optimization of Sec. II C 1 (Prop. 1, Eqs. (13)-(19)) restricted to the
physically accessible controls of each model, and the response bounds of chosen readouts.
hbar=beta=1; gamma = g^2 (wide-band hybridization). Fock-space spectral seminorm throughout.
WB MFG covariance: N = 1/2 - [psi(1/2 - beta A/2pi) - h.c.]/(2 pi i), A = -i h - (g^2/2) J.

Controls:
 RLM : parity superselection => admissible kappa in span{1, n_d}; kappa = n_d is optimal.
 DQD : quadratic number-conserving controls kappa = d^dag k d (gate voltages + interdot
       tunnelling). ||kappa|| = sum_i |eig(k)|, so the unit ball is the trace-norm ball of k;
       its extreme points are +-|v><v|, i.e. single-mode occupations n_v. Since sqrt(F) is
       convex in kappa (Prop. 1), the constrained optimum is max_v F(n_v).
Run: python src/fermionic/local_sensitivity/local_sensitivity.py
"""
import sys,os,json,csv,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[2]/'data'/'fermionic'/'local_sensitivity'
ENGINE=HERE.parent/'engine'
sys.path.insert(0,str(ENGINE))
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import expit
from wideband import original
from continuum import density,hermitian

def equilibrium(h,gamma):
    n=len(h); a=-1j*h-gamma*np.ones((n,n))/2
    psi=original.matrix_digamma(.5*np.eye(n)-a/(2*np.pi))
    return hermitian(.5*np.eye(n)-(psi-psi.conj().T)/(2j*np.pi))

def deriv(fun,step):
    return (fun(-2*step)-8*fun(-step)+8*fun(step)-fun(2*step))/(12*step)

def sld_qfi(rho,drho):
    p,u=eigh(rho); x=u.conj().T@drho@u; den=p[:,None]+p[None,:]
    return float(np.sum(np.divide(2*abs(x)**2,den,out=np.zeros_like(den),where=den>1e-13)))

def span(k):  # Fock-space spectral range of d^dag k d
    return float(np.abs(eigh(hermitian(k),eigvals_only=True)).sum())

def local(h,gamma,k,readouts=(),step=2e-4):
    """QFI of the MFG family along kappa=d^dag k d, and response bounds for quadratic readouts."""
    n=equilibrium(h,gamma)
    dn=deriv(lambda s:equilibrium(h+s*k,gamma),step)
    drho=deriv(lambda s:density(equilibrium(h+s*k,gamma)),step)
    F=sld_qfi(density(n),drho)
    out=dict(F=F,chi_tilde=np.sqrt(F)/span(k))
    for name,a in readouts:
        xi=-np.trace(a@dn).real                       # xi_{A kappa} = -d<A>/d delta
        var=float(np.trace(n@a@(np.eye(len(n))-n)@a).real)
        out[f'resp_{name}']=xi*xi/var                  # xi^2/Var(A) <= F
    return out

def mode(theta,phi):
    v=np.array([np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)]);return np.outer(v,v.conj())

I2=np.eye(2); X=np.array([[0,1],[1,0]]); Z=np.diag([1.,-1.])
P_dark=(I2-X)/2; P_bright=(I2+X)/2; P1=np.diag([1.,0]); P2=np.diag([0,1.])
DIRECTIONS={'common_shift':I2,'dot1':P1,'detuning':Z/2,'tunnelling':X/2,'dark':P_dark,'bright':P_bright}
READOUTS=[('N',I2),('n1',P1),('dark',P_dark),('bright',P_bright)]

def dqd_optimum(h,gamma):
    f=lambda x:-local(h,gamma,mode(*x))['F']
    starts=[(t,p) for t in np.linspace(0,np.pi,7) for p in (0.,np.pi/2,np.pi)]
    best=min((minimize(f,s,method='Nelder-Mead',options=dict(xatol=1e-7,fatol=1e-12)) for s in starts),key=lambda r:r.fun)
    th,ph=best.x; return -best.fun,float(np.mod(th,2*np.pi)),float(np.mod(ph,2*np.pi))

gammas=np.unique(np.r_[np.logspace(-2,3,41),[.2]])
rows=[];checks={}
# ---- RLM: kappa=n_d, readout n_d (Eq. (19)) ----
for E in [0.,1.,3.]:
    for g in gammas:
        h=np.array([[E]]); q=local(h,g,np.eye(1),[('n',np.eye(1))])
        rows.append(dict(model='rlm',E=E,mean=np.nan,gamma=g,direction='level',**q))
# ---- DQD: fixed directions, readouts, and constrained optimum ----
for mean in [1.1,2.35]:
    h=np.diag([mean+.1,mean-.1])
    for g in gammas:
        for name,k in DIRECTIONS.items():
            q=local(h,g,k,READOUTS+[('self',k)])
            rows.append(dict(model='dqd',E=np.nan,mean=mean,gamma=g,direction=name,**q))
        F,th,ph=dqd_optimum(h,g); kopt=mode(th,ph)
        q=local(h,g,kopt,READOUTS+[('self',kopt)])
        v_dark=abs(np.array([1,-1])/np.sqrt(2)@np.array([np.cos(th/2),np.exp(1j*ph)*np.sin(th/2)]))**2
        rows.append(dict(model='dqd',E=np.nan,mean=mean,gamma=g,direction='optimum_single_mode',
                         theta=th,phi=ph,dark_weight=v_dark,**q))
# ---- checks ----
def get(model,direction,g,**kw):
    return next(r for r in rows if r['model']==model and r['direction']==direction and abs(r['gamma']-g)<1e-12
                and all((np.isnan(r[k]) and np.isnan(v)) or r[k]==v for k,v in kw.items()))
fd=[]
for g in [.01,1.,1000.]:
    h=np.diag([1.2,1.0]); a=local(h,g,P_dark,step=2e-4)['F'];b=local(h,g,P_dark,step=1e-4)['F'];fd.append(abs(a-b)/b)
checks['finite_difference_relative_change']=max(fd)
checks['rlm_occupation_readout_saturates']=max(abs(r['resp_n']-r['F'])/r['F'] for r in rows if r['model']=='rlm' and r['F']>1e-12)
checks['convexity_common_shift_le_optimum']=all(get('dqd','common_shift',r['gamma'],mean=r['mean'])['chi_tilde']<=r['chi_tilde']*(1+1e-6)
      for r in rows if r['direction']=='optimum_single_mode')
checks['optimum_ge_all_directions']=all(max(get('dqd',d,r['gamma'],mean=r['mean'])['chi_tilde'] for d in DIRECTIONS)<=r['chi_tilde']*(1+1e-6)
      for r in rows if r['direction']=='optimum_single_mode')
checks['response_le_qfi']=all(r[k]<=r['F']*(1+1e-6)+1e-14 for r in rows for k in r if k.startswith('resp_') and np.isfinite(r[k]))
for mean in [1.1,2.35]:
    q=expit(mean)*(1-expit(mean))   # f(1-f) with f=1/(1+e^{beta*mean})
    gmax=gammas[-1]
    checks[f'strong_coupling_mean{mean}']=dict(q=q,
        chi_dark=get('dqd','dark',gmax,mean=mean)['chi_tilde'],chi_dark_limit=np.sqrt(q),
        chi_common=get('dqd','common_shift',gmax,mean=mean)['chi_tilde'],chi_common_limit=np.sqrt(q)/2,
        chi_opt=get('dqd','optimum_single_mode',gmax,mean=mean)['chi_tilde'],
        dark_weight_opt=get('dqd','optimum_single_mode',gmax,mean=mean)['dark_weight'])
out=DATA;out.mkdir(exist_ok=True)
keys=sorted({k for r in rows for k in r},key=lambda k:(k not in ['model','E','mean','gamma','direction'],k))
with (out/'local_sensitivity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys,restval='');w.writeheader();w.writerows(rows)
checks.update(units='hbar=beta=1, gamma=g^2, wide-band MFG',
   script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   engine_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ENGINE/'wideband_reference.py',ENGINE/'continuum.py']})
(out/'local_sensitivity_checks.json').write_text(json.dumps(checks,indent=2,default=float)+'\n')
print(json.dumps(checks,indent=1,default=float))
