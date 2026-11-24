"""Fermionic results: coupling sweeps of the finite-error bound at a FIXED Hamiltonian ball.

Task (fixed for the whole sweep; only g^2 varies), hbar=beta=1, gamma=g^2, wide-band, Fock-space seminorm:
  RLM        H(x) = x n_d,                  |x| <= delta, preparation p(0)=1/2
  DQD common h(x) = hbar + x N_S/2,         |x| <= delta, preparation (n1,n2)=(1,0)
  DQD dark   h(x) = hbar + x n_-,           |x| <= delta, preparation (1,0)
  hbar = diag(1.2,1.0) (eps_bar=1.1, Delta=0.2), delta = 8*eps_max = 2*pi, ||H(x)-H(x')|| = |x-x'|.
For each g^2 and accuracy eps:
  bound  chi(delta,eps) = max over pairs (x1,x2) inside [-delta,delta] of 2[D(w1,w2)-2eps]_+/|x1-x2|
         (dense pair grid + constrained Nelder-Mead refinement)
  time   tau_eps = min{t : max_H D(rho(t,H), w(H)) <= eps}, H over K=21 equally spaced members AND the optimal
         pair, so tau_eps >= tau_eps(pair) >= tau_Pl*chi holds by construction.
         Each member's error is evaluated on its own grid that resolves its oscillation frequencies; the
         intervals where it is within eps are refined with brentq and intersected over members.
         Also stored: tau_sustained (all members stay within eps afterwards), the last member to enter.
Outputs: data/sweep.csv, data/checks.json, data/config.json.
Run: python src/fermionic/families/compute.py   [~10 min, 8 processes]
"""
import sys,json,csv,hashlib,time,os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import brentq,minimize
from scipy.special import digamma
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
DATA=HERE.parents[2]/'data'/'fermionic'/'families'
from dqd_solver import GeneralH,WideBand,mfg,density,bures,validate as validate_solver   # noqa: E402

# ------------------------------------------------------------------ configuration
DELTA=8*np.pi/4                      # delta = 8 eps_max, eps_max = pi/4
EPS=[.01,.05,.1,.15,.2]              # eps / eps_max
K=21                                 # sampled members of the fixed ball
HBAR=np.diag([1.2,1.0]);I2=np.eye(2);X=np.array([[0,1.],[1,0]])
DIRECTION={'dqd_common':I2/2,'dqd_dark':(I2-X)/2}
P0_RLM=.5;N0_DQD=np.diag([1.,0.])
G2_BASE=np.unique(np.r_[np.geomspace(.01,1000,31),[.1,1.,10.,100.]])
FAMILIES=['rlm','dqd_common','dqd_dark']
NGRID_BOUND={'rlm':801,'dqd_common':241,'dqd_dark':241}

# ------------------------------------------------------------------ RLM equilibrium (exact)
def p_th(e,g2):return .5-np.imag(digamma(.5+(g2/2+1j*np.asarray(e,float))/(2*np.pi)))/np.pi
def theta(e,g2):return np.arcsin(np.sqrt(np.clip(p_th(e,g2),0,1)))
def rlm_thresholds():
    """g^2 at which the bound and tau vanish for the symmetric ball: D(-delta,delta)=2eps (exact)."""
    return {e:brentq(lambda g:2*theta(-DELTA,g)-np.pi/2-2*e*np.pi/4,1e-3,1e7) for e in EPS}

# ------------------------------------------------------------------ family members
def member(fam,x,g2):
    if fam=='rlm':return WideBand([float(x)],g2,np.array([[P0_RLM]]))
    return GeneralH(HBAR+x*DIRECTION[fam],g2,N0_DQD)

def targets(fam,xs,g2):
    if fam=='rlm':return theta(xs,g2)
    return [density(mfg(HBAR+x*DIRECTION[fam],g2)) for x in xs]

def chi_pair(fam,x1,x2,g2,eps):
    s=abs(x2-x1)
    if s<1e-12:return 0.
    if fam=='rlm':D=abs(theta(x1,g2)-theta(x2,g2))
    else:D=bures(*targets(fam,[x1,x2],g2))
    return 2*max(D-2*eps,0.)/s

def distance_matrix(fam,xs,g2):
    T=targets(fam,xs,g2)
    if fam=='rlm':return abs(T[:,None]-T[None,:])
    S=[]                                  # sqrt of each target, then fidelity with all others (batched)
    for r in T:
        w,u=np.linalg.eigh(r);S.append((u*np.sqrt(np.clip(w,0,None)))@u.conj().T)
    R=np.array(T);M=np.zeros((len(T),len(T)))
    for i,s in enumerate(S):
        w=np.linalg.eigvalsh(s@R@s);M[i]=np.arccos(np.clip(np.sqrt(np.clip(w,0,None)).sum(-1),0,1))
    return (M+M.T)/2

def bound(fam,g2,Dm,xs,eps):
    """max of chi over pairs inside [-delta,delta]: dense grid, then constrained Nelder-Mead."""
    dx=abs(xs[:,None]-xs[None,:]);C=np.where(dx>0,2*np.maximum(Dm-2*eps,0)/np.where(dx>0,dx,1),0)
    i,j=np.unravel_index(np.argmax(C),C.shape)
    if C[i,j]==0:return 0.,np.nan,np.nan
    f=lambda p:1. if max(abs(p[0]),abs(p[1]))>DELTA else -chi_pair(fam,p[0],p[1],g2,eps)
    r=minimize(f,[xs[min(i,j)],xs[max(i,j)]],method='Nelder-Mead',options=dict(xatol=1e-9,fatol=1e-13,maxiter=4000))
    best=(-r.fun,*sorted(r.x)) if -r.fun>=C[i,j] else (C[i,j],*sorted((xs[i],xs[j])))
    return best

# ------------------------------------------------------------------ vectorized errors on dense grids
def _covariances(b,ts):
    """N(t) for an array of times (engine conventions; matches b.at(t)[0])."""
    n=b.n;out=np.empty((len(ts),n,n),complex)
    early=ts<=b.t_switch;late=~early
    if np.any(early):
        y=b.solution.sol(ts[early])[:n*n].T.reshape(-1,n,n);out[early]=(y+np.conj(np.swapaxes(y,1,2)))/2
    if np.any(late):
        s=ts[late]-b.t_switch;M=b.n_switch-b.n_mfg
        if n==1:
            out[late]=b.n_mfg+np.exp(2*b.poles[0].real*s)[:,None,None]*M
        else:
            a,c=b.poles
            if abs(a-c)<1e-8*max(1,b.gamma):U=np.array([b.propagator(x) for x in s])
            else:
                U=(np.exp(a*s)[:,None,None]*(b.a-c*np.eye(2))-np.exp(c*s)[:,None,None]*(b.a-a*np.eye(2)))/(a-c)
            d=U@M@np.conj(np.swapaxes(U,1,2));out[late]=b.n_mfg+(d+np.conj(np.swapaxes(d,1,2)))/2
    return out

def errors(b,ts,chunk=40000):
    if getattr(b,'stationary',False):return np.zeros(len(ts))
    if len(ts)>chunk:return np.concatenate([errors(b,ts[i:i+chunk],chunk) for i in range(0,len(ts),chunk)])
    N=_covariances(b,ts)
    if b.n==1:
        p=np.clip(N[:,0,0].real,0,1);q=b.n_mfg[0,0].real
        return np.arccos(np.clip(np.sqrt(p*q)+np.sqrt((1-p)*(1-q)),0,1))
    p11=(N[:,0,0]*N[:,1,1]-N[:,0,1]*N[:,1,0]).real
    rho=np.zeros((len(ts),4,4),complex)
    rho[:,0,0]=1-(N[:,0,0]+N[:,1,1]).real+p11;rho[:,1:3,1:3]=N-p11[:,None,None]*np.eye(2);rho[:,3,3]=p11
    w,u=np.linalg.eigh(b.target);S=(u*np.sqrt(np.clip(w,0,None)))@u.conj().T
    ev=np.linalg.eigvalsh(S@rho@S)
    return np.arccos(np.clip(np.sqrt(np.clip(ev,0,None)).sum(-1),0,1))

def time_grid(b):
    scale=max(1.,b.gamma,float(np.max(np.abs(np.linalg.eigvalsh(np.atleast_2d(b.h).astype(complex))))))
    t_end=80*b.slow_time
    parts=[[0.],np.geomspace(1e-6/scale,t_end,8000),np.linspace(0,b.t_switch,4000)]
    if b.n==2:
        a,c=b.poles;freq=abs((a-c).imag);decay=-(a.real+c.real)
        if freq>1e-12:
            t_osc=min(t_end,40/decay);parts.append(np.linspace(0,t_osc,min(400000,int(t_osc*freq*16/(2*np.pi))+2)))
    return np.unique(np.concatenate(parts)),t_end

def inside_intervals(b,eps,cache):
    """Sorted list of [start,end] (end may be inf) where D(rho(t),w) <= eps; edges refined with brentq."""
    if 'grid' not in cache:
        ts,t_end=time_grid(b);cache['grid']=(ts,errors(b,ts))
    ts,e=cache['grid'];ok=e<=eps
    if not ok[-1]:raise RuntimeError('member not within eps at the end of its grid; extend t_end')
    f=lambda t:b.error(t)-eps
    def edge(lo,hi):
        flo,fhi=f(lo),f(hi)
        if flo*fhi>0:return hi if fhi<=0 else lo     # vectorized/scalar roundoff at the threshold
        return brentq(f,lo,hi,xtol=1e-12,rtol=1e-12)
    out=[];i=0;n=len(ts)
    while i<n:
        if not ok[i]:i+=1;continue
        j=i
        while j+1<n and ok[j+1]:j+=1
        start=0. if i==0 else edge(ts[i-1],ts[i])
        end=np.inf if j==n-1 else edge(ts[j],ts[j+1])
        out.append([start,end]);i=j+1
    return out

def intersect(A,B):
    out=[];i=j=0
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]);hi=min(A[i][1],B[j][1])
        if lo<hi or (lo==hi==np.inf):out.append([lo,hi])
        if A[i][1]<B[j][1]:i+=1
        else:j+=1
    return out

# ------------------------------------------------------------------ one (family, g^2) task
def task(args):
    fam,g2=args;t0=time.monotonic()
    xs=np.linspace(-DELTA,DELTA,NGRID_BOUND[fam]);Dm=distance_matrix(fam,xs,g2)
    grid=np.linspace(-DELTA,DELTA,K);members={};caches={}
    def get(x):
        key=round(float(x),12)
        if key not in members:members[key]=member(fam,key,g2);caches[key]={}
        return key
    for x in grid:get(x)
    rows=[]
    for e in EPS:
        eps=e*np.pi/4;chi,x1,x2=bound(fam,g2,Dm,xs,eps)
        keys=[round(float(x),12) for x in grid]
        if chi>0:keys+= [get(x1),get(x2)]
        keys=sorted(set(keys))
        ivs={k:inside_intervals(members[k],eps,caches[k]) for k in keys}
        common=[[0.,np.inf]]
        for k in keys:common=intersect(common,ivs[k])
        tau=common[0][0];tau_sus=max(ivs[k][-1][0] for k in keys)
        entry={k:next(iv[0] for iv in ivs[k] if iv[0]<=tau<=iv[1]) for k in keys}
        last=max(keys,key=lambda k:entry[k])
        tau_pair=intersect(ivs[get(x1)],ivs[get(x2)])[0][0] if chi>0 else 0.
        slow=min(-2*members[k].poles.real.max() for k in keys)   # slowest population decay rate in the family
        rows.append(dict(family=fam,gamma=g2,eps_fraction=e,epsilon=eps,delta=DELTA,chi=chi,x1=x1,x2=x2,
                         tau=tau,tau_sustained=tau_sus,tau_pair=tau_pair,last_member=last,
                         n_members=len(keys),slowest_rate=slow,
                         max_initial_distance=max(bures(members[k].rho0,members[k].target) if fam!='rlm'
                             else float(np.arccos(np.sqrt(P0_RLM*members[k].n_mfg[0,0].real)+np.sqrt((1-P0_RLM)*(1-members[k].n_mfg[0,0].real))))
                             for k in keys),
                         ratio=chi/tau if tau>0 else np.nan))
    # consistency of the vectorized errors with the engine's scalar error (random times)
    k=keys[0];ts=caches[k]['grid'][0];rng=np.random.default_rng(1);sel=rng.choice(len(ts),40,replace=False)
    verr=float(max(abs(errors(members[k],ts[sel])-np.array([members[k].error(t) for t in ts[sel]]))))
    for r in rows:r['vectorized_vs_scalar_error']=verr
    return rows,fam,g2,time.monotonic()-t0

def main():
    th=rlm_thresholds()
    extra=[g*(1+s) for g in th.values() for s in (-.3,-.1,-.03,-.01,-.003,-.001,.001,.01)]
    g2_rlm=np.unique(np.r_[G2_BASE,[g for g in extra if .01<=g<=1000]])
    tasks=[('rlm',g) for g in g2_rlm]+[(f,g) for f in ['dqd_common','dqd_dark'] for g in G2_BASE]
    tasks.sort(key=lambda t:(t[0]=='rlm',-t[1]))      # heavy DQD weak-coupling tasks first
    rows=[];t0=time.monotonic()
    with Pool(min(8,os.cpu_count() or 1)) as pool:
        for rr,fam,g2,dt in pool.imap_unordered(task,tasks):
            rows+=rr;r=rr[1]
            print(f'{fam:10s} g2={g2:9.4g}: chi={r["chi"]:.4g} tau={r["tau"]:.4g} (eps=0.05) [{dt:.0f}s, total {time.monotonic()-t0:.0f}s]',flush=True)
    rows.sort(key=lambda r:(FAMILIES.index(r['family']),r['gamma'],r['eps_fraction']))
    (DATA).mkdir(exist_ok=True)
    with (DATA/'sweep.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # ---------------- checks
    bad=[r for r in rows if r['chi']>r['tau']*(1+1e-6)+1e-10]
    checks=dict(
        bound_le_tau_violations=len(bad),
        bound_le_tau_pair_violations=sum(r['chi']>r['tau_pair']*(1+1e-6)+1e-10 for r in rows),
        tau_ge_tau_pair=all(r['tau']>=r['tau_pair']*(1-1e-9) for r in rows),
        rows_with_reentry_after_tau={f:sum(r['tau_sustained']>r['tau']*(1+1e-9)+1e-12 for r in rows if r['family']==f) for f in FAMILIES},
        max_reentry_gap={f:max([r['tau_sustained']-r['tau'] for r in rows if r['family']==f]+[0.]) for f in FAMILIES},
        max_vectorized_vs_scalar_error=max(r['vectorized_vs_scalar_error'] for r in rows),
        rlm_thresholds_g2c=th,
        rlm_tau_zero_iff_chi_zero=all((r['tau']==0)==(r['chi']==0) for r in rows if r['family']=='rlm'),
        nontriviality_condition=dict(delta=DELTA,eps_max_used=max(EPS)*np.pi/4,beta_delta_over_8=DELTA/8,beta_delta_over_4=DELTA/4),
        solver_validation=validate_solver())
    # convergence in the number of sampled members (K -> 2K-1) at a few couplings
    conv=[]
    for fam,g2 in [('rlm',.1),('rlm',10.),('dqd_common',.1),('dqd_common',100.),('dqd_dark',.1),('dqd_dark',100.)]:
        dense=np.linspace(-DELTA,DELTA,2*K-1);ms=[member(fam,x,g2) for x in dense]
        for e in [.01,.2]:
            eps=e*np.pi/4;common=[[0.,np.inf]]
            for m in ms:common=intersect(common,inside_intervals(m,eps,{}))
            ref=next(r for r in rows if r['family']==fam and r['gamma']==g2 and r['eps_fraction']==e)
            conv.append(dict(family=fam,gamma=g2,eps_fraction=e,tau_K=ref['tau'],tau_2K=common[0][0],
                             rel=abs(common[0][0]-ref['tau'])/max(ref['tau'],1e-12)))
    checks['member_sampling_convergence']=conv
    checks['member_sampling_max_rel_change']=max(c['rel'] for c in conv)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'compute.py',HERE/'dqd_solver.py']+
            list((HERE.parent/'engine').glob('*.py'))}
    (DATA/'checks.json').write_text(json.dumps(checks,indent=1,default=float)+'\n')
    (DATA/'config.json').write_text(json.dumps(dict(
        units='hbar=beta=1; gamma=g^2 (wide-band hybridization); energies in k_BT, times in tau_Pl',
        delta=DELTA,delta_definition='8*eps_max, eps_max=pi/4',eps_fractions=EPS,K=K,
        families={'rlm':'H(x)=x n_d, |x|<=delta, p(0)=1/2',
                  'dqd_common':'h(x)=diag(1.2,1.0)+x*N_S/2, |x|<=delta, (n1,n2)=(1,0)',
                  'dqd_dark':'h(x)=diag(1.2,1.0)+x*n_-, n_-=(1-X)/2, |x|<=delta, (n1,n2)=(1,0)'},
        tau_definition='min{t: max_H D(rho(t,H),w(H))<=eps} over K grid members + the optimal pair',
        bound_definition='max over pairs in [-delta,delta] of 2[D-2eps]_+/|x1-x2|',
        g2_base=list(G2_BASE),hashes=hashes),indent=1,default=float)+'\n')
    print(json.dumps({k:v for k,v in checks.items() if k not in ('member_sampling_convergence','solver_validation')},indent=1,default=float))

if __name__=='__main__':
    main()
