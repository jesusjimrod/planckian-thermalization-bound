"""Speed-limit benchmark for the fixed Hamiltonian families of ../families, at finite bandwidth.

Model: dots coupled to a Lorentzian continuum of width Lambda (validated engine
../engine/continuum.py, unmodified). hbar=beta=1, g^2 = hybridization at the Fermi level.
Families (as in ../families; only g^2 varies along a sweep):
  rlm : H(x) = x n_d,                               |x| <= delta, p(0) = 1/2
  dqd : h(x) = diag(1.2,1.0) + x N_S/2 (common shift), |x| <= delta, (n1,n2) = (1,0)
  delta = 8 eps_max = 2 pi, eps/eps_max in {0.01,0.05,0.1,0.15,0.2}.
Everything below is evaluated in the SAME finite-Lambda model:
  tau_eps  = min{t : max_x D(rho(t,x), w_x) <= eps}  over K=21 members + the maximizing pair (family time)
  chi      = max over pairs in [-delta,delta] of 2[D(w1,w2)-2eps]_+/|x1-x2|   (finite-Lambda MFG targets)
  SS var   = max_x hbar [D(rho0,w_x)-eps]_+ / <dE_S(x)>_[0,tau_eps],  dE_S = std of H_S(x)+V in the global state
  SS static= max_x 2 hbar [D(rho0,w_x)-eps]_+ / ||H_S(x)+V||
Both SS versions are lower bounds on the same tau_eps: at tau_eps every member is eps-close, so
D(rho0,rho_x(tau_eps)) >= D(rho0,w_x)-eps, and Shiraishi-Saito bounds the left side by int dE_S/hbar.
Outputs data/benchmark.csv, data/checks.json, data/config.json.  Run from the project root:
  python src/fermionic/speedlimits/compute.py        [~10 min, 8 processes]
"""
import sys,os,json,csv,hashlib,time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import brentq,minimize
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import digamma
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[2]/'data'/'fermionic'/'speedlimits'
# Cross-check against the earlier fixed-pair run of the project archive. That raw run is not
# distributed with this repository, so the check is skipped when the file is absent.
REFERENCE_RUN=HERE/'reference'/'coupling_benchmark.csv'
ENGINE=HERE.parent/'engine';sys.path.insert(0,str(ENGINE))
from continuum import LorentzianContinuum,stable_modes,density,bures,hermitian   # noqa: E402

# ------------------------------------------------------------------ configuration
DELTA=2*np.pi
EPS=[.01,.05,.1,.15,.2]
K=21
WIDTHS=[1e2,1e3,1e4,1e5,1e6,1e8]
G2_BASE=np.unique(np.r_[np.geomspace(.01,1000,31),[.1,1.,10.,100.]])   # same grid as ../families
INITIAL={'rlm':np.diag([.5]).astype(complex),'dqd':np.diag([1.,0.]).astype(complex)}
NGRID={'rlm':801,'dqd':241}
MAX_ENERGY={'rlm':DELTA,'dqd':1.2+DELTA/2}          # largest |level energy| in each family

def energies(fam,x):
    return [float(x)] if fam=='rlm' else [1.2+x/2,1.0+x/2]

def member(fam,x,g2,W):
    return LorentzianContinuum(energies(fam,x),np.sqrt(g2),W,initial=INITIAL[fam])

# ------------------------------------------------------------------ finite-Lambda MFG targets (no dynamics)
def finite_equilibrium(e,g2,W):
    """Same analytic expression as LorentzianContinuum.equilibrium (dots + collective mode c), without the ODE."""
    e=np.asarray(e,float);n=len(e);m=n+1;lam=np.sqrt(g2)*np.sqrt(W/2)
    k=np.diag(np.r_[e,0.]).astype(complex);k[:n,-1]=lam;k[-1,:n]=lam
    ec=np.eye(m,dtype=complex)[:,-1];a=-1j*k-W*np.outer(ec,ec)
    poles,vec=stable_modes(a,e,lam,W)
    if np.linalg.cond(vec)>=1e7:   # defective point: use the engine's spectral integral
        return LorentzianContinuum(e,np.sqrt(g2),W,initial=np.diag([.5]*n)).equilibrium
    psi=(vec*digamma(.5-poles/(2*np.pi)))@np.linalg.inv(vec)
    return hermitian(.5*np.eye(m)-(psi-psi.conj().T)/(2j*np.pi))

def target(fam,x,g2,W):
    n=1 if fam=='rlm' else 2
    return density(finite_equilibrium(energies(fam,x),g2,W)[:n,:n])

def distance_matrix(fam,xs,g2,W):
    T=[target(fam,x,g2,W) for x in xs]
    if fam=='rlm':
        th=np.arcsin(np.sqrt(np.clip([t[1,1].real for t in T],0,1)));return abs(th[:,None]-th[None,:])
    R=np.array(T);M=np.zeros((len(T),len(T)))
    for i,r in enumerate(T):
        w,u=np.linalg.eigh(r);s=(u*np.sqrt(np.clip(w,0,None)))@u.conj().T
        M[i]=np.arccos(np.clip(np.sqrt(np.clip(np.linalg.eigvalsh(s@R@s),0,None)).sum(-1),0,1))
    return (M+M.T)/2

def chi_pair(fam,x1,x2,g2,W,eps):
    s=abs(x2-x1)
    return 0. if s<1e-12 else 2*max(bures(target(fam,x1,g2,W),target(fam,x2,g2,W))-2*eps,0.)/s

def bound(fam,g2,W,Dm,xs,eps):
    dx=abs(xs[:,None]-xs[None,:]);C=np.where(dx>0,2*np.maximum(Dm-2*eps,0)/np.where(dx>0,dx,1),0)
    i,j=np.unravel_index(np.argmax(C),C.shape)
    if C[i,j]==0:return 0.,np.nan,np.nan
    f=lambda p:1. if max(abs(p[0]),abs(p[1]))>DELTA else -chi_pair(fam,p[0],p[1],g2,W,eps)
    r=minimize(f,[xs[min(i,j)],xs[max(i,j)]],method='Nelder-Mead',options=dict(xatol=1e-9,fatol=1e-13,maxiter=4000))
    return (-r.fun,*sorted(r.x)) if -r.fun>=C[i,j] else (C[i,j],*sorted((xs[i],xs[j])))

# ------------------------------------------------------------------ vectorized reduced-state errors
def covariances(b,ts):
    n,m=b.n,b.m;out=np.empty((len(ts),n,n),complex);early=ts<=b.t_switch;late=~early
    if np.any(early):
        y=b.solution.sol(np.sqrt(ts[early]))[:m*m].T.reshape(-1,m,m)
        out[early]=((y+np.conj(np.swapaxes(y,1,2)))/2)[:,:n,:n]
    if np.any(late):
        s=ts[late]-b.t_switch;M=b.x_switch-b.equilibrium
        if b.condition_number<1e7:
            U=(b.vectors[None]*np.exp(np.outer(s,b.poles))[:,None,:])@b.inverse_vectors
        else:U=np.array([expm(b.a*x) for x in s])
        d=U@M@np.conj(np.swapaxes(U,1,2));out[late]=(b.equilibrium+(d+np.conj(np.swapaxes(d,1,2)))/2)[:,:n,:n]
    return out

def errors(b,ts,chunk=40000):
    if len(ts)>chunk:return np.concatenate([errors(b,ts[i:i+chunk]) for i in range(0,len(ts),chunk)])
    N=covariances(b,ts)
    if b.n==1:
        p=np.clip(N[:,0,0].real,0,1);q=b.equilibrium[0,0].real
        return np.arccos(np.clip(np.sqrt(p*q)+np.sqrt((1-p)*(1-q)),0,1))
    p11=(N[:,0,0]*N[:,1,1]-N[:,0,1]*N[:,1,0]).real;rho=np.zeros((len(ts),4,4),complex)
    rho[:,0,0]=1-(N[:,0,0]+N[:,1,1]).real+p11;rho[:,1:3,1:3]=N-p11[:,None,None]*np.eye(2);rho[:,3,3]=p11
    w,u=np.linalg.eigh(b.target);S=(u*np.sqrt(np.clip(w,0,None)))@u.conj().T
    return np.arccos(np.clip(np.sqrt(np.clip(np.linalg.eigvalsh(S@rho@S),0,None)).sum(-1),0,1))

def time_grid(b,stretch=1.):
    scale=max(1.,b.width,b.gamma,float(np.max(np.abs(b.energies))))
    t_end=80*b.slow_time*stretch
    parts=[[0.],np.geomspace(1e-6/scale,t_end,8000),np.linspace(0,b.t_switch,4000)]
    for i in range(b.m):
        for j in range(i):
            freq=abs((b.poles[i]-b.poles[j]).imag);decay=-(b.poles[i].real+b.poles[j].real)
            if freq>1e-12 and decay>0:
                t_osc=min(t_end,40/decay)
                parts.append(np.linspace(0,t_osc,min(400000,int(t_osc*freq*16/(2*np.pi))+2)))
    return np.unique(np.concatenate(parts))

def inside_intervals(b,eps,cache):
    """Sorted [start,end] intervals (end may be inf) with D(rho(t),w) <= eps; edges refined on the scalar error."""
    if 'grid' not in cache:
        for stretch in (1.,10.,100.):
            ts=time_grid(b,stretch);e=errors(b,ts)
            if e[-1]<=min(EPS)*np.pi/4/10:break
        cache['grid']=(ts,e)
    ts,e=cache['grid'];ok=e<=eps
    if not ok[-1]:raise RuntimeError('member not eps-close at the end of its grid')
    f=lambda t:b.error(t)-eps
    def edge(lo,hi):
        flo,fhi=f(lo),f(hi)
        if flo*fhi>0:return hi if fhi<=0 else lo
        return brentq(f,lo,hi,xtol=1e-13,rtol=1e-12)
    out=[];i=0;n=len(ts)
    while i<n:
        if not ok[i]:i+=1;continue
        j=i
        while j+1<n and ok[j+1]:j+=1
        out.append([0. if i==0 else edge(ts[i-1],ts[i]),np.inf if j==n-1 else edge(ts[j],ts[j+1])]);i=j+1
    return out

def intersect(A,B):
    out=[];i=j=0
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]);hi=min(A[i][1],B[j][1])
        if lo<hi or lo==hi==np.inf:out.append([lo,hi])
        if A[i][1]<B[j][1]:i+=1
        else:j+=1
    return out

def family_time(keys,members,caches,eps):
    ivs={k:inside_intervals(members[k],eps,caches[k]) for k in keys};common=[[0.,np.inf]]
    for k in keys:common=intersect(common,ivs[k])
    return common[0][0],max(ivs[k][-1][0] for k in keys)

# ------------------------------------------------------------------ Shiraishi-Saito
def cumulative_speed(b,ends):
    """int_0^t sqrt(Var(H_S+V)) dt at every t in ends (same quadrature landmarks as the engine)."""
    ends=sorted(set(float(t) for t in ends if t>0))
    if not ends:return {},0.
    top=ends[-1]
    pts=[0.,min(top,1/max(1.,b.width,b.lam)),min(top,1/max(1.,b.gamma)),min(top,b.beta),min(top,b.t_switch)]
    pts+=[float(t) for t in np.geomspace(max(1e-5,b.t_switch),max(b.t_switch,top),24) if t<top]
    pts=sorted(set(pts+ends));total=err=0.;out={}
    for lo,hi in zip(pts[:-1],pts[1:]):
        v,e=quad(lambda t:np.sqrt(b.variance(t)),lo,hi,epsabs=2e-8,epsrel=2e-8,limit=150);total+=v;err+=e
        if hi in ends:out[hi]=total
    return out,err

# ------------------------------------------------------------------ one (family, Lambda, g^2) task
def task(args):
    fam,W,g2=args;t0=time.monotonic()
    xs=np.linspace(-DELTA,DELTA,NGRID[fam]);Dm=distance_matrix(fam,xs,g2,W)
    members={};caches={}
    def get(x):
        key=round(float(x),12)
        if key not in members:members[key]=member(fam,key,g2,W);caches[key]={}
        return key
    grid=[get(x) for x in np.linspace(-DELTA,DELTA,K)]
    per_eps=[]
    for e in EPS:
        eps=e*np.pi/4;chi,x1,x2=bound(fam,g2,W,Dm,xs,eps)
        keys=sorted(set(grid+([get(x1),get(x2)] if chi>0 else [])))
        tau,tau_sus=family_time(keys,members,caches,eps)
        per_eps.append((e,eps,chi,x1,x2,keys,tau,tau_sus))
    # Shiraishi-Saito for every member, at every family time
    taus=[p[6] for p in per_eps];speed={};qerr=0.
    for k in members:
        speed[k],er=cumulative_speed(members[k],taus);qerr=max(qerr,er)
    dist0={k:bures(members[k].rho0,members[k].target) for k in members}
    rows=[]
    for e,eps,chi,x1,x2,keys,tau,tau_sus in per_eps:
        st={k:2*max(dist0[k]-eps,0)/members[k].span for k in keys}
        if tau>0:va={k:(tau*max(dist0[k]-eps,0)/speed[k][tau] if dist0[k]>eps else 0.) for k in keys}
        else:va={k:0. for k in keys}
        ks=max(keys,key=lambda k:st[k]);kv=max(keys,key=lambda k:va[k])
        rows.append(dict(family=fam,width=W,gamma=g2,eps_fraction=e,epsilon=eps,delta=DELTA,
            tau=tau,tau_sustained=tau_sus,chi=chi,x1=x1,x2=x2,ss_static=st[ks],ss_variance=va[kv],
            x_ss_static=ks,x_ss_variance=kv,max_numerator=max(max(dist0[k]-eps,0) for k in keys),n_members=len(keys),
            wb_regime=bool(W>=100*max(g2,MAX_ENERGY[fam],1.)),quadrature_error=qerr,
            ratio_bound_over_ss=chi/va[kv] if va[kv]>0 else np.nan))
    return rows,fam,W,g2,time.monotonic()-t0

# ------------------------------------------------------------------ validation against the manuscript run
def reproduce_manuscript(points=((.1,),(1.,),(10.,))):
    """Old fixed pairs recomputed with this code, against the archived reference run at Lambda=1e4."""
    ref=list(csv.DictReader(open(REFERENCE_RUN)))
    out=[]
    for model,ee,init,gap in [('rlm',[[0.],[3.]],INITIAL['rlm'],3.),('dqd',[[1.2,1.0],[2.45,2.25]],INITIAL['dqd'],2.5)]:
        for (g2,) in points:
            bs=[LorentzianContinuum(e,np.sqrt(g2),1e4,initial=init) for e in ee];cs=[{} for _ in bs]
            for f in [.05,.2]:
                eps=f*np.pi/4;common=[[0.,np.inf]]
                for b,c in zip(bs,cs):common=intersect(common,inside_intervals(b,eps,c))
                tau=common[0][0]
                va=max(tau*max(bures(b.rho0,b.target)-eps,0)/b.integrated_speed(tau)[0] for b in bs) if tau>0 else 0.
                st=max(2*max(bures(b.rho0,b.target)-eps,0)/b.span for b in bs)
                r=next(x for x in ref if x['model']==model and abs(float(x['g_squared'])-g2)<1e-9 and abs(float(x['epsilon_fraction'])-f)<1e-12)
                out.append(dict(model=model,g2=g2,eps_fraction=f,tau_new=tau,tau_ref=float(r['tau_finite']),
                                ss_var_new=va,ss_var_ref=float(r['ss_variance']),ss_static_new=st,ss_static_ref=float(r['ss_static'])))
    return out

def main():
    # RLM: add couplings next to the finite-Lambda zero thresholds of the symmetric ball
    tasks=[]
    for W in WIDTHS:
        extra=[]
        for e in EPS:
            f=lambda g:bures(target('rlm',-DELTA,g,W),target('rlm',DELTA,g,W))-2*e*np.pi/4
            try:
                gc=brentq(f,1e-3,1e4);extra+=[gc*(1+s) for s in (-.1,-.01,-.001,.001)]
            except ValueError:pass
        tasks+=[('rlm',W,g) for g in np.unique(np.r_[G2_BASE,[g for g in extra if .01<=g<=1000]])]
        tasks+=[('dqd',W,g) for g in G2_BASE]
    tasks.sort(key=lambda t:(t[0]=='rlm',-t[2]))
    rows=[];t0=time.monotonic()
    with Pool(min(8,os.cpu_count() or 1)) as pool:
        for rr,fam,W,g2,dt in pool.imap_unordered(task,tasks):
            rows+=rr;r=rr[1]
            print(f'{fam} W={W:.0e} g2={g2:9.4g}: tau={r["tau"]:.4g} chi={r["chi"]:.4g} ssV={r["ss_variance"]:.3g} ssS={r["ss_static"]:.3g} '
                  f'[{dt:.0f}s, total {time.monotonic()-t0:.0f}s]',flush=True)
    rows.sort(key=lambda r:(r['family'],r['width'],r['gamma'],r['eps_fraction']))
    (DATA).mkdir(exist_ok=True)
    with (DATA/'benchmark.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # ---------------- checks
    tol=lambda t:t*(1+1e-6)+1e-10
    wb=list(csv.DictReader(open(HERE.parents[2]/'data'/'fermionic'/'families'/'sweep.csv')))
    def wb_row(fam,g2,e):
        name='rlm' if fam=='rlm' else 'dqd_common'
        return next((x for x in wb if x['family']==name and abs(float(x['gamma'])-g2)<1e-9 and abs(float(x['eps_fraction'])-e)<1e-12),None)
    agree={}
    for W in WIDTHS:
        for fam in ['rlm','dqd']:
            dt=[];dc=[]
            for r in rows:
                if r['family']!=fam or r['width']!=W or not r['wb_regime']:continue
                q=wb_row(fam,r['gamma'],r['eps_fraction'])
                if q is None:continue
                if float(q['tau'])>0:dt.append(abs(r['tau']-float(q['tau']))/float(q['tau']))
                if float(q['chi'])>0:dc.append(abs(r['chi']-float(q['chi']))/float(q['chi']))
            agree[f'{fam} Lambda={W:g}']=dict(max_rel_tau_vs_WB=max(dt,default=np.nan),max_rel_chi_vs_WB=max(dc,default=np.nan),points=len(dt))
    checks=dict(
        bound_le_tau_violations=sum(r['chi']>tol(r['tau']) for r in rows),
        ss_variance_le_tau_violations=sum(r['ss_variance']>tol(r['tau']) for r in rows),
        ss_static_le_ss_variance_violations=sum(r['ss_static']>tol(r['ss_variance']) for r in rows if r['tau']>0),
        tau_zero_with_positive_ss_numerator=sum(r['tau']==0 and r['max_numerator']>0 for r in rows),
        reentries_after_tau=sum(r['tau_sustained']>r['tau']*(1+1e-9)+1e-12 for r in rows),
        rlm_threshold_coincidence=all((r['tau']==0)==(r['chi']==0)==(r['max_numerator']==0) for r in rows if r['family']=='rlm'),
        finite_vs_wideband_in_WB_regime=agree,
        max_quadrature_error=max(r['quadrature_error'] for r in rows),
        reproduce_manuscript_run_20261008_182421=reproduce_manuscript())
    (DATA/'checks.json').write_text(json.dumps(checks,indent=1,default=float)+'\n')
    (DATA/'config.json').write_text(json.dumps(dict(
        units='hbar=beta=1; g^2 = hybridization at the Fermi level; Lorentzian width Lambda in k_BT; times in tau_Pl',
        families={'rlm':'H(x)=x n_d, |x|<=delta, p(0)=1/2','dqd':'h(x)=diag(1.2,1.0)+x N_S/2, |x|<=delta, (1,0)'},
        delta=DELTA,eps_fractions=EPS,K=K,widths=WIDTHS,g2_base=list(G2_BASE),
        wb_regime_flag='Lambda >= 100 max(g^2, max|level energy|, k_BT)',
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'compute.py']+list(ENGINE.glob('*.py'))}),
        indent=1,default=float)+'\n')
    print(json.dumps({k:v for k,v in checks.items() if k!='reproduce_manuscript_run_20261008_182421'},indent=1,default=float))
    for r in checks['reproduce_manuscript_run_20261008_182421']:print(r)

if __name__=='__main__':
    main()
