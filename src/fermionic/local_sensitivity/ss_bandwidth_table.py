"""Bandwidth dependence of Shiraishi--Saito (SS) bounds vs the equilibrium bound.

Imports the validated manuscript engine unchanged (src/fermionic/engine).
Physical setup = selected manuscript run run_20261008_182421 except the swept Lorentzian
width Lambda: RLM energies (0,3), p0=1/2; DQD means (1.1,2.35), Delta=0.2, occupations (1,0).
hbar=beta=1, g^2 = wide-band hybridization. Precisions are fractions of pi/4.
All finite-width quantities (targets, common time, our bound, SS) are computed in the SAME
finite-Lambda model; WB values are stored alongside with agreement diagnostics.
Run: python src/fermionic/local_sensitivity/ss_bandwidth_table.py
"""
import sys,os,json,csv,hashlib,time
os.environ.setdefault('MPLCONFIGDIR','/tmp/prx-results-mpl')
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[2]/'data'/'fermionic'/'local_sensitivity'
ENGINE=HERE.parent/'engine'
sys.path.insert(0,str(ENGINE))
import numpy as np
from benchmark import evaluate,STATE_BUDGET,TIME_BUDGET

WIDTHS=[1e2,1e3,1e4,1e5,1e6,1e8]
G2=[0.1,1.,10.,100.,1000.]
PRECISIONS=[.01,.05,.1]
KEEP=['model','g_squared','width','epsilon_fraction','tau_wb','tau_finite','equilibrium_wb',
      'equilibrium_finite','ss_static','ss_variance','ratio_static','ratio_variance',
      'target_error','trajectory_error','tau_relative_error','tau_valid','spectral_span_1',
      'spectral_span_2','integral_sigma_1','integral_sigma_2','early_integral_fraction']
rows=[]
for model in ['rlm','dqd']:
    for W in WIDTHS:
        for g2 in G2:
            t=time.monotonic()
            try:
                rr=evaluate(model,g2,W,rlm_energy=3.,precisions=PRECISIONS)
            except Exception as e:   # record, do not hide
                rows.append(dict(model=model,g_squared=g2,width=W,status=f'error: {e}'));continue
            for r in rr:
                d={k:r[k] for k in KEEP};d['status']='ok'
                d['wb_regime']=bool(W>=100*max(g2,3.,1.))   # Lambda >> g^2, |E|, k_BT
                d['endpoint_ok']=bool(r['target_error']<STATE_BUDGET and r['tau_valid'])
                rows.append(d)
            print(model,W,g2,f'{time.monotonic()-t:.1f}s',flush=True)
out=DATA;out.mkdir(exist_ok=True)
keys=sorted({k for r in rows for k in r},key=lambda k:(k not in KEEP,KEEP.index(k) if k in KEEP else 0,k))
with (out/'ss_bandwidth_table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
# Scaling check: SS*sqrt(Lambda) should approach a constant when Lambda >> g^2 Lambda-independent scales
meta=dict(widths=WIDTHS,g_squared=G2,precisions=PRECISIONS,rlm=dict(energies=[0,3],p0=.5),
    dqd=dict(means=[1.1,2.35],delta=.2,occupations=[1,0]),beta=1.,hbar=1.,
    ss_definition='Shiraishi-Saito PRR 3,023074 Eq.(6): L(rho_S(0),rho_S(tau))<=hbar^-1 int dE_S, '
                  'E_S=H_S+V in joint state; numerator [D(rho0,target)-eps]_+; static: dE_S<=||H_S+V||/2',
    our_bound='2[D(target1,target2)-2eps]_+/||H1-H2||, RLM ||.||=3, DQD ||.||=2.5',
    wb_regime_flag='Lambda>=100*max(g^2,|E2|=3,k_BT=1)',state_budget=STATE_BUDGET,time_budget=TIME_BUDGET,
    engine_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ENGINE.glob('*.py')},
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'ss_bandwidth_table.json').write_text(json.dumps(meta,indent=2)+'\n')
print('saved',out)
