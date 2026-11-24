"""Controlled WB primary curves, with matching finite-width microscopic bounds.
Numerical evaluation helpers. Launch simulations through ../run.py.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/prx-audit-mpl')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from pathlib import Path
import argparse,csv,json,time,hashlib
import numpy as np
from scipy.optimize import brentq
from continuum import LorentzianContinuum,Numerics,bures,density,physical
from wideband import WideBand
from model_parameters import dqd_energies,dqd_preparation
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
PRECISIONS=[.01,.05,.1,.15,.2]
# Declared BEFORE selecting bandwidth; applies to sampled full trajectories incl. t~1/W.
STATE_BUDGET=.1*min(PRECISIONS)*np.pi/4
TIME_BUDGET=.005

def save(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def pair(model,gamma,width,numerics=Numerics(),rlm_energy=1.,rlm_occupation=.5,dqd_delta=.2,dqd_occupations=(1.,0.)):
    ee=[[0.],[float(rlm_energy)]] if model=='rlm' else dqd_energies(dqd_delta)
    if not 0 <= rlm_occupation <= 1: raise ValueError("RLM occupation must lie in [0,1]")
    initial=np.diag([rlm_occupation] if model=='rlm' else dqd_preparation(dqd_occupations))
    wb=[WideBand(e,gamma,initial,numerics) for e in ee]
    finite=[LorentzianContinuum(e,np.sqrt(gamma),width,initial=initial,numerics=numerics) for e in ee]
    return wb,finite

def grid(bs,width,gamma,resolution=1):
    end=max(20.,20*max(b.slow_time for b in bs))
    arrays=[np.array([0.]),np.geomspace(1e-5/width,end,int(900*resolution)),
            np.geomspace(.001/width,40/width,int(160*resolution)),
            np.linspace(0,min(end,20/max(gamma,1)),int(300*resolution))]
    for b in bs:
        for i in range(len(b.poles)):
            for j in range(i):
                frequency=abs(b.poles[i].imag-b.poles[j].imag)
                decay=-b.poles[i].real-b.poles[j].real
                if frequency<1e-12 or decay<=0:continue
                stop=min(end,28/decay)
                count=min(30000,max(2,int(stop*frequency*24*resolution/(2*np.pi))+1))
                arrays.append(np.linspace(0,stop,count))
    return np.unique(np.concatenate(arrays))

def crossing(bs,errors,times,eps):
    inside=np.flatnonzero(np.max(errors,axis=0)<=eps)
    if not len(inside):return np.nan,np.nan
    j=int(inside[0]);f=lambda t:max(b.error(t) for b in bs)-eps
    tau=0. if j==0 else brentq(f,times[j-1],times[j],xtol=2e-12,rtol=2e-12)
    outside=np.flatnonzero(np.max(errors[:,j+1:],axis=0)>eps)
    exit_time=np.nan
    if len(outside):
        k=j+1+int(outside[0]);exit_time=brentq(f,times[k-1],times[k],xtol=2e-12)
    return tau,exit_time

def evaluate(model,gamma,width,numerics=Numerics(),resolution=1,variance=True,trajectories=False,
             rlm_energy=1.,precisions=PRECISIONS,rlm_occupation=.5,dqd_delta=.2,dqd_occupations=(1.,0.)):
    wb,finite=pair(model,gamma,width,numerics,rlm_energy=rlm_energy,rlm_occupation=rlm_occupation,dqd_delta=dqd_delta,dqd_occupations=dqd_occupations)
    ts=grid(wb+finite,width,gamma,resolution)
    ew=np.array([[b.error(t) for t in ts] for b in wb])
    ef=np.array([[b.error(t) for t in ts] for b in finite])
    target_errors=[bures(w.target,f.target) for w,f in zip(wb,finite)]
    # Use a fully resolved early mesh and logarithmic later samples for WB validation.
    differences=[];cov_diff=[]
    for w,f in zip(wb,finite):
        dist=[];cd=[]
        for t in ts:
            a=w.at(t)[0];b=f.at(t)[0][:f.n,:f.n]
            physical(a);physical(b)
            dist.append(bures(density(a),density(b)));cd.append(np.max(abs(a-b)))
        differences.append(dist);cov_diff.append(cd)
    maximum=float(np.max(differences))
    gap=abs(rlm_energy) if model=='rlm' else 2.5
    sep_w=bures(wb[0].target,wb[1].target);sep_f=bures(finite[0].target,finite[1].target)
    rows=[]
    for f_eps in precisions:
        eps=f_eps*np.pi/4
        tw,xw=crossing(wb,ew,ts,eps);tf,xf=crossing(finite,ef,ts,eps)
        rel=abs(tf-tw)/max(tf,tw,1e-12)
        time_ok=(tf==tw==0) or (tf>0 and tw>0 and rel<TIME_BUDGET)
        static=[];var=[];integrals=[];qerrs=[];early=[]
        for b in finite:
            numerator=max(bures(b.rho0,b.target)-eps,0)
            static.append(2*numerator/b.span)
            if variance and tf>0 and np.isfinite(tf):
                integ,err=b.integrated_speed(tf)
                endpoint=bures(b.rho0,density(b.at(tf)[0][:b.n,:b.n]))
                assert endpoint<=integ+1e-6 and integ<=tf*b.span/2+1e-6*max(1,integ)
                integrals.append(integ);qerrs.append(err);var.append(tf*numerator/integ)
                cut=min(tf,20/width)
                short,_=b.integrated_speed(cut);early.append(short/integ)
            elif tf==0:
                integrals.append(0.);qerrs.append(0.);var.append(0.);early.append(0.)
            else:
                integrals.append(np.nan);qerrs.append(np.nan);var.append(np.nan);early.append(np.nan)
        pw=2*max(sep_w-2*eps,0)/gap
        pf=2*max(sep_f-2*eps,0)/gap
        corrected=2*max(sep_w-2*eps-sum(target_errors),0)/gap
        if variance and np.isfinite(tf):
            assert max(pf,max(static),max(var))<=tf+1e-6*max(1,tf)
            assert max(static)<=max(var)+1e-9
        rows.append(dict(model=model,g=np.sqrt(gamma),g_squared=gamma,width=width,beta=1.,j0=1.,
            epsilon_fraction=f_eps,epsilon=eps,tau_wb=tw,tau_finite=tf,
            equilibrium_wb=pw,equilibrium_finite=pf,equilibrium_corrected_estimate=corrected,
            ss_static=max(static),ss_variance=max(var),
            ratio_static=pf/max(static) if max(static)>0 else np.nan,
            ratio_variance=pf/max(var) if max(var)>0 else np.nan,
            target_error=max(target_errors),trajectory_error=maximum,covariance_error=float(np.max(cov_diff)),
            tau_relative_error=rel,tau_valid=bool(time_ok),wb_valid=bool(time_ok and maximum<STATE_BUDGET and max(target_errors)<STATE_BUDGET),
            exit_wb=xw,exit_finite=xf,window_end=ts[-1],samples=len(ts),
            spectral_span_1=finite[0].span,spectral_span_2=finite[1].span,
            integral_sigma_1=integrals[0],integral_sigma_2=integrals[1],
            integral_error=max(qerrs),early_integral_fraction=max(early),
            min_decay=min(-b.poles.real.max() for b in finite),
            bound_modes=sum(len(b.bound_weights) for b in finite),
            source_residual_relative=max(b.source_residual/width for b in finite)))
    if trajectories:return rows,wb,finite,ts,ew,ef,np.array(differences)
    return rows
