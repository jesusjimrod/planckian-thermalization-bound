"""Independent check of Eq. (mfg_correlation): wide-band MFG correlation matrix.

Formula (hbar=1):  C = 1/2 - [psi(1/2 - beta A/2pi) - h.c.]/(2 pi i),  A = -i h - (g^2/2) u u^T.
Reference: real-frequency spectral integral  C = int dw/2pi f(w) G^R(w) g^2 u u^T G^R(w)^dagger,
G^R(w) = (w - h + i g^2 u u^T/2)^{-1}. Convention C_mn = Tr[rho d_n^dag d_m].
Matrix digamma via eigendecomposition (A is diagonalizable away from exceptional points);
at the exceptional point we use scipy funm. Run: venv/bin/python <this file>
"""
import numpy as np, json
from pathlib import Path
from scipy.integrate import quad_vec
from scipy.special import digamma, expit
from scipy.linalg import funm

def formula(h,g2,beta=1.):
    n=len(h);u=np.ones((n,1));A=-1j*h-g2/2*(u@u.T)
    X=.5*np.eye(n)-beta*A/(2*np.pi)
    w,V=np.linalg.eig(X)
    psi=(V*digamma(w))@np.linalg.inv(V) if np.linalg.cond(V)<1e8 else funm(X,digamma)
    return .5*np.eye(n)-(psi-psi.conj().T)/(2j*np.pi)

def spectral(h,g2,beta=1.):
    n=len(h);u=np.ones((n,1));G=g2*(u@u.T)
    def f(w):
        R=np.linalg.inv(w*np.eye(n)-h+.5j*G);return expit(-beta*w)*R@G@R.conj().T/(2*np.pi)
    pts=sorted(set([0.]+list(np.linalg.eigvals(h-.5j*G).real)))
    edges=[-np.inf]+[p+s for p in pts for s in (-50,-5,-.5,0,.5,5,50)]+[np.inf]
    edges=sorted(set(edges));tot=0
    for a,b in zip(edges[:-1],edges[1:]):
        v,_=quad_vec(f,a,b,epsabs=1e-13,epsrel=1e-12,limit=500);tot=tot+v
    return tot

cases={'RLM eps=3, g2=1':(np.array([[3.]]),1.),
 'RLM eps=0.7, g2=50':(np.array([[.7]]),50.),
 'DQD diag (1.2,1.0), g2=0.01':(np.diag([1.2,1.]),.01),
 'DQD diag (1.2,1.0), g2=10':(np.diag([1.2,1.]),10.),
 'DQD near exceptional point g2=Delta=0.2 (+1e-6)':(np.diag([1.2,1.]),.2+1e-6),
 'DQD with real tunnelling 0.3':(np.array([[1.2,.3],[.3,1.]]),2.),
 'DQD with complex tunnelling 0.3i':(np.array([[1.2,.3j],[-.3j,1.]]),2.),
 'DQD negative energies':(np.array([[-2.,.1],[.1,-.5]]),5.)}
out={}
for k,(h,g2) in cases.items():
    a,b=formula(h,g2),spectral(h,g2);err=float(np.max(abs(a-b)))
    out[k]=err;print(f'{k:50s} max|formula - spectral| = {err:.2e}')
assert max(out.values())<1e-8
Path(__file__).with_name('data').mkdir(exist_ok=True)
(Path(__file__).parent/'data'/'verify_mfg_formula.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS')
