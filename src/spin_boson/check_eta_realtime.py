"""Check 2: closed-form real-time eta(t) (sb_core.MatsubaraSD.eta_real_time) for J = alpha nu e^{-nu/wc}.

(a) eta'' = C: second difference of eta vs C(t) from an independent quadrature of
    int J [coth(beta nu/2) cos(nu t) - i sin(nu t)] (short times, where quadrature is reliable).
(b) eta(t) vs a direct double integral of the same quadrature C (t <= 3).
(c) eta vs OQuPy's own quad at short times (t <= 1), where OQuPy's integrals do not warn.
Failure controls: dropping the Gamma-function (thermal) term, or using Im instead of Re in it,
must fail (a). Exit 1 on failure."""
import sys, warnings
import numpy as np
from scipy.integrate import quad
from scipy.special import loggamma
import oqupy
import sb_core as sc
warnings.filterwarnings('ignore')

def C_quad(t, a, wc, beta):
    re = quad(lambda n: a * n * np.exp(-n / wc) / np.tanh(beta * n / 2) * np.cos(n * t), 0, 60 * wc, limit=4000, epsabs=1e-13)[0]
    im = quad(lambda n: a * n * np.exp(-n / wc) * np.sin(n * t), 0, 60 * wc, limit=4000, epsabs=1e-13)[0]
    return re - 1j * im

def eta_mut(t, a, wc, beta, kind):
    z = 1 / (beta * wc); base = a * (np.log(1 + 1j * wc * t) - 1j * wc * t)
    if kind == 'nothermal':
        return base
    if kind == 'imag':
        return base + 2 * a * (loggamma(1 + z).real - loggamma(1 + z + 1j * t / beta).imag)
    raise ValueError

fails = 0
for a, wc, beta in [(0.2, 10., 1.), (0.5, 5., 2.), (0.05, 20., 0.5)]:
    sd = sc.MatsubaraSD(a, wc, 1 / beta)
    h = 1e-3
    errA = 0.0; ctrl = {'nothermal': 0.0, 'imag': 0.0}
    for t in [0.05, 0.2, 0.7, 1.5, 3.0]:
        Cq = C_quad(t, a, wc, beta)
        d2 = lambda f: (f(t + h) - 2 * f(t) + f(t - h)) / h ** 2
        errA = max(errA, abs(d2(sd.eta_real_time) - Cq) / max(1, abs(Cq)))
        for k in ctrl:
            ctrl[k] = max(ctrl[k], abs(d2(lambda u: eta_mut(u, a, wc, beta, k)) - Cq) / max(1, abs(Cq)))
    # (b) double integral eta(t) = int_0^t (t-u) C(u) du
    errB = 0.0
    for t in [0.3, 1.0, 3.0]:
        us = np.linspace(0, t, 1201); Cs = np.array([C_quad(u, a, wc, beta) for u in us])
        num = np.trapz((t - us) * Cs, us)
        errB = max(errB, abs(num - sd.eta_real_time(t)) / max(1, abs(num)))
    # (c) OQuPy quad at short times
    ref = oqupy.CustomSD(j_function=lambda w: a * w, cutoff=wc, cutoff_type='exponential', temperature=1 / beta)
    errC = max(abs(ref.eta_function(t) - sd.eta_real_time(t)) / max(1, abs(sd.eta_real_time(t))) for t in [0.05, 0.3, 1.0])
    okA, okB, okC = errA < 1e-3, errB < 1e-3, errC < 1e-6   # (a),(b) limited by the quadrature reference (~1e-4)
    fails += not (okA and okB and okC)
    print(f"alpha={a} wc={wc} beta={beta}: (a) eta''-C rel {errA:.1e}  (b) double-int rel {errB:.1e}  (c) vs OQuPy rel {errC:.1e}  ok={okA and okB and okC}")
    for k, v in ctrl.items():
        rej = v > 1e-3
        print(f"   failure control '{k}': {v:.1e} -> {'rejected (good)' if rej else 'NOT rejected'}"); fails += not rej
# how wrong is OQuPy's quad at long times? (informational)
sd = sc.MatsubaraSD(0.2, 10., 1.); ref = oqupy.CustomSD(j_function=lambda w: 0.2 * w, cutoff=10., cutoff_type='exponential', temperature=1.)
for t in [2.5, 5.0, 10.0, 20.0]:
    print(f"   info t={t:5.1f}: eta closed={sd.eta_real_time(t):.10f}  OQuPy quad={ref.eta_function(t):.10f}  diff={abs(sd.eta_real_time(t)-ref.eta_function(t)):.1e}")
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})"); sys.exit(1 if fails else 0)
