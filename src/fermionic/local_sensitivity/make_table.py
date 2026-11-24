"""LaTeX body of the speed-limit comparison table from data/ss_bandwidth_table.csv (eps=0.05 pi/4).
tau_eps and our bound: wide-band values (Lambda-independent). SS: variance form, finite-Lambda model,
evaluated at that model's own common time. Output: data/ss_table_body.tex"""
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=[r for r in csv.DictReader(open(HERE/'data/ss_bandwidth_table.csv')) if r['status']=='ok' and float(r['epsilon_fraction'])==.05]
W=[1e2,1e4,1e6,1e8]
def fmt(x):
    x=float(x)
    if x==0:return '$0$'
    if 1e-2<=abs(x)<1e3:return f'${x:.3g}$'
    m,e=f'{x:.2e}'.split('e');return f'${float(m):.2g}\\times10^{{{int(e)}}}$'
def row(model,g2):
    rr={float(r['width']):r for r in R if r['model']==model and float(r['g_squared'])==g2}
    ref=rr[1e8]
    cells=[fmt(ref['tau_wb']),fmt(ref['equilibrium_wb'])]
    for w in W:
        c=fmt(rr[w]['ss_variance'])
        if rr[w]['wb_regime']!='True' and float(rr[w]['ss_variance'])>0:c+='$^\\dagger$'
        cells.append(c)
    return cells
lines=[]
for model,label,gs in [('rlm','RLM',[.1,1.,10.,100.]),('dqd','DQD',[.1,1.,10.,100.,1000.])]:
    for i,g in enumerate(gs):
        name=(label if i==0 else '')
        lines.append(' & '.join([name,f'${g:g}$']+row(model,g))+r' \\')
    lines.append(r'\midrule' if model=='rlm' else '')
(HERE/'data/ss_table_body.tex').write_text('\n'.join(l for l in lines if l)+'\n')
print((HERE/'data/ss_table_body.tex').read_text())
