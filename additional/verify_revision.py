"""Independent checks of added configurations and the low-floor custody runs."""
from pathlib import Path
import sys, json
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'results_v05'))
from compute import generators, simulate, INITIAL, plain
from validate import independent_orders

def main():
    local=np.load(ROOT/'local_configurations.npz');r=json.loads((ROOT/'revision_results.json').read_text())
    picks={('O',4931),('T',84234),('T',83653)}
    for panel in r['augmented'][1:]:
        picks.add(tuple(panel['planning_class']['nearest']))
        picks.update(tuple(q['nearest']) for q in panel['queries'])
    old={m:np.load(ROOT.parent/f'results_v05/outputs/{m}_refined.npz') for m in ['O','T']}
    results=[]
    for m,i in sorted(picks):
        n=len(old[m]['params'])
        p,st=(old[m]['params'][i],old[m]['states'][i]) if i<n else (local[f'{m}_params'][i-n],local[f'{m}_states'][i-n])
        prev=INITIAL.copy();err=0.
        for t in range(1,61):
            h=prev[1:4]+[0,0,.1];c=prev[4:7]+[p[0],.08,.02];a=((120+t)/120)**p[2]
            def orders(P):return independent_orders(m,p,P,h,c,t,prev[7],np.zeros(3))
            P=brentq(lambda P:orders(P).sum(),a/16,a*16,xtol=1e-13)
            q=orders(P);lp=np.log(P);prev=np.r_[lp,h+q,c-P*q,lp-prev[0]]
            err=max(err,float(abs(prev-st[t]).max()))
        assert err<1e-8
        results.append({'model':m,'id':i,'max_state_difference':err})
    cases=0;maxdiff=0.;minmargin=float('inf')
    for name,g in generators().items():
        if g['model']=='S':continue
        p=g['params'];kind=g['model'];initial=g['states'][36]
        for pulse in [0.,1.]:
            control,_,_=simulate(kind,p,36,48,initial[None,:],0.,pulse)
            for chi in [.01,.05,.1]:
                restricted,_,_=simulate(kind,p,36,48,initial[None,:],chi,pulse)
                maxdiff=max(maxdiff,float(abs(control-restricted).max()))
                k=chi*(initial[1:4]+[0,0,.1])
                for t in range(37,49):
                    prev=restricted[0,t-37];P=np.exp(restricted[0,t-36,0])
                    h=prev[1:4]+[0,0,.1];c=prev[4:7]+[p[0]+(pulse if t==37 else 0),.08,.02]
                    left=independent_orders(kind,p,P*np.exp(-1e-7),h,c,t,prev[7],k).sum()
                    right=independent_orders(kind,p,P*np.exp(1e-7),h,c,t,prev[7],k).sum()
                    assert left>0 and right<0
                    minmargin=min(minmargin,left,-right);cases+=1
    assert maxdiff<1e-8
    out={'independent_paths':results,'custody_checked_auctions':cases,'custody_price_perturbation_log_units':1e-7,
         'custody_min_order_sign_margin':minmargin,'custody_max_control_restricted_state_difference':maxdiff}
    (ROOT/'verification.json').write_text(json.dumps(plain(out),indent=2));print(json.dumps(plain(out),indent=2))
if __name__=='__main__':main()
