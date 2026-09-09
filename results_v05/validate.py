"""Independent numerical checks for the model implementation and inference."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from compute import *

def independent_orders(kind,p,price,h,c,t,lag,k):
    anchor=((120+t)/120)**p[2];q=[]
    for i in range(3):
        lower=k[i]-h[i];upper=c[i]/price
        if kind=='O' or (kind=='H' and i<2):
            vals=anchor*np.array([.5,2]);pi=(1+np.array([.75,.375,.25])[i])/3+(p[1] if i==0 else 0)
            probs=np.array([1-pi,pi])
            def foc(x):return np.sum(probs*(vals-price)/(c[i]-price*x+vals*(h[i]+x)))
            if foc(lower)<=0:x=lower
            elif foc(upper)>=0:x=upper
            else:x=brentq(foc,lower,upper,xtol=1e-12)
        else:
            eta=3 if kind=='R' and t>=43 else p[3]
            a=expit(logit(np.array([.75,.375,.25])[i])+[1,0,-1][i]*p[4]*lag+eta*np.log(anchor/price))
            x=np.clip([.25,.6,1][i]*(a*(h[i]+c[i]/price)-h[i]),lower,upper)
        q.append(x)
    return np.array(q)

def main():
    rng=np.random.default_rng(20260906);errors=[];monotone=0.;cases=0
    for kind in ['O','T','H','R']:
      for j in range(20):
        p=np.array([rng.uniform(0,1.6),rng.uniform(-.08,.08),rng.uniform(0,2),rng.uniform(0,4),rng.uniform(0,4)])
        initial=np.array([rng.uniform(-.3,.3),*rng.uniform(1,100,3),*rng.uniform(1,100,3),rng.uniform(-.1,.1)])
        t=int(rng.integers(1,60));chi=.1 if j%2 else 0.;pulse=float(j%3==0)
        ans,vol,met=simulate(kind,p,t-1,t,initial[None,:],chi,pulse)
        h=initial[1:4]+[0,0,.1];c=initial[4:7]+[p[0]+pulse,.08,.02];k=chi*h
        A=((120+t)/120)**p[2]
        def order(x):return independent_orders(kind,p,x,h,c,t,initial[7],k)
        root=brentq(lambda x:order(x).sum(),A/16,16*A,xtol=1e-12)
        q=order(root);errors.append([abs(np.exp(ans[0,1,0])-root),max(abs(ans[0,1,1:4]-h-q)),max(abs(ans[0,1,4:7]-c+root*q))])
        zs=np.array([order(x).sum() for x in np.geomspace(A/16,16*A,41)])
        monotone=max(monotone,np.diff(zs).max());cases+=1
    errors=np.array(errors)
    assert errors.max()<1e-8 and monotone<1e-8
    for model in ['O','T','S']:
        coarse=menu(model,'coarse');fine=menu(model,'refined');trend=menu(model,'trend')
        sets=[set(map(tuple,np.round(a,7))) for a in [coarse,fine,trend]]
        assert sets[0]<=sets[1] and sets[0]<=sets[2]
        if model!='S':assert sets[0]<=set(map(tuple,np.round(menu(model,'funding'),7)))
    # Restarting the same state must reproduce the original continuation.
    continuation_error=0.;stock_error=0.;cash_error=0.
    for name,g in generators().items():
        if g['model']=='S':continue
        ans,v,m=simulate(g['model'],g['params'],36,60,g['states'][None,36,:])
        continuation_error=max(continuation_error,abs(ans[0]-g['states'][36:]).max())
        stock_error=max(stock_error,abs(g['states'][:,1:4].sum(axis=1)-(100+.1*np.arange(61))).max())
        cash_error=max(cash_error,abs(g['states'][:,4:7].sum(axis=1)-(100+(g['params'][0]+.1)*np.arange(61))).max())
    assert continuation_error<1e-9 and max(stock_error,cash_error)<1e-8
    record={'independent_auction_cases':cases,'max_price_difference':errors[:,0].max(),'max_holdings_difference':errors[:,1].max(),'max_cash_difference':errors[:,2].max(),'max_positive_order_increment':monotone,'continuation_error':continuation_error,'generator_stock_error':stock_error,'generator_cash_error':cash_error,'menus_nested':True}
    (OUT/'validation.json').write_text(json.dumps(plain(record),indent=2));print(json.dumps(plain(record),indent=2))

if __name__=='__main__':main()
