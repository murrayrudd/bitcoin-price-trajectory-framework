"""Analytical diagnostics, inference checks, and manuscript-ready summaries."""
from compute import *
from validate import independent_orders
from scipy.optimize import brentq
from scipy.special import expit,logit
from collections import deque

def best_joint(beta_range,x,z):
    lo,hi=beta_range
    candidates=[lo,hi]+[(zi+zj)/(xi+xj) for xi,zi in zip(x,z) for xj,zj in zip(x,z)]
    b=np.clip(candidates,lo,hi);d=np.max(abs(b[:,None]*x-z),axis=1);j=np.argmin(d)
    return float(d[j]),float(b[j])

def continuous_scaling(gs):
    rows=[];x=np.log((120+np.arange(61))/120)
    for name,g in gs.items():
      for B in [0.,.005,.02]:
       obs=g['states'][:,0].copy();obs[1:]+=.0025*(-1.)**np.arange(1,61)
       for origin in [36,42,48]:
        for panel in ['P1','P2']:
         ts=np.array([origin]) if panel=='P1' else np.arange(1,origin+1)
         lo=max(-2.,np.max((obs[ts]-(.005+B))/x[ts]));hi=min(6.,np.min((obs[ts]+(.005+B))/x[ts]))
         r=dict(generator=name,B=B,origin=origin,panel=panel,beta_low=float(lo),beta_high=float(hi),compatible=bool(lo<=hi))
         if lo<=hi:
            hd=np.array([1,6,12]);xx=x[origin+hd];z=g['states'][origin+hd,0];d,b=best_joint((lo,hi),xx,z)
            r.update(future_low=lo*xx,future_high=hi*xx,joint_distance=d,joint_beta=b,pointwise=((z>=lo*xx-1e-10)&(z<=hi*xx+1e-10)),joint_approx=bool(d<=.005))
         rows.append(r)
    return rows

def equivalence():
    # Reference: the main O generator's first auction; full adjustment.
    pars=np.array([.6,.04,.5,0,0]);state,vol,met=simulate('O',pars,end=1)
    h=np.array([60.,30.,10.1]);c=np.array([20.6,50.08,30.02]);P=np.exp(state[0,1,0]);a=P*state[0,1,1:4]/(c+P*h)
    K0=np.sum(a*c/P);K1=np.sum(a*(1-a)*(h+c/P));rows=[]
    for nu in [0.,1.,4.,16.]:
        def share(p):return expit(logit(a)-nu*np.log(p/P))
        def clear(cash):return brentq(lambda p:np.sum(share(p)*(h+cash/p)-h),P/10,P*10,xtol=1e-13)
        fitted=clear(c);new=clear(c+[1,0,0]);eps=1e-4
        exact=a[0]/(P*(K0+nu*K1))
        numerical=(np.log(clear(c+[eps,0,0]))-np.log(clear(c-[eps,0,0])))/(2*eps)
        assert abs(exact-numerical)<1e-9 and abs(fitted-P)<1e-10
        rows.append(dict(nu=nu,price=fitted,local_percent=100*exact,unit_percent=100*(new/P-1),target_price=new))
    return dict(P=P,holdings=state[0,1,1:4],cash=state[0,1,4:7],pre_h=h,pre_c=c,shares=a,K0=K0,K1=K1,rows=rows)

def width_after(values,target,eps):
    # Pairwise interval overlap is equivalent to a possible common observation.
    ids=np.argsort(values);v=values[ids];q=target[ids];lo=deque();hi=deque();left=0;best=-1.;pair=None
    for right in range(len(v)):
        while lo and q[lo[-1]]>=q[right]:lo.pop()
        while hi and q[hi[-1]]<=q[right]:hi.pop()
        lo.append(right);hi.append(right)
        while v[right]-v[left]>2*eps+1e-12:
            if lo[0]==left:lo.popleft()
            if hi[0]==left:hi.popleft()
            left+=1
        width=q[hi[0]]-q[lo[0]]
        if width>best:best=width;pair=[int(ids[lo[0]]),int(ids[hi[0]])]
    return float(best),pair

def main():
    gs=generators();records=json.loads((OUT/'records.json').read_text())
    analytic=equivalence();continuous=continuous_scaling(gs);info=[];truthresp=[];pairchecks=[];accountsummary=[]
    for name,g in gs.items():
        if g['model']=='S':continue
        data={'params':g['params'][None,:],'states':g['states'][None,:,:],'volume':g['volume'][None,:]}
        resp=interventions(g['model'],data)
        pre=g['states'][36,1:4]+[0,0,.1]
        threshold=float(np.min(g['states'][37:49,1:4]/pre))
        for chi in [0.,.01,.05,.1]:
            for h in [1,12]:
                truthresp.append(dict(generator=name,chi=chi,horizon=h,custody=float(resp[f'custody_{chi:.2f}'][0,h]),funding=float(resp[f'funding_{chi:.2f}'][0,h]),volume=float(resp[f'volume_{chi:.2f}'][0,h]),holdings=resp[f'holdings_{chi:.2f}'][0,h],custody_threshold=threshold))
        accountsummary.append(dict(generator=name,price36=float(np.exp(g['states'][36,0])),price48=float(np.exp(g['states'][48,0])),h36=g['states'][36,1:4],cash36=g['states'][36,4:7],volume37=g['volume'][37],volume48=g['volume'][48],custody_threshold=threshold))
    # Scope fixed at P2-compatible O/T configurations, refined menus.
    data_by={m:histories(m,'refined') for m in ['O','T']}
    resp_by={m:interventions(m,data_by[m]) for m in ['O','T']}
    ids=np.flatnonzero(diagnostics(data_by['T'],gs['T_tau2'])['P2']<=1+1e-10)
    pairs=resp_by['T']['funding_0.00'][ids][:,[1,12]]
    corner=np.array([pairs[:,0].min(),pairs[:,1].max()]);distance=np.max(abs(pairs-corner),axis=1)
    (OUT/'joint_response_example.json').write_text(json.dumps({'generator':'T_tau2','model':'T','version':'refined','panel':'P2','corner':corner.tolist(),'distance_pp':float(distance.min()),'nearest_id':int(ids[distance.argmin()]),'nearest_pair':pairs[distance.argmin()].tolist()},indent=2))
    for m,data in data_by.items():
        for g in gs.values():
            mask=diagnostics(data,g)['P2']<=1+1e-10
            if mask.any():
                assert max(np.max(abs(resp_by[m][f'custody_{chi:.2f}'][mask][:,[1,12]])) for chi in [.01,.05,.1])<1e-9
    for name,g in gs.items():
        values=[];q=[];config=[]
        for m,data in data_by.items():
            mask=diagnostics(data,g)['P2']<=1+1e-10;ids=np.flatnonzero(mask)
            values.extend(data['states'][ids,36,4:5].reshape(-1).tolist())
            q.extend(resp_by[m]['funding_0.00'][ids,1].tolist());config.extend([(m,int(j)) for j in ids])
        if not q:continue
        q=np.array(q);n=len(q);before=float(np.ptp(q))
        for label,col,t,eps in [('A_cash36',4,36,2.),('A_holdings36',1,36,.5),('log_price37',0,37,.005)]:
            v=np.array([data_by[m]['states'][j,t,col] for m,j in config]);w,pair=width_after(v,q,eps)
            assert w<=before+1e-12
            if len(q)<1000:
                brute=np.max(np.where(abs(v[:,None]-v[None,:])<=2*eps+1e-12,abs(q[:,None]-q[None,:]),0))
                assert abs(w-brute)<1e-10
            info.append(dict(generator=name,observation=label,n=n,prior_width=before,worst_width=w,guaranteed_reduction=before-w,witness=[config[i] for i in pair]))
    # Continuous S is included for common-price forecasts, not interventions.
    combined=[]
    for name,g in gs.items():
      for origin in [36,42,48]:
       parts=[r for r in records['forecasts'] if r['version']=='refined' and r['generator']==name and r['origin']==origin and r['panel']=='P2' and r['model']!='S' and r['n']]
       sr=next(r for r in continuous if r['generator']==name and r['origin']==origin and r['panel']=='P2' and r['B']==0)
       lows=[p['low'] for p in parts];highs=[p['high'] for p in parts];ds=[(p['joint_distance'],p['model'],p['joint_witness']) for p in parts]
       if sr['compatible']:lows.append(sr['future_low']);highs.append(sr['future_high']);ds.append((sr['joint_distance'],'S_continuous',sr['joint_beta']))
       item=dict(generator=name,origin=origin,scope=[p['model'] for p in parts]+(['S_continuous'] if sr['compatible'] else []))
       if lows:
         low=np.min(lows,axis=0);high=np.max(highs,axis=0);z=g['states'][origin+np.array([1,6,12]),0];d,m,w=min(ds)
         item.update(low=low,high=high,truth=z,joint_distance=d,joint_witness=(m,w),pointwise=((z>=low-1e-10)&(z<=high+1e-10)),joint_approx=d<=.005)
       obs_origin=g['states'][origin,0]+.0025*(-1.)**origin
       item['last_price_abs_log_errors']=abs(g['states'][origin+np.array([1,6,12]),0]-obs_origin)
       combined.append(item)
    # All model masks obey panel nesting; all refined grids contain coarse states.
    nesting=0;refinement_max=0.;metrics=[]
    for m in ['O','T','S']:
        co=histories(m,'coarse');fi=histories(m,'refined');lookup={tuple(np.round(p,7)):i for i,p in enumerate(fi['params'])}
        loc=np.array([lookup[tuple(np.round(p,7))] for p in co['params']])
        refinement_max=max(refinement_max,np.nanmax(abs(co['states']-fi['states'][loc])))
        for name,g in gs.items():
          for B in [0.,.005,.02]:
            ds=diagnostics(fi,g,B=B);last=None
            for p in ['P1','P2','PH','PHC']:
              if np.isnan(ds[p]).all():continue
              mask=ds[p]<=1+1e-10
              if last is not None:assert not np.any(mask&~last)
              last=mask;nesting+=1
        if m!='S':
            st=fi['states'];times=np.arange(61)
            stockerr=np.max(abs(st[:,:,1:4].sum(axis=2)-(100+.1*times)))
            casherr=np.max(abs(st[:,:,4:7].sum(axis=2)-(100+(fi['params'][:,0,None]+.1)*times)))
            metrics.append(dict(model=m,stock_error=float(stockerr),cash_error=float(casherr)))
            assert max(stockerr,casherr)<1e-8
    assert refinement_max<1e-10
    # Truth configurations retained where included, for every nested compatible panel.
    for name,g in gs.items():
        if g['model'] not in ['O','T','S']:continue
        data=histories(g['model'],'refined');j=np.flatnonzero(np.max(abs(data['params']-g['params']),axis=1)<1e-8)[0]
        for zero in [True,False]:
            for v in diagnostics(data,g,zero=zero).values():
                if not np.isnan(v).all():assert v[j]<=1+1e-10
    out=dict(equivalence=analytic,continuous_scaling=continuous,information=info,generator_responses=truthresp,accounts=accountsummary,combined_forecasts=combined,verification=dict(nested_panels_checked=nesting,coarse_refined_state_difference=refinement_max,account_errors=metrics,all_generators_retained=True))
    (OUT/'analysis.json').write_text(json.dumps(plain(out),indent=2))
    print('Analytical checks and summaries complete.')
    for r in accountsummary:print(r['generator'],'custody threshold',r['custody_threshold'])
    for r in info:print(r['generator'],r['observation'],r['n'],r['prior_width'],r['worst_width'])

if __name__=='__main__':main()
