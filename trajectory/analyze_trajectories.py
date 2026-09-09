"""Supplementary trajectory interrogation for manuscript v0.7.

Uses the unchanged v0.5 economic engine and refined parameter domains.
Selects complete lower/upper month-60 paths descriptively after historical
screening; no empirical likelihood, out-of-sample selection claim, or dollar
calibration attaches to these witnesses. Future target tolerance is 0.005
at every month 37--60, fixed to the preceding protocol's joint tolerance.
"""
from pathlib import Path
import sys, json, csv
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'results_v05'))
from compute import histories, generators, diagnostics, simulate, plain, menu
from analyze import width_after

def main():
    gs=generators(); gen=gs['T_tau2']; data={m:histories(m,'refined') for m in ['O','T']}
    masks={}; verification={}
    for m,d in data.items():
        dg=diagnostics(d,gen)
        masks[m]={p:v<=1+1e-10 for p,v in dg.items()}
        for p,mask in masks[m].items():
            archived=np.load(ROOT.parent/'results_v05'/'outputs'/f'mask_refined_{m}_T_tau2_{p}_B0.0.npy')
            assert np.array_equal(mask,archived)
        t=np.arange(61)
        stock=float(np.max(abs(d['states'][:,:,1:4].sum(2)-(100+.1*t))))
        cash=float(np.max(abs(d['states'][:,:,4:7].sum(2)-(100+(d['params'][:,0,None]+.1)*t))))
        assert max(stock,cash)<1e-8
        verification[m]={'evaluated':len(d['params']),'stock_error':stock,'cash_error':cash,'archived_masks_match':True}
    configs=[(m,int(i)) for m in ['O','T'] for i in np.flatnonzero(masks[m]['P2'])]
    states=np.array([data[m]['states'][i] for m,i in configs]); pars=np.array([data[m]['params'][i] for m,i in configs])
    p60=states[:,60,0]; witness_ids=[int(np.argmin(p60)),int(np.argmax(p60))]
    targets={'Lower endpoint':states[witness_ids[0],37:61,0],
             'Generating continuation':gen['states'][37:61,0],
             'Upper endpoint':states[witness_ids[1],37:61,0],
             'Changed behavior':gs['Regime']['states'][37:61,0]}
    reports=[]; paths=[]
    for panel in ['P1','P2','PH','PHC']:
        cs=[(m,int(i)) for m in ['O','T'] for i in np.flatnonzero(masks[m][panel])]
        st=np.array([data[m]['states'][i] for m,i in cs]); ps=np.array([data[m]['params'][i] for m,i in cs])
        q=st[:,37:61,0]
        row={'panel':panel,'n':len(cs),'counts':{m:sum(c[0]==m for c in cs) for m in ['O','T']},
             'index60_range':[float(100*np.exp(st[:,60,0].min()-gen['states'][36,0])),float(100*np.exp(st[:,60,0].max()-gen['states'][36,0]))],
             'parameters_min':ps.min(0),'parameters_max':ps.max(0),'targets':[]}
        for name,z in targets.items():
            dist=abs(q-z).max(1); k=int(dist.argmin()); keep=dist<=.005+1e-12
            r={'name':name,'min_distance':float(dist[k]),'nearest':cs[k],'nearest_params':ps[k],
               'support_n':int(keep.sum()),'support_counts':{m:sum(keep[j] and c[0]==m for j,c in enumerate(cs)) for m in ['O','T']}}
            if keep.any():r.update(parameters_min=ps[keep].min(0),parameters_max=ps[keep].max(0),support_configs=[cs[j] for j in np.flatnonzero(keep)])
            row['targets'].append(r)
        reports.append(row)
    witnesses=[]
    for title,k in zip(['Lower endpoint','Upper endpoint'],witness_ids):
        m,i=configs[k]; st=states[k]; h36=st[36,1:4]; c36=st[36,4:7]
        witnesses.append({'name':title,'model':m,'id':i,'params':pars[k],
                          'index37':float(100*np.exp(st[37,0]-gen['states'][36,0])),
                          'index48':float(100*np.exp(st[48,0]-gen['states'][36,0])),
                          'index60':float(100*np.exp(st[60,0]-gen['states'][36,0])),
                          'h36':h36,'cash36':c36,'survives_PH':bool(masks[m]['PH'][i])})
    # Prospective measurement now addresses month-60 price, using a common index.
    target=100*np.exp(states[:,60,0]-gen['states'][36,0]); information=[]
    for label,col,t,eps in [('A cash at 36',4,36,2.),('A holdings at 36',1,36,.5),('Log price at 37',0,37,.005)]:
        v=states[:,t,col];w,pair=width_after(v,target,eps)
        brute=float(np.max(np.where(abs(v[:,None]-v[None,:])<=2*eps+1e-12,abs(target[:,None]-target[None,:]),0)))
        assert abs(w-brute)<1e-10
        information.append({'measurement':label,'initial_width':float(np.ptp(target)),'worst_width':w,'reduction':float(np.ptp(target)-w),'witness':[configs[k] for k in pair]})
    # A sequentially independent check of both extreme paths through all months.
    from validate import independent_orders
    from scipy.optimize import brentq
    independent_errors=[]
    for k in witness_ids:
        m,i=configs[k]; p=pars[k]; st=states[k]; prev=st[0].copy();err=0.
        for t in range(1,61):
            h=prev[1:4]+[0,0,.1];c=prev[4:7]+[p[0],.08,.02];a=((120+t)/120)**p[2]
            def q(price):return independent_orders(m,p,price,h,c,t,prev[7],np.zeros(3))
            price=brentq(lambda price:q(price).sum(),a/16,16*a,xtol=1e-13)
            orders=q(price); lp=np.log(price); prev=np.r_[lp,h+orders,c-price*orders,lp-prev[0]]
            err=max(err,float(abs(prev-st[t]).max()))
        assert err<1e-8;independent_errors.append({'model':m,'id':i,'max_state_error':err})
    # Report the implication of replacing the anchor extrapolation after month 36.
    # Anchor remains continuous: freeze A at A_36; use the same T rule and accounts.
    # This is an explicit new continuation, not an estimated historical parameter.
    frozen=gen['states'][36].copy(); frozen_path=[frozen.copy()]; fixed_anchor=(156/120)**.5
    from scipy.special import expit,logit
    for t in range(37,61):
        h=frozen[1:4]+[0,0,.1];c=frozen[4:7]+[.6,.08,.02]
        def orders(P):
            a=expit(logit(np.array([.75,.375,.25]))+np.array([1,0,-1])*2*frozen[7]+np.log(fixed_anchor/P))
            return np.clip(np.array([.25,.6,1.])*(a*(h+c/P)-h),-h,c/P)
        P=brentq(lambda P:orders(P).sum(),fixed_anchor/16,fixed_anchor*16,xtol=1e-13)
        q=orders(P);lp=np.log(P);frozen=np.r_[lp,h+q,c-P*q,lp-frozen[0]];frozen_path.append(frozen.copy())
    frozen_path=np.array(frozen_path)
    assert np.max(abs(frozen_path[:,1:4].sum(1)-(100+.1*np.arange(36,61))))<1e-8
    assert np.max(abs(frozen_path[:,4:7].sum(1)-(100+.7*np.arange(36,61))))<1e-8
    distances=abs(states[:,37:61,0]-frozen_path[None,1:,0]).max(1)
    freeze={'index60':float(100*np.exp(frozen_path[-1,0]-gen['states'][36,0])),'index48':float(100*np.exp(frozen_path[12,0]-gen['states'][36,0])),
            'min_P2_distance':float(distances.min()),'nearest':configs[int(distances.argmin())],
            'description':'At month 36 retain the generating state, funding, eta=1 and tau=2; hold the valuation reference at its month-36 level for months 37--60.'}
    replay=gen['states'][None,36,:].copy(); freeze_error=0.
    for t in range(37,61):
        p=gen['params'].copy();p[2]=np.log(fixed_anchor)/np.log((120+t)/120)
        step,_,_=simulate('T',p,t-1,t,replay)
        replay=step[:,-1,:];freeze_error=max(freeze_error,float(abs(replay[0]-frozen_path[t-36]).max()))
    assert freeze_error<1e-8
    freeze['independent_engine_error']=freeze_error
    sensitivity=[]
    for panel in ['P2','PH']:
        cs=[(m,int(i)) for m in ['O','T'] for i in np.flatnonzero(masks[m][panel])]
        q=np.array([data[m]['states'][i,37:61,0] for m,i in cs])
        for name,z in targets.items():
            dist=abs(q-z).max(1)
            sensitivity.append({'panel':panel,'target':name,'counts_by_tolerance':{str(e):int((dist<=e+1e-12).sum()) for e in [.0025,.005,.01]}})
    result={'design':{'origin':36,'future_months':[37,60],'tolerance':.005,'price_index_base':'Main T latent month-36 price = 100','candidate_scope':['O','T'],'selection':'Lowest and highest month-60 prices on P2 refined menus, retaining each entire path.'},
            'panels':reports,'witnesses':witnesses,'information':information,'verification':verification,'independent_path_checks':independent_errors,
            'generator_index60':float(100*np.exp(gen['states'][60,0]-gen['states'][36,0])), 'frozen_anchor':freeze,'tolerance_sensitivity':sensitivity}
    (ROOT/'trajectory_results.json').write_text(json.dumps(plain(result),indent=2))
    np.savez_compressed(ROOT/'trajectory_paths.npz',states=states,params=pars,models=np.array([m for m,i in configs]),ids=np.array([i for m,i in configs]),
                        ph_mask=np.array([masks[m]['PH'][i] for m,i in configs]),truth=gen['states'],regime=gs['Regime']['states'],frozen_anchor=frozen_path)
    with (ROOT/'trajectory_configurations.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['model','candidate_id','f','d','g','eta','tau','survives_holdings','price_index_60'])
        for j,(m,i) in enumerate(configs):writer.writerow([m,i,*pars[j],int(masks[m]['PH'][i]),target[j]])
    brief={k:v for k,v in result.items() if k!='panels'}
    brief['panels']=[{**{k:v for k,v in r.items() if k!='targets'},'targets':[{k:v for k,v in t.items() if k!='support_configs'} for t in r['targets']]} for r in reports]
    print(json.dumps(plain(brief),indent=2))

if __name__=='__main__':main()
