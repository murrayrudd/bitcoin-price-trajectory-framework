"""Execute the frozen revision protocol; retain original and locally augmented results.

Run from any directory. Original calculations remain in results_v05 and trajectory.
The local seed choice is adaptive, as declared in protocol.json; query definitions
and tolerances are fixed before any supplementary query evaluation.
"""
from pathlib import Path
import sys, json, itertools, hashlib, csv
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'results_v05'))
from compute import histories, generators, diagnostics, simulate, plain
from analyze import width_after

def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    gen = generators()['T_tau2']; base = gen['states'][36,0]
    legacy = json.loads((ROOT.parent/'trajectory/trajectory_results.json').read_text())
    data = {m:histories(m,'refined') for m in ['O','T']}
    masks = {m:{p:v<=1+1e-10 for p,v in diagnostics(d,gen).items()} for m,d in data.items()}
    qpaths = {name:np.log(np.interp(np.arange(37,61),q['months'],q['indices'])/100)+base
              for name,q in protocol['complete_path_queries'].items()}
    seeds = set()
    def panel_data(panel, source=data, allowed=masks):
        configs=[(m,int(i)) for m in ['O','T'] for i in np.flatnonzero(allowed[m][panel])]
        return configs, np.array([source[m]['states'][i] for m,i in configs]), np.array([source[m]['params'][i] for m,i in configs])
    def assess(panel, source=data, allowed=masks, seed=False):
        cs,st,ps=panel_data(panel,source,allowed)
        ix=100*np.exp(st[:,:,0]-base); acq=ix[:,48]<=105.5+1e-12; term=ix[:,60]>=112-1e-12
        deficit=np.maximum.reduce([ix[:,48]-105.5,112-ix[:,60],np.zeros(len(cs))])
        j=int(deficit.argmin())
        if seed:seeds.add(cs[j])
        result={'panel':panel,'n':len(cs),'counts':{m:sum(c[0]==m for c in cs) for m in ['O','T']},
                'index60_range':[ix[:,60].min(),ix[:,60].max()],
                'planning_class':{'acquisition_only_n':int(acq.sum()),'terminal_only_n':int(term.sum()),'joint_n':int((acq&term).sum()),
                                  'minimum_max_violation_index_points':deficit[j], 'nearest':cs[j], 'nearest_params':ps[j],
                                  'nearest_indices':[ix[j,48],ix[j,60]], 'support_configs':[cs[i] for i in np.flatnonzero(acq&term)]},'queries':[]}
        if acq.any():result['planning_class']['maximum_terminal_under_acquisition_ceiling']=ix[acq,60].max()
        if term.any():result['planning_class']['minimum_acquisition_under_terminal_floor']=ix[term,48].min()
        for name,z in qpaths.items():
            dist=abs(st[:,37:61,0]-z).max(1); k=int(dist.argmin())
            if seed:seeds.add(cs[k])
            lo=st[:,37:61,0].min(0);hi=st[:,37:61,0].max(0)
            result['queries'].append({'name':name,'min_distance':dist[k],'nearest':cs[k],'nearest_params':ps[k],
              'pointwise_envelope_min_tolerance':max(0,float(np.maximum(lo-z,z-hi).max())),
              'terminal_min_distance':abs(st[:,60,0]-z[-1]).min(),
              'support_by_tolerance':{str(e):int((dist<=e+1e-12).sum()) for e in protocol['tolerances']}})
        return result
    original=[assess(p,seed=True) for p in ['P1','P2','PH']]
    cs,st,ps=panel_data('P2'); target=100*np.exp(st[:,60,0]-base)
    measurements=[('A cash at 36',4,36,2.),('A holdings at 36',1,36,.5),('Log price at 37',0,37,.005)]
    def information(configs,states):
        target=100*np.exp(states[:,60,0]-base);result=[]
        for label,col,t,eps in measurements:
            for mult in protocol['measurement_allowance_multipliers']:
                v=states[:,t,col];allow=eps*mult;w,pair=width_after(v,target,allow)
                brute=np.max(np.where(abs(v[:,None]-v[None,:])<=2*allow+1e-12,abs(target[:,None]-target[None,:]),0))
                assert abs(w-brute)<1e-10
                result.append({'measurement':label,'multiplier':mult,'allowance':allow,'initial_width':np.ptp(target),
                               'worst_width':w,'reduction':np.ptp(target)-w,'witness':[configs[k] for k in pair]})
        return result
    info=information(cs,st)
    for row in info:
        if row['multiplier']==1:seeds.update(tuple(c) for c in row['witness'])
    for w in legacy['witnesses']:seeds.add((w['model'],w['id']))
    # A single extra half-step neighborhood, with no iteration after its results.
    seeds=sorted(seeds);new={};local_checks={};seed_records=[]
    for m in ['O','T']:
        proposals=[]
        delta=[.05,.005,.0625,0,0] if m=='O' else [.05,0,.0625,.125,.125]
        offsets=np.array(list(itertools.product(*[[-d,0,d] if d else [0] for d in delta])))
        for mm,i in seeds:
            if mm!=m:continue
            p=data[m]['params'][i];seed_records.append({'model':m,'id':i,'params':p})
            proposals.extend(p+offsets)
        oldkeys={tuple(np.round(p,8)) for p in data[m]['params']}
        lo=np.array([0,-.08,0,0,0]);hi=np.array([1.6,.08,2,4,4])
        pars=np.array(sorted({tuple(np.round(p,8)) for p in proposals if np.all(p>=lo-1e-12) and np.all(p<=hi+1e-12)}-oldkeys))
        if len(pars):
            states,vol,met=simulate(m,pars)
            stock=np.max(abs(states[:,:,1:4].sum(2)-(100+.1*np.arange(61))))
            cash=np.max(abs(states[:,:,4:7].sum(2)-(100+(pars[:,0,None]+.1)*np.arange(61))))
            assert max(stock,cash)<1e-8
            local_checks[m]={'new_evaluated':len(pars),'stock_error':stock,'cash_error':cash}
            new[m]={'params':pars,'states':states,'metrics':met}
        else:new[m]={'params':np.empty((0,5)),'states':np.empty((0,61,8))}
    merged={m:{k:np.concatenate([data[m][k],new[m][k]]) for k in ['params','states']} for m in ['O','T']}
    mmasks={m:{p:v<=1+1e-10 for p,v in diagnostics(d,gen).items()} for m,d in merged.items()}
    augmented=[assess(p,merged,mmasks) for p in ['P1','P2','PH']]
    mcs,mst,mps=panel_data('P2',merged,mmasks); augmented_info=information(mcs,mst)
    old_target_results=[]
    for p in ['P2','PH']:
        pcs,pst,pps=panel_data(p,merged,mmasks)
        for w in legacy['witnesses']:
            z=data[w['model']]['states'][w['id'],37:61,0]
            dist=abs(pst[:,37:61,0]-z).max(1)
            old_target_results.append({'panel':p,'target':w['name'],'min_distance':dist.min(),
              'support_by_tolerance':{str(e):int((dist<=e+1e-12).sum()) for e in protocol['tolerances']}})
    archived=json.loads((ROOT.parent/'results_v05/outputs/analysis.json').read_text())
    benchmarks=[]
    for row in archived['combined_forecasts']:
        if row['origin']==36:
            benchmarks.append({'generator':row['generator'],'candidate_joint_distance':row['joint_distance'],
                               'benchmark_joint_distance':max(row['last_price_abs_log_errors']),
                               'horizons':[1,6,12],'benchmark_component_errors':row['last_price_abs_log_errors']})
    from validate import independent_orders
    h=np.array([60.,30.,10.]);c=np.array([20.,50.,30.]);p=np.array([0.,0.,0.,0.,0.])
    custody=[{'price':price,'unrestricted_orders':independent_orders('O',p,price,h,c,0,0,np.zeros(3)).sum(),
               'restricted_orders':independent_orders('O',p,price,h,c,0,0,h).sum()} for price in [.99,1,1.01,1.1,2.]]
    assert all(abs(r['restricted_orders'])<1e-10 for r in custody if r['price']>=1)
    result={'protocol':protocol,'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
            'original':original,'measurement_sensitivity':info,'local_seed_records':seed_records,
            'local_checks':local_checks,'augmented':augmented,'augmented_measurement_sensitivity':augmented_info,
            'augmented_original_target_support':old_target_results,'matched_benchmarks':benchmarks,'custody_counterexample':custody}
    (ROOT/'revision_results.json').write_text(json.dumps(plain(result),indent=2))
    np.savez_compressed(ROOT/'local_configurations.npz',**{f'{m}_{k}':v for m,d in new.items() for k,v in d.items()},
                       **{name:z for name,z in qpaths.items()})
    with (ROOT/'planning_configurations.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['model','candidate_id','f','d','g','eta','tau','local_addition','survives_PH','index48','index60','class_support'])
        for (m,i),state,pars in zip(mcs,mst,mps):
            ix=100*np.exp(state[[48,60],0]-base)
            w.writerow([m,i,*pars,int(i>=len(data[m]['params'])),int(mmasks[m]['PH'][i]),*ix,int(ix[0]<=105.5+1e-12 and ix[1]>=112-1e-12)])
    brief={k:v for k,v in result.items() if k not in ['local_seed_records','protocol','measurement_sensitivity','augmented_measurement_sensitivity']}
    print(json.dumps(plain(brief),indent=2))

if __name__=='__main__':main()
