"""Execute the predeclared Methods v0.5 synthetic design, without market data."""
from pathlib import Path
import ctypes, subprocess, json, itertools, time, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
SO=ROOT/'engine.so'
if not SO.exists() or SO.stat().st_mtime<(ROOT/'engine.cpp').stat().st_mtime:
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'engine.cpp'),'-o',str(SO)],check=True)
lib=ctypes.CDLL(str(SO));arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.simulate.argtypes=[ctypes.c_int,ctypes.c_int,arr,ctypes.c_int,ctypes.c_int,arr,ctypes.c_double,ctypes.c_double,arr,arr,arr]
lib.simulate.restype=None
INITIAL=np.array([0.,60.,30.,10.,20.,50.,30.,0.])
KINDS={'O':0,'T':1,'H':2,'R':3}

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def simulate(kind,pars,start=0,end=60,initial=None,chi=0.,pulse=0.):
    pars=np.ascontiguousarray(np.atleast_2d(pars),dtype=float);n=len(pars)
    initial=np.ascontiguousarray(np.tile(INITIAL,(n,1)) if initial is None else initial,dtype=float)
    ans=np.empty((n,end-start+1,8));vol=np.empty(ans.shape[:2]);met=np.empty((n,6))
    lib.simulate(KINDS[kind],n,pars,start,end,initial,chi,pulse,ans,vol,met)
    if np.any(met[:,5]):raise RuntimeError(f'Unresolved auctions in {kind}: {np.flatnonzero(met[:,5])[:10]}')
    assert met[:,0].max()<1e-10 and min(met[:,3:5].min(),0)>-1e-9
    return ans,vol,met

def menu(model,version):
    refine=version=='refined';step=.1 if refine else .2
    fs=np.round(np.arange(0,3.2+step/2 if version=='funding' else 1.6+step/2,step),8)
    gs=np.round(np.arange(0,4.001 if version=='trend' else 2.001,.125 if refine else .25),8)
    if model=='O':
        ds=np.round(np.arange(-.08,.08001,.01 if refine else .02),8)
        return np.array([[f,d,g,0.,0.] for f,d,g in itertools.product(fs,ds,gs)])
    if model=='T':
        es=np.arange(0,4.001,.25 if refine else .5)
        return np.array([[f,0.,g,e,t] for f,e,t,g in itertools.product(fs,es,es,gs)])
    bs=np.round(np.arange(-.04,.04001,.005 if refine else .01),8)
    betas=np.arange(-4 if version=='trend' else -2,8.001 if version=='trend' else 6.001,.25 if refine else .5)
    return np.array(list(itertools.product(bs,betas)))

def histories(model,version):
    path=OUT/f'{model}_{version}.npz'
    if path.exists():return dict(np.load(path))
    p=menu(model,version)
    if model=='S':
        states=np.full((len(p),61,8),np.nan);states[:,:,0]=p[:,0,None]+p[:,1,None]*np.log((120+np.arange(61))/120)
        vol=np.full((len(p),61),np.nan);metrics=np.zeros((len(p),6))
    else:states,vol,metrics=simulate(model,p)
    data=dict(params=p,states=states,volume=vol,metrics=metrics)
    np.savez_compressed(path,**data)
    return data

def generators():
    gs={}
    for d in [-.04,0.,.04]:gs[f'O_d{d:+.2f}']=('O',np.array([.6,d,.5,0.,0.]))
    for tau in [0.,2.,4.]:gs[f'T_tau{tau:.0f}']=('T',np.array([.6,0.,.5,1.,tau]))
    for beta in [.5,1.5,3.]:gs[f'S_beta{beta:.1f}']=('S',np.array([0.,beta]))
    gs['Hybrid']=('H',np.array([.6,.04,.5,1.,2.]))
    gs['Regime']=('R',np.array([.6,0.,.5,1.,2.]))
    result={}
    for name,(kind,p) in gs.items():
        if kind=='S':
            state=np.full((61,8),np.nan);state[:,0]=p[0]+p[1]*np.log((120+np.arange(61))/120);vol=np.full(61,np.nan);met=np.zeros(6)
        else:
            a,v,m=simulate(kind,p);state=a[0];vol=v[0];met=m[0]
        result[name]={'model':kind,'params':p,'states':state,'volume':vol,'metrics':met}
    return result

def diagnostics(data,gen,origin=36,zero=False,B=0.):
    """Return normalized residuals of price, holdings and A cash blocks."""
    state=data['states'];truth=gen['states'];n=len(state)
    obs=truth[:,0].copy()
    if not zero:obs[1:]+=.0025*(-1.)**np.arange(1,61)
    # The exact initial price remains an exact restriction on S's intercept.
    initial=np.abs(state[:,0,0]-obs[0])/1e-8
    ptol=(1e-8 if zero else .005)+B
    price=np.maximum(initial,np.abs(state[:,origin,0]-obs[origin])/ptol)
    monthly=np.maximum(initial,np.max(np.abs(state[:,1:origin+1,0]-obs[None,1:origin+1]),axis=1)/ptol)
    hd=[t for t in [18,36,42,48,60] if t<=origin]
    cd=[t for t in [36,42,48,60] if t<=origin]
    if np.isnan(state[0,0,1]) or np.isnan(truth[0,1]):
        hold=cash=np.full(n,np.nan)
    else:
        ho=truth[hd,1:4].copy()
        if not zero:
            ho+=np.array([[.25,-.25,0] if t in [18,42,60] else [-.25,0,.25] for t in hd])
        hold=np.max(abs(state[:,hd,1:4]-ho[None,:,:]),axis=(1,2))/((1e-8 if zero else .5)+100*B)
        co=truth[cd,4]+(0 if zero else 1.)
        cash=np.max(abs(state[:,cd,4]-co),axis=1)/((1e-8 if zero else 2.)+100*B)
    return {'P1':price,'P2':monthly,'PH':np.maximum(monthly,hold),'PHC':np.maximum.reduce([monthly,hold,cash])}

def interventions(model,data):
    digest=hashlib.sha256(data['params'].tobytes()).hexdigest()[:12]
    path=OUT/f'{model}_responses_{digest}.npz'
    if path.exists():return dict(np.load(path))
    pars=data['params'];initial=data['states'][:,36,:]
    control=data['states'][:,36:49,:]
    result={}
    for chi in [0.,.01,.05,.1]:
        if chi==0:base=control;vol=data['volume'][:,36:49]
        else:
            base,vol,met=simulate(model,pars,36,48,initial,chi,0.)
            result[f'base_metrics_{chi:.2f}']=met
        fund,vfund,met=simulate(model,pars,36,48,initial,chi,1.)
        key=f'{chi:.2f}'
        result[f'custody_{key}']=100*np.expm1(base[:,:,0]-control[:,:,0])
        result[f'funding_{key}']=100*np.expm1(fund[:,:,0]-base[:,:,0])
        result[f'volume_{key}']=vol
        result[f'holdings_{key}']=base[:,:,[1,2,3]]
        result[f'funding_holdings_{key}']=fund[:,:,[1,2,3]]
        result[f'metrics_{key}']=met
    np.savez_compressed(path,**result)
    return result

def range_record(values,mask):
    ids=np.flatnonzero(mask)
    if not len(ids):return {'n':0,'range':None,'witness':None}
    a=values[ids];lo=np.argmin(a);hi=np.argmax(a)
    return {'n':len(ids),'range':[a[lo],a[hi]],'witness':[int(ids[lo]),int(ids[hi])]}

def main():
    start=time.time();gs=generators()
    np.savez_compressed(OUT/'generators.npz',**{f'{k}__{a}':v[a] for k,v in gs.items() for a in ['params','states','volume','metrics']})
    records=[];forecasts=[];responses=[];info=[];verification={}
    for version in ['coarse','refined','funding','trend']:
      for model in ['O','T','S']:
        if version=='funding' and model=='S':continue
        print(f'{version} {model} history',flush=True)
        data=histories(model,version)
        if model!='S':
            print(f'{version} {model} responses',flush=True);resp=interventions(model,data)
        else:resp=None
        verification[f'{model}_{version}']={'n':len(data['params']),'max_metrics':data['metrics'].max(axis=0),'min_balances':data['metrics'][:,3:5].min(axis=0)}
        for name,gen in gs.items():
          for B in ([0.,.005,.02] if version in ['coarse','refined'] else [0.]):
            dg=diagnostics(data,gen,B=B)
            for panel,violation in dg.items():
              if np.isnan(violation).all():continue
              mask=violation<=1+1e-10
              ids=np.flatnonzero(mask)
              item={'version':version,'model':model,'generator':name,'B':B,'panel':panel,'n':len(ids),'min_violation':float(violation.min()),'best_id':int(np.argmin(violation))}
              if len(ids):item['parameter_ranges']=[data['params'][ids].min(axis=0),data['params'][ids].max(axis=0)]
              records.append(item)
              if resp is not None:
                for chi in [0.,.01,.05,.1]:
                  for target in (['funding'] if chi==0 else ['custody','funding']):
                    vals=resp[f'{target}_{chi:.2f}']
                    for h in [1,12]:responses.append({**{k:item[k] for k in ['version','model','generator','B','panel']},'target':target,'chi':chi,'horizon':h,**range_record(vals[:,h],mask)})
              np.save(OUT/f'mask_{version}_{model}_{name}_{panel}_B{B}.npy',mask)
          if version in ['coarse','refined']:
            dz=diagnostics(data,gen,zero=True)
            for panel,v in dz.items():
              if not np.isnan(v).all():records.append({'version':version,'model':model,'generator':name,'B':'zero_error','panel':panel,'n':int((v<=1+1e-10).sum()),'min_violation':float(v.min()),'best_id':int(np.argmin(v))})
            for origin in [36,42,48]:
              dg=diagnostics(data,gen,origin=origin)
              for panel in ['P2','PH','PHC']:
                v=dg[panel]
                if np.isnan(v).all():continue
                mask=v<=1+1e-10;ids=np.flatnonzero(mask);hd=np.array([1,6,12]);z=gen['states'][origin+hd,0]
                item={'version':version,'model':model,'generator':name,'origin':origin,'panel':panel,'n':len(ids)}
                if len(ids):
                    q=data['states'][ids[:,None],(origin+hd)[None,:],0]
                    lo=q.min(axis=0);hi=q.max(axis=0);dist=np.max(abs(q-z),axis=1);best=np.argmin(dist)
                    item.update(low=lo,high=hi,truth=z,joint_distance=float(dist[best]),joint_witness=int(ids[best]),pointwise=((z>=lo-1e-10)&(z<=hi+1e-10)),joint_approx=bool(dist[best]<=.005),joint_exact=bool(dist[best]<=1e-8))
                forecasts.append(item)
        print(f'completed {version} {model}, elapsed {time.time()-start:.1f}s',flush=True)
        (OUT/'records.json').write_text(json.dumps(plain({'compatibility':records,'responses':responses,'forecasts':forecasts,'verification':verification}),indent=2))
    print('All registered menus completed.',flush=True)

if __name__=='__main__':main()
