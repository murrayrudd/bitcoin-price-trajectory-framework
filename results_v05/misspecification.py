from compute import *

gs=generators();rows=[]
for version in ['coarse','refined','funding','trend']:
 for m in ['O','T']:
  d=histories(m,version);r=interventions(m,d)
  for name in ['Hybrid','Regime']:
   g=gs[name];gd={'params':g['params'][None,:],'states':g['states'][None,:,:],'volume':g['volume'][None,:]};tr=interventions(g['model'],gd)
   for B in ([0.,.005,.02] if version in ['coarse','refined'] else [0.]):
    dg=diagnostics(d,g,B=B)
    for panel,v in dg.items():
     ids=np.flatnonzero(v<=1+1e-10)
     for key in ['funding_0.00','custody_0.01','custody_0.05','custody_0.10','funding_0.01','funding_0.05','funding_0.10']:
      z=tr[key][0,[1,12]];item=dict(version=version,model=m,generator=name,B=B,panel=panel,target=key,n=len(ids),truth=z.tolist())
      if len(ids):
       q=r[key][ids][:,[1,12]];dist=np.max(abs(q-z),axis=1);best=np.argmin(dist);lo=q.min(axis=0);hi=q.max(axis=0)
       item.update(low=lo.tolist(),high=hi.tolist(),width=(hi-lo).tolist(),inside=((z>=lo-1e-9)&(z<=hi+1e-9)).tolist(),nearest_each=np.min(abs(q-z),axis=0).tolist(),joint_distance=float(dist[best]),joint_witness=int(ids[best]))
      rows.append(item)
(OUT/'misspecification_targets.json').write_text(json.dumps(rows,indent=2))
print('Misspecification response diagnostics:',len(rows))
