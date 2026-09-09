from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
d=np.load(ROOT/'trajectory_paths.npz'); r=json.loads((ROOT/'trajectory_results.json').read_text())
s=d['states'];p=d['params'];truth=d['truth'];ph=d['ph_mask'];base=truth[36,0]
idx=100*np.exp(s[:,:,0]-base);actual=100*np.exp(truth[:,0]-base)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9,'legend.fontsize':8,'savefig.dpi':220})
blue='#28658C'; orange='#BF6737';green='#3D7E67';gray='#A5ABB2';purple='#80549A'
def save(fig,name):
 fig.savefig(ROOT/(name+'.png'),bbox_inches='tight');fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(7.5,3.4),layout='constrained')
tt=np.arange(37); q=100*np.exp((truth[:,0]+.0025*(-1.)**np.arange(61))-base);q[0]=actual[0]
ax[0].plot(tt,idx[:,:37].T,color=gray,alpha=.06,lw=.6)
ax[0].plot(tt,actual[:37],color='black',lw=1.4,label='Generating history')
ax[0].scatter(tt[1:],q[1:37],s=7,color=blue,zorder=3,label='Price observations')
ax[0].set(title='A. Monthly prices through month 36',xlabel='Synthetic month',ylabel='Price index (generating month 36 = 100)')
ax[0].legend(loc='upper left',frameon=False)
tt=np.arange(36,61)
ax[1].fill_between(tt,idx[:,36:].min(0),idx[:,36:].max(0),color=gray,alpha=.3,label='Monthly-price envelope')
ax[1].fill_between(tt,idx[ph,36:].min(0),idx[ph,36:].max(0),color=orange,alpha=.38,label='With holdings')
for name,col in [('Lower endpoint',blue),('Upper endpoint',purple)]:
 w=next(w for w in r['witnesses'] if w['name']==name);j=np.flatnonzero((d['ids']==w['id'])&(d['models']==w['model']))[0]
 ax[1].plot(tt,idx[j,36:],color=col,lw=1.4,label=name+' path')
ax[1].plot(tt,actual[36:],color='black',lw=1.3,ls='--',label='Generating continuation')
ax[1].set(title='B. Complete conditional continuations',xlabel='Synthetic month',ylabel='Price index')
ax[1].legend(loc='upper left',frameon=False)
save(fig,'figure_1_trajectories')
fig,ax=plt.subplots(1,2,figsize=(7.5,3.4),layout='constrained')
ax[0].scatter(p[:,0],p[:,2],s=32,facecolors='none',edgecolors=gray,linewidths=.6,label='Other price-compatible')
targets=r['panels'][1]['targets']
for label,col in [('Lower endpoint',blue),('Upper endpoint',purple)]:
 ids=set(tuple(x) for x in next(t for t in targets if t['name']==label)['support_configs'])
 mask=np.array([(m,int(i)) in ids for m,i in zip(d['models'],d['ids'])])
 ax[0].scatter(p[mask,0],p[mask,2],s=30,color=col,label=label+' support')
ax[0].scatter(p[ph,0],p[ph,2],s=64,facecolors='none',edgecolors=orange,linewidths=1.5,label='Survives holdings')
ax[0].scatter([.6],[.5],marker='*',s=115,color='black',label='Generator',zorder=5)
ax[0].set(title='A. Funding and valuation assumptions',xlabel='A monthly funding, f',ylabel='Valuation-growth exponent, g')
ax[0].legend(loc='upper right',frameon=False,fontsize=7)
meas=r['information'];ys=np.arange(3)
ax[1].barh(ys,[x['worst_width'] for x in meas],height=.5,color=[gray,orange,gray])
ax[1].axvline(meas[0]['initial_width'],color='black',lw=1,ls='--',label='Before measurement')
for j,x in enumerate(meas):ax[1].text(x['worst_width']+.03,j,f"{x['worst_width']:.2f}",va='center',fontsize=8)
ax[1].set(yticks=ys,yticklabels=['A cash, month 36','A holdings, month 36','Price, month 37'],xlim=(0,3.18),xlabel='Worst remaining month-60 width\n(price-index points)',title='B. Prospective measurement')
ax[1].invert_yaxis()
save(fig,'figure_2_conditions_measurement')
fig,ax=plt.subplots(1,2,figsize=(7.5,3.25),layout='constrained');tt=np.arange(36,61)
ax[0].fill_between(tt,idx[:,36:].min(0),idx[:,36:].max(0),color=gray,alpha=.3,label='Monthly-price envelope')
ax[0].fill_between(tt,idx[ph,36:].min(0),idx[ph,36:].max(0),color=orange,alpha=.4,label='With holdings')
ax[0].plot(tt,actual[36:],color='black',lw=1.4,label='Unchanged continuation')
ax[0].plot(tt,100*np.exp(d['regime'][36:,0]-base),color=purple,lw=1.4,label='Behavior changes at 43')
ax[0].plot(tt,100*np.exp(d['frozen_anchor'][:,0]-base),color=green,lw=1.4,label='Valuation reference freezes')
ax[0].set(title='A. Price consequences',xlabel='Synthetic month',ylabel='Price index');ax[0].legend(loc='upper left',frameon=False,fontsize=7)
anchor=((120+tt)/156)**.5*100
ax[1].plot(tt,anchor,color='black',lw=1.4,label='Original valuation reference')
ax[1].plot(tt,np.repeat(100.,25),color=green,lw=1.4,label='Frozen valuation reference')
ax[1].set(title='B. Valuation continuation assumption',xlabel='Synthetic month',ylabel='Valuation-reference index (month 36 = 100)')
ax[1].legend(loc='upper left',frameon=False,fontsize=7)
save(fig,'figure_4_continuation')
