from compute import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit,logit

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
C={'O':'#245772','T':'#B06C22','S':'#74628D'}
def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/f'{name}.png');fig.savefig(OUT/f'{name}.pdf');plt.close(fig)

a=json.loads((OUT/'analysis.json').read_text());r=json.loads((OUT/'records.json').read_text());gs=generators()
eq=a['equivalence'];P=eq['P'];shares=np.array(eq['shares']);h=np.array(eq['pre_h']);c=np.array(eq['pre_c'])
fig,axes=plt.subplots(1,2,figsize=(6.5,3.05))
grid=np.linspace(.96*P,1.04*P,151)
colors=['#444444','#245772','#B06C22','#74628D']
for nu,col in zip([0,1,4,16],colors):
    aa=expit(logit(shares)[None,:]-nu*np.log(grid[:,None]/P));z=(aa*(h+c/grid[:,None])-h).sum(axis=1)
    axes[0].plot(grid,z,color=col,label=f'ν = {nu}')
axes[0].axhline(0,color='#888',lw=.6);axes[0].axvline(P,color='#888',lw=.6,ls=':')
axes[0].set(xlabel='Candidate price',ylabel='Net order (Bitcoin units)',title='A. Same observed equilibrium');axes[0].legend(frameon=False,fontsize=8)
axes[1].bar(['0','1','4','16'],[x['unit_percent'] for x in eq['rows']],color=colors)
axes[1].set(xlabel='Allocation sensitivity ν',ylabel='Price response (%)',title='B. One-unit funding response')
finish(fig,'figure_1_equivalence')

fig,axes=plt.subplots(1,2,figsize=(6.5,3.1),sharey=True)
panels=['P1','P2','PH','PHC'];labels=['Initial + terminal price','Monthly prices','+ Group holdings','+ A cash']
for ax,hh in zip(axes,[1,12]):
 for i,panel in enumerate(panels):
  for m,offset in [('O',-.10),('T',.10)]:
   z=next(x for x in r['responses'] if x['version']=='refined' and x['generator']=='T_tau2' and x['panel']==panel and x['model']==m and x['B']==0 and x['target']=='funding' and x['chi']==0 and x['horizon']==hh)
   if z['range']:
    ax.plot(z['range'],[i+offset]*2,color=C[m],lw=3,solid_capstyle='round',label=m if i==0 else None)
    if z['range'][0]==z['range'][1]:ax.scatter([z['range'][0]],[i+offset],color=C[m],s=12)
 truth=next(x['funding'] for x in a['generator_responses'] if x['generator']=='T_tau2' and x['chi']==0 and x['horizon']==hh)
 ax.axvline(truth,color='#222',ls='--',lw=1,label='Generator')
 ax.set(xlabel='Price response (%)',title=f'{hh}-month response');ax.set_yticks(range(4),labels)
axes[0].invert_yaxis();axes[1].legend(frameon=False,fontsize=8,loc='lower right')
finish(fig,'figure_2_information')

data=histories('T','refined');resp=interventions('T',data);dg=diagnostics(data,gs['T_tau2']);mask=dg['P2']<=1+1e-10;full=dg['PHC']<=1+1e-10
pairs=resp['funding_0.00'][mask][:,[1,12]];fullpairs=resp['funding_0.00'][full][:,[1,12]]
corner=np.array([pairs[:,0].min(),pairs[:,1].max()])
fig,ax=plt.subplots(figsize=(6.5,3.55))
ax.scatter(pairs[:,0],pairs[:,1],s=14,color='#A8BEC9',label='Monthly prices (277 configurations)')
ax.scatter(fullpairs[:,0],fullpairs[:,1],s=19,color=C['T'],label='Prices, holdings and cash (43)')
ax.scatter(*corner,marker='x',s=70,color='#9D3F44',linewidths=2,label='Combination of separate extrema')
truth=[next(x['funding'] for x in a['generator_responses'] if x['generator']=='T_tau2' and x['chi']==0 and x['horizon']==hh) for hh in [1,12]]
ax.scatter(*truth,marker='*',s=100,color='#222222',label='Generating configuration',zorder=4)
ax.set(xlabel='Immediate funding response (%)',ylabel='12-month funding response (%)');ax.legend(frameon=False,fontsize=8,loc='lower right')
finish(fig,'figure_3_joint_responses')

fig,axes=plt.subplots(1,2,figsize=(6.5,3.25),gridspec_kw={'width_ratios':[1.35,1]})
xx=np.arange(37,49);mask=dg['P2']<=1+1e-10
lp=data['states'][mask,37:49,0]
axes[0].fill_between(xx,np.exp(lp.min(axis=0)),np.exp(lp.max(axis=0)),color='#CAD7DE',label='T: monthly-price envelope')
lp=data['states'][full,37:49,0]
axes[0].fill_between(xx,np.exp(lp.min(axis=0)),np.exp(lp.max(axis=0)),color='#D7AD74',alpha=.8,label='T: full-account envelope')
axes[0].plot(xx,np.exp(gs['Regime']['states'][37:49,0]),color='#9D3F44',lw=1.7,label='Changed behavior')
axes[0].axvline(43,color='#777',lw=.8,ls=':');axes[0].set(xlabel='Synthetic month',ylabel='Price',title='A. Forecast issued at month 36')
axes[0].legend(frameon=False,fontsize=7.3,loc='upper left')
for i,panel in enumerate(['P2','PHC']):
 z=next(x for x in r['responses'] if x['version']=='refined' and x['generator']=='Regime' and x['panel']==panel and x['model']=='T' and x['B']==0 and x['target']=='funding' and x['chi']==0 and x['horizon']==12)
 axes[1].plot(z['range'],[i]*2,color=C['T'],lw=5,solid_capstyle='round')
truth=next(x['funding'] for x in a['generator_responses'] if x['generator']=='Regime' and x['chi']==0 and x['horizon']==12)
axes[1].axvline(truth,color='#9D3F44',ls='--',lw=1.3,label='Changed generator')
axes[1].set(yticks=[0,1],yticklabels=['Prices','+ Accounts'],xlabel='12-month response (%)',title='B. Funding response');axes[1].invert_yaxis();axes[1].legend(frameon=False,fontsize=7.3,loc='center right')
finish(fig,'figure_4_regime')
print('Four figures generated.')
