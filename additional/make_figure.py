"""Plot the fixed acquisition and terminal-price conditions on the original menu."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parent
d=np.load(ROOT.parent/'trajectory/trajectory_paths.npz')
ix=100*np.exp(d['states'][:,:,0]-d['truth'][36,0]);ph=d['ph_mask']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
fig,ax=plt.subplots(figsize=(6.5,4.2))
ax.add_patch(Rectangle((105.2,112),.3,1.45,facecolor='#e5ebe4',edgecolor='none'))
ax.scatter(ix[:,48],ix[:,60],s=19,color='#355c78',alpha=.65,label='Monthly prices (P2)',linewidths=0)
ax.scatter(ix[ph,48],ix[ph,60],s=37,facecolors='none',edgecolors='#c77830',linewidths=1.1,label='Prices and holdings (PH)')
ax.axvline(105.5,color='#687865',ls='--',lw=1);ax.axhline(112,color='#687865',ls='--',lw=1)
ax.text(105.23,113.15,'Planning\nconditions',fontsize=9,color='#435640')
ax.set(xlim=(105.2,107.05),ylim=(110.3,113.45),xlabel='Month-48 acquisition price index',ylabel='Month-60 price index')
ax.legend(loc='lower right',frameon=False,fontsize=9)
ax.grid(color='#e8eaed',lw=.55);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(ROOT/'figure_C1_planning.png');fig.savefig(ROOT/'figure_C1_planning.pdf');plt.close(fig)

assert (ROOT/'figure_C1_planning.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
