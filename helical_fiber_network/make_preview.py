"""Export inspectable waveform examples and training curves."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiber_net.data import KINDS
from fiber_net.physics import geometry
out=Path('outputs/overview');out.mkdir(parents=True,exist_ok=True)
with np.load('data/synthetic/train.npz') as f:a={k:f[k] for k in ('y','kind','coef','contact')}
s=geometry(a['y'].shape[1])[0]
fig,axes=plt.subplots(4,1,figsize=(13,10),layout='constrained',sharex=True)
for i,kind in enumerate(KINDS):
    candidates=np.flatnonzero(a['kind']==i)
    if i<2:
        candidates=candidates[np.linalg.norm(a['coef'][candidates,:2],axis=1)<.6]
    j=int(candidates[0])
    axes[i].plot(s,a['y'][j]*100,lw=.7,label='Synthetic measurement')
    axes[i].plot(s,a['contact'][j]*100,lw=.8,label='Contact ground truth')
    axes[i].set_title(kind.replace('_',' '));axes[i].legend(loc='upper right')
real=np.genfromtxt('data/reference/bending_contact.csv',delimiter=',',names=True)
axes[3].plot(real['s_m'],real['strain'],lw=.7,color='black')
axes[3].set_title('Real bending + contact reference (no ground-truth labels)')
for ax in axes:ax.set_ylabel('Microstrain');ax.grid(alpha=.2)
axes[-1].set_xlabel('Fiber arc length (m)');fig.savefig(out/'data_examples.png',dpi=150);plt.close(fig)
h=json.loads(Path('runs/joint/history.json').read_text())
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
ax.semilogy([r['epoch'] for r in h],[r['train_loss'] for r in h],label='Train')
ax.semilogy([r['epoch'] for r in h],[r['val_loss'] for r in h],label='Validation')
ax.set(xlabel='Epoch',ylabel='Weighted joint loss');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'training.png',dpi=150);plt.close(fig)
print(out.resolve())
