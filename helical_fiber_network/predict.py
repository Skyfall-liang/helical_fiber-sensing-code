import argparse,json
from pathlib import Path
import numpy as np
import torch
from fiber_net.inference import load_model,read_csv,predict,extract_contacts
from fiber_net.physics import angles,STRAIN_UNIT

def plot_prediction(s,y,pred,contacts,path,truth=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,9),layout='constrained',sharex=True)
    axes[0].plot(s,y*100,lw=.7,label='Measured')
    axes[0].plot(s,pred['reconstruction']*100,lw=.9,label='Physics reconstruction')
    axes[0].plot(s,pred['bending']*100,lw=.7,label='Estimated bending + offset')
    axes[1].plot(s,pred['contact']*100,label='Predicted contact strain',lw=1)
    if truth is not None:axes[1].plot(s,truth*100,'--',label='Synthetic contact truth',lw=.8)
    for c in contacts:
        axes[1].plot(c['s_m'],c['peak_strain']*1e6,'ro')
        axes[1].annotate(f"{c['force_n']:.2f} N",(c['s_m'],c['peak_strain']*1e6),fontsize=8)
    axes[2].plot(s,(y-pred['reconstruction'])*100,lw=.8,label='Unexplained residual')
    for ax in axes:ax.set_ylabel('Strain (microstrain)');ax.legend(loc='upper right');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Fiber arc length (m)');fig.savefig(path,dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/reference/bending_contact.csv')
    p.add_argument('--unit',choices=['strain','microstrain'],default='microstrain')
    p.add_argument('--checkpoint',default='runs/joint/best.pt');p.add_argument('--out',default='outputs/real_bending_contact')
    a=p.parse_args();torch.set_num_threads(4)
    model,ckpt=load_model(a.checkpoint);s,y=read_csv(a.input,model.n,a.unit);pred=predict(model,y)
    theta,phi=angles(pred['coef']);contacts=extract_contacts(pred,s)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    report={'input':str(Path(a.input).resolve()),'theta_deg':theta,'phi_deg':phi,
      'kappa_per_m':float(np.linalg.norm(pred['coef'][:2])), 'reconstruction_rmse_microstrain':float(np.sqrt(np.mean((y-pred['reconstruction'])**2))*100),
      'contacts':contacts,'checkpoint_epoch':ckpt['epoch'],
      'limitations':['Synthetic-trained; no real shape/force ground truth used or validated.',
      'Contact entries are fiber-peak effective forces, not independent object forces.',
      'Phi is null below 0.5 degree bending; this threshold is configurable in angles().']}
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savetxt(out/'decoupled.csv',np.column_stack((s,y*STRAIN_UNIT,pred['bending']*STRAIN_UNIT,pred['contact']*STRAIN_UNIT,
       pred['reconstruction']*STRAIN_UNIT,pred['support'],pred['force'])),delimiter=',',
       header='s_m,strain,bending_with_offset,contact_strain,reconstructed_strain,signed_support,local_force_n',comments='')
    plot_prediction(s,y,pred,contacts,out/'analysis.png');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
