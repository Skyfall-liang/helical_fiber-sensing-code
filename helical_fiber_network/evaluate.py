"""Held-out scene evaluation, including matching misses and theoretical baseline."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from fiber_net.physics import geometry,STRAIN_UNIT,angles
from fiber_net.inference import load_model,extract_contacts
from fiber_net.theory_reference import HelicalFiberModel
from fiber_net.data import KINDS
from predict import plot_prediction

def match_events(pred,truth,tolerance=.02):
    if not pred or not truth:return [],0
    cost=np.abs(np.array([p['s_m'] for p in pred])[:,None]-np.array([t['s_m'] for t in truth])[None,:])
    # Large penalty maximizes valid matches before minimizing distance.
    cost2=np.where(cost<=tolerance,cost,1000+cost)
    rows,cols=linear_sum_assignment(cost2)
    matches=[(pred[i],truth[j]) for i,j in zip(rows,cols) if cost[i,j]<=tolerance]
    return matches,len(matches)

def metric_rows(preds,truth_coef,events,kinds):
    result={}
    for kind in [-1,0,1,2]:
        ids=np.arange(len(kinds)) if kind==-1 else np.flatnonzero(kinds==kind)
        if not len(ids):continue
        theta_errors=[];phi_errors=[];loc=[];force=[];tp=0;npred=0;ntrue=0;empty_fp=0;empty_n=0
        for i in ids:
            p=preds[i];theta,phi=angles(p['coef']);tt,pt=angles(truth_coef[i])
            theta_errors.append(theta-tt)
            if pt is not None:
                # Evaluate phase from coefficients even if predicted theta is near zero.
                rawphi=np.rad2deg(np.arctan2(p['coef'][1],p['coef'][0]))
                phi_errors.append((rawphi-pt+180)%360-180)
            matches,count=match_events(p['contacts'],events[i]);tp+=count;npred+=len(p['contacts']);ntrue+=len(events[i])
            if not events[i]:empty_n+=1;empty_fp+=bool(p['contacts'])
            for observed,true in matches:
                loc.append((observed['s_m']-true['s_m'])*1000);force.append(observed['force_n']-true['force_n'])
        rmse=lambda a:float(np.sqrt(np.mean(np.square(a)))) if len(a) else None
        result['all' if kind==-1 else KINDS[kind]]={
          'frames':len(ids),'theta_rmse_deg':rmse(theta_errors),'phi_rmse_deg_nonzero_truth':rmse(phi_errors),
          'matched_position_rmse_fiber_mm':rmse(loc),'matched_force_rmse_n':rmse(force),
          'contact_precision':tp/npred if npred else None,'contact_recall':tp/ntrue if ntrue else None,
          'matched':tp,'predicted_peaks':npred,'true_peaks':ntrue,
          'false_contact_frame_rate_on_no_contact':empty_fp/empty_n if empty_n else None}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/joint/best.pt');p.add_argument('--data',default='data/synthetic')
    p.add_argument('--split',choices=['test','stress'],default='test');p.add_argument('--out',default='outputs/evaluation')
    a=p.parse_args();torch.set_num_threads(4);model,ckpt=load_model(a.checkpoint)
    with np.load(Path(a.data)/f'{a.split}.npz') as file:data={k:file[k] for k in file.files}
    events=json.loads((Path(a.data)/f'{a.split}_events.json').read_text())
    s=geometry(model.n)[0];preds=[];arrays=[];baseline=[];theory=HelicalFiberModel();recon=[]
    with torch.no_grad():
        for start in range(0,len(data['y']),32):
            out=model(torch.from_numpy(data['y'][start:start+32]));out={k:v.numpy() for k,v in out.items()}
            for j in range(len(out['coef'])):
                row={k:v[j] for k,v in out.items()};arrays.append(row)
                preds.append({'coef':row['coef'],'contacts':extract_contacts(row,s)})
                recon.append(np.mean((row['reconstruction']-data['y'][start+j])**2))
    for y in data['y']:
        r=theory.fit(s,y*STRAIN_UNIT);k=r['summary']['kappa_per_m'];phi=r['summary']['phi_rad'] or 0.
        baseline.append({'coef':np.array([k*np.cos(phi),k*np.sin(phi),0]),'contacts':r['contacts']})
    report={'split':a.split,'checkpoint_epoch':ckpt['epoch'],'match_tolerance_fiber_m':.02,
      'network':metric_rows(preds,data['coef'],events,data['kind']),
      'tukey_baseline':metric_rows(baseline,data['coef'],events,data['kind']),
      'network_reconstruction_rmse_microstrain':float(np.sqrt(np.mean(recon))*100),
      'note':'Force and localization errors are conditional on matched peaks; inspect precision/recall. Synthetic only.'}
    outdir=Path(a.out)/a.split;outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    for kind in range(3):
        i=int(np.flatnonzero(data['kind']==kind)[0]);plot_prediction(s,data['y'][i],arrays[i],preds[i]['contacts'],outdir/f'{KINDS[kind]}.png',data['contact'][i])
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
