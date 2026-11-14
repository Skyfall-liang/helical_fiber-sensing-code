from pathlib import Path
import numpy as np
import torch
from scipy.signal import find_peaks
from .network import JointModel
from .physics import geometry, angles, STRAIN_UNIT, LF
from .theory_reference import HelicalFiberModel

def load_model(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    model=JointModel(ckpt['n'],ckpt['channels']);model.load_state_dict(ckpt['state_dict']);model.eval()
    return model,ckpt

def extract_contacts(pred,s):
    # Support confidence is a learned signed footprint, not calibrated probability.
    support=np.abs(pred['support'])
    peaks,_=find_peaks(support,height=.35,prominence=.20,distance=max(1,round(.025/(s[1]-s[0]))))
    theory=HelicalFiberModel();theta,phi=angles(pred['coef'])
    kappa=float(np.linalg.norm(pred['coef'][:2]));phase=np.arctan2(pred['coef'][1],pred['coef'][0])
    out=[]
    for i in peaks:
        if abs(pred['contact'][i])*STRAIN_UNIT<50e-6: continue
        slope, intercept = theory.calibration(s[i])
        peak_force = max(0., (abs(pred['contact'][i])*STRAIN_UNIT-intercept)/slope)
        out.append(dict(force_from_peak_n=float(peak_force),
            force_head_minus_peak_n=float(pred['force'][i]-peak_force), sample_index=int(i),s_m=float(s[i]),backbone_s_m=float(s[i]*.2/LF),
            force_n=float(pred['force'][i]),signed_support=float(pred['support'][i]),
            peak_strain=float(pred['contact'][i]*STRAIN_UNIT),zone=theory.zone(s[i]),
            calibration_extrapolated=not 1<=pred['force'][i]<=3,
            xyz_m=theory.helix_xyz([s[i]],kappa,phase)[0].tolist()))
    return out

def predict(model,y):
    with torch.no_grad():
        out=model(torch.as_tensor(y,dtype=torch.float32).reshape(1,-1))
    return {k:v[0].numpy() for k,v in out.items()}

def read_csv(path,n,unit):
    a=np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8-sig')
    if not {'s_m','strain'}.issubset(a.dtype.names or ()):
        raise ValueError('CSV requires s_m,strain columns')
    s,y=np.atleast_1d(a['s_m']),np.atleast_1d(a['strain'])
    if len(s)<10 or not np.isfinite(s).all() or np.any(np.diff(s)<=0):
        raise ValueError('at least 10 ordered finite coordinates required')
    if not np.isfinite(y).all():
        raise ValueError('This trained network needs complete frames; missing strain must be handled separately')
    if s[0]>1e-6 or s[-1]<LF-1e-6 or s[0]<-1e-6 or s[-1]>LF+1e-6:
        raise ValueError('Input must cover the calibrated sensing span [0,4.277] m exactly')
    if unit=='microstrain':y=y*1e-6
    target=geometry(n)[0]
    # Polyphase antialiasing for uniform oversampled full-frame input.
    if len(s)>n and np.allclose(np.diff(s),np.diff(s).mean(),rtol=1e-3):
        from scipy.signal import resample_poly
        import math
        divisor=math.gcd(n,len(s));y2=resample_poly(y,n//divisor,len(s)//divisor,padtype='line')
        s2=np.arange(len(y2))*(s[1]-s[0])*len(s)/n
        y=np.interp(target,s2,y2)
    else:y=np.interp(target,s,y)
    return target,y/STRAIN_UNIT
