"""Nominal paper model. Internal neural units: strain / 100 microstrain."""
import numpy as np
LF, L, R, H, N, C, OMEGA = 4.277, .2, .017, .005, 8, .012, .999
STRAIN_UNIT = 100e-6
LAM = np.hypot(R,H/(2*np.pi))
PERIOD = 2*np.pi*LAM/OMEGA

def geometry(n=1536):
    s=np.linspace(0,LF,n)
    q=OMEGA*s/LAM
    env=np.abs(np.sin(np.pi*N*s/LF))
    x=np.column_stack((C*R*env*np.sin(q)/STRAIN_UNIT,
                       C*R*env*np.cos(q)/STRAIN_UNIT,np.ones(n)))
    disk=(s*L/LF % .026)>=.017
    slope=np.where(disk,27.914,16.845)/100
    intercept=np.where(disk,35.526,42.556)/100
    return s,x,slope,intercept

def angles(ab, min_theta_deg=.5):
    ab=np.asarray(ab)
    theta=np.rad2deg(np.linalg.norm(ab[:2])*L)
    phi=float(np.rad2deg(np.arctan2(ab[1],ab[0]))) if theta>=min_theta_deg else None
    return float(theta),phi
