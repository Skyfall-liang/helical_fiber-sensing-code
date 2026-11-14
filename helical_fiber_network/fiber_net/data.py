"""Independent procedural scenes; no real curves used as training labels."""
from pathlib import Path
import json
import numpy as np
from .physics import geometry, PERIOD, LF, STRAIN_UNIT
KINDS=('pure_bending','bending_contact','pure_contact')

def generate(count, seed, n=1536, stress=False):
    rng=np.random.default_rng(seed); s,x,slope,intercept=geometry(n)
    ys=[]; coefs=[]; contacts=[]; supports=[]; forces=[]; kinds=[]; events=[]; nuisances=[]
    for i in range(count):
        kind=i%3
        # Most bending amplitudes resemble real samples (about 100 microstrain).
        kappa=0. if kind==2 else (rng.uniform(.12,1.0) if rng.random()<.8 else rng.uniform(1.,4.0))
        phi=rng.uniform(-np.pi,np.pi)
        coef=np.array([kappa*np.cos(phi),kappa*np.sin(phi),rng.uniform(-.12,.12)])
        support=np.zeros(n); force=np.zeros(n); scene=[]
        if kind!=0:
            centers=[rng.uniform(.45,LF-.45)]
            if rng.random()<.25:
                alternatives=np.linspace(.45,LF-.45,30)
                far=alternatives[np.abs(alternatives-centers[0])>.9]
                centers.append(rng.choice(far))
            for cluster,center in enumerate(centers):
                peaks=int(rng.integers(1,5))
                sign=1 if rng.random()<.85 else -1
                for j in range(peaks):
                    pos=center+(j-(peaks-1)/2)*PERIOD+rng.normal(0,.002)
                    width=rng.uniform(.008,.017) if not stress else rng.uniform(.005,.025)
                    f=rng.uniform(1.,3.)
                    g=np.exp(-.5*((s-pos)/width)**2)
                    replace=g>np.abs(support)
                    support[replace]=sign*g[replace]
                    force[replace]=f
                    scene.append(dict(s_m=float(pos),force_n=float(f),width_m=float(width),sign=sign,cluster_id=cluster))
        # Local strain calibration uses each fiber sample's disk/gap sensitivity.
        # Thus support peaks recover the exact paper peak-force relation.
        contact=support*(slope*force+intercept)
        bending=x@coef
        strength=2. if stress else 1.
        # Model mismatch is labeled nuisance, not contact or hidden bending truth.
        nuisance=strength*(rng.uniform(0,.12)*np.sin(2*np.pi*s/LF+rng.uniform(-np.pi,np.pi))
                +rng.uniform(0,.055)*np.sin(2*.999*s/.0170186+rng.uniform(-np.pi,np.pi)))
        nuisance+=strength*rng.uniform(-.025,.025)*s/LF*bending
        noise=rng.normal(0,rng.uniform(.005,.03)*strength,n)
        ys.append(bending+contact+nuisance+noise); coefs.append(coef)
        contacts.append(contact); supports.append(support); forces.append(force)
        kinds.append(kind); events.append(scene); nuisances.append(nuisance)
    arrays={key:np.asarray(val,dtype=np.float32) for key,val in
            dict(y=ys,coef=coefs,contact=contacts,support=supports,force=forces,nuisance=nuisances).items()}
    arrays['kind']=np.array(kinds,dtype=np.int64)
    return arrays,events

def save_dataset(folder, train=4096, val=512, test=768, n=1536):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    meta={'strain_unit':STRAIN_UNIT,'samples_per_frame':n,'kinds':KINDS,'splits':{},
          'contact_semantics':'per-fiber-peak effective normal force; cluster is a synthetic footprint, not an identified independent object',
          'real_data_use':'visual morphology reference only; no real frame or label is included in any split'}
    for split,count,seed,stress in [('train',train,1102,False),('val',val,2102,False),('test',test,3102,False),('stress',test,4102,True)]:
        arrays,events=generate(count,seed,n,stress)
        np.savez_compressed(folder/f'{split}.npz',**arrays)
        (folder/f'{split}_events.json').write_text(json.dumps(events),encoding='utf8')
        meta['splits'][split]={'count':count,'seed':seed,'stress':stress}
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    # One full-resolution CSV per class for inspection and external use.
    a,events=generate(3,5102,6395)
    for i,kind in enumerate(KINDS):
        np.savetxt(folder/f'example_{kind}.csv',np.column_stack((np.linspace(0,LF,6395),a['y'][i]*STRAIN_UNIT)),
                   delimiter=',',header='s_m,strain',comments='')
    (folder/'example_truth.json').write_text(json.dumps({'coef':a['coef'].tolist(),'events':events}),encoding='utf8')
