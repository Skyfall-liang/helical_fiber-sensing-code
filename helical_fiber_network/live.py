"""Real-time learned shape/contact inference using shared odisi_tcp acquisition."""
import logging
from pathlib import Path
import numpy as np
import torch
from odisi_tcp.stream import parser,run_processor
from fiber_net.inference import load_model,prepare_arrays,predict,extract_contacts
from fiber_net.physics import LF,angles

def main():
    p=parser(__doc__);p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'runs/joint/best.pt')
    args=p.parse_args()
    if not np.isclose(args.fiber_length,LF):p.error('--fiber-length must match the trained geometry (4.277 m)')
    torch.set_num_threads(4);model,_=load_model(args.checkpoint)
    def process(m):
        y=np.asarray(m.data[args.start_index:args.stop_index],dtype=float)
        s=np.linspace(0,args.fiber_length,len(y))
        try:s,y=prepare_arrays(s,y,model.n,args.unit)
        except ValueError as e:
            logging.warning('Skipped measurement: %s',e);return
        pred=predict(model,y);theta,phi=angles(pred['coef'])
        return dict(theta_deg=theta,phi_deg=phi,contacts=extract_contacts(pred,s),
            reconstruction_rmse_microstrain=float(np.sqrt(np.mean((y-pred['reconstruction'])**2))*100))
    run_processor(args,process)
if __name__=='__main__':main()
