"""Real-time theoretical decoupling using the shared odisi_tcp package."""
import json,logging
from pathlib import Path
import numpy as np
from odisi_tcp.stream import parser,run_processor
from helical_fiber import Config,HelicalFiberModel

def main():
    p=parser(__doc__);p.add_argument('--config',type=Path,default=Path(__file__).parent/'config/segment1.json')
    args=p.parse_args();model=HelicalFiberModel(Config(**json.loads(args.config.read_text(encoding='utf-8'))))
    if not np.isclose(args.fiber_length,model.config.fiber_length_m):
        p.error('--fiber-length must match the model config')
    def process(m):
        y=np.asarray(m.data[args.start_index:args.stop_index],dtype=float)
        if len(y)<10:
            logging.warning('Skipped channel %d: fewer than 10 cropped samples',m.channel);return
        if args.unit=='microstrain':y=y*1e-6
        s=np.linspace(0,args.fiber_length,len(y))
        try:r=model.fit(s,y)
        except ValueError as e:
            logging.warning('Skipped measurement: %s',e);return
        return dict(summary=r['summary'],contacts=r['contacts'])
    run_processor(args,process)
if __name__=='__main__':main()
