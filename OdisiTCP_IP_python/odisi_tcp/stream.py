"""Small shared loop for user-supplied model callbacks."""
import argparse,json,time
from .receiver import ODISIReceiver,json_safe

def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--host',default='192.168.1.134');p.add_argument('--port',type=int,default=50000)
    p.add_argument('--channel',type=int,default=1);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--start-index',type=int,default=0);p.add_argument('--stop-index',type=int,default=None)
    p.add_argument('--fiber-length',type=float,required=True,help='confirmed length of the cropped uniformly sampled sensing span (m)')
    p.add_argument('--unit',choices=['microstrain','strain'],required=True)
    p.add_argument('--start-command',default='START');p.add_argument('--stop-command',default='remote:stop')
    return p

def run_processor(args,process):
    """Callback receives one Measurement; returns a JSON-serializable result or None.
    Synchronous: heavy inference can lag acquisition. No silent drop or auto-reconnect.
    """
    if args.duration<0 or args.start_index<0 or args.fiber_length<=0:
        raise ValueError('invalid duration, crop or fiber length')
    if args.stop_index is not None and args.stop_index<=args.start_index:
        raise ValueError('stop-index must exceed start-index')
    with ODISIReceiver(args.host,args.port,channels=[args.channel],
                       start_command=args.start_command,stop_command=args.stop_command) as receiver:
        start=time.monotonic()
        try:
            while not args.duration or time.monotonic()-start<args.duration:
                remaining=None if not args.duration else args.duration-(time.monotonic()-start)
                for m in receiver.poll(remaining):
                    result=process(m)
                    if result is not None:
                        print(json.dumps(json_safe(dict(channel=m.channel,received_at=m.received_at,
                            device_timestamp=m.device_timestamp,result=result)),ensure_ascii=False,allow_nan=False),flush=True)
        except KeyboardInterrupt:pass
