"""Small shared loop for user-supplied model callbacks."""
import argparse,json,time
from threading import Event, Lock, Thread
from .receiver import ODISIReceiver,json_safe

def parser(description, *, require_fiber_length=True):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--host',default='192.168.1.134');p.add_argument('--port',type=int,default=50000)
    p.add_argument('--channel',type=int,default=1);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--start-index',type=int,default=0);p.add_argument('--stop-index',type=int,default=None)
    p.add_argument('--fiber-length',type=float,required=require_fiber_length,help='confirmed length of the cropped uniformly sampled sensing span (m)')
    p.add_argument('--unit',choices=['microstrain','strain'],required=True)
    p.add_argument('--start-command',default='START');p.add_argument('--stop-command',default='remote:stop')
    return p

def emit_result(measurement, result, *, received_at=None):
    if result is not None:
        print(json.dumps(json_safe(dict(channel=measurement.channel, received_at=received_at or measurement.received_at,
            device_timestamp=measurement.device_timestamp, result=result)),
            ensure_ascii=False, allow_nan=False), flush=True)


def run_processor(args,process,*,idle=None,on_message=None,channels=None,latest_interval=None):
    """Callback receives one Measurement; returns a JSON-serializable result or None.
    Synchronous: heavy inference can lag acquisition. No silent drop or auto-reconnect.
    """
    if args.duration<0 or args.start_index<0 or (args.fiber_length is not None and args.fiber_length<=0):
        raise ValueError('invalid duration, crop or fiber length')
    if args.stop_index is not None and args.stop_index<=args.start_index:
        raise ValueError('stop-index must exceed start-index')
    with ODISIReceiver(args.host,args.port,channels=[args.channel] if channels is None else channels,
                       start_command=args.start_command,stop_command=args.stop_command) as receiver:
        receiver.on_message = on_message
        if latest_interval is not None:
            return _run_latest(receiver, args.duration, process, idle, latest_interval)
        receiver.on_measurement = lambda measurement: emit_result(measurement, process(measurement))
        start=time.monotonic()
        try:
            while not args.duration or time.monotonic()-start<args.duration:
                remaining=None if not args.duration else args.duration-(time.monotonic()-start)
                receiver.poll(remaining)
                if idle is not None and not idle():
                    break
        except KeyboardInterrupt:pass


def _run_latest(receiver, duration, process, idle, interval):
    """Drain TCP independently and fit only the newest measurement per channel."""
    pending = {}
    lock = Lock()
    stopped = Event()
    errors = []

    def enqueue(measurement):
        with lock:
            pending[measurement.channel] = measurement

    def read():
        try:
            while not stopped.is_set():
                receiver.poll()
        except Exception as exc:
            errors.append(exc)
            stopped.set()

    receiver.on_measurement = enqueue
    worker = Thread(target=read)
    worker.start()
    start = time.monotonic()
    next_fit = start
    try:
        while not duration or time.monotonic() - start < duration:
            now = time.monotonic()
            if now >= next_fit or stopped.is_set():
                with lock:
                    batch = list(pending.values())
                    pending.clear()
                for measurement in batch:
                    emit_result(measurement, process(measurement))
                next_fit = now + interval
            if idle is not None and not idle():
                break
            if stopped.wait(.02):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        worker.join()
    if errors:
        raise errors[0]
