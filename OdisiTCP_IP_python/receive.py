"""Run the receiver; Ctrl+C saves the latest channel snapshots and closes TCP."""
import argparse,json,logging,time
from datetime import datetime
from pathlib import Path
from odisi_receiver import ODISIReceiver

class LivePlot:
    def __init__(self,unit):
        import matplotlib.pyplot as plt
        self.plt=plt;self.fig,self.ax=plt.subplots();self.lines={}
        self.ax.set(xlabel='Sensing point index (zero-based)',ylabel=unit,title='Luna ODiSI latest channel measurements (asynchronous)')
        self.ax.grid(alpha=.3);plt.ion();plt.show(block=False)
    def update(self,receiver):
        if not self.plt.fignum_exists(self.fig.number): return False
        for ch,m in receiver.latest.items():
            if ch not in self.lines:
                self.lines[ch],=self.ax.plot([],[],label=f'Channel {ch}');self.ax.legend()
            self.lines[ch].set_data(range(len(m.data)),m.data)
        self.ax.relim();self.ax.autoscale_view();self.fig.canvas.draw_idle();self.plt.pause(.001)
        return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='192.168.1.134');p.add_argument('--port',type=int,default=50000)
    p.add_argument('--duration',type=float,default=0,help='seconds; 0 = until Ctrl+C')
    p.add_argument('--channels',type=int,nargs='+',help='only keep these channels; default discover all')
    p.add_argument('--start-command',default='START');p.add_argument('--stop-command',default='remote:stop')
    p.add_argument('--plot',action='store_true');p.add_argument('--plot-hz',type=float,default=10)
    p.add_argument('--unit-label',default='Strain (microstrain; verify device setting)',help='display label only; no conversion')
    p.add_argument('--output',type=Path,help='new output directory; defaults to timestamped recordings directory')
    p.add_argument('--history-size',type=int,default=10)
    p.add_argument('--fiber-length',type=float,help='optional assumed sensing length for snapshot s_m coordinate')
    p.add_argument('--start-index',type=int,default=0,help='zero-based crop for snapshots only')
    a=p.parse_args()
    if a.duration<0 or a.plot_hz<=0 or a.start_index<0 or (a.fiber_length is not None and a.fiber_length<=0):
        p.error('invalid duration, plot rate, start index or fiber length')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    receiver=ODISIReceiver(a.host,a.port,start_command=a.start_command,stop_command=a.stop_command,
                           channels=a.channels,history_size=a.history_size)
    plot=LivePlot(a.unit_label) if a.plot else None
    output=a.output or Path(__file__).parent/'recordings'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();last_status=start;last_count=0;last_plot=0.;exit_code=0
    with (output/'messages.jsonl').open('w',encoding='utf-8') as log:
        receiver.on_message=lambda event:log.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        try:
            receiver.connect();start=time.monotonic();last_status=start
            while not a.duration or time.monotonic()-start<a.duration:
                wait=None if not a.duration else a.duration-(time.monotonic()-start)
                receiver.poll(wait)
                now=time.monotonic()
                if plot and now-last_plot>=1/a.plot_hz:
                    last_plot=now
                    if not plot.update(receiver): break
                if now-last_status>=1:
                    count=receiver.stats['measurements']
                    detail={ch:{'points':len(m.data),'age_s':round(now-m.received_monotonic,2)} for ch,m in receiver.latest.items()}
                    logging.info('Measurements=%d rate=%.1f msg/s channels=%s',count,(count-last_count)/(now-last_status),detail)
                    last_count=count;last_status=now;log.flush()
        except KeyboardInterrupt: logging.info('Stopped by user')
        except OSError as exc:
            logging.error('%s',exc);exit_code=1
        finally:
            receiver.close()
            receiver.snapshot(output/'latest',fiber_length=a.fiber_length,start_index=a.start_index)
            (output/'stats.json').write_text(json.dumps(dict(receiver.stats),indent=2),encoding='utf-8')
            logging.info('Saved to %s',output.resolve())
    return exit_code
if __name__=='__main__': raise SystemExit(main())
