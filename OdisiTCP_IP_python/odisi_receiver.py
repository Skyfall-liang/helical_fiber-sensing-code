"""Luna ODiSI LF-delimited JSON TCP receiver, ported from this repo's MATLAB.
Core acquisition requires Python standard library only. No inferred channel sync.
"""
from __future__ import annotations
import csv
import json
import logging
import math
import socket
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

LOG = logging.getLogger(__name__)

def key(name):
    return ''.join(c for c in name.lower() if c.isalnum())

def fields(message):
    return {key(k): v for k, v in message.items()}

def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value

class LineDecoder:
    """Retains partial TCP lines. Oversized lines are discarded through next LF."""
    def __init__(self, max_line_bytes=8*1024*1024):
        if max_line_bytes < 1: raise ValueError('max_line_bytes must be positive')
        self.limit = max_line_bytes
        self.buffer = bytearray()
        self.discarding = False
        self.stats = Counter()

    def feed(self, chunk: bytes):
        output = []
        for i, part in enumerate(chunk.split(b'\n')):
            if i:
                if not self.discarding and self.buffer:
                    try:
                        text = self.buffer.decode('utf-8').strip()
                        start = text.find('{')
                        if start < 0:
                            self.stats['non_json_lines'] += 1
                        else:
                            obj, _ = json.JSONDecoder().raw_decode(text[start:])
                            if not isinstance(obj, dict): raise ValueError('object required')
                            output.append(obj)
                    except (UnicodeError, ValueError):
                        self.stats['malformed_lines'] += 1
                self.buffer.clear()
                self.discarding = False
            if not self.discarding:
                if len(self.buffer)+len(part) > self.limit:
                    self.buffer.clear(); self.discarding = True
                    self.stats['oversized_lines'] += 1
                else:
                    self.buffer.extend(part)
        return output

@dataclass(frozen=True)
class Measurement:
    channel: int
    data: tuple[float, ...]
    received_at: str
    received_monotonic: float
    message_number: int  # local counter, NOT the instrument's synchronized frame ID
    device_timestamp: object = None

class ODISIReceiver:
    def __init__(self, host='192.168.1.134', port=50000, *,
                 start_command='START', stop_command='remote:stop',
                 channels=None, history_size=10, timeout=.2, connect_timeout=5.,
                 max_line_bytes=8*1024*1024):
        if not 1 <= port <= 65535 or timeout <= 0 or connect_timeout <= 0 or history_size < 0:
            raise ValueError('invalid port, timeout or history_size')
        if any('\n' in x or '\r' in x for x in (start_command,stop_command)):
            raise ValueError('commands must not contain line breaks')
        if channels is not None and (not channels or any(type(c) is not int or c<1 for c in channels)):
            raise ValueError('channels must contain positive integers')
        self.host, self.port = host, port
        self.start_command, self.stop_command = start_command, stop_command
        self.channels = set(channels) if channels is not None else None
        self.history_size, self.timeout, self.connect_timeout = history_size, timeout, connect_timeout
        self.max_line_bytes = max_line_bytes
        self.decoder = LineDecoder(max_line_bytes)
        self.sock = None
        self.latest = {}
        self.history = {}
        self.metadata = {}
        self.tare = {}
        self.stats = Counter()
        self.connection_number = 0
        self.on_measurement: Callable[[Measurement], None] | None = None
        self.on_message: Callable[[dict], None] | None = None

    def connect(self):
        if self.sock is not None: raise RuntimeError('already connected')
        sock = socket.create_connection((self.host,self.port),timeout=self.connect_timeout)
        try:
            sock.settimeout(self.timeout)
            if self.start_command: sock.sendall((self.start_command+'\n').encode('utf-8'))
        except BaseException:
            sock.close(); raise
        self.sock = sock
        # Do not mix cached frames/partial bytes between independent TCP sessions.
        self.decoder = LineDecoder(self.max_line_bytes)
        self.latest.clear(); self.history.clear(); self.metadata.clear(); self.tare.clear()
        self.connection_number += 1
        LOG.info('Connected to %s:%s',self.host,self.port)

    def handle_message(self, message):
        now = datetime.now(timezone.utc).isoformat()
        f = fields(message)
        kind = str(f.get('messagetype','')).strip().lower()
        self.stats['messages'] += 1
        # Preserve metadata and original field names for firmware-specific processing.
        if self.on_message:
            self.on_message(dict(received_at=now,connection_number=self.connection_number,message=json_safe(message)))
        if kind == 'metadata':
            self.metadata = message
            self.stats['metadata'] += 1
            return None
        if kind not in ('measurement','tare'):
            self.stats['other_messages'] += 1
            return None
        channel = f.get('channel')
        try:
            if isinstance(channel,bool): raise ValueError()
            integer = int(channel)
            if float(channel) != integer or integer < 1: raise ValueError()
            channel = integer
        except (ValueError,TypeError,OverflowError):
            self.stats['invalid_messages'] += 1; return None
        if kind == 'tare':
            self.tare[channel] = message
            self.stats['tare'] += 1
            return None  # retained, not automatically subtracted a second time
        if self.channels is not None and channel not in self.channels:
            self.stats['filtered_measurements'] += 1; return None
        values = f.get('data')
        try:
            if not isinstance(values,list) or not values: raise ValueError('nonempty numeric data array required')
            if any(isinstance(v,(list,dict,bool)) for v in values): raise ValueError('flat numeric array required')
            data = tuple(float(v) if v is not None else math.nan for v in values)
            data = tuple(v if math.isfinite(v) else math.nan for v in data)
        except (ValueError,TypeError,OverflowError):
            self.stats['invalid_messages'] += 1; return None
        self.stats['measurements'] += 1
        measurement = Measurement(channel,data,now,time.monotonic(),self.stats['measurements'],
                                  f.get('timestamp',f.get('time')))
        self.latest[channel] = measurement
        self.history.setdefault(channel,deque(maxlen=self.history_size)).append(measurement)
        if self.on_measurement: self.on_measurement(measurement)
        return measurement

    def poll(self, wait=None):
        """Receive one TCP chunk, emit every complete measurement, at most timeout wait.
        EOF raises ConnectionError. Callback exceptions deliberately propagate.
        """
        if self.sock is None: raise RuntimeError('connect first')
        self.sock.settimeout(self.timeout if wait is None else max(.001,min(wait,self.timeout)))
        try: chunk = self.sock.recv(65536)
        except socket.timeout: return []
        if not chunk:
            if self.decoder.buffer or self.decoder.discarding:
                self.stats['truncated_lines'] += 1
            raise ConnectionError('ODiSI closed the TCP connection')
        self.stats['bytes_received'] += len(chunk)
        before = self.decoder.stats.copy()
        messages = self.decoder.feed(chunk)
        self.stats.update(self.decoder.stats-before)
        return [m for msg in messages if (m := self.handle_message(msg)) is not None]

    def snapshot(self, directory, *, fiber_length=None, start_index=0):
        """Each CSV is a channel's latest measurement; these are NOT synchronized.
        Without explicit fiber_length, exports index,strain; never invent coordinates.
        """
        if fiber_length is not None and (not math.isfinite(fiber_length) or fiber_length<=0):
            raise ValueError('fiber_length must be finite and positive')
        if start_index < 0: raise ValueError('start_index must be nonnegative (Python zero-based)')
        directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
        index = {'synchronized':False,'unit':'as received; see metadata/device configuration',
                 'metadata':json_safe(self.metadata),'channels':{}}
        for channel,m in sorted(self.latest.items()):
            data=m.data[start_index:]
            if not data: continue
            filename=f'channel_{channel}.csv'
            with (directory/filename).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream)
                writer.writerow(['s_m' if fiber_length is not None else 'index','strain'])
                for i,value in enumerate(data):
                    coord=fiber_length*i/(len(data)-1) if fiber_length is not None and len(data)>1 else (0 if fiber_length is not None else i+start_index)
                    writer.writerow([coord,value])
            index['channels'][str(channel)]={'file':filename,'received_at':m.received_at,
                 'device_timestamp':json_safe(m.device_timestamp),'message_number':m.message_number,
                 'points':len(data),'start_index':start_index,'assumed_uniform_fiber_length_m':fiber_length}
        (directory/'snapshot.json').write_text(json.dumps(index,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        return index

    def close(self, send_stop=True):
        sock,self.sock=self.sock,None
        if sock is not None:
            try:
                if send_stop and self.stop_command:
                    sock.sendall((self.stop_command+'\n').encode('utf-8'))
            except OSError: pass
            finally: sock.close()

    def __enter__(self): self.connect(); return self
    def __exit__(self,*exc): self.close()
