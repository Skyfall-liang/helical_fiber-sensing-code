import json,math,socket,tempfile,threading,time,unittest
from pathlib import Path
from odisi_tcp import LineDecoder,ODISIReceiver

def packet(ch=1,data=None):
    return {'messageType':'measurement','channel':ch,'data':[1,None,-2.5] if data is None else data}

class Tests(unittest.TestCase):
    def test_every_byte_fragment_and_utf8(self):
        a=packet();a['name']='鍏夌氦 {test}'
        raw=('data: '+json.dumps(a,ensure_ascii=False)+'\r\n'+json.dumps(packet(2))+'\n').encode()
        d=LineDecoder();out=[]
        for b in raw:out.extend(d.feed(bytes([b])))
        self.assertEqual([v['channel'] for v in out],[1,2]);self.assertEqual(out[0]['name'],a['name'])
    def test_malformed_and_oversize_recovery(self):
        d=LineDecoder(200)
        out=d.feed(b'ACK\n{broken}\n'+b'x'*250+b'\n'+json.dumps(packet()).encode()+b'\n')
        self.assertEqual(len(out),1);self.assertEqual(d.stats['malformed_lines'],1)
        self.assertEqual(d.stats['oversized_lines'],1);self.assertEqual(d.stats['non_json_lines'],1)
    def test_dispatch_no_duplicate_callback_and_history(self):
        r=ODISIReceiver(history_size=2);calls=[];r.on_measurement=calls.append
        r.handle_message(packet())
        r.handle_message({'message type':'metadata','testName':'x'})
        r.handle_message({'message_type':'tare','channel':1,'data':[0]})
        r.handle_message(packet(data=[3]));r.handle_message(packet(data=[4,5]))
        self.assertEqual(len(calls),3);self.assertEqual(len(r.history[1]),2)
        self.assertTrue(math.isnan(calls[0].data[1]));self.assertEqual(r.latest[1].data,(4.,5.))
        self.assertEqual(r.metadata['testName'],'x');self.assertIn(1,r.tare)
    def test_invalid_and_filter(self):
        r=ODISIReceiver(channels=[2]);r.handle_message(packet(1));r.handle_message(packet(2,[[1]]))
        r.handle_message(packet(True));self.assertEqual(r.latest,{})
        self.assertEqual(r.stats['filtered_measurements'],1);self.assertEqual(r.stats['invalid_messages'],2)
    def test_snapshot_units_and_coordinates(self):
        r=ODISIReceiver();r.handle_message(packet(data=[8,10,12,14]))
        with tempfile.TemporaryDirectory() as t:
            info=r.snapshot(t,start_index=1,fiber_length=.2)
            lines=(Path(t)/'channel_1.csv').read_text().splitlines()
            self.assertEqual(lines,['s_m,strain','0.0,10.0','0.1,12.0','0.2,14.0'])
            self.assertFalse(info['synchronized'])
    def test_real_tcp_and_timeout_retains_partial(self):
        server=socket.socket();server.bind(('127.0.0.1',0));server.listen(1)
        port=server.getsockname()[1];commands=[];errors=[]
        def peer():
            try:
                with server:
                    conn,_=server.accept()
                    with conn:
                        conn.settimeout(3);commands.append(conn.recv(1024))
                        raw=(json.dumps(packet())+'\n'+json.dumps(packet(2))+'\n').encode()
                        conn.sendall(raw[:20]);time.sleep(.08);conn.sendall(raw[20:])
                        commands.append(conn.recv(1024))
            except BaseException as e: errors.append(e)
        thread=threading.Thread(target=peer,daemon=True);thread.start()
        r=ODISIReceiver('127.0.0.1',port,timeout=.02);got=[]
        try:
            r.connect();deadline=time.monotonic()+2
            while len(got)<2 and time.monotonic()<deadline:got.extend(r.poll())
        finally:r.close()
        thread.join(3)
        self.assertFalse(thread.is_alive());self.assertFalse(errors)
        self.assertEqual([m.channel for m in got],[1,2]);self.assertEqual(commands,[b'START\n',b'remote:stop\n'])
    def test_eof_truncated(self):
        a,b=socket.socketpair();r=ODISIReceiver();r.sock=a
        b.sendall(b'{"messageType":');b.close();self.assertEqual(r.poll(),[])
        with self.assertRaises(ConnectionError):r.poll()
        r.close();self.assertEqual(r.stats['truncated_lines'],1)
if __name__=='__main__':unittest.main()
