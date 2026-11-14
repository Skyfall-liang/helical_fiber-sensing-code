"""Local-only simulator. Emits prefixed JSON, null, fragmented and joined lines."""
import argparse,json,math,socket,time

def serve(host='127.0.0.1',port=50001,frames=100,hz=10.,points=2000):
    with socket.socket() as server:
        server.bind((host,port));server.listen(1)
        print(f'Mock listening on {host}:{server.getsockname()[1]}',flush=True)
        conn,_=server.accept()
        with conn:
            conn.settimeout(5)
            # No makefile wrapper: avoid read buffering swallowing later commands.
            command=bytearray()
            while not command.endswith(b'\n'):
                b=conn.recv(1)
                if not b:return
                command.extend(b)
                if len(command)>1024:raise ValueError('command too long')
            print('Received:',command.decode().strip(),flush=True)
            conn.sendall((json.dumps({'messageType':'metadata','testName':'Python mock',
                         'sensors':[{'channel':1,'sensorName':'Mock1'},{'channel':2,'sensorName':'Mock2'}]})+'\n').encode())
            try:
                for frame in range(frames):
                    payload=[]
                    for ch in (1,2):
                        data=[round(80*math.sin(j*.08+frame*.03+ch)*abs(math.sin(j*.006)),3) for j in range(points)]
                        data[3]=None
                        msg={'message type':'measurement','channel':ch,'timestamp':frame/hz,'data':data}
                        payload.append('data: '+json.dumps(msg)+'\r\n')
                    packet=''.join(payload).encode()
                    conn.sendall(packet[:17]);conn.sendall(packet[17:])
                    time.sleep(1/hz)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError): pass
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=50001)
    p.add_argument('--frames',type=int,default=100);p.add_argument('--hz',type=float,default=10.)
    a=p.parse_args()
    if a.frames<1 or a.hz<=0:p.error('frames and hz must be positive')
    serve(a.host,a.port,a.frames,a.hz)
