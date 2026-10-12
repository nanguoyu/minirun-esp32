"""USB client. Text tokenization stays on Mac; STEP executes the model on ESP32."""
import argparse, os, pathlib, re, struct, time
from serial.tools import list_ports
import serial
from fetch_lfm import ROOT, verify, verify_aligned

def resolve_port(port=None):
    if port or os.environ.get('MINIRUN_PORT'):
        return port or os.environ['MINIRUN_PORT']
    matches = [p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x303A, 0x1001)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError('No ESP32-S3 USB serial device found. Connect the board, or pass --port PORT.')
    raise ValueError('More than one ESP32-S3 is connected. Choose one with --port PORT.')

class Device:
    def __init__(self, port=None, verbose=True, on_line=None):
        self.verbose=verbose
        self.on_line=on_line
        self.s=serial.Serial(resolve_port(port),115200,timeout=1,write_timeout=30)
    def line(self, deadline=120):
        end=time.monotonic()+deadline
        while time.monotonic()<end:
            line=self.s.readline()
            if line:
                text=line.decode(errors='replace').strip()
                if self.verbose:print(text,flush=True)
                if self.on_line:self.on_line(text)
                if text.startswith('ERROR'):raise RuntimeError(text)
                return text
        raise TimeoutError('device silent')
    def until(self, prefix, deadline=120):
        end=time.monotonic()+deadline
        while time.monotonic()<end:
            text=self.line(max(1,end-time.monotonic()))
            if text.startswith(prefix):return text
        raise TimeoutError(prefix)
    def command(self, command):self.s.write((command+'\n').encode())
    def close(self):self.s.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',help='USB serial port; detected automatically when one ESP32-S3 is connected')
    p.add_argument('--aligned',action='store_true',help='Upload the derived 4096-byte-aligned GGUF')
    p.add_argument('action',choices=['info','upload','run','test-tx'])
    p.add_argument('--tokens',type=int,nargs='+')
    p.add_argument('--batch-size',type=int,choices=range(1,9),default=1)
    p.add_argument('--mode',choices=['SERIAL','PIPELINE'],default='PIPELINE')
    p.add_argument('--output',type=pathlib.Path,default=ROOT/'build/device-logits.bin')
    a=p.parse_args()
    if a.action=='run' and (not a.tokens or len(a.tokens)>1024 or any(t<0 or t>=65536 for t in a.tokens)):
        p.error('run needs 1..1024 token IDs, each in 0..65535')
    try:d=Device(a.port)
    except (ValueError,serial.SerialException) as error:p.exit(1,str(error)+'\n')
    try:
        d.command('INFO');info=d.until('READY LFM230')
        capacity_match=re.search(r' context=(\d+)',info)
        capacity=int(capacity_match.group(1)) if capacity_match else 64
        if a.action=='run' and len(a.tokens)>capacity:
            p.error(f'device context capacity is {capacity}')
        if a.action=='upload':
            model=ROOT/('artifacts/model-aligned.gguf' if a.aligned else 'artifacts/model.gguf')
            (verify_aligned if a.aligned else verify)(model)
            d.command('PUT');d.until('READY 16384')
            start=time.monotonic();done=0
            with model.open('rb') as f:
                while block:=f.read(16384):
                    if d.s.write(block)!=len(block):raise IOError('short serial write')
                    # Do not print thousands of acknowledgements.
                    ack=d.s.readline()
                    if ack.strip()!=b'ACK':raise IOError(repr(ack))
                    done+=len(block)
                    if done%1048576==0:print(f'upload_bytes={done} seconds={time.monotonic()-start:.1f}',flush=True)
            d.until('UPLOADED')
        elif a.action=='test-tx':
            d.command('TEST_TX');d.until('TEST_TX 262144')
            data=bytearray();end=time.monotonic()+30
            while len(data)<262144 and time.monotonic()<end:
                data.extend(d.s.read(262144-len(data)))
            expected=bytes(i%251 for i in range(32768))*8
            if bytes(data)!=expected:raise IOError('binary transfer pattern mismatch')
            d.until('TEST_TX_END');print('PASS: all 262144 bytes match')
        elif a.action=='run':
            d.command('MODE '+a.mode);d.until('MODE_OK')
            d.command('OPEN');d.until('OPENED',600)
            for i in range(0,len(a.tokens),a.batch_size):
                chunk=a.tokens[i:i+a.batch_size];head=int(i+len(chunk)==len(a.tokens))
                if a.batch_size==1:d.command(f'STEP {chunk[0]} {head}')
                else:d.command('PREFILL '+str(head)+' '+' '.join(map(str,chunk)))
                d.until('TOKEN ',900)
            d.command('LOGITS');d.until('LOGITS 262144')
            data=bytearray();end=time.monotonic()+60
            while len(data)<262144 and time.monotonic()<end:
                data.extend(d.s.read(262144-len(data)))
            if len(data)!=262144:raise IOError('short logits')
            d.until('LOGITS_END')
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(data)
            print('saved',a.output)
    except KeyboardInterrupt:
        d.s.write(b'!');raise
    finally:d.close()
