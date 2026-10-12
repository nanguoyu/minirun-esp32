"""USB terminal: ESP32 runs model forward passes; Mac handles tokens, rendering and control."""
import argparse, re, subprocess
from device import Device
from fetch_lfm import ROOT, verify

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--port',required=True)
p.add_argument('--prompt',default='Hello!')
p.add_argument('--new-tokens',type=int,default=8)
p.add_argument('--batch-size',type=int,choices=range(1,9),default=8)
p.add_argument('--reverify',action='store_true',help='Force a fresh full-file verification')
p.add_argument('--mode',choices=['PIPELINE','SERIAL'],default='PIPELINE')
a=p.parse_args()
model=ROOT/'artifacts/model.gguf';verify(model)
def show(text):
    frame=subprocess.run([str(ROOT/'build/render_text'),text],capture_output=True,check=True).stdout
    if len(frame)!=17280:raise RuntimeError('invalid LCD frame')
    d.command('FRAME');d.until('FRAME_READY')
    if d.s.write(frame)!=len(frame):raise IOError('short LCD frame write')
    d.until('FRAME_OK')
def vocab(mode,args):
    result=subprocess.run([str(ROOT/'build/reference'),str(model),'-',mode]+args,
                          capture_output=True)
    if mode=='--tokenize' and result.returncode==4:
        p.error('prompt exceeds the 1024-token tokenizer limit')
    result.check_returncode()
    return result.stdout.decode('utf-8',errors='replace')
# Exact plain single-user/no-tools branch of this pinned GGUF's chat template.
text='<|im_start|>user\n'+a.prompt+'<|im_end|>\n<|im_start|>assistant\n'
line=next(l for l in vocab('--tokenize',[text]).splitlines() if l.startswith('TOKENS'))
ids=list(map(int,line.split()[1:]))
if a.new_tokens<1:p.error('new-tokens must be positive')
print('prompt_tokens=',ids,flush=True)
d=Device(a.port);generated=[]
try:
    d.command('INFO');info=d.until('READY LFM230')
    match=re.search(r' context=(\d+)',info)
    capacity=int(match.group(1)) if match else 64  # Legacy firmware.
    if len(ids)+a.new_tokens-1>capacity:
        p.error(f'prompt plus generated positions exceeds the {capacity}-position budget')
    show('You: '+a.prompt+'\n\nThinking...')
    d.command('MODE '+a.mode);d.until('MODE_OK')
    reuse=False
    if not a.reverify:
        d.command('RESET');answer=d.line()
        reuse=answer.startswith('RESET_OK')
        if not reuse and answer!='RESET_NEEDS_OPEN':raise RuntimeError(answer)
    if not reuse:
        d.command('PHASE VERIFYING');d.until('PHASE_OK')
        d.command('OPEN');d.until('OPENED',600)
    for i in range(0,len(ids),a.batch_size):
        chunk=ids[i:i+a.batch_size]
        d.command(f'PHASE PREFILL {i+1}-{i+len(chunk)}');d.until('PHASE_OK')
        d.command('PREFILL '+str(int(i+len(chunk)==len(ids)))+' '+' '.join(map(str,chunk)))
        result=d.until('TOKEN ',900)
    for step in range(a.new_tokens):
        token=int(re.search(r'argmax=(\d+)',result).group(1))
        if token in (2,7):  # Both EOG tokens in the pinned GGUF.
            print('EOS',flush=True);break
        generated.append(token)
        answer=vocab('--decode',list(map(str,generated)))
        print('TEXT:',answer,flush=True)
        show('You: '+a.prompt+'\n\n'+answer)
        if step==a.new_tokens-1:break
        d.command(f'PHASE REPLY {step+2}/{a.new_tokens}');d.until('PHASE_OK')
        d.command(f'STEP {token} 1');result=d.until('TOKEN ',900)
    d.command('PHASE DONE');d.until('PHASE_OK')
except KeyboardInterrupt:
    d.s.write(b'!');raise
finally:d.close()
