"""Compare scalar streaming logits with the independent llama.cpp CPU backend."""
import struct, math, subprocess, pathlib, json, sys
from fetch_lfm import verify
root=pathlib.Path(__file__).resolve().parents[1]
p=root/'build'
model=root/'artifacts/model.gguf'
verify(model)
report=[]
for prompt in ['<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n','<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n','<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n']:
 r=subprocess.run([str(p/'reference'),str(model),str(p/'reference.bin'),'--text',prompt],capture_output=True,text=True,errors='replace',check=True)
 ids=next(l for l in r.stdout.splitlines() if l.startswith('TOKENS')).split()[1:]
 subprocess.run([str(p/'host'),str(model),str(p/'logits.bin')]+ids,check=True,capture_output=True)
 a=struct.unpack('<65536f',(p/'reference.bin').read_bytes());b=struct.unpack('<65536f',(p/'logits.bin').read_bytes())
 rmse=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a));cos=sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
 row={'prompt':prompt,'tokens':list(map(int,ids)),'reference_argmax':max(range(len(a)),key=a.__getitem__),'streaming_argmax':max(range(len(b)),key=b.__getitem__),'rmse':rmse,'max_abs':max(abs(x-y) for x,y in zip(a,b)),'cosine':cos};report.append(row);print(json.dumps(row),flush=True)
 # Different CPU matmul arithmetic: compatibility gate, not bit equality.
 assert row['reference_argmax']==row['streaming_argmax'] and cos>0.999 and rmse<0.15
(p/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
