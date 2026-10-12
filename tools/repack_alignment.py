import struct,pathlib,json,hashlib,sys,os
from fetch_lfm import verify, verify_aligned, ROOT
root=ROOT;src=root/'artifacts/model.gguf';verify(src)
f=src.open('rb')
def u(t):return struct.unpack('<'+t,f.read(struct.calcsize('<'+t)))[0]
def s():return f.read(u('Q')).decode()
def val(t):
 if t==8:return s()
 if t==9:
  typ,n=u('I'),u('Q');return [val(typ) for _ in range(n)]
 return u({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
def string(s):b=s.encode();return struct.pack('<Q',len(b))+b
assert f.read(4)==b'GGUF' and u('I')==3
nt,nk=u('Q'),u('Q');records=[];oldalignment=32
for _ in range(nk):
 start=f.tell();key=s();v=val(u('I'));end=f.tell();f.seek(start);raw=f.read(end-start)
 if key=='general.alignment':oldalignment=v
 else:records.append(raw)
records.append(string('general.alignment')+struct.pack('<II',4,4096));ts=[]
for _ in range(nt):
 name=s();dims=[u('Q') for _ in range(u('I'))];typ=u('I');off=u('Q');count=1
 for n in dims:count*=n
 block,size={0:(1,4),2:(32,18),14:(256,210)}[typ];nb=count//block*size
 ts.append(dict(name=name,dims=dims,type=typ,old_offset=off,bytes=nb))
oldstart=(f.tell()+oldalignment-1)//oldalignment*oldalignment
header=b'GGUF'+struct.pack('<IQQ',3,nt,len(records))+b''.join(records);offset=0
for t in ts:
 t['offset']=offset;header+=string(t['name'])+struct.pack('<I',len(t['dims']))+b''.join(struct.pack('<Q',x) for x in t['dims'])+struct.pack('<IQ',t['type'],offset)
 offset=(offset+t['bytes']+4095)//4096*4096
start=(len(header)+4095)//4096*4096
out=root/'artifacts/model-aligned.gguf'
if out.exists():
 verify_aligned(out);print('Verified existing aligned artifact:',out);raise SystemExit(0)
part=out.with_suffix('.gguf.part')
with part.open('xb') as g:
 g.write(header);g.write(bytes(start-len(header)))
 for t in ts:
  f.seek(oldstart+t['old_offset']);payload=f.read(t['bytes']);assert len(payload)==t['bytes']
  g.write(bytes(start+t['offset']-g.tell()));g.write(payload)
  t['sha256']=hashlib.sha256(payload).hexdigest();t['offset']+=start
 g.flush();os.fsync(g.fileno())
with part.open('rb') as g:
 for t in ts:
  g.seek(t['offset']);assert hashlib.sha256(g.read(t['bytes'])).hexdigest()==t['sha256']
report={'source_sha256':verify(src)['file']['lfs']['sha256'],'sha256':hashlib.sha256(part.read_bytes()).hexdigest(),'bytes':part.stat().st_size,'alignment':4096,'tensors':ts}
verify_aligned(part)
part.rename(out)
# The release manifest and tensor index are fixed; setup never edits source files.
print({k:v for k,v in report.items() if k!='tensors'})

header='#ifndef MODEL_INDEX_ALIGNED_H\n#define MODEL_INDEX_ALIGNED_H\n#define MODEL_BYTES '+str(report['bytes'])+'\n#define MODEL_SHA256 "'+report['sha256']+'"\nstatic const Tensor tensors[] = {\n'
for t in ts:
 header+='{"%s", %d, %d, %d, %d},\n'%(t['name'],t['offset'],t['dims'][0],t['dims'][1] if len(t['dims'])>1 else 1,t['type'])
header+='};\n#endif\n'
if (root/'src/model_index_aligned.h').read_text()!=header:
 raise RuntimeError('aligned tensor index differs from the pinned artifact')
