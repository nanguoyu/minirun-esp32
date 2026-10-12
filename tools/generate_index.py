"""Regenerate the fixed C tensor index only from the SHA-256-pinned GGUF."""
import json, math, pathlib, struct
from fetch_lfm import verify, ROOT

def index(path):
    verify(path)
    with path.open('rb') as f:
        def read(n):
            data=f.read(n)
            if len(data)!=n:raise ValueError('short GGUF read')
            return data
        def unpack(fmt):return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
        def string():
            n=unpack('Q')
            if n>2**22:raise ValueError('oversized metadata string')
            return read(n).decode('utf-8')
        def value(t):
            if t==8:return string()
            if t==9:
                typ,n=unpack('I'),unpack('Q')
                if n>1000000:raise ValueError('oversized metadata array')
                return [value(typ) for _ in range(n)]
            return unpack({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
        if read(4)!=b'GGUF' or unpack('I')!=3:raise ValueError('not GGUF v3')
        nt,nk=unpack('Q'),unpack('Q')
        if nt>1000 or nk>1000:raise ValueError('unexpected census')
        metadata={}
        for _ in range(nk):key=string();metadata[key]=value(unpack('I'))
        tensors=[]
        for _ in range(nt):
            name=string();nd=unpack('I')
            if nd not in (1,2):raise ValueError('unsupported shape')
            dims=[unpack('Q') for _ in range(nd)]
            typ,offset=unpack('I'),unpack('Q')
            if typ not in (0,2,14):raise ValueError('unsupported quantization')
            block,bytes_per_block={0:(1,4),2:(32,18),14:(256,210)}[typ]
            if dims[0]%block:raise ValueError('partial quant block')
            tensors.append({'name':name,'dims':dims,'type':typ,'offset':offset})
        alignment=metadata.get('general.alignment',32)
        start=(f.tell()+alignment-1)//alignment*alignment
        for t in tensors:
            t['offset']+=start
            block,bpb={0:(1,4),2:(32,18),14:(256,210)}[t['type']]
            if t['offset']+math.prod(t['dims'])//block*bpb>path.stat().st_size:raise ValueError('tensor outside file')
    return tensors

if __name__=='__main__':
    pin=json.loads((ROOT/'research/lfm230-artifact.json').read_text())
    tensors=index(ROOT/'artifacts/model.gguf')
    text='#ifndef MODEL_INDEX_H\n#define MODEL_INDEX_H\n#define MODEL_BYTES '+str(pin['file']['size'])+'\n#define MODEL_SHA256 "'+pin['file']['lfs']['sha256']+'"\nstatic const Tensor tensors[] = {\n'
    for t in tensors:
        text+='{"%s", %d, %d, %d, %d},\n'%(t['name'],t['offset'],t['dims'][0],t['dims'][1] if len(t['dims'])>1 else 1,t['type'])
    text+='};\n#endif\n'
    target=ROOT/'src/model_index.h'
    if target.read_text()!=text:raise SystemExit('committed index differs from pinned artifact')
    print('pinned tensor index matches')
