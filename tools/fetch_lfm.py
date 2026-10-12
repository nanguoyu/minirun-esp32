"""Fetch the pinned official artifact, atomically, after size and SHA-256 checks."""
import hashlib, json, os, pathlib, sys, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
def verify(path):
 pin=json.loads((ROOT/'models/lfm230-artifact.json').read_text());f=pin['file']
 if path.stat().st_size!=f['size']:raise ValueError('wrong model size')
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 if h.hexdigest()!=f['lfs']['sha256']:raise ValueError('wrong model SHA-256')
 return pin
if __name__=='__main__':
 out=ROOT/'artifacts/model.gguf';out.parent.mkdir(exist_ok=True)
 if out.exists():verify(out);print(out);sys.exit(0)
 pin=json.loads((ROOT/'models/lfm230-artifact.json').read_text())
 url='https://huggingface.co/'+pin['repo']+'/resolve/'+pin['revision']+'/'+pin['file']['rfilename']
 part=out.with_suffix('.gguf.part')
 with urllib.request.urlopen(url,timeout=60) as src,part.open('xb') as dst:
  total=0
  while block:=src.read(1<<20):
   total+=len(block)
   if total>pin['file']['size']:raise ValueError('oversized response')
   dst.write(block)
  dst.flush();os.fsync(dst.fileno())
 verify(part);part.rename(out);print(out)

def verify_aligned(path):
    pin=json.loads((ROOT/'models/lfm230-aligned-artifact.json').read_text())
    source=json.loads((ROOT/'models/lfm230-artifact.json').read_text())
    if pin['source_sha256']!=source['file']['lfs']['sha256']:raise ValueError('wrong source revision')
    if path.stat().st_size!=pin['bytes']:raise ValueError('wrong aligned size')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    if h.hexdigest()!=pin['sha256']:raise ValueError('wrong aligned SHA-256')
    return pin
