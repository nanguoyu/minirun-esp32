"""Run the scalar host implementation only after pinned-artifact verification."""
import argparse, subprocess
from fetch_lfm import ROOT, verify, verify_aligned
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('tokens',type=int,nargs='+')
p.add_argument('--aligned',action='store_true')
p.add_argument('--output',default=str(ROOT/'build/logits.bin'))
a=p.parse_args()
if not 1<=len(a.tokens)<=128 or any(not 0<=t<65536 for t in a.tokens):p.error('expected 1..128 token IDs in 0..65535')
model=ROOT/('artifacts/model-aligned.gguf' if a.aligned else 'artifacts/model.gguf')
(verify_aligned if a.aligned else verify)(model)
subprocess.run([str(ROOT/('build/host-aligned' if a.aligned else 'build/host')),str(model),a.output]+list(map(str,a.tokens)),check=True)
