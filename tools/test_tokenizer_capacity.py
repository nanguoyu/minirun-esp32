"""Exercise native tokenizer capacity without running a neural forward pass."""
import subprocess
from fetch_lfm import ROOT, verify
model=ROOT/'artifacts/model.gguf';verify(model)
for words,expected_code in [(511,0),(1023,0),(1024,4)]:
    p=subprocess.run([str(ROOT/'build/reference'),str(model),'-','--tokenize',' the'*words],capture_output=True,text=True,errors='replace')
    assert p.returncode==expected_code,(words,p.returncode,p.stderr[-500:])
    if expected_code==0:
        ids=list(map(int,next(x for x in p.stdout.splitlines() if x.startswith('TOKENS')).split()[1:]))
        assert len(ids)==words+1 and ids[0]==1,(words,len(ids))
    else:assert '1024-token helper limit' in p.stderr
print('PASS tokenizer accepts 512/1024 tokens and explicitly rejects 1025')
