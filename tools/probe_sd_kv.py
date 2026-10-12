"""Real-device full-capacity FP32 SD KV test. Cache files are session-owned and deleted.
Build firmware with the requested context, KV_BITS=32, KV_FILE=ON; run test_kv_model first.
"""
import argparse,hashlib,json,pathlib,re,subprocess,time
from device import Device
from fetch_lfm import ROOT
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',required=True)
p.add_argument('--firmware-revision',required=True)
p.add_argument('--context',type=int,choices=[256,512,1024],default=256)
p.add_argument('--positions',type=int,help='Explicit partial exploration; defaults to full capacity')
p.add_argument('--reference-prefix',type=pathlib.Path,default=ROOT/'build/sd-kv/host32')
p.add_argument('--device-reference-dir',type=pathlib.Path,help='Compare overlapping checkpoints from an earlier device run')
p.add_argument('--output',type=pathlib.Path,default=ROOT/'build/sd-kv/device')
a=p.parse_args();limit=a.context if a.positions is None else a.positions
if not 1<=limit<=a.context:p.error('positions must be in 1..context')
a.output.mkdir(parents=True,exist_ok=True)
seq=[1,6,6423,708,22998,768,3290,4161,1337,768,15481,523,7,708,6,64015,708,
     1286,779,14587,4887,803,51957,522,52323,521,1695,25583,2644,1886,1264,576,4643,871]
report={'firmware_revision':a.firmware_revision,'tool_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'context':a.context,'tested_limit':limit,'full_capacity':limit==a.context,'kv_bits':32,'fixture':'repeated 34-token boundary fixture; not a language-quality benchmark',
        'sequence':seq,'batch_max':8,'checkpoints':[]}
d=Device(a.port)
def save(): (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
def logits(position):
    d.command('LOGITS');d.until('LOGITS 262144');data=bytearray();end=time.monotonic()+60
    while len(data)<262144 and time.monotonic()<end:data.extend(d.s.read(262144-len(data)))
    assert len(data)==262144;d.until('LOGITS_END');raw=bytes(data)
    (a.output/f'{position}.f32').write_bytes(raw)
    import struct,math
    x=struct.unpack('<65536f',raw);y=struct.unpack('<65536f',pathlib.Path(str(a.reference_prefix)+f'-{position}.f32').read_bytes())
    assert all(math.isfinite(v) for v in x)
    row={'position':position,'sha256':hashlib.sha256(raw).hexdigest(),
         'argmax_match':x.index(max(x))==y.index(max(y)),
         'max_abs':max(abs(u-v) for u,v in zip(x,y)),
         'rmse':math.sqrt(sum((u-v)**2 for u,v in zip(x,y))/len(x))}
    assert row['argmax_match'] and row['max_abs']<0.001,row
    # Same bytes as the earlier on-device RAM cache wherever a saved checkpoint exists.
    previous=ROOT/f'data/2026-09-28-context-kv/device-{position}.f32'
    if previous.exists():
        row['device_ram_bitwise']=raw==previous.read_bytes();assert row['device_ram_bitwise']
    if a.device_reference_dir:
        prior=a.device_reference_dir/f'{position}.f32'
        if prior.exists():
            row['prior_device_bitwise']=raw==prior.read_bytes()
            assert row['prior_device_bitwise']
    return row
try:
    d.command('INFO');report['info']=d.until('READY LFM230')
    assert f'context={a.context} ' in report['info'] and 'kv_backend=FILE' in report['info'] and 'kv_bits=32' in report['info']
    d.command('MODE PIPELINE');d.until('MODE_OK');d.command('OPEN');d.until('OPENED',600)
    shared=bool(re.search(r'kv_shared=1(?:\s|$)',report['info']))
    report['shared_kv_reads']=shared
    expected_read=0
    base=0
    for boundary in sorted({n for n in [8,9,16,17,64,65,128,129,256,257,511,512,513,1023,1024,limit] if n<=limit}):
        while base<boundary:
            n=min(8,boundary-base);ids=[seq[i%len(seq)] for i in range(base,base+n)]
            d.command(f'PHASE SD KV {base+1}-{base+n}');d.until('PHASE_OK')
            d.command('PREFILL '+str(int(base+n==boundary))+' '+' '.join(map(str,ids)))
            result=d.until('TOKEN ',900)
            assert int(re.search(r'position=(\d+)',result)[1])==base+n
            expected_read+=6*32768*((base+n-1)//8 if shared else sum(i//8 for i in range(base,base+n)))
            base+=n
        row=logits(base);row['result']=result
        d.command('KVINFO');row['cache']=d.until('KV backend=')
        expected_write=(base//8)*6*32768
        fields=dict(re.findall(r'(\w+)=(\S+)',row['cache']))
        assert int(fields['file_bytes'])==expected_write and int(fields['write_bytes'])==expected_write
        assert int(fields['read_bytes'])==expected_read
        report['checkpoints'].append(row);save();print('CHECKPOINT',json.dumps(row),flush=True)
    if limit==a.context:
        d.command('STEP 1 1')
        try:d.until('TOKEN ',10);raise AssertionError('overflow accepted')
        except RuntimeError as e:assert str(e)=='ERROR step'
        d.command('RESET');assert d.until('RESET_')=='RESET_NEEDS_OPEN'
        d.command('OPEN');d.until('OPENED',600)
        report['overflow_recovery']=True
    else:
        d.command('RESET');assert d.until('RESET_').startswith('RESET_OK')
        report['warm_reset']=True
    d.command('KVINFO');fresh=d.until('KV backend=');assert 'file_bytes=0' in fresh and 'read_bytes=0' in fresh and 'write_bytes=0' in fresh
    # Cancel a real model read, then verify no failed cache is reused.
    d.command('PREFILL 0 1 6 6423 708 22998 768 3290 4161')
    d.until('LAYER 2',120);d.s.write(b'!')
    try:d.until('TOKEN ',120);raise AssertionError('cancellation ignored')
    except RuntimeError as e:assert str(e)=='ERROR step'
    d.command('RESET');assert d.until('RESET_')=='RESET_NEEDS_OPEN'
    d.command('OPEN');d.until('OPENED',600)
    report['cancellation_recovery']=True
    d.command('KVCLOSE');d.until('KV_CLOSED');d.command('KVINFO');closed=d.until('KV backend=')
    assert 'file_bytes=-1' in closed and 'failed=1' in closed
    report['owned_cache_closed']=True;save()
    print(f'PASS SD KV tested_positions={limit} capacity={a.context} full={limit==a.context} outputs/recovery/cleanup',flush=True)
except KeyboardInterrupt:
    d.s.write(b'!');raise
finally:d.close()
