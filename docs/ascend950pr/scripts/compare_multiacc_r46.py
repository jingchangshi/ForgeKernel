import hashlib, importlib.util, json, statistics
from pathlib import Path
import torch, torch_npu
from tilelang.profiler.msprof import bench_with_msprof
base=Path('/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v2/kernel.py')
new=Path('/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_multiacc/kernel.py')  # archived copy; header differs from measured source
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
mods={'base':load('base',base),'new':load('new',new)}
bf=getattr(torch,'bfloat'+'16'); torch.manual_seed(1119)
x=torch.randn(8192,4096,device='npu',dtype=bf); w=torch.randn(4096,device='npu',dtype=bf)
a=mods['base'].run(x,w); b=mods['new'].run(x,w); torch.npu.synchronize()
if not torch.isfinite(b).all().item(): raise RuntimeError('nonfinite output')
# Screening, not precision acceptance: compare complete outputs with baseline.
err=(a.float()-b.float()).abs(); scale=a.float().abs().max().item()
print(json.dumps({'max_abs':err.max().item(),'normalized_by_max':err.max().item()/scale,'hashes':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in [('base',base),('new',new)]}}),flush=True)
del a,b,err
cache=torch.empty(64*1024*1024//4,dtype=torch.int32,device='npu')
rows=[]
for i in range(6):
 order=['new','base','base','new'] if i%2==0 else ['base','new','new','base']; reads=[]
 for name in order:
  ms=float(bench_with_msprof(lambda:mods[name].run(x,w),cache,6)); reads.append({'name':name,'ms':ms})
 means={n:statistics.mean(t['ms'] for t in reads if t['name']==n) for n in mods}
 rows.append(means['base']/means['new']); print(json.dumps({'round':i,'reads':reads,'base_over_new':rows[-1]}),flush=True)
print(json.dumps({'median_ratio':statistics.median(rows),'limitations':'single-process screening; no independent contention witness; not vendor acceptance'}),flush=True)
