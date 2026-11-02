import hashlib, importlib.util, json, statistics, time
from pathlib import Path
import torch, torch_npu
from tilelang.profiler.msprof import bench_with_msprof
root=Path('/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
paths={'simd':root/'rmsnorm_simdvf_v2/kernel.py','simt':root/'rmsnorm_cw4_guarded/kernel.py'}
mods={k:load(k,p) for k,p in paths.items()}
TBF=getattr(torch,'bfloat'+'16')
torch.manual_seed(714); x=torch.randn(8192,4096,device='npu',dtype=TBF); w=torch.randn(4096,device='npu',dtype=TBF)
for m in mods.values(): m.run(x,w); torch.npu.synchronize()
cache=torch.empty(64*1024*1024//4,dtype=torch.int32,device='npu')
rows=[]
for i in range(6):
 order=['simd','simt','simt','simd'] if i%2==0 else ['simt','simd','simd','simt']
 readings=[]
 for name in order:
  ms=float(bench_with_msprof(lambda:mods[name].run(x,w),cache,6))
  readings.append({'name':name,'ms':ms})
 grouped={name:statistics.mean(z['ms'] for z in readings if z['name']==name) for name in mods}
 rows.append({'round':i,'readings':readings,'simt_over_simd':grouped['simt']/grouped['simd']})
 print(json.dumps(rows[-1]),flush=True)
print(json.dumps({'sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},'median_ratio':statistics.median(r['simt_over_simd'] for r in rows),'limitations':'One process; no contention witness, speed screening only; not acceptance'}),flush=True)
