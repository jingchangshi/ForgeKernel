"""Precision: rel_l2 of v33 and v34 against float64 truth. They must be equal."""
import importlib.util, torch, tilelang

V33 = "/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v33/kernel.py"
V34 = "/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v34/kernel.py"

def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m

def rel_l2(y, t):
    return (torch.norm(y.float() - t.float()) / torch.norm(t.float())).item()

for batch, d in ((8192, 4096), (64, 4096), (256, 4096)):
    torch.manual_seed(3)
    x = torch.randn(batch * d, dtype=torch.bfloat16, device="npu")
    w = torch.randn(d, dtype=torch.bfloat16, device="npu")
    # fp64 truth from the same inputs
    xh = x.cpu().double().view(batch, d); wh = w.cpu().double()
    truth = ((xh * torch.rsqrt((xh*xh).mean(1, keepdim=True) + 1e-6)) * wh).view(-1)
    out = {}
    for name, path in (("v33", V33), ("v34", V34)):
        m = load(path, name)
        k = tilelang.compile(m._build(batch, d), target="ascend", out_idx=[2])
        y = k(x, w, 1e-6); torch.npu.synchronize()
        out[name] = rel_l2(y.cpu(), truth)
    print("%6dx%-5d  rel_l2 v33=%.6e  v34=%.6e  equal=%s"
          % (batch, d, out["v33"], out["v34"], out["v33"] == out["v34"]))
