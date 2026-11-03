"""v34 vs v33: bit-identical check on both shapes, then paired timing."""
import importlib.util, os, re, statistics, tempfile
import torch, tilelang

V33 = "/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v33/kernel.py"
V34 = "/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v34/kernel.py"

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

m33, m34 = load(V33, "k33"), load(V34, "k34")
print("vcadd calls: v33=%d v34=%d" % (
    open(V33).read().count("T.simd.vcadd("), open(V34).read().count("T.simd.vcadd(")))

def build(m, batch, d):
    return tilelang.compile(m._build(batch, d), target="ascend", out_idx=[2])

print("\n=== BIT-IDENTICAL CHECK ===")
for batch, d in ((8192, 4096), (64, 4096)):
    torch.manual_seed(0)
    x = torch.randn(batch * d, dtype=torch.bfloat16, device="npu")
    w = torch.randn(d, dtype=torch.bfloat16, device="npu")
    k33, k34 = build(m33, batch, d), build(m34, batch, d)
    y33 = k33(x, w, 1e-6); torch.npu.synchronize()
    y34 = k34(x, w, 1e-6); torch.npu.synchronize()
    diff = (y33.float() - y34.float()).abs()
    print("  %dx%d: max_abs=%.3e mismatches=%d bit_identical=%s"
          % (batch, d, diff.max().item(), int((diff != 0).sum().item()),
             bool((diff == 0).all().item())))

def paired(batch, d, n_pairs, n_inner, label):
    torch.manual_seed(0)
    x = torch.randn(batch * d, dtype=torch.bfloat16, device="npu")
    w = torch.randn(d, dtype=torch.bfloat16, device="npu")
    k33, k34 = build(m33, batch, d), build(m34, batch, d)
    for _ in range(8):
        k33(x, w, 1e-6); k34(x, w, 1e-6)
    torch.npu.synchronize()
    def once(k):
        s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(n_inner): k(x, w, 1e-6)
        e.record(); torch.npu.synchronize()
        return s.elapsed_time(e) * 1000.0 / n_inner   # ms -> us
    ratios, a_s, b_s = [], [], []
    for i in range(n_pairs):
        if i % 2 == 0: ta, tb = once(k33), once(k34)
        else:          tb, ta = once(k34), once(k33)
        a_s.append(ta); b_s.append(tb); ratios.append(ta / tb)
    ratios.sort()
    med = statistics.median(ratios)
    import random; random.seed(11)
    boots = sorted(statistics.median([random.choice(ratios) for _ in ratios]) for _ in range(4000))
    lo, hi = boots[int(.025*len(boots))], boots[int(.975*len(boots))]
    print("\n=== %s %dx%d ===" % (label, batch, d))
    print("  v33 (30 vcadd) %.3f us | v34 (5 vcadd) %.3f us" % (statistics.median(a_s), statistics.median(b_s)))
    print("  per-pair v33/v34 median %.4f [%.4f, %.4f]  (%d pairs)" % (med, lo, hi, n_pairs))
    print("  v34 faster by %.2f%%   | pairs agreeing %d/%d" % ((med-1)*100, sum(1 for r_ in ratios if r_ > 1.0), len(ratios)))

paired(8192, 4096, 24, 20, "PREFILL")
paired(64, 4096, 24, 200, "DECODE")
