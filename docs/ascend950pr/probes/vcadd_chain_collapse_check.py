"""Does collapsing every 6-call vcadd chain in v33 to ONE call change the output?

The probe established vcadd is a FULL reduction into lane 0 and is idempotent:
calls 1,2,3,6 all return the same lane-0 value. If that holds, the 5 chains in
kernel.py (30 vcadd calls total) are numerically redundant and collapsing them
must leave the output bit-identical.

This is the correctness half of the operator's challenge. It measures max_abs
between the shipped kernel and a chain-collapsed variant on the same inputs.
"""
import importlib.util, os, re, tempfile, time
import torch, tilelang

SRC = "/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v33/kernel.py"
src = open(SRC).read()
print("T.simd.vcadd in source:", src.count("T.simd.vcadd("))

# Each chain is written as exactly two source lines:  a{r}_1..a{r}_3 then a{r}_4..a{r}_6
# Replace with one call and an alias so the later vdupv(a{r}_6) still resolves.
patched = src
total = 0
for r in range(8):
    pat = re.compile(
        r"( *)a%d_1 = T\.simd\.vcadd\(acc%d, full\); a%d_2 = T\.simd\.vcadd\(a%d_1, full\); a%d_3 = T\.simd\.vcadd\(a%d_2, full\)\n"
        r" *a%d_4 = T\.simd\.vcadd\(a%d_3, full\); a%d_5 = T\.simd\.vcadd\(a%d_4, full\); a%d_6 = T\.simd\.vcadd\(a%d_5, full\)\n"
        % (r, r, r, r, r, r, r, r, r, r, r, r))
    patched, n = pat.subn(
        lambda m: "%sa%d_1 = T.simd.vcadd(acc%d, full)\n%sa%d_6 = a%d_1\n" % (m.group(1), r, r, m.group(1), r, r),
        patched, count=1)
    total += n
    print("  chain in acc%d: %s" % (r, "collapsed" if n else "not found"))

# The decode branch (rows==1) writes its chain without a row prefix: a1..a6.
pat = re.compile(
    r"( *)a1 = T\.simd\.vcadd\(acc, full\); a2 = T\.simd\.vcadd\(a1, full\); a3 = T\.simd\.vcadd\(a2, full\)\n"
    r" *a4 = T\.simd\.vcadd\(a3, full\); a5 = T\.simd\.vcadd\(a4, full\); a6 = T\.simd\.vcadd\(a5, full\)\n")
patched, n = pat.subn(
    lambda m: "%sa1 = T.simd.vcadd(acc, full)\n%sa6 = a1\n" % (m.group(1), m.group(1)),
    patched, count=1)
total += n
print("  chain in decode branch (a1..a6):", "collapsed" if n else "not found")

print("total chains collapsed:", total)
print("T.simd.vcadd in patched:", patched.count("T.simd.vcadd("))
assert total == 5, "expected 5 chains, found %d" % total


def load(text, name):
    fd, p = tempfile.mkstemp(suffix=".py"); os.close(fd)
    open(p, "w").write(text)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run(module, rows_kernel, batch, d, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch * d, dtype=torch.bfloat16, device="npu")
    w = torch.randn(d, dtype=torch.bfloat16, device="npu")
    prog = module._build(batch, d)
    k = tilelang.compile(prog, target="ascend", out_idx=[2])
    y = k(x, w, 1e-6)          # out_idx=[2]: Y is produced, not passed
    torch.npu.synchronize()
    return y.float().cpu()


# rows==4 path is the certified prefill shape family: batch 8192, d 4096.
BATCH, D = 8192, 4096
print("\n=== building both variants (batch=%d d=%d) ===" % (BATCH, D))
orig = load(src, "orig_v33")
coll = load(patched, "coll_v33")

print("=== running ORIGINAL (30 vcadd) ===")
t0 = time.time(); y_orig = run(orig, 4, BATCH, D); t_orig = time.time() - t0
print("  %.1fs" % t_orig)
print("=== running COLLAPSED (5 vcadd) ===")
t0 = time.time(); y_coll = run(coll, 4, BATCH, D); t_coll = time.time() - t0
print("  %.1fs" % t_coll)

diff = (y_orig - y_coll).abs()
print("\n=== output comparison (bf16) ===")
print("  max_abs      :", diff.max().item())
print("  numel        :", diff.numel())
print("  mismatches   :", int((diff != 0).sum().item()))
print("  bit-identical:", bool((diff == 0).all().item()))

# A CPU reference so a shared-wrong-answer cannot masquerade as agreement.
# The INPUTS must be the ones the device actually consumed: NPU and CPU RNGs do
# not agree, so re-seeding on the host would compare against different data.
torch.manual_seed(0)
xh = torch.randn(BATCH * D, dtype=torch.bfloat16, device="npu").cpu()
wh = torch.randn(D, dtype=torch.bfloat16, device="npu").cpu()
xh32 = xh.float().view(BATCH, D); wh32 = wh.float()
ref = (xh32 * torch.rsqrt((xh32 * xh32).mean(dim=1, keepdim=True) + 1e-6)) * wh32
ref = ref.view(-1).to(torch.bfloat16)

prog = orig._build(BATCH, D)
k = tilelang.compile(prog, target="ascend", out_idx=[2])
y_dev = k(xh.to("npu"), wh.to("npu"), 1e-6); torch.npu.synchronize()
d_ref = (y_dev.float().cpu() - ref.float()).abs()
print("\n=== vs independent CPU float32 reference (same inputs) ===")
print("  max_abs :", d_ref.max().item())
print("  mean_abs:", d_ref.mean().item())
