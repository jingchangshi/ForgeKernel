"""Paired, alternating A/B timing of the redundant vcadd chain in v33.

Correctness is already settled: collapsing the 5 chains (30 vcadd -> 5) is
bit-identical over 33.5M elements. So this measures ONLY what the redundancy
costs, using the project's discipline: alternate A and B within one process so
drift hits both arms equally, take many pairs, and report the median per-pair
ratio with a bootstrap interval. A single A-then-B reading is not admissible
here because the effect is around 1%.

A = shipped kernel (30 vcadd, 6-call chains)
B = collapsed kernel (5 vcadd, 1 call each)
"""
import importlib.util, os, re, statistics, tempfile
import torch, tilelang

SRC = "/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v33/kernel.py"
src = open(SRC).read()

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
pat = re.compile(
    r"( *)a1 = T\.simd\.vcadd\(acc, full\); a2 = T\.simd\.vcadd\(a1, full\); a3 = T\.simd\.vcadd\(a2, full\)\n"
    r" *a4 = T\.simd\.vcadd\(a3, full\); a5 = T\.simd\.vcadd\(a4, full\); a6 = T\.simd\.vcadd\(a5, full\)\n")
patched, n = pat.subn(lambda m: "%sa1 = T.simd.vcadd(acc, full)\n%sa6 = a1\n" % (m.group(1), m.group(1)), patched, count=1)
total += n
assert total == 5, total
print("collapsed %d chains: vcadd %d -> %d" % (total, src.count("T.simd.vcadd("), patched.count("T.simd.vcadd(")))


def load(text, name):
    fd, p = tempfile.mkstemp(suffix=".py"); os.close(fd)
    open(p, "w").write(text)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def bench(batch, d, n_pairs, n_inner, label):
    torch.manual_seed(0)
    x = torch.randn(batch * d, dtype=torch.bfloat16, device="npu")
    w = torch.randn(d, dtype=torch.bfloat16, device="npu")
    ka = tilelang.compile(load(src, "vA")._build(batch, d), target="ascend", out_idx=[2])
    kb = tilelang.compile(load(patched, "vB")._build(batch, d), target="ascend", out_idx=[2])

    # Warm both, so JIT and allocation are not inside the measurement.
    for _ in range(8):
        ka(x, w, 1e-6); kb(x, w, 1e-6)
    torch.npu.synchronize()

    def once(k):
        s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(n_inner):
            k(x, w, 1e-6)
        e.record(); torch.npu.synchronize()
        # elapsed_time returns MILLISECONDS; scale to microseconds here so every
        # number this script prints is in us and cannot be mislabelled later.
        return s.elapsed_time(e) * 1000.0 / n_inner

    ratios, ta_s, tb_s = [], [], []
    for i in range(n_pairs):
        if i % 2 == 0:                      # alternate which arm goes first
            ta = once(ka); tb = once(kb)
        else:
            tb = once(kb); ta = once(ka)
        ta_s.append(ta); tb_s.append(tb)
        ratios.append(ta / tb)

    ratios.sort()
    med = statistics.median(ratios)
    # Bootstrap the median for a distribution-free interval.
    import random
    random.seed(11)
    boots = []
    for _ in range(4000):
        sample = [random.choice(ratios) for _ in ratios]
        boots.append(statistics.median(sample))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    print("\n=== %s  batch=%d d=%d ===" % (label, batch, d))
    print("  A (30 vcadd) median : %.4f us" % statistics.median(ta_s))
    print("  B ( 5 vcadd) median : %.4f us" % statistics.median(tb_s))
    print("  per-pair A/B median : %.4f  [%.4f, %.4f]  (%d pairs)" % (med, lo, hi, n_pairs))
    print("  A is slower by      : %.2f%%" % ((med - 1.0) * 100.0))
    slow = sum(1 for r in ratios if r > 1.0)
    print("  pairs where A slower: %d/%d" % (slow, len(ratios)))
    return med, lo, hi, ratios


if __name__ == "__main__":
    # Prefill family (certified shape) and decode family.
    bench(8192, 4096, 24, 20, "PREFILL")
    bench(64, 4096, 24, 200, "DECODE")
