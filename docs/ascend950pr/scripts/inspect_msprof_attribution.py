"""Inspect what msprof actually attributes to each arm.

Qualification question for the timing basis: does the vendor call appear as one kernel or
several, and does the TileLang arm look like a single kernel? If the basis silently summed
only part of an arm, the comparison would be unfair to the other side.
"""
import json, torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof
from paired_bench import rms_norm_bf16

B, D = 4096, 4096
x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
kern = tilelang.compile(rms_norm_bf16(B, D), target="ascend", out_idx=[1])
cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")

def tiled():
    kern(x.flatten(), w, 1e-6)

def vendor():
    torch_npu.npu_rms_norm(x, w, 1e-6)

tiled(); vendor(); torch.npu.synchronize()
out = {"batch": B, "d": D}
for name, fn in (("tiled", tiled), ("vendor", vendor)):
    prof = bench_with_msprof(fn, cache, 20, detailed=True)
    out[name] = {
        "kernels": len(prof.kernels) if hasattr(prof, "kernels") else None,
        "dur_ns": getattr(prof, "dur_ns", None),
        "dur_us": round(prof.dur_ns / 1000, 3) if getattr(prof, "dur_ns", None) else None,
        "repr": str(prof)[:600],
    }
print("MS_DETAIL " + json.dumps(out))