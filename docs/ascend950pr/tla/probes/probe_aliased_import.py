"""Can the SOURCE tla tree be made importable as triton.language.extra.tla,
and if so, does compilation actually work against the installed libtriton?

This distinguishes "wrong sys.path" (fixable) from "missing native builder"
(a different wheel). No pip install, no modification of the env.
"""
import sys, types, traceback, os
sys.path.insert(0, "/home/s00653124/workspace/triton-ascend/third_party/ascend")

import torch
import torch_npu
import triton, triton.language as tl

# Alias the on-disk source tree into the import path the wheel hardcodes.
import language.tla as _tla_src
SRC = "/home/s00653124/workspace/triton-ascend/third_party/ascend/language/tla"
pkg = types.ModuleType("triton.language.extra.tla")
pkg.__path__ = [SRC]
pkg.__package__ = "triton.language.extra.tla"
sys.modules["triton.language.extra.tla"] = pkg

import importlib
for sub in ("compiler", "jit", "core", "dma", "simd", "semantic" if False else "scope",
            "alloc", "copy", "view", "aux_ops", "math", "debug", "sync", "cube",
            "dispatch", "builder", "code_generator", "simd_semantic"):
    try:
        m = importlib.import_module("triton.language.extra.tla." + sub)
        setattr(pkg, sub, m)
    except Exception as e:
        print("SUBMODULE_FAIL", sub, type(e).__name__, str(e)[:160])

for name in ("alloc_buffer", "scope", "copy", "view", "simd", "jit",
             "annotate_buffer_versions", "EvictionPolicy", "buffer_type"):
    if hasattr(_tla_src, name):
        setattr(pkg, name, getattr(_tla_src, name))
print("STEP alias: OK, triton.language.extra.tla is now the source tree")

try:
    from triton.language.extra import tla
    print("STEP import_tla: OK ->", getattr(tla, "__file__", "<aliased pkg>"))
except Exception as e:
    print("STEP import_tla: FAIL", type(e).__name__, str(e)[:200]); raise SystemExit(0)

@tla.jit
def probe(X, Y, N: tl.constexpr, D: tl.constexpr):
    x_ub = tla.alloc_buffer((1, D), tl.bfloat16, space="ub", versions=2)
    y_ub = tla.alloc_buffer((1, D), tl.bfloat16, space="ub", versions=2)
    row = tl.program_id(0)
    src = tla.view(X, (1, D), strides=(D, 1), offsets=(row, 0))
    dst = tla.view(Y, (1, D), strides=(D, 1), offsets=(row, 0))
    tla.copy(src, x_ub)
    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0, 0], dist="UNPK_B16")
        vf = tla.simd.vextf(v, tl.float32, part="part_even")
        ss = tla.simd.vreduce_sum(tla.simd.vmul(vf, vf))
        b = tla.simd.vdupv(ss, pos=tla.simd.POS.LOWEST)
        sc = tla.simd.vdiv(tla.simd.vdup(1.0, tl.float32), tla.simd.vsqrt(b))
        tla.simd.vsts(y_ub[0, 0], tla.simd.vmul(vf, sc), dist="PK_B32")
    tla.copy(y_ub, dst)
print("STEP jit_decorate: OK")

x = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
y = torch.empty(4, 4096, dtype=torch.bfloat16, device="npu")
try:
    k = probe.warmup(x, y, N=4, D=4096, grid=(4,))
    print("STEP compile_warmup: OK")
except Exception as e:
    print("STEP compile_warmup: FAIL", type(e).__name__)
    print(str(e)[:600])
    tb = traceback.format_exc()
    print("--- tail ---")
    print(tb[-1200:])
