"""Attempt a real TLA compile using the source package against the installed wheel.
Records the exact failure, if any. Tiny capability probe, not a v33 port.
"""
import os, sys, traceback
sys.path.insert(0, "/home/s00653124/workspace/triton-ascend/third_party/ascend")
import torch
import torch_npu  # explicit: TORCH_DEVICE_BACKEND_AUTOLOAD=0 disables the auto-load
import triton, triton.language as tl

stage = {"where": None}
try:
    import language.tla as tla
    stage["where"] = "import language.tla OK (python layer only)"
    print("STEP import_tla: OK")
except Exception as e:
    print("STEP import_tla: FAIL", type(e).__name__, str(e)[:200])
    raise SystemExit(0)

try:
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
    stage["where"] = "decorator/jit OK"
    print("STEP jit_decorate: OK")
    print("STAGE:", stage["where"])
except Exception as e:
    print("STEP jit_decorate: FAIL", type(e).__name__)
    traceback.print_exc()
    raise SystemExit(0)

try:
    c = probe.warmup(torch.empty((4, 4096), dtype=torch.bfloat16, device="npu"),
                     torch.empty((4, 4096), dtype=torch.bfloat16, device="npu"),
                     N=4, D=4096, grid=(4,))
    print("STEP compile: OK")
except Exception as e:
    print("STEP compile: FAIL", type(e).__name__, str(e)[:400])
    tb = traceback.format_exc()
    print(tb[-1500:])
