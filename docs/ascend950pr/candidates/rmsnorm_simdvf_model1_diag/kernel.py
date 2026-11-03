"""RMSNorm (bf16) on Ascend950PR, DIAGNOSTIC ONLY: minimal model with the second pass removed to test whether the simulator load metric is load-proportional. Output is WRONG by construction. (NORMAL_FV on every DMA path: best prefill and best decode so far).

Single-phase kernel (the only form that has been proven race-free) with the rows=4
fast path written out explicitly, so every row keeps its own accumulators and its own
scale in registers. Nothing is parked in shared memory between a reduce stage and a
scale stage, which is what made the v10 family racy.

Only the batch sizes that map to rows=4 are served by that path; rows<=2 fall back to
the loop form. Performance outside the certified decode shape is NOT established.
"""
import torch
import tilelang
import tilelang.ascend.language as T

BF = "bfloat16"
N_CORES = 1
CH = 64


def _rows_per_core(batch):
    for rows in (4, 2, 1):
        if batch > 0 and batch % (N_CORES * rows) == 0:
            return rows
    raise ValueError("batch must be divisible by 64; remainder handling is unsupported")


def _build(batch, d):
    rows = _rows_per_core(batch)
    if d <= 0 or d % (4 * CH):
        raise ValueError("width must be a positive multiple of 256")
    NCH = d // CH
    iters = batch // (N_CORES * rows)
    total = batch * d
    stages, yver = (4, 2) if rows > 1 else (2, 1)

    if rows == 4:
        @T.prim_func
        def main(X: T.Tensor((total,), BF), W: T.Tensor((d,), BF), Y: T.Tensor((total,), BF), eps: T.float32):
            with T.Kernel(N_CORES) as core:
                w_ub = T.alloc_shared((d,), BF)
                wf = T.alloc_shared((d,), "float32")
                x_ub = T.alloc_shared(rows * d, BF)
                y_ub = T.alloc_shared(rows * d, BF)
                T.annotate_buffer_versions({x_ub: stages, y_ub: yver})
                T.copy(W[:d], w_ub[:d])
                with T.SimdVF():
                    full = T.simd.pset(32)
                            T.simd.vsts(y_ub[r0], T.simd.vcvt(sc0, BF), full, dist="PK_B32")  # DIAGNOSTIC: pass 2 removed