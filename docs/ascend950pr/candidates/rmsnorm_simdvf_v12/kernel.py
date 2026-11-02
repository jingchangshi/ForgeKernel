"""RMSNorm (bf16) on Ascend950PR, TileLang-Ascend SimdVF, v12 (two-phase, eight accumulation chains).

Two-phase schedule: every row's sum-of-squares runs back to back (phase A), the
per-row inverse RMS is parked in a small shared buffer, and only then does phase B
apply the scale and the weight. The point is not fewer instructions - the per-element
count is unchanged - but that the reduction chain latency of one row no longer sits
between the vector work of two other rows.

Core-contiguous partitioning is kept from v5. Performance outside the certified
decode shape is NOT established.
"""
import torch
import tilelang
import tilelang.ascend.language as T

BF = "bfloat16"
N_CORES = 64
CH = 64
NACC = 8


def _plan(batch):
    for rows in (4, 2, 1):
        if batch > 0 and batch % (N_CORES * rows) == 0:
            iters = batch // (N_CORES * rows)
            if rows == 1:
                return rows, 2, 1, iters
            return rows, 3, 2, iters
    raise ValueError("batch must be divisible by 64; remainder handling is unsupported")


def _build(batch, d):
    rows, stages, yver, iters = _plan(batch)
    if d <= 0 or d % (NACC * CH):
        raise ValueError("width must be a positive multiple of 256")
    NCH = d // CH
    total = batch * d

    @T.prim_func
    def main(X: T.Tensor((total,), BF), W: T.Tensor((d,), BF), Y: T.Tensor((total,), BF), eps: T.float32):
        with T.Kernel(N_CORES) as core:
            w_ub = T.alloc_shared((d,), BF)
            wf = T.alloc_shared((d,), "float32")
            rs_ub = T.alloc_shared((rows * CH,), "float32")
            x_ub = T.alloc_shared(rows * d, BF)
            y_ub = T.alloc_shared(rows * d, BF)
            T.annotate_buffer_versions({x_ub: stages, y_ub: yver})
            T.copy(W[:d], w_ub[:d])
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in T.serial(NCH):
                    T.simd.vsts(wf[i * CH],
                                T.simd.vcvt(T.simd.vld(w_ub[i * CH], dist="UNPK_B16"), "float32", part=0),
                                full)
            for step in T.Pipelined(iters, num_stages=stages):
                base = ((core * iters + step) * rows) * d
                T.copy(X[base:base + rows * d], x_ub)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    # phase A: reductions for every row, back to back
                    for r in T.serial(rows):
                        ro = r * d
                        acc0 = T.simd.alloc_var(T.float32); acc0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc1 = T.simd.alloc_var(T.float32); acc1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc2 = T.simd.alloc_var(T.float32); acc2 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc3 = T.simd.alloc_var(T.float32); acc3 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc4 = T.simd.alloc_var(T.float32); acc4 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc5 = T.simd.alloc_var(T.float32); acc5 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc6 = T.simd.alloc_var(T.float32); acc6 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc7 = T.simd.alloc_var(T.float32); acc7 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH // NACC):
                            v0 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 0) * CH], dist="UNPK_B16"), "float32", part=0)
                            v1 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                            v2 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 2) * CH], dist="UNPK_B16"), "float32", part=0)
                            v3 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 3) * CH], dist="UNPK_B16"), "float32", part=0)
                            v4 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 4) * CH], dist="UNPK_B16"), "float32", part=0)
                            v5 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 5) * CH], dist="UNPK_B16"), "float32", part=0)
                            v6 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 6) * CH], dist="UNPK_B16"), "float32", part=0)
                            v7 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 8 + 7) * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc0, v0, v0, full)
                            T.simd.vmula(acc1, v1, v1, full)
                            T.simd.vmula(acc2, v2, v2, full)
                            T.simd.vmula(acc3, v3, v3, full)
                            T.simd.vmula(acc4, v4, v4, full)
                            T.simd.vmula(acc5, v5, v5, full)
                            T.simd.vmula(acc6, v6, v6, full)
                            T.simd.vmula(acc7, v7, v7, full)
                        acc = T.simd.vadd(T.simd.vadd(T.simd.vadd(acc0, acc1, full), T.simd.vadd(acc2, acc3, full), full), T.simd.vadd(T.simd.vadd(acc4, acc5, full), T.simd.vadd(acc6, acc7, full), full), full)
                        a1 = T.simd.vcadd(acc, full); a2 = T.simd.vcadd(a1, full); a3 = T.simd.vcadd(a2, full)
                        a4 = T.simd.vcadd(a3, full); a5 = T.simd.vcadd(a4, full); a6 = T.simd.vcadd(a5, full)
                        b = T.simd.vdupv(a6, full, pos="POS_LOWEST")
                        rs = T.simd.vdiv(
                            T.simd.vdup(T.float32(1.0), "float32", full),
                            T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b, T.float32(1.0 / d), full), eps, full), full),
                            full)
                        T.simd.vsts(rs_ub[r * CH], rs, full)
                    # phase B: scale and weight, every row back to back
                    for r in T.serial(rows):
                        ro = r * d
                        rsv = T.simd.vld(rs_ub[r * CH])
                        for i in T.serial(NCH):
                            vf = T.simd.vcvt(T.simd.vld(x_ub[ro + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wv = T.simd.vld(wf[i * CH])
                            yv = T.simd.vmul(T.simd.vmul(vf, rsv, full), wv, full)
                            T.simd.vsts(y_ub[ro + i * CH], T.simd.vcvt(yv, BF), full, dist="PK_B32")
                T.copy(y_ub, Y[base:base + rows * d])

    return main


_KERNELS = {}


def run(x, w, eps=1e-6):
    batch, d = x.shape
    key = (batch, d)
    if key not in _KERNELS:
        _KERNELS[key] = tilelang.compile(_build(batch, d), target="ascend", out_idx=[2])
    y = _KERNELS[key](x.flatten(), w, eps)
    return y.reshape(batch, d)
