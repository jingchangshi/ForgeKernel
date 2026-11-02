"""RMSNorm (bf16) on Ascend950PR, TileLang-Ascend SimdVF, v24 (NORMAL_FV on every DMA path: best prefill and best decode so far).

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
N_CORES = 64
CH = 64


def _rows_per_core(batch):
    for rows in (8, 4, 2, 1):
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
                    for i in T.serial(NCH):
                        T.simd.vsts(wf[i * CH], T.simd.vcvt(T.simd.vld(w_ub[i * CH], dist="UNPK_B16"), "float32", part=0), full)
                for step in T.Pipelined(iters, num_stages=stages):
                    base = ((core * iters + step) * rows) * d
                    T.copy(X[base:base + rows * d], x_ub, l2_cache_ctrl="NORMAL_FV")
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        r0 = 0 * d
                        acc0_0 = T.simd.alloc_var(T.float32); acc0_0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc0_1 = T.simd.alloc_var(T.float32); acc0_1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc0_2 = T.simd.alloc_var(T.float32); acc0_2 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc0_3 = T.simd.alloc_var(T.float32); acc0_3 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH // 4):
                            v0_0 = T.simd.vcvt(T.simd.vld(x_ub[r0 + (i * 4 + 0) * CH], dist="UNPK_B16"), "float32", part=0)
                            v0_1 = T.simd.vcvt(T.simd.vld(x_ub[r0 + (i * 4 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                            v0_2 = T.simd.vcvt(T.simd.vld(x_ub[r0 + (i * 4 + 2) * CH], dist="UNPK_B16"), "float32", part=0)
                            v0_3 = T.simd.vcvt(T.simd.vld(x_ub[r0 + (i * 4 + 3) * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc0_0, v0_0, v0_0, full)
                            T.simd.vmula(acc0_1, v0_1, v0_1, full)
                            T.simd.vmula(acc0_2, v0_2, v0_2, full)
                            T.simd.vmula(acc0_3, v0_3, v0_3, full)
                        acc0 = T.simd.vadd(T.simd.vadd(acc0_0, acc0_1, full), T.simd.vadd(acc0_2, acc0_3, full), full)
                        a0_1 = T.simd.vcadd(acc0, full); a0_2 = T.simd.vcadd(a0_1, full); a0_3 = T.simd.vcadd(a0_2, full)
                        a0_4 = T.simd.vcadd(a0_3, full); a0_5 = T.simd.vcadd(a0_4, full); a0_6 = T.simd.vcadd(a0_5, full)
                        b0 = T.simd.vdupv(a0_6, full, pos="POS_LOWEST")
                        sc0 = T.simd.alloc_var(T.float32)
                        sc0 = T.simd.vdiv(T.simd.vdup(T.float32(1.0), "float32", full), T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b0, T.float32(1.0 / d), full), eps, full), full), full)
                        r1 = 1 * d
                        acc1_0 = T.simd.alloc_var(T.float32); acc1_0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc1_1 = T.simd.alloc_var(T.float32); acc1_1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc1_2 = T.simd.alloc_var(T.float32); acc1_2 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc1_3 = T.simd.alloc_var(T.float32); acc1_3 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH // 4):
                            v1_0 = T.simd.vcvt(T.simd.vld(x_ub[r1 + (i * 4 + 0) * CH], dist="UNPK_B16"), "float32", part=0)
                            v1_1 = T.simd.vcvt(T.simd.vld(x_ub[r1 + (i * 4 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                            v1_2 = T.simd.vcvt(T.simd.vld(x_ub[r1 + (i * 4 + 2) * CH], dist="UNPK_B16"), "float32", part=0)
                            v1_3 = T.simd.vcvt(T.simd.vld(x_ub[r1 + (i * 4 + 3) * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc1_0, v1_0, v1_0, full)
                            T.simd.vmula(acc1_1, v1_1, v1_1, full)
                            T.simd.vmula(acc1_2, v1_2, v1_2, full)
                            T.simd.vmula(acc1_3, v1_3, v1_3, full)
                        acc1 = T.simd.vadd(T.simd.vadd(acc1_0, acc1_1, full), T.simd.vadd(acc1_2, acc1_3, full), full)
                        a1_1 = T.simd.vcadd(acc1, full); a1_2 = T.simd.vcadd(a1_1, full); a1_3 = T.simd.vcadd(a1_2, full)
                        a1_4 = T.simd.vcadd(a1_3, full); a1_5 = T.simd.vcadd(a1_4, full); a1_6 = T.simd.vcadd(a1_5, full)
                        b1 = T.simd.vdupv(a1_6, full, pos="POS_LOWEST")
                        sc1 = T.simd.alloc_var(T.float32)
                        sc1 = T.simd.vdiv(T.simd.vdup(T.float32(1.0), "float32", full), T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b1, T.float32(1.0 / d), full), eps, full), full), full)
                        r2 = 2 * d
                        acc2_0 = T.simd.alloc_var(T.float32); acc2_0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc2_1 = T.simd.alloc_var(T.float32); acc2_1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc2_2 = T.simd.alloc_var(T.float32); acc2_2 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc2_3 = T.simd.alloc_var(T.float32); acc2_3 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH // 4):
                            v2_0 = T.simd.vcvt(T.simd.vld(x_ub[r2 + (i * 4 + 0) * CH], dist="UNPK_B16"), "float32", part=0)
                            v2_1 = T.simd.vcvt(T.simd.vld(x_ub[r2 + (i * 4 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                            v2_2 = T.simd.vcvt(T.simd.vld(x_ub[r2 + (i * 4 + 2) * CH], dist="UNPK_B16"), "float32", part=0)
                            v2_3 = T.simd.vcvt(T.simd.vld(x_ub[r2 + (i * 4 + 3) * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc2_0, v2_0, v2_0, full)
                            T.simd.vmula(acc2_1, v2_1, v2_1, full)
                            T.simd.vmula(acc2_2, v2_2, v2_2, full)
                            T.simd.vmula(acc2_3, v2_3, v2_3, full)
                        acc2 = T.simd.vadd(T.simd.vadd(acc2_0, acc2_1, full), T.simd.vadd(acc2_2, acc2_3, full), full)
                        a2_1 = T.simd.vcadd(acc2, full); a2_2 = T.simd.vcadd(a2_1, full); a2_3 = T.simd.vcadd(a2_2, full)
                        a2_4 = T.simd.vcadd(a2_3, full); a2_5 = T.simd.vcadd(a2_4, full); a2_6 = T.simd.vcadd(a2_5, full)
                        b2 = T.simd.vdupv(a2_6, full, pos="POS_LOWEST")
                        sc2 = T.simd.alloc_var(T.float32)
                        sc2 = T.simd.vdiv(T.simd.vdup(T.float32(1.0), "float32", full), T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b2, T.float32(1.0 / d), full), eps, full), full), full)
                        r3 = 3 * d
                        acc3_0 = T.simd.alloc_var(T.float32); acc3_0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc3_1 = T.simd.alloc_var(T.float32); acc3_1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc3_2 = T.simd.alloc_var(T.float32); acc3_2 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc3_3 = T.simd.alloc_var(T.float32); acc3_3 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH // 4):
                            v3_0 = T.simd.vcvt(T.simd.vld(x_ub[r3 + (i * 4 + 0) * CH], dist="UNPK_B16"), "float32", part=0)
                            v3_1 = T.simd.vcvt(T.simd.vld(x_ub[r3 + (i * 4 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                            v3_2 = T.simd.vcvt(T.simd.vld(x_ub[r3 + (i * 4 + 2) * CH], dist="UNPK_B16"), "float32", part=0)
                            v3_3 = T.simd.vcvt(T.simd.vld(x_ub[r3 + (i * 4 + 3) * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc3_0, v3_0, v3_0, full)
                            T.simd.vmula(acc3_1, v3_1, v3_1, full)
                            T.simd.vmula(acc3_2, v3_2, v3_2, full)
                            T.simd.vmula(acc3_3, v3_3, v3_3, full)
                        acc3 = T.simd.vadd(T.simd.vadd(acc3_0, acc3_1, full), T.simd.vadd(acc3_2, acc3_3, full), full)
                        a3_1 = T.simd.vcadd(acc3, full); a3_2 = T.simd.vcadd(a3_1, full); a3_3 = T.simd.vcadd(a3_2, full)
                        a3_4 = T.simd.vcadd(a3_3, full); a3_5 = T.simd.vcadd(a3_4, full); a3_6 = T.simd.vcadd(a3_5, full)
                        b3 = T.simd.vdupv(a3_6, full, pos="POS_LOWEST")
                        sc3 = T.simd.alloc_var(T.float32)
                        sc3 = T.simd.vdiv(T.simd.vdup(T.float32(1.0), "float32", full), T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b3, T.float32(1.0 / d), full), eps, full), full), full)
                        for i in T.serial(NCH):
                            vf0 = T.simd.vcvt(T.simd.vld(x_ub[r0 + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wv0 = T.simd.vld(wf[i * CH])
                            yv0 = T.simd.vmul(T.simd.vmul(vf0, sc0, full), wv0, full)
                            T.simd.vsts(y_ub[r0 + i * CH], T.simd.vcvt(yv0, BF), full, dist="PK_B32")
                        for i in T.serial(NCH):
                            vf1 = T.simd.vcvt(T.simd.vld(x_ub[r1 + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wv1 = T.simd.vld(wf[i * CH])
                            yv1 = T.simd.vmul(T.simd.vmul(vf1, sc1, full), wv1, full)
                            T.simd.vsts(y_ub[r1 + i * CH], T.simd.vcvt(yv1, BF), full, dist="PK_B32")
                        for i in T.serial(NCH):
                            vf2 = T.simd.vcvt(T.simd.vld(x_ub[r2 + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wv2 = T.simd.vld(wf[i * CH])
                            yv2 = T.simd.vmul(T.simd.vmul(vf2, sc2, full), wv2, full)
                            T.simd.vsts(y_ub[r2 + i * CH], T.simd.vcvt(yv2, BF), full, dist="PK_B32")
                        for i in T.serial(NCH):
                            vf3 = T.simd.vcvt(T.simd.vld(x_ub[r3 + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wv3 = T.simd.vld(wf[i * CH])
                            yv3 = T.simd.vmul(T.simd.vmul(vf3, sc3, full), wv3, full)
                            T.simd.vsts(y_ub[r3 + i * CH], T.simd.vcvt(yv3, BF), full, dist="PK_B32")
                    T.copy(y_ub, Y[base:base + rows * d], l2_cache_ctrl="NORMAL_FV")
        return main

    @T.prim_func
    def looped(X: T.Tensor((total,), BF), W: T.Tensor((d,), BF), Y: T.Tensor((total,), BF), eps: T.float32):
        with T.Kernel(N_CORES) as core:
            w_ub = T.alloc_shared((d,), BF)
            wf = T.alloc_shared((d,), "float32")
            x_ub = T.alloc_shared(rows * d, BF)
            y_ub = T.alloc_shared(rows * d, BF)
            T.annotate_buffer_versions({x_ub: stages, y_ub: yver})
            T.copy(W[:d], w_ub[:d])
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in T.serial(NCH):
                    T.simd.vsts(wf[i * CH], T.simd.vcvt(T.simd.vld(w_ub[i * CH], dist="UNPK_B16"), "float32", part=0), full)
            for step in T.Pipelined(iters, num_stages=stages):
                base = ((core * iters + step) * rows) * d
                T.copy(X[base:base + rows * d], x_ub, l2_cache_ctrl="NORMAL_FV")
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for r in T.serial(rows):
                        ro = r * d
                        acc0 = T.simd.alloc_var(T.float32); acc0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc1 = T.simd.alloc_var(T.float32); acc1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH // 2):
                            v0 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 2) * CH], dist="UNPK_B16"), "float32", part=0)
                            v1 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 2 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc0, v0, v0, full)
                            T.simd.vmula(acc1, v1, v1, full)
                        acc = T.simd.vadd(acc0, acc1, full)
                        a1 = T.simd.vcadd(acc, full); a2 = T.simd.vcadd(a1, full); a3 = T.simd.vcadd(a2, full)
                        a4 = T.simd.vcadd(a3, full); a5 = T.simd.vcadd(a4, full); a6 = T.simd.vcadd(a5, full)
                        b = T.simd.vdupv(a6, full, pos="POS_LOWEST")
                        rs = T.simd.vdiv(T.simd.vdup(T.float32(1.0), "float32", full),
                            T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b, T.float32(1.0 / d), full), eps, full), full), full)
                        for i in T.serial(NCH):
                            vf = T.simd.vcvt(T.simd.vld(x_ub[ro + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wv = T.simd.vld(wf[i * CH])
                            yv = T.simd.vmul(T.simd.vmul(vf, rs, full), wv, full)
                            T.simd.vsts(y_ub[ro + i * CH], T.simd.vcvt(yv, BF), full, dist="PK_B32")
                T.copy(y_ub, Y[base:base + rows * d], l2_cache_ctrl="NORMAL_FV")
    return looped


_KERNELS = {}


def run(x, w, eps=1e-6):
    batch, d = x.shape
    key = (batch, d)
    if key not in _KERNELS:
        _KERNELS[key] = tilelang.compile(_build(batch, d), target="ascend", out_idx=[2])
    y = _KERNELS[key](x.flatten(), w, eps)
    return y.reshape(batch, d)
