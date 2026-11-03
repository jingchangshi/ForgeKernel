"""RMSNorm (bf16) on Ascend950PR, TileLang-Ascend SimdVF, v33 (56-block grid: the core-count fix).

v23 launched T.Kernel(64). The Ascend950PR_9579 platform config declares
ai_core_cnt=28 / vector_core_cnt=56, torch_npu reports vector_core_num=56 and
multi_processor_count=56, and a step-detection probe (fixed per-block work, grid
1..224, 6 repeats) shows the time is flat from grid=1 to grid=56 (~10.0 us) and
then jumps +82% at grid=57 (~18.5 us), staying flat out to 84. So 56 is the true
concurrent AIV core count, and v23's 64-block launch was the single largest prefill
and decode regression: on 8192x4096, 15 paired quiet windows gave 0.0549-0.0556 ms
at 56 blocks versus 0.0760-0.0766 ms at 64 blocks against a 0.0555 ms vendor
reference (ratio 1.01 MEET versus 0.726 FAIL, ~1.14x), and on decode 64x4096 the
ratio is 1.53-1.54 at 56 blocks versus 1.22 at 64. A fixed-total-work sweep agrees:
13440 tiles of 4096 bf16 take 187.2 us at grid=56 (1176 GB/s) versus 277.4 us at
grid=64 (794 GB/s).

2048 prefill tiles do not divide by 56, so the split is ragged: nbase = tiles // 56
tiles each, and the first tiles % 56 cores take one extra (8192x4096 with rows=4 is
36 base, with 32 cores holding 37). The per-core loop bound follows the per-core
count, selected by the module-level LOOP_MODE constant:

  "dyn"       (default) loop over the per-core count itself -> best decode (1.66x)
  "recompute" loop over the uniform max, clamp the tile index -> best prefill (1.005x)

Both modes were correctness-verified (3 seeds x 4 runs, max_abs 0.0078-0.0156).

The read path additionally moved from NORMAL_FV to NOTALLOC_CLEAN (L2_READ_CTRL); the
write path stays on NORMAL_FV (L2_WRITE_CTRL). Measured against vendor at 56 blocks over
20 paired quiet windows this is 1.103x on prefill (1.1185 [1.1179,1.1193] versus 1.0170
[1.0144,1.0179]) and 1.009x on decode, at 0 bad runs out of 64 per shape. See the
constant's comment for the policies that are unsafe.

Single-phase kernel (the only form that has been proven race-free) with the rows=4
fast path written out explicitly, so every row keeps its own accumulators and its own
scale in registers. Nothing is parked in shared memory between a reduce stage and a
scale stage, which is what made the v10 family racy.

Only the batch sizes that map to rows=4 are served by that path; rows<=2 fall back to
the loop form. Performance outside the certified prefill (8192x4096) and decode
(64x4096) shapes is NOT established.
"""
import torch
import tilelang
import tilelang.ascend.language as T

BF = "bfloat16"
N_CORES = 56          # v23 used 64; 56 is the measured concurrent AIV core count.
CH = 64
LOOP_MODE = "dyn"     # "dyn" (default, best decode) | "recompute" (best prefill)

# L2 policy. Read (GM->UB, the X tile) uses NOTALLOC_CLEAN; the Y write (UB->GM) stays
# on NORMAL_FV. CLEAN keeps the line and marks it Clean, so evicting it needs no
# write-back to GM. Our prefill footprint is 64MB X + 64MB Y = 128MB = exactly the L2
# size and every line is touched once, so the allocating policy (NORMAL_FV) forces
# dirty-line write-backs inside the read stream; CLEAN removes that and still measures
# 0 bad runs out of 64 on both shapes. Measured against vendor at 56 blocks, 20 paired
# quiet windows: prefill vendor/cand 1.0170 [1.0144,1.0179] with a NORMAL_FV read vs
# 1.1185 [1.1179,1.1193] with NOTALLOC_CLEAN (CLEAN is 1.103x the candidate), decode
# 1.6500 -> 1.6699 (1.009x).
#
# L2 policy trap, do not re-try: NOTALLOC_DROP and NORMAL_LV on the read path are
# BROKEN. DROP benchmarked fastest and then produced wrong results in 56 of 64 decode
# runs (worst max_abs 16.6); NORMAL_LV is equally unsafe. They only look fast because
# they silently corrupt, so their benchmark numbers are not wins. Both DMA copies were
# once set to NORMAL_FV for this reason; only the read moved to CLEAN, which is the one
# non-allocating policy that stayed correct in all 64 runs of each shape.
L2_READ_CTRL = "NOTALLOC_CLEAN"   # X tile, GM->UB
L2_WRITE_CTRL = "NORMAL_FV"       # Y tile, UB->GM


def _rows_per_core(batch):
    """Largest row tile that still gives every one of the 56 cores work to do.

    v23 gated this on batch % (N_CORES * rows) == 0, which no 56-core shape can
    satisfy (8192 % 224 == 128). The grid is ragged now, so the only requirements are
    that the row tile packs the batch exactly and that there is at least one tile per
    core -- with fewer tiles than cores a zero-count core would have no live tile, and
    the "dyn" mode's T.max(cnt, 1) would run one iteration against a bad index.
    """
    if batch <= 0:
        raise ValueError("batch must be positive")
    for rows in (4, 2, 1):
        if batch % rows == 0 and batch // rows >= N_CORES:
            return rows
    raise ValueError(
        "batch must map to >= 56 row tiles; remainder handling below one tile per core is unsupported")


def _build(batch, d):
    rows = _rows_per_core(batch)
    if d <= 0 or d % (4 * CH):
        raise ValueError("width must be a positive multiple of 256")
    NCH = d // CH
    # Ragged split: 2048 prefill tiles over 56 cores is 36 base + 32 cores holding one
    # more. v23 instead required an exactly even split and refused anything else.
    tiles = batch // rows
    nbase = tiles // N_CORES
    rem = tiles % N_CORES
    iters = nbase + (1 if rem else 0)   # uniform max, the bound used by "recompute"
    dyn = LOOP_MODE == "dyn"
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
                start = core * nbase + T.min(core, rem)
                cnt = nbase + T.if_then_else(core < rem, T.int32(1), T.int32(0))
                for step in T.Pipelined(T.max(cnt, 1) if dyn else iters, num_stages=stages):
                    # "recompute" rewrites the last tile cnt-1 times; the store is idempotent
                    s = step if dyn else T.min(step, cnt - 1)
                    base = ((start + s) * rows) * d
                    T.copy(X[base:base + rows * d], x_ub, l2_cache_ctrl=L2_READ_CTRL)
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
                    T.copy(y_ub, Y[base:base + rows * d], l2_cache_ctrl=L2_WRITE_CTRL)
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
            start = core * nbase + T.min(core, rem)
            cnt = nbase + T.if_then_else(core < rem, T.int32(1), T.int32(0))
            for step in T.Pipelined(T.max(cnt, 1) if dyn else iters, num_stages=stages):
                # "recompute" rewrites the last tile cnt-1 times; the store is idempotent
                s = step if dyn else T.min(step, cnt - 1)
                base = ((start + s) * rows) * d
                T.copy(X[base:base + rows * d], x_ub, l2_cache_ctrl=L2_READ_CTRL)
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
                T.copy(y_ub, Y[base:base + rows * d], l2_cache_ctrl=L2_WRITE_CTRL)
    return looped


_KERNELS = {}


def run(x, w, eps=1e-6):
    batch, d = x.shape
    key = (batch, d)
    if key not in _KERNELS:
        _KERNELS[key] = tilelang.compile(_build(batch, d), target="ascend", out_idx=[2])
    y = _KERNELS[key](x.flatten(), w, eps)
    return y.reshape(batch, d)
