REJECTED (R59): this two-phase kernel parks each row's inverse RMS in a shared buffer
that is written in phase A and read in phase B inside a software-pipelined loop. That
construct is racy: the rows=2 form produced wrong output at every tested batch size and
was NONDETERMINISTIC (batch=128, rel=1.0 across repeats). The rows=4 form happened to be
deterministic over 10 repeats, which is luck, not proof. Do not use this candidate; the
single-phase per-row kernel (rmsnorm_simdvf_v5) is the accepted prefill candidate.

"""RMSNorm (bf16) on Ascend950PR, TileLang-Ascend SimdVF, v13.

Two-phase schedule with each row's inverse RMS held in its own register.

History: v10 parked the per-row scale in a shared buffer between phase A and phase B.
That buffer was never versioned against the software-pipelined loop, and its rows=2 form
produced wrong output at EVERY tested batch size (256/512/1024/8192), so the shared
buffer form is abandoned as unsafe. The rows=4 form of v10 happened to agree with the
float64 reference, which is luck, not proof.

Here the row loop is unrolled at Python level inside the kernel, so every row owns its
own accumulators and its own scale variable - no shared state crosses the pipeline
boundary. Core-contiguous partitioning is kept.

Performance outside the certified decode shape is NOT established.
"""
import torch
import tilelang
import tilelang.ascend.language as T

BF = "bfloat16"
N_CORES = 64
CH = 64
NACC = 4


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
                    scales = []
                    for r in range(rows):
                        ro = r * d
                        acc0 = T.simd.alloc_var(T.float32); acc0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc1 = T.simd.alloc_var(T.float32); acc1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc2 = T.simd.alloc_var(T.float32); acc2 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        acc3 = T.simd.alloc_var(T.float32); acc3 = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH // NACC):
                            v0 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 0) * CH], dist="UNPK_B16"), "float32", part=0)
                            v1 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                            v2 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 2) * CH], dist="UNPK_B16"), "float32", part=0)
                            v3 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 3) * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc0, v0, v0, full)
                            T.simd.vmula(acc1, v1, v1, full)
                            T.simd.vmula(acc2, v2, v2, full)
                            T.simd.vmula(acc3, v3, v3, full)
                        acc = T.simd.vadd(T.simd.vadd(acc0, acc1, full), T.simd.vadd(acc2, acc3, full), full)
                        a1 = T.simd.vcadd(acc, full); a2 = T.simd.vcadd(a1, full); a3 = T.simd.vcadd(a2, full)
                        a4 = T.simd.vcadd(a3, full); a5 = T.simd.vcadd(a4, full); a6 = T.simd.vcadd(a5, full)
                        bcast = T.simd.vdupv(a6, full, pos="POS_LOWEST")
                        sc = T.simd.alloc_var(T.float32)
                        sc = T.simd.vdiv(
                            T.simd.vdup(T.float32(1.0), "float32", full),
                            T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(bcast, T.float32(1.0 / d), full), eps, full), full),
                            full)
                        scales.append(sc)
                    for r in range(rows):
                        ro = r * d
                        rsc = scales[r]
                        for i in T.serial(NCH):
                            vf = T.simd.vcvt(T.simd.vld(x_ub[ro + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wv = T.simd.vld(wf[i * CH])
                            yv = T.simd.vmul(T.simd.vmul(vf, rsc, full), wv, full)
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
