"""RMSNorm (bf16) on Ascend950PR as a TileLang-Ascend SimdVF kernel (v1, exploratory).

Correctness: single row matches CPU float64 exactly; 8192x4096 rel err 0.00208 <= 2^-7.
Performance: ~0.123 ms device vs vendor ~0.068-0.072 ms in the same window -> ~0.58x.
NOT at parity. Ding - not acceptance tested, no witness audit.

Design: DMA copies a tile of rows into shared outside the SimdVF region; inside
T.SimdVF the per-row pass accumulates squares per lane with vmula, reduces 64 lanes
to a scalar with 6 vcadd steps, computes rstd with vmuls/vadds/vsqrt/vdiv, then a
second pass scales and stores bf16. The weight is converted to f32 once per core.
"""
import torch
import tilelang
import tilelang.ascend.language as T

BF = "bf16"
N_CORES = 64
CH = 64  # elements per vector op (64 f32 lanes)


def _rows_per_core(batch):
    rows = max(1, min(4, batch // N_CORES))
    if batch <= 0 or batch % (N_CORES * rows):
        raise ValueError("batch must be divisible by 64 * rows; remainder handling is unsupported")
    return rows


def _build(batch, d):
    rows = _rows_per_core(batch)
    NCH = d // CH
    iters = batch // (N_CORES * rows)
    total = batch * d

    @T.prim_func
    def main(X: T.Tensor((total,), BF), W: T.Tensor((d,), BF), Y: T.Tensor((total,), BF), eps: T.float32):
        with T.Kernel(N_CORES) as core:
            w_ub = T.alloc_shared((d,), BF)
            wf_ub = T.alloc_shared((d,), "float32")
            x_ub = T.alloc_shared(rows * d, BF)
            y_ub = T.alloc_shared(rows * d, BF)
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in T.serial(NCH):
                    T.simd.vsts(wf_ub[i * CH],
                                T.simd.vcvt(T.simd.vld(w_ub[i * CH], dist="UNPK_B16"), "float32", part=0),
                                full)
            for step in T.Pipelined(iters, num_stages=2):
                base = ((step * N_CORES + core) * rows) * d
                T.copy(X[base:base + rows * d], x_ub)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for r in T.serial(rows):
                        ro = r * d
                        acc = T.simd.alloc_var(T.float32)
                        acc = T.simd.vdup(T.float32(0.0), T.float32, full)
                        for i in T.serial(NCH):
                            vf = T.simd.vcvt(T.simd.vld(x_ub[ro + i * CH], dist="UNPK_B16"), "float32", part=0)
                            T.simd.vmula(acc, vf, vf, full)
                        a1 = T.simd.vcadd(acc, full); a2 = T.simd.vcadd(a1, full); a3 = T.simd.vcadd(a2, full)
                        a4 = T.simd.vcadd(a3, full); a5 = T.simd.vcadd(a4, full); a6 = T.simd.vcadd(a5, full)
                        b = T.simd.vdupv(a6, full, pos="POS_LOWEST")
                        rs = T.simd.vdiv(
                            T.simd.vdup(T.float32(1.0), "float32", full),
                            T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b, T.float32(1.0 / d), full), eps, full), full),
                            full)
                        for i in T.serial(NCH):
                            vf = T.simd.vcvt(T.simd.vld(x_ub[ro + i * CH], dist="UNPK_B16"), "float32", part=0)
                            wf = T.simd.vld(wf_ub[i * CH])
                            yv = T.simd.vmul(T.simd.vmul(vf, rs, full), wf, full)
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
