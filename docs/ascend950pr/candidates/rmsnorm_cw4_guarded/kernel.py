"""Experimental cw4 variant with explicit batch divisibility validation.

The original historical candidate is preserved separately. Unsupported batches fail
rather than returning uncomputed output rows. Performance and correctness of this
variant have not been independently accepted.
"""
import torch
import tilelang
import tilelang.ascend.language as T

N_CORES = 64


def _rows_per_core(batch):
    rows = max(1, min(4, batch // N_CORES))
    if batch <= 0 or batch % (N_CORES * rows):
        raise ValueError("batch must be divisible by 64 * rows; remainder handling is unsupported")
    return rows


def _build(batch, d):
    tile = tilelang.next_power_of_2(d)
    rows = _rows_per_core(batch)
    iterations = batch // (N_CORES * rows)
    total = batch * d
    CW = T.int32(4)

    @T.prim_func
    def main(X: T.Tensor((total,), "bfloat16"), Y: T.Tensor((total,), "bfloat16"),
             W: T.Tensor((d,), "bfloat16"), eps: T.float32):
        with T.Kernel(N_CORES) as core:
            w_ub = T.alloc_shared((d,), "bfloat16")
            x_ub = T.alloc_shared(rows * d, "bfloat16")
            y_ub = T.alloc_shared(rows * d, "bfloat16")
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            for step in T.Pipelined(iterations, num_stages=2):
                base = ((step * N_CORES + core) * rows) * d
                T.copy(X[base:base + rows * d], x_ub)
                with T.SimtVF(threads=256):
                    xf = T.alloc_fragment((rows, tile), "float32")
                    for r, i in T.Parallel(rows, tile, coalesced_width=CW):
                        xf[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    wf = T.alloc_fragment((tile,), "float32")
                    for i in T.Parallel(tile, coalesced_width=CW):
                        wf[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    acc = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(acc)
                    for r, i in T.Parallel(rows, tile, coalesced_width=CW):
                        if i < d:
                            T.reducer_update(acc[r], xf[r, i] * xf[r, i])
                    rstd = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(acc, rstd)
                    for r in T.Parallel(rows):
                        rstd[r] = T.rsqrt(rstd[r] / d + eps)
                    for r, i in T.Parallel(rows, tile, coalesced_width=CW):
                        if i < d:
                            y_ub[r * d + i] = T.cast(xf[r, i] * rstd[r] * wf[i], "bfloat16")
                T.copy(y_ub, Y[base:base + rows * d])
    return main


_KERNELS = {}


def run(x, w, eps=1e-6):
    batch, d = x.shape
    key = (batch, d)
    if key not in _KERNELS:
        _KERNELS[key] = tilelang.compile(_build(batch, d), target="ascend", out_idx=[1])
    y = _KERNELS[key](x.flatten(), w, eps)
    return y.reshape(batch, d)