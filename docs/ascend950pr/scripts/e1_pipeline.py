
import os, json, statistics, torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof

DT = "bfloat16"
N_CORES = 64

def build(batch, d, rows, versions, stages, threads):
    tile = tilelang.next_power_of_2(d)
    iterations = batch // (N_CORES * rows)
    total = batch * d
    @T.prim_func
    def main(X: T.Tensor((total,), DT), Y: T.Tensor((total,), DT),
             W: T.Tensor((d,), DT), eps: T.float32):
        with T.Kernel(N_CORES) as core:
            w_ub = T.alloc_shared((d,), DT)
            x_ub = T.alloc_shared(rows * d, DT)
            y_ub = T.alloc_shared(rows * d, DT)
            T.annotate_buffer_versions({x_ub: versions, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            for step in T.Pipelined(iterations, num_stages=stages):
                base = ((step * N_CORES + core) * rows) * d
                T.copy(X[base:base + rows * d], x_ub)
                with T.SimtVF(threads=threads):
                    xf = T.alloc_fragment((rows, tile), "float32")
                    for r, i in T.Parallel(rows, tile, coalesced_width=T.int32(1)):
                        xf[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    wf = T.alloc_fragment((tile,), "float32")
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        wf[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    acc = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(acc)
                    for r, i in T.Parallel(rows, tile, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(acc[r], xf[r, i] * xf[r, i])
                    rstd = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(acc, rstd)
                    for r in T.Parallel(rows):
                        rstd[r] = T.rsqrt(rstd[r] / d + eps)
                    for r, i in T.Parallel(rows, tile, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.cast(xf[r, i] * rstd[r] * wf[i], DT)
                T.copy(y_ub, Y[base:base + rows * d])
    return main

def torch_ref(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * w.float()).to(x.dtype)

batch, d = 8192, 4096
torch.manual_seed(0)
x = torch.randn(batch, d, dtype=torch.bfloat16, device="npu")
w = torch.randn(d, dtype=torch.bfloat16, device="npu")
ref = torch_ref(x, w)
cache = torch.empty(64*1024*1024//4, dtype=torch.int32, device="npu")

def vendor():
    return torch_npu.npu_rms_norm(x, w, 1e-6)[0]

def measure(fn, n=20):
    return float(bench_with_msprof(fn, cache, n))

# configs: (name, rows, versions, stages, threads)
configs = [
  ("base_v2_s2_r4", 4, 2, 2, 256),
  ("e1_v3_s3_r4",   4, 3, 3, 256),
  ("e1_v3_s3_t512", 4, 3, 3, 512),
  ("e2_v2_s2_r6",   6, 2, 2, 256),
]
results = []
# vendor baseline (median of two measurements bracketing)
v0 = measure(vendor)
for name, rows, versions, stages, threads in configs:
    try:
        kern = tilelang.compile(build(batch, d, rows, versions, stages, threads), target="ascend", out_idx=[1])
        y = kern(x.flatten(), w, 1e-6).reshape(batch, d)
        err = float((y.float()-ref.float()).abs().max())
        ms = measure(lambda: kern(x.flatten(), w, 1e-6).reshape(batch, d))
        results.append({"name": name, "rows": rows, "versions": versions, "stages": stages,
                        "threads": threads, "device_ms": round(ms,6), "max_abs_err": err})
    except Exception as e:
        results.append({"name": name, "rows": rows, "versions": versions, "stages": stages,
                        "threads": threads, "error": str(e)[:180]})
v1 = measure(vendor)
vmed = statistics.median([v0, v1])
for r in results:
    if "device_ms" in r:
        r["ratio_vs_vendor"] = round(vmed / r["device_ms"], 4)
print("E1JSON " + json.dumps({"vendor_ms": round(vmed,6), "vendor_samples":[round(v0,6),round(v1,6)], "results": results}))