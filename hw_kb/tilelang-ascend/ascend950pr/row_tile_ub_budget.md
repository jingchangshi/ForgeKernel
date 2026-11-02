# row_tile_ub_budget (per-core row batching as the MTE2 lever, the UB budget that caps it, and the effective-bandwidth arithmetic)

> **Certification source: measured on Ascend950PR by this repo's device probe** (a rows-per-core sweep on
> 8192x4096 bf16, and the bandwidth arithmetic derived from it). These are measured readings, not a
> `max_diff` numerical contract.
>
> One file per hardware interface. Operator-agnostic; mechanism only. For the ceiling itself see
> [`ub_capacity`](ub_capacity.md); for the pipe reading that motivates it see
> [`msprof_pipe_attribution`](msprof_pipe_attribution.md).

## Purpose

When the vector pipe is already near saturation and the gap is data movement (`mte2` low), the lever is
**how many rows one core processes per iteration**: more rows means more in-flight bulk copy per core to
feed the vector pipe. It is capped by the per-core UB budget — this entry records both the measured gain
and where the budget stops it.

## When needed (trigger)

- `mte2` is well below the baseline's while `vec` is near saturation (the signature of too few in-flight
  bulk copies per core).
- You are choosing the rows-per-core / tile depth of a streaming kernel.
- A row-batch size overflows UB with an error like `dyn_ubuf 271360 bytes > budget`.

## Minimal runnable example

```python
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof

def build(batch, d, rows):
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)
    threads = min(256 if rows >= 4 else (512 if rows == 2 else 128), TILE)
    iterations = batch // (N_CORES * rows)
    N = batch * d

    @T.prim_func
    def main(X: T.Tensor((N,), "bf16"), Y: T.Tensor((N,), "bf16"),
             W: T.Tensor((d,), "bf16"), eps: T.float32):
        with T.Kernel(N_CORES) as core_id:
            w_ub = T.alloc_shared((d,), "bf16")
            x_ub = T.alloc_shared(rows * d, "bf16")     # the row batch lives here
            y_ub = T.alloc_shared(rows * d, "bf16")
            T.annotate_buffer_versions({x_ub: 3, y_ub: 1})   # pipelined versions share the budget
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iterations, num_stages=3):
                base = (t * N_CORES + core_id) * rows * d
                T.copy(X[base:base + rows * d], x_ub)
                # ... vector work over x_ub -> y_ub ...
                T.copy(y_ub, Y[base:base + rows * d])
    return main

cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")
for rows in (1, 2, 4, 8):
    try:
        kern = tilelang.compile(build(8192, 4096, rows), target="ascend", out_idx=[1])
        prof = bench_with_msprof(lambda: kern(x.flatten(), w, 1e-6), cache, 20, detailed=True)
        print(rows, round(prof.dur_ns / 1e6, 3), "ms")
    except Exception as exc:
        print(rows, "ERROR", str(exc)[:160])
```

Measured sweep (8192x4096 bf16; `mte2` share and device time):

| rows / core | `mte2` | device time |
|---|---|---|
| 1 | 34.2% | 289 us |
| 2 | 58.9% | 190 us |
| 4 | 53.4% | **164 us** |
| 8 | — | UB overflow: `dyn_ubuf 271360 bytes > budget` |

- **rows 1 -> 4 buys 1.76x** (289 -> 164 us), confirming the `mte2`-starvation reading.
- **rows = 8 requests 271360 B of dynamic UB**, over the 262144 B per-core ceiling (see
  [`ub_capacity`](ub_capacity.md)), so per-core row batching alone hits a structural ceiling there.

**Effective bandwidth arithmetic** (the diagnostic, not an operator result): 8192x4096 bf16 = 64 Mi
elements x 2 B = **128 MiB moved (read once, write once)** — the "bidirectional 128 MiB" basis.
Baseline ~51 us implies ~2.5-2.6 TB/s; the candidate ~164 us implies ~0.8 TB/s. The gap is DMA / pipeline
efficiency, not vector math (`vec` is already ~85%) — so the target is **UB budget + copy pipelining**, not
more arithmetic tuning.

## Gotchas

1. **`mte2` share is not the score — pick by device time.** rows=2 reads a higher `mte2` share (58.9%)
   than rows=4 (53.4%), yet rows=4 is faster (164 vs 190 us). The pipe share is not monotone with wall /
   device time; the time is the criterion.
2. **The UB budget holds all simultaneously-resident copies.** Input bf16 + the fp32 working fragment(s) +
   every pipelined version (`annotate_buffer_versions`) draw on the same budget. A "rows*2 + rows*4*2"
   style accounting overflows before the byte count you first think of; count versions and fp32 widening
   too.
3. **Divisibility.** `batch % (N_CORES * rows)` must be 0 (and the per-row width a power-of-two tile for
   the vector pass); guard the sweep and skip a rows value that does not divide, rather than mis-tiling.
4. **248 KiB is the conservative practical figure.** The hardware ceiling is 256 KiB; design against
   ~248 KiB (253952 B) once compiler-managed static shared memory is subtracted.
5. **A deeper pipeline is not free.** Each extra stage is another copy of the row batch in UB; on this
   ceiling the extra version can be exactly what pushes rows=8 over.

## Sources

- TileLang public API: `T.alloc_shared`, `T.Pipelined`, `T.annotate_buffer_versions`,
  `tilelang.compile(target="ascend")`, `tilelang.profiler.msprof.bench_with_msprof`.
- Device-measured on this repo's Ascend950PR probe: `docs/ascend950pr/scripts/sweep_rows.py` (rows sweep),
  evidence JSONs under `docs/ascend950pr/evidence/`.
- The per-core ceiling used here: [`ub_capacity`](ub_capacity.md) (256 KiB = 262144 B).
- Detailed provenance: docs/kb-provenance/tilelang-ascend/row_tile_ub_budget.md (for human review, not in the agent read set).
