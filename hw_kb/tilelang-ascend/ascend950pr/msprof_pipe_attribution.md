# msprof_pipe_attribution (reading AIV pipe attribution and the device-time basis on Ascend: vec / mte2 / mte3, per-core device time excluding cache-flush cores)

> **Certification source: measured on Ascend950PR by this repo's device probe** (same-window attribution of
> one candidate arm and one vendor arm, each aggregated to exactly one kernel profile). These are measured
> readings and the timing-basis definition, not a `max_diff` numerical contract.
>
> One file per hardware interface. Operator-agnostic; mechanism / measurement method only.

## Purpose

Turn "this kernel is slow" into a measured **pipe imbalance and a timing basis** rather than a guess:
on the Ascend AIV (AI vector core) the attribution distinguishes the vector pipe (`vec`) from the
data-movement pipes (`mte2` = load/in, `mte3` = store/out), and the comparison is only honest if both arms
use the same host-free device-time basis.

## When needed (trigger)

- Performance feedback reports a time but not a direction; you must choose whether to optimize compute
  (vector) or data movement (load/store).
- The vector pipe already looks near saturation yet the kernel is still far from the baseline.
- You need to pick a timing basis for an A/B, or you suspect a timing number includes host enqueue time.

## Minimal runnable example

```python
import torch
from tilelang.profiler.msprof import bench_with_msprof

# int32 cache is mandatory: msprof's own cache-flush kernel cannot lower int8 vectors on Ascend
# ("SimtVF cannot convert vector type int8x8").
cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")

prof = bench_with_msprof(lambda: kern(x.flatten(), w, 1e-6), cache, 20, detailed=True)
# prof.dur_ns is the per-kernel device time (sum over device tasks, repeats averaged),
# already excluding the profiler's cache-flush kernels.
# detailed=True also carries the per-pipe split: aiv_vec / aiv_mte2 / aiv_mte3.
```

**Timing basis (the definition to reuse):** `tilelang.profiler.msprof.bench_with_msprof` reports the
**per-kernel device time** parsed from `torch_npu.profiler` FFTS per-core records (`dur_ns / repeats`),
**excluding the cache-flush kernels**. It is a host-free device time — the host enqueue time is not in it —
so Python scheduling jitter does not enter. Both arms must use this same basis.

Measured pipe attribution (4096x4096 bf16, candidate vs baseline, same card and window):

| pipe | candidate | baseline |
|---|---|---|
| AIV `vec` | 84.9% | 80.7% |
| AIV `mte2` (load / in) | 32.0% | 64.2% |
| AIV `mte3` (store / out) | 12.6% | 44.9% |

Reading: both arms are AIV-vector-bound (`vec` ~80-85%), so the gap is **not** arithmetic — it is data
movement. The candidate under-feeds the vector pipe (`mte2` 32% vs 64%) and under-uses the store path
(`mte3` 12.6% vs 44.9%). Direction: raise `mte2` (in-flight bulk copy) and `mte3` (store path) while keeping
`vec` saturated; see [`row_tile_ub_budget`](row_tile_ub_budget.md) for the lever that raises `mte2`.

## Gotchas

1. **The cache buffer must be int32.** msprof's cache-flush kernel cannot lower an int8 vector on Ascend
   (`SimtVF cannot convert vector type int8x8`), so an int8 cache breaks the measurement path.
2. **One KernelProfile per arm, or the percentages are partial.** Confirm each arm is a single kernel
   (`len(prof.kernels)`); an arm that is silently several kernels makes a percentage cover only part of
   the work. In the measured check both arms aggregate to exactly one KernelProfile each.
3. **Pipe shares are directions, not speed-ups.** A percentage is the pipe's active share; it tells you
   where to look, not how much faster a change will be. Only a timing sweep prices a change.
4. **Excluding cache-flush cores is what makes it the kernel's own time.** A naive sum that includes the
   flush kernels inflates the reading; the basis must exclude them.
5. **Absolute device time is not evidence across runs.** On a shared card it drifts run to run; only a
   ratio formed inside one window is comparable — see [`shared_card_contention`](shared_card_contention.md).
6. **The baseline can do strictly more work.** A vendor arm that also writes an extra output tensor is not
   flattered by a candidate that returns less; state the interface asymmetry rather than hiding it.

## Sources

- TileLang public API `tilelang.profiler.msprof.bench_with_msprof` (returns per-kernel device duration;
  `detailed=True` carries the per-pipe split).
- `torch_npu.profiler` FFTS per-core device-time records (public infra API surface).
- Device-measured on this repo's Ascend950PR probe: `docs/ascend950pr/scripts/inspect_msprof_attribution.py`
  (per-pipe attribution), evidence JSONs under `docs/ascend950pr/evidence/`.
- Detailed provenance: docs/kb-provenance/tilelang-ascend/msprof_pipe_attribution.md (for human review, not in the agent read set).
