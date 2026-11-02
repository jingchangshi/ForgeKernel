# ub_capacity (Unified Buffer (UB) per-core capacity ceiling: 256 KiB, and how to measure it)

> **Certification source: measured on Ascend950PR by this repo's device probe.** This is a resource-ceiling
> measurement, not a `max_diff` numerical contract: a single-shared-buffer kernel is compiled AND run at
> each candidate size and the launch is what accepts or rejects it. The companion pipe/timing readings are
> in [`msprof_pipe_attribution`](msprof_pipe_attribution.md).
>
> One file per hardware interface. Operator-agnostic; mechanism only.

## Purpose

State the hard per-core UB (unified buffer, the Ascend on-chip scratch space shared by the vector and
data-movement pipes) ceiling so that a kernel's tile / row-batch / double-buffer budget is a **prior**, not
re-discovered (or guessed) each iteration.

## When needed (trigger)

- You are about to allocate or enlarge a shared / UB buffer (`T.alloc_shared`): a tile, a row batch, an
  input copy, a fp32 working fragment, a multi-version (pipelined) buffer.
- A compile or launch fails with a UB / shared-memory overflow such as
  `UB overflow: dyn_ubuf 271360 bytes > budget`, and you need to know what the budget actually is.
- You are choosing rows per core, pipeline depth, or how many simultaneously-resident copies to keep.

## Minimal runnable example

The ceiling is enforced at **launch**, not at compile, so a probe must run the kernel. Allocate one shared
buffer of a candidate byte size in a trivial kernel and bisect the size (sketch adapted from this repo's
Ascend950PR probe scripts):

```python
import torch, tilelang
import tilelang.ascend.language as T

def build(nbytes):
    n = nbytes // 2  # bf16 elements
    @T.prim_func
    def main(X: T.Tensor((n,), "bf16"), Y: T.Tensor((n,), "bf16")):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((n,), "bf16")
            T.copy(X, buf)
            T.copy(buf, Y)
    return main

x = torch.randn(262144 // 2, dtype=torch.bfloat16, device="npu")
for size in (253952, 256000, 262144, 263168, 266240, 270336, 271360):
    n = size // 2
    try:
        kern = tilelang.compile(build(size), target="ascend", out_idx=[1])
        kern(x[:n], x[:n])          # RUN it: only the launch checks dynamic+static UB
        print(size, "ok")
    except Exception as exc:
        print(size, "REJECTED", str(exc)[:160])
```

Measured ladder (the actual reading):

| bytes | KiB | verdict |
|---|---|---|
| 253952 | 248 | ok |
| 256000 | ~250 | ok |
| 262144 | 256 | ok |
| 263168 | 257 | REJECTED — `dyn_ubuf_bytes=263168` |
| 266240 | 260 | REJECTED |
| 270336 | 264 | REJECTED |
| 271360 | 265 | REJECTED |

**Conclusion: the per-kernel shared-memory ceiling is exactly 256 KiB = 262144 B** (256 KiB passes, 257 KiB
fails).

## Gotchas

1. **Compilation alone does not enforce the limit — you must run.** The runtime launch check
   (`aclrtLaunchKernelWithHostArgs`) is what validates dynamic + static shared memory; a build that merely
   compiles can sit above the ceiling and tell you nothing.
2. **Static shared memory draws on the same budget.** A real kernel's dynamic budget is smaller than the
   bare 262144 B ceiling once compiler-managed static shared memory is subtracted. 248 KiB = 253952 B is a
   plausible conservative *effective* figure; treat 256 KiB as the hard ceiling and design tiles against a
   budget you have subtracted from it, not against the ceiling itself.
3. **The ceiling is per core, not per process.** It is the per-kernel shared-memory budget of one AI core;
   do not multiply it by the core count when sizing one core's tile.
4. **Do not carry an "about 216 KiB" figure.** That number was an unsupported guess, was never printed by
   any error, and is retracted. The only error text seen was `dyn_ubuf <n> bytes > budget`, with no
   budget stated. Measure, do not infer the budget from the absence of a number.

## Sources

- CANN header `include/pto/common/buffer_limits.hpp` (public infra API surface): defines
  `PTO_UBUF_SIZE_BYTES` as `256*1024` for `PTO_NPU_ARCH_A5` (the 950PR generation); A2A3 (910B) is
  `192*1024`.
- Ascend runtime API `aclrtLaunchKernelWithHostArgs`: validates dynamic + static shared memory at launch.
- Device-measured on this repo's Ascend950PR probe: the bisection above (probe scripts under
  `docs/ascend950pr/scripts/`; evidence JSONs under `docs/ascend950pr/evidence/`). 262144 B passes,
  263168 B is rejected with `dyn_ubuf_bytes=263168`.
- Detailed provenance: docs/kb-provenance/tilelang-ascend/ub_capacity.md (for human review, not in the agent read set).
