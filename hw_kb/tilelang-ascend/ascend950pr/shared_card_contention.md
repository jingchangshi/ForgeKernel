# shared_card_contention (how much a busy shared NPU can bias a paired A/B, and why pairing alone is not enough)

> **Certification source: measured on Ascend950PR by a controlled contention experiment in this repo**
> (the same candidate artifact measured idle, then while a background load held another NPU partition near
> 100% AICore). These are measured readings about the measurement platform, not a `max_diff` contract.
>
> One file per hardware interface. Operator-agnostic; measurement-validity only.

## Purpose

The evaluation machine is a **shared** host: other tenants' work can be running on the same card. This
entry records how much a saturated neighbour can move a paired candidate/baseline ratio, so a comparison
**excludes contended windows** instead of trusting a ratio that a co-tenant has shifted.

## When needed (trigger)

- A/B timing drifts between runs, or the same artifact reads differently at two times.
- A paired ratio moves when another process appears on the card.
- You are deciding whether a timing window is usable for a KEEP/REVERT verdict.

## Minimal runnable example

```python
import subprocess, statistics, time
from tilelang.profiler.msprof import bench_with_msprof

def aicore_witness():
    # npu-smi is slow (sub-second to a few seconds per call), so sample it as an out-of-process witness.
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30).stdout
    return out   # parse AICore utilisation + device process table from here

cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")
ratios = []
for _ in range(15):
    before = aicore_witness()
    c_ms = bench_with_msprof(lambda: cand(), cache, 10)   # candidate arm
    v_ms = bench_with_msprof(lambda: base(), cache, 10)   # baseline arm
    after = aicore_witness()
    # interleave A/B and B/A inside one window; keep the per-window ratio, never an absolute time
    ratios.append(c_ms / v_ms)
print("paired median", statistics.median(ratios))
```

Measured (same candidate artifact, idle vs a controlled background load at AICore ~99%):

| shape | idle ratio (candidate/baseline) | contended ratio | shift | direction |
|---|---|---|---|---|
| 8192x4096 | 0.322 | 0.661 | +105% | favours the candidate |
| 64x4096 | 0.994 | 0.827 | -17% | penalises the candidate |

- **The bias direction reverses between shapes**, so contention is not a conservative error.
- **Pairing does not cancel it**: the two arms are not equally sensitive (victim factors 2.68x vs 1.31x at
  8192x4096; 15.71x vs 18.78x at 64x4096), so their ratio does not cancel the disturbance.
- **False-pass risk**: a ratio inflation of **2.052** means a candidate whose true ratio is 0.478 would be
  reported at 0.98 and pass — a 2.05x contention level can certify an implementation that is twice as slow.
- **Light vs sustained is a different regime**: a bursty neighbour is absorbed by the paired median (it
  moved ~0.8%), a sustained saturated one is not (+105%). Intensity reconciles the two readings; they are
  not contradictory.
- **The reliable detector** is an out-of-process AICore trajectory plus the device process table (100% of
  windows flagged when saturated). A short in-process canary kernel is a **supporting positive signal only,
  never a negative one** — at small size its detection rate was ~33%.

## Gotchas

1. **A median resists random noise but not a systematic shift**, and this shift is systematic (2.052x), not
   random. Do not average contended windows in.
2. **Do not carry "pairing cancels external load" from a light-load window to a saturated one.** That claim
   is only true for light / bursty load.
3. **Exclude contended windows entirely** from the point estimate and bootstrap; record and display them but
   never let them enter the judged sample. If quiet windows are below the minimum sample size, or are under
   70% of all windows, RETEST the shape rather than judge a selectively biased sample.
4. **Process names do not identify owners.** `npu-smi` reports pid / name / memory only; ownership needs
   `ps` with the user column.
5. **A card can show no processes yet hold several GB of HBM** (unreleased allocation); that is not a running
   tenant. Conversely an NPU-shaped command from another user does not prove *this* card is contended if it
   is pinned to a different device.
6. **Never compare absolute device times across runs**; only a ratio inside one window is meaningful.

## Sources

- `npu-smi info` (out-of-process AICore/utilisation and device process table) — the CANN platform tool.
- `tilelang.profiler.msprof.bench_with_msprof` — per-kernel device time for each arm (see
  [`msprof_pipe_attribution`](msprof_pipe_attribution.md)).
- Device-measured on this repo's Ascend950PR probes: `docs/ascend950pr/scripts/paired_bench.py`,
  `docs/ascend950pr/scripts/contention_bias.py`, `docs/ascend950pr/scripts/verified_contention.py`; evidence
  JSONs under `docs/ascend950pr/evidence/`.
- Detailed provenance: docs/kb-provenance/tilelang-ascend/shared_card_contention.md (for human review, not in the agent read set).
