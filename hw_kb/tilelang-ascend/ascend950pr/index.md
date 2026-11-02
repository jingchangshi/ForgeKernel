# ascend950pr (Ascend 950PR) interface index — TileLang mode

The `ascend950pr` tree of TileLang mode (a Python `kernel.py` student, built by tilelang's Ascend
compiler). Demand-driven: an entry is written when the student gets stuck on that interface or needs a
measured hardware prior, following `../TEMPLATE.md` (examples are TileLang Python), and is marked ✅ only
on **measured device evidence from this repo's Ascend950PR probes** (a resource ceiling / pipe census /
timing sweep, or a `PROBE PASS` from such a probe). This file is not a plan; it is the ledger of what has
already been earned.

This tree is separate from `hw_kb/cutedsl/` and `hw_kb/cuda/`: the readings here were taken on the Ascend
NPU with TileLang, and are not carried across from the NVIDIA trees. **Not finding something here does not
mean it cannot be done** — the CANN / Ascend runtime / TileLang sources and your own probes are
authoritative; what you win on a probe gets written in here in the finish round.

## Look up by symptom first (this section is the entry point, not a table of contents)

| What you see / what you are about to do | Look first at |
|---|---|
| **About to allocate or enlarge a shared / UB buffer**, or a compile/launch reports a UB overflow such as `dyn_ubuf 271360 bytes > budget`; need the per-core ceiling and the margin to design against | [`ub_capacity`](ub_capacity.md): single-buffer launch observations; check the actual kernel rather than assuming a general ceiling |
| **Performance is slow but you don't know the direction** — vector pipe looks near saturation, or you need the device-time basis for an A/B | [`msprof_pipe_attribution`](msprof_pipe_attribution.md): per-core device time excluding cache-flush cores; `vec` / `mte2` / `mte3` split; both arms vector-bound, gap is data movement |
| **Data movement is the gap** (`mte2` below the baseline) and you are choosing rows-per-core / tile depth, or a row batch overflows UB | [`row_tile_ub_budget`](row_tile_ub_budget.md): rows 1->4 gives 1.76x (289->164 us); rows=8 overflows `dyn_ubuf` 271360 B; effective bandwidth 0.8 vs 2.6 TB/s |
| **Timing drifts between runs** or a paired ratio moves when another tenant appears; deciding whether a window is usable for a verdict | [`shared_card_contention`](shared_card_contention.md): a saturated neighbour shifted a paired ratio by up to 2.05x and the direction reversed with shape; exclude contended windows |

## Full table

| Interface | Purpose (one line) | Status |
|---|---|---|
| [`ub_capacity`](ub_capacity.md) | UB per-core capacity ceiling and measurement method | ✅ Measured (Ascend950PR device probe) |
| [`msprof_pipe_attribution`](msprof_pipe_attribution.md) | AIV pipe attribution (vec / mte2 / mte3) and device-time basis | ✅ Measured (Ascend950PR device probe) |
| [`row_tile_ub_budget`](row_tile_ub_budget.md) | per-core row batching vs UB budget; effective-bandwidth arithmetic | ✅ Measured (Ascend950PR device probe) |
| [`shared_card_contention`](shared_card_contention.md) | shared-card contention bias of a paired A/B; window validity | ✅ Measured (controlled contention experiment) |
