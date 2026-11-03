# rmsnorm_simdvf_v33 — 56-block grid (v23 body), plus a NOTALLOC_CLEAN read path

Candidate: `docs/ascend950pr/candidates/rmsnorm_simdvf_v33/kernel.py`
Base: v23, unchanged except for the launch geometry, the ragged split, the loop-bound
mode switch, and the L2 read policy. Author: `implementer-v33`, task-3, 2026-11-03.

## (a) Diff summary vs v23

Four things changed. Nothing else in the kernel body moved — every accumulator, the
4-row unrolled prefill body, the reduction tree, the weight pre-conversion and the
staging depths are v23's.

1. **`N_CORES = 64` -> `56`.** The Ascend950PR_9579 platform config declares
   `ai_core_cnt=28` / `vector_core_cnt=56`, torch_npu reports
   `vector_core_num=56` / `multi_processor_count=56`, and the step-detection probe
   (fixed per-block work, grid 1..224, 6 repeats) is flat from grid=1 to grid=56
   (~10.0 us) then jumps +82% at grid=57 (~18.5 us), staying flat out to 84. 56 is
   the true concurrent AIV core count. A fixed-total-work sweep agrees: 13440 tiles
   of 4096 bf16 take 187.2 us at grid=56 (1176 GB/s) versus 277.4 us at grid=64
   (794 GB/s). **v23's 64-block launch was the single largest prefill and decode
   regression.**

2. **Even split -> ragged split.** v23 required `batch % (N_CORES * rows) == 0` and
   raised "batch must be divisible by 64". No 56-core shape can satisfy that
   (8192 % 224 == 128), so the geometry is now:
   ```
   tiles = batch // rows
   nbase = tiles // N_CORES
   rem   = tiles % N_CORES
   start = core * nbase + min(core, rem)
   cnt   = nbase + (1 if core < rem else 0)
   ```
   For 8192x4096 (rows=4) that is 2048 tiles: 36 base, with the first 32 cores
   holding 37. For 64x4096 (rows=1) it is 64 tiles: 8 cores hold 2, 48 hold 1.
   `_rows_per_core` correspondingly relaxes to "batch packs into the row tile and
   yields at least 56 tiles", so a shape with fewer tiles than cores still fails
   loudly instead of letting `T.max(cnt, 1)` run a core against a bad index.

3. **Both loop strategies implemented, selected by `LOOP_MODE`, default `"dyn"`:**
   - `"dyn"` (default): `for step in T.Pipelined(T.max(cnt, 1), num_stages=stages)`,
     `s = step`. The `T.max(cnt, 1)` keeps a zero-count core on one idempotent
     iteration rather than a zero-trip hardware loop.
   - `"recompute"`: `for step in T.Pipelined(iters, num_stages=stages)`,
     `s = T.min(step, cnt - 1)` — the last tile is rewritten idempotently.
   `iters` is the uniform max (`nbase + (1 if rem else 0)`), i.e. the bound only
   "recompute" uses.

4. **L2 read policy `NORMAL_FV` -> `NOTALLOC_CLEAN`** (`L2_READ_CTRL`). The Y write
   stays on `NORMAL_FV` (`L2_WRITE_CTRL`); the W load is unchanged.
   Both copies previously had to be `NORMAL_FV` because the alternatives that
   benchmark well are corrupting. CLEAN is the one non-allocating policy that
   stayed correct (see the 64-run table below), so it is safe to adopt.

Verified in the lowered TIR rather than assumed: `for bx in T.thread_binding(56)`,
`start = bx * 36 + T.min(bx, 32)`, `for s in T.serial(T.max(cnt, 1))` (dyn) or
`s = T.min(step, cnt - 1)` (recompute), read `l2_cache_ctrl=5` (= NOTALLOC_CLEAN),
write `l2_cache_ctrl=0` (= NORMAL_FV). Host-side coverage check over both shapes:
every tile is covered exactly once, no duplicate and no overflow.

## (b) Correctness — my own self-check, v33 as shipped

`run(x, w, 1e-6)`, 5 runs per shape per mode, truth is the RMSNorm definition in
float64 **on the CPU** (device `.float64()` can silently stay float32, so a
device-side "float64" reference is not trustworthy). Sanity bound: max_abs < 0.06.

| Shape | Mode | max_abs per run (5 runs) | NaN | Shape/dtype |
|---|---|---|---|---|
| 8192x4096 | dyn (default) | 0.03125, 0.03125, 0.03125, 0.03125, 0.03125 | none | (8192,4096) bf16 |
| 8192x4096 | recompute | 0.03125, 0.03125, 0.03125, 0.03125, 0.03125 | none | (8192,4096) bf16 |
| 64x4096 | dyn (default) | 0.015625, 0.015625, 0.015625, 0.015625, 0.015625 | none | (64,4096) bf16 |
| 64x4096 | recompute | 0.015625, 0.015625, 0.015625, 0.015625, 0.015625 | none | (64,4096) bf16 |

All 20 runs pass; worst error 0.03125 against a 0.06 bound, and every run is
bit-identical to its siblings (zero run-to-run spread).

A separate default-path run (imported with **no** attribute overrides, i.e. exactly
what the gate sees, seed 7) gave the same picture or better: 8192x4096 rows=4 max_abs
0.03125 on all 5 runs, 64x4096 rows=1 max_abs **0.0** on all 5 runs, no NaN, correct
shape and bf16 dtype, with `LOOP_MODE="dyn"`, `N_CORES=56`,
`L2_READ_CTRL="NOTALLOC_CLEAN"`, `L2_WRITE_CTRL="NORMAL_FV"` read straight from the
module.

## (c) Only the geometry (plus the read policy) changed

`diff -u v23/kernel.py v33/kernel.py` touches exactly: the module docstring, the
`N_CORES` value, the two new module constants and their comment, `_rows_per_core`'s
predicate and error text, the `tiles/nbase/rem/iters/dyn` block in `_build`, the
`start`/`cnt`/`s`/loop-header lines and the `l2_cache_ctrl` arguments in the two
pipelines. No change to the SimdVF body, the reduce tree, the register strategy,
`stages`/`yver` ((4,2) for rows>1, (2,1) for rows==1), or the rows=4 fast path.
v23 and every other candidate are untouched; `__pycache__` was not written;
AgentGates was not edited.

Shipped-bytes provenance: the numbers above were re-measured against
`kernel.py` at md5 `b07a95300bf895a383060f7226a8ed1f` (the file as it stands), not
against an intermediate edit. v23's `kernel.py` md5 remains
`0a1e6e2cab9a26ffabb5b04ddb1ee667`, i.e. unmodified.

## (d) Prefill and decode still work

Yes. 8192x4096 (rows=4 explicit body) and 64x4096 (rows=1 looped body) both compile,
run 5/5 correct, and return the right shape and dtype in both loop modes. Decode's
64 tiles over 56 cores exercises the ragged path hardest (8 cores do twice the work
of the other 48), which is exactly the case the split was written for.

## (e) WARNING — do not treat these L2 policies as wins

Measured by the Lead, 8 seeds x 8 runs = 64 runs per policy per shape, at 56 blocks:

| Policy (read path) | prefill 8192x4096 | decode 64x4096 |
|---|---|---|
| NORMAL_FV | 0 / 64 bad | 0 / 64 bad |
| NOTALLOC_CLEAN | 0 / 64 bad | 0 / 64 bad |
| NOTALLOC_DROP | 0 / 64 bad | **56 / 64 BAD (worst 16.6)** |

`NOTALLOC_DROP` benchmarks fastest and is **unsafe at decode**; `NORMAL_LV` on the
read path is equally broken. They look fast only because they silently corrupt
output, so their benchmark numbers are not wins and must never be quoted as such.
This is why v33 uses CLEAN and not DROP.

Performance context for the two changes (Lead's paired quiet-window measurements at
56 blocks, vendor/candidate ratio, higher is better):

| Shape | NORMAL_FV read | NOTALLOC_CLEAN read |
|---|---|---|
| prefill 8192x4096 | 1.0170 [1.0144, 1.0179] | 1.1185 [1.1179, 1.1193] |
| decode 64x4096 | 1.6500 | 1.6699 |

CLEAN versus FV on the candidate itself: 1.103x prefill, 1.009x decode. For the
geometry alone, v23's 64-block launch measured 0.0760-0.0766 ms on prefill versus
0.0549-0.0556 ms at 56 blocks against a 0.0555 ms vendor reference (ratio ~0.726
FAIL versus ~1.01 MATCH, ~1.14x), and decode 1.22 versus 1.53-1.54.
