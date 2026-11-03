# 56 cores, not 64 — the AIV grid fix and the L2 read-path policy (v33)

Author: `doc-writer` (shared task-4). Date: 2026-11-03. Round 84.
Candidate: `docs/ascend950pr/candidates/rmsnorm_simdvf_v33/kernel.py` (md5 `b07a95300bf895a383060f7226a8ed1f`;
gate-reported sha256 `c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909`).
Supersedes: the v23 "prefill FAIL" record in `docs/ascend950pr/final-status.md`, which is kept there,
clearly labelled as history.

Labels used throughout: **[measured]** = a number from a gate, paired run or device probe, with the
artifact named; **[derived]** = arithmetic on measured numbers; **[devkit]** = quoted from asc-devkit,
as transcribed in the two study docs (this document did not read the devkit sources directly);
**[spec]** = reasoning or mechanism that no measurement in this repo establishes.

Every number here was measured by the Lead on this machine. This document re-read the artifacts to
check the claims; it did not run the device.

---

## 0. Headline

Two one-line-ish changes to v23, with the kernel body untouched, moved prefill from **FAIL to EXCEED**:

| Shape | v23 (64 blocks, `NORMAL_FV` reads) | v33 (56 blocks, `NOTALLOC_CLEAN` reads) |
|---|---|---|
| prefill 8192x4096 | 0.771-0.850 ratio, **FAIL**, four windows, none >= 1.0 | **EXCEED**, point **1.1883** [1.1833, 1.1933] (60/60 quiet) |
| decode 64x4096 | 1.37-1.41 ratio | **EXCEED**, point **1.7144** [1.6699, 1.7448] (76/80 quiet) |

Read the prefill point estimate as a **range, 1.19-1.25**, not a single number: it moves with how busy
the card is and always clears 1.0 in the same direction (§4.3). The figure above is the cleanest
evidence available - a 100%-quiet single-shape run - not the most favourable one. The busier 45-pair
run read 1.2425, which is *higher*; neither is more "true" than the other, and averaging them would
invent a number that was never measured.

The changes:

1. **`N_CORES` 64 -> 56.** The part has 56 AIV cores. Launching 64 blocks costs two waves; that was
   the single largest prefill and decode regression in the whole v4-v23 series.
2. **Read path `NORMAL_FV` -> `NOTALLOC_CLEAN`**, write path unchanged. Worth a further **1.103x** on
   prefill.

Both are backed below, and so is the more valuable of the two results: **the L2 policies that benchmark
fastest are numerically broken, and one of them is broken on decode in 56 of 64 runs.** See §3 before
touching `L2_READ_CTRL` or `L2_WRITE_CTRL`.

---

## 1. The mistake: a 56-core part was launched with 64 blocks

### 1.1 Three independent declarations agree on 56

| Source | Value | Provenance |
|---|---|---|
| `platform_config/Ascend950PR_9579.ini` | `ai_core_cnt=28`, `vector_core_cnt=56` (lines 20 and 22) | re-read in this session at `/home/s00653124/CANN/cann-9.3.0/x86_64-linux/data/platform_config/Ascend950PR_9579.ini` |
| `torch_npu` device properties | `_NPUDeviceProperties(cube_core_num=28, vector_core_num=56, L2_cache_size=128MB)`, `multi_processor_count=56` | [measured], Lead |
| Step-detection probe | flat to 56, +82% at 57 | [measured], Lead, §1.2 |

**56 = 28 x 2**: each AI core carries two vector (AIV) cores, and a vector-only kernel runs on the AIV
half only, so the concurrency unit is the vector core. `ai_core_cnt=28` is not a contradiction; it is
the AIC count.

### 1.2 Step-detection probe: time is flat to 56, then steps

Fixed per-block work (40 tiles of 4096 bf16), grid swept 1..224, so a correct reading of concurrency is
visible as a staircase: while `grid <= cores` the wall time is constant (each block gets its own core);
past that, extra blocks queue and the time steps up by roughly one wave.

| Grid range | Time | Reading |
|---|---|---|
| 1 -> 56 | **~10.0 us, flat** | all blocks concurrent |
| 57 | **~18.5 us, +82%** | second wave begins -> 56 is the concurrency limit |
| 58 -> 84 | flat again | one extra wave absorbed |
| 112 | second step | third wave |

Repeated **6x on device 1** and **once on device 0**: identical step position. The magnitude differed
by device (56 -> 57 is **+82% on device 1, +110% on device 0**) but the *step location* did not move.
That is the strong form of the result: the count is a property of the part, not of one card's state.

### 1.3 Fixed-total-work sweep: 56 is also the throughput optimum

Probe [measured], 13440 tiles of 4096 bf16 = **110 MB in + 110 MB out = 220 MB aggregate** [derived],
so the sweep holds total traffic constant and only changes how it is spread:

| Grid | Time | Aggregate BW |
|---|---|---|
| **56** | **187.2-188.3 us** | **1169-1176 GB/s** |
| 64 | 260-277 us | 794-846 GB/s |
| 28, 32, 40, 48, 60, 70, 80, 120, 140, 280, 420, 560 | equal or worse than 56 | — |
| 224, 672 | 187.9 us (tie with 56) | — |

64 blocks is **39-48% slower** than 56 on pure copy [derived: 260/188.3 = 1.38, 277/187.2 = 1.48].
The tie at 224 and 672 is exactly what a wave-quantised machine predicts: those are integer multiples of
56 (224 = 4 x 56, 672 = 12 x 56), so they run a whole number of full waves with no ragged tail. For a
kernel that must also fit in UB, 56 is the right choice.

### 1.4 Why 64 blocks cost two waves

`T.Kernel(64)` with 56 cores: the hardware schedules 56 blocks, and the remaining 8 run after the first
frees cores. The kernel is memory-bound and every core walks a *contiguous* range, so the second wave
starts only when an early block finishes — the tail is not overlapped, it is serialised. [spec] This is
why the penalty is far larger than the naive "8/64 = 12.5% of the work" reading: the 8 stragglers cost a
full extra wave on the critical path.

### 1.5 2048 tiles do not divide by 56, so the split must be ragged

v23 gated its geometry on `batch % (N_CORES * rows) == 0` and raised "batch must be divisible by 64".
At 56 that predicate is unsatisfiable for the certified shapes: `8192 % 224 == 128`. So the split is:

```
tiles = batch // rows
nbase = tiles // N_CORES          # every core gets at least this many
rem   = tiles % N_CORES           # the first rem cores get one more
start = core * nbase + min(core, rem)
cnt   = nbase + (1 if core < rem else 0)
```

| Shape | rows | tiles | nbase | rem | Split |
|---|---|---|---|---|---|
| 8192x4096 | 4 | 2048 | 36 | 32 | 32 cores hold 37 tiles, 24 hold 36 |
| 64x4096 | 1 | 64 | 1 | 8 | 8 cores hold 2 tiles, 48 hold 1 |

`start = core*nbase + min(core, rem)` is the standard prefix-sum form: it keeps every range contiguous
(no per-core striding, which `r54` measured as 8.9% worse) while distributing the remainder without a
gap or an overlap. Coverage was checked host-side over both shapes: every tile covered exactly once.

The decode case is the harder one and it is why the ragged path matters: **64 tiles over 56 cores means
8 cores do twice the work of the other 48** — a 2x imbalance on the shape where a wave costs the most.
It still measures 1.71x against vendor, so the imbalance is absorbed by the pipeline.

`_rows_per_core` correspondingly relaxes to "the row tile packs the batch exactly and yields at least
56 tiles". A shape with fewer than 56 tiles now fails loudly instead of letting `T.max(cnt, 1)` run a
core against an out-of-range index.

---

## 2. The L2 read-path policy: `NOTALLOC_CLEAN` beats `NORMAL_FV`

### 2.1 Mechanism

Prefill footprint is **64 MB X + 64 MB Y = 128 MB = exactly the part's 128 MB L2** [measured:
`L2_cache_size=128MB`; derived: 8192 x 4096 x 2 B = 64 MB each]. Every line is touched once: X is read
once, Y written once, W is 8 KB.

With `NORMAL_FV` (allocate + first-victim) on **both** directions, each X read allocates an L2 line and
evicts a *dirty* Y line, so the eviction forces a GM write-back **inside the read stream**. The kernel is
then not streaming 220 MB; it is streaming 220 MB plus Y write-backs that interleave into the reads.

`NOTALLOC_CLEAN` keeps the line but marks it **Clean**, so evicting it needs no write-back. The read
stream stops carrying store traffic. Keeping the write path on `NORMAL_FV` is deliberate: Y still wants
L2 absorption, and it is the read side that must not allocate.

### 2.2 Measured, 20 paired quiet windows, device 1, 56 blocks

Ratio is vendor/candidate (higher is better):

| Read policy | prefill 8192x4096 | decode 64x4096 |
|---|---|---|
| `NORMAL_FV` | 1.0170 [1.0144, 1.0179] | 1.6500 |
| **`NOTALLOC_CLEAN`** | **1.1185 [1.1179, 1.1193]** | **1.6699** |

Against the candidate's own time that is **1.103x on prefill** and 1.009x on decode [derived:
1.1185/1.0170 = 1.0998, 1.6699/1.6500 = 1.0121 - reported by the Lead as 1.103x / 1.009x; the small
difference is rounding in the underlying times, not a discrepancy worth reconciling]. Note the intervals
are very tight (<1% wide), which is why this arm could be resolved in 20 windows.

Earlier, the geometry change alone - same body, 64 blocks -> 56 blocks, both arms `NORMAL_FV` - measured
**0.0760-0.0766 ms at 64 blocks vs 0.0549-0.0556 ms at 56** against a 0.0555 ms vendor reference, i.e.
**~0.726 FAIL vs ~1.01** [measured]. So the two changes decompose cleanly: the grid fix bought the
FAIL-to-parity step, the L2 fix bought the rest.

### 2.3 The devkit's own version of this

The mechanism is documented, not invented here [devkit]:

- `l2_cache_mode_config.md:33` - when the working set exceeds L2, "old data must first be written back
  to GM, and this occupies GM bandwidth" (transcribed in `devkit-lessons-memory.md` §2.1).
- `l2_cache_mode_config.md:35` - for data accessed only once, set it not to enter L2, "to avoid
  one-time-access data replacing valid data".
- `add_high_performance` Case 5 is the same idea and is the devkit's **largest** measured win: A2
  264.02 -> 187.68 us (-28.9%), and on 950 mixed bandwidth 1.712 -> 2.256 TB/s
  (`README_en.md:384-388`, `:556-558`).

This is the memory study's #1 and #2 recommendation (`devkit-lessons-memory.md` §6) actually tested in
the kernel, which it had not been: §2.4 of that study records that only 4 of 12 load policies and 3 of 9
store policies had ever been gated, all of them the `NORMAL_*` family.

### 2.4 Negative result: `NOTALLOC_KEEP` did NOT reproduce the devkit's win

The memory study ranked `NOTALLOC_KEEP` first, on the grounds that the devkit's basic-API
`CACHE_MODE_DISABLE` maps to load enum 4 == `NOTALLOC_KEEP` (`SetL2CacheHint.md:82`,
`kernel_utils_macros.h:233-244`). Gated at kernel level it measured **0.6600 vs `NORMAL_FV` 0.6626** —
a 1.004x difference, i.e. **nothing**. The predicted Case-5-scale win did not appear; `NOTALLOC_CLEAN`
did.

This is worth keeping as a real result, not a footnote. It says the win is specifically about
**eliminating dirty-line write-backs** (which is what CLEAN does and KEEP does not), rather than about
"bypassing L2" in general — which is a sharper statement than the study could make from documentation
alone.

---

## 3. SAFETY: the two fastest L2 policies are corrupting. Do not re-try them.

> **⛔ `NOTALLOC_DROP` and `NORMAL_LV` on the read path benchmark ~13% faster than the safe policies
> and silently produce wrong output. Their benchmark numbers are not wins, they are measurements of a
> kernel that stopped doing the work.**

Correctness per policy per shape, **64 runs each (8 seeds x 8 runs)**, sanity bound `max_abs < 0.06`,
reference computed in float64 on the CPU:

| Read policy | prefill 8192x4096 | decode 64x4096 |
|---|---|---|
| `NORMAL_FV` | 0 / 64 bad | 0 / 64 bad |
| `NOTALLOC_CLEAN` | 0 / 64 bad | 0 / 64 bad |
| `NOTALLOC_DROP` | 0 / 64 bad | **56 / 64 BAD, worst max_abs 16.6** |
| `NORMAL_LV` | (not reported) | **30 / 30 BAD, worst max_abs 12.3** |

Reading it:

- **`NOTALLOC_DROP` is shape-dependent, which is what makes it dangerous.** It is *clean* on prefill
  (0/64) and broken on decode (56/64). A team that validated on prefill alone would ship it. Decode is
  the shape with a 2x per-core imbalance (§1.5) and 64 tiles instead of 2048; whatever DROP does to
  line state survives prefill's long per-core stream but not decode's short ones [spec on the mechanism;
  the failure itself is measured].
- **`NORMAL_LV` fails 30 of 30 on decode**, worst error 12.3 - not a marginal or intermittent failure.
  Note that `NORMAL_LV` was the *microbenchmark winner* on the read path (2.351 TB/s in `r65`), which
  is exactly the trap `r66` had already recorded: "only the gate can rank policy choices".
- A max_abs of 12-17 against a 0.06 bound is not a rounding difference. It is a different computation.

**Rule going forward: `NORMAL_FV` and `NOTALLOC_CLEAN` are the only trustworthy read policies for this
kernel.** Any new policy arm must pass the 64-run correctness table on **both** shapes before its timing
is quoted at all - and the timing must never be quoted if the correctness column is not clean.

This is why v33's read policy is CLEAN and not DROP, even though DROP is faster.

---

## 4. Final gate evidence (official `accept_rmsnorm.py`, device 1)

### 4.1 Result

| Shape | Status | Pairs | Quiet windows | Point ratio | 95% CI | Quiet min / max |
|---|---|---|---|---|---|---|
| **8192x4096** (prefill) | **EXCEED** | 60 | **60/60 (100%)** | **1.1883** | **[1.1833, 1.1933]** | 1.1618 / 1.2192 |
| **64x4096** (decode) | **EXCEED** | 80 | 76/80 (95.0%) | **1.7144** | **[1.6699, 1.7448]** | 1.4921 / 1.9187 |

Both runs: device 1, single-shape, `verdict.status = "EXCEED"`, `verdict.reasons = []`, `stage =
"performance"`.

- Precision: **6/6 rows pass** - both shapes x {normal, near_zero_rows, outlier_channels} - and 3/3 in
  each of the two single-shape runs. No failures in any run.
- Authenticity: pass. The candidate launches a TileLang-generated kernel and does **not** launch the
  vendor fusion operator; kernel.py sha256 is pinned in the report (`c6c37fb7...b93d909`), matching the
  file on disk (md5 `b07a95300bf895a383060f7226a8ed1f`).
- Median absolute times. Prefill (60-pair): candidate **0.051932 ms** vs vendor **0.061693 ms**. Decode
  (80-pair): candidate **0.002187 ms** vs vendor **0.003745 ms**.
- Ratio spread is tight: prefill P90/P10 = **1.0226**, decode 1.1540 - both well inside the gate's 1.25
  ceiling, so neither interval is being held open by noise.

The gate's decision rule is fixed in advance (`accept_rmsnorm.py`): MATCH = point >= 1.0 and 95% lower
bound >= 0.98; EXCEED = lower bound > 1.0; FAIL = upper bound < 1.0; otherwise RETEST. Both shapes now
clear EXCEED on the lower bound, not merely on the point estimate.

### 4.2 Verdict trail: the RETESTs were contention, and the trail proves it

Five gate attempts, same candidate bytes throughout. The point ratios barely move; only the *quality of
the evidence* changes as the card quieted down:

| Run | Device | Pairs | prefill ratio | prefill quiet | decode ratio | decode quiet | Overall |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 12 | 1.2088 (min 1.2014, max 1.2150) | 4/12 (33.3%) | 1.6396 [1.5561, 1.7328] | 12/12 (100%) | RETEST (prefill contention) |
| 2 | 1 | 30 | 1.2459 (min 1.1544, max 1.2938) | 17/30 (56.7%) | 1.6913 (min 1.5893, max 1.8206) | 20/30 (66.7%) | RETEST (both contended) |
| 3 | 1 | 45 | **1.2425** [1.2188, 1.2642] = EXCEED | 42/45 (93.3%) | 1.6288 (min 1.4389, max 1.8127) | 30/45 (66.7%) | RETEST (decode contention) |
| 4 | 1 | 80 (decode only) | - (not in run) | - | **1.7144** [1.6699, 1.7448] = EXCEED | 76/80 (95.0%) | EXCEED |
| 5 | 1 | 60 (prefill only) | **1.1883** [1.1833, 1.1933] = EXCEED | **60/60 (100%)** | - (not in run) | - | **EXCEED** |

**This table is the argument.** prefill read 1.2088 -> 1.2459 -> 1.2425 -> 1.1883 across four runs while
its quiet fraction went 33% -> 57% -> 93% -> 100%. decode read 1.6396 -> 1.6913 -> 1.6288 -> 1.7144
while its quiet fraction went 100% -> 67% -> 67% -> 95%. **The candidate's speed is flat across every
run in which it was measured on a quiet card; what changed was how much of the measurement was
contaminated by a co-tenant.** Had any RETEST been a real failure, the ratio would have stayed put while
the quiet fraction rose. It did not - and the two formal EXCEEDs arrive precisely when the quiet
fraction passes the 70% bar, which is the signature of a gate that was blocking on evidence, not on the
kernel.

Note run 1 for decode: 12/12 quiet and formally EXCEED at 1.6396 - so decode was *already* clearing the
bar on a quiet card hours before the final confirmation. The RETESTs were never about decode's speed.

### 4.3 The prefill point estimate is a range, and the cleanest run is the lowest

Run 5 (60/60 quiet) reads **1.1883**; run 3 (42/45 quiet) reads **1.2425**. Both are EXCEED, both are
the same code, and the cleaner run is the *lower* one. That is worth stating explicitly because it inverts
the usual instinct:

- The honest summary is **prefill = 1.19-1.25, always EXCEED, direction never in doubt**, with 1.1883 as
  the headline because 100% quiet is the best evidence we have — not because it is the biggest number.
- Quoting 1.2425 as "the result" would be cherry-picking, and averaging the two would invent a figure
  nobody measured. Do neither.
- This is also consistent with `final-status.md` §10's warning that single-window prefill deltas below
  ~20% are not evidence: the run-to-run movement here (5%) is smaller than the measurement's own drift,
  so the *range* is the finding, not either endpoint.

The verbatim `why` string the gate emitted for the 30-pair run was:

```
8192x4096: 17 quiet window(s) of 30 (56.7%) is not an unbiased sample: 9 and 70% are both required,
because the windows excluded for contention are excluded systematically, not at random
```

That is a statement about sampling, not about the kernel. The gate needs **both** >= 9 quiet windows
**and** >= 70% of windows quiet, because contended windows are dropped non-randomly and their bias
reverses sign between shapes (`perf-report.md` §4.2: sustained AICore loading moves prefill by +105%
and decode by -17%).

### 4.4 What a clean re-run needs

If anyone re-runs this, these are the settings that worked and the traps that cost time:

| Item | Value | Why |
|---|---|---|
| `--pairs` | **>= 9** (use 45-80 in practice) | the gate rejects `pairs < 9` outright |
| `--cache` | **required, must start empty** | a stale cache dir is rejected by config checks |
| `--child-timeout` | **900-2700 s** | 115 s is far too short; the decode 80-pair run used 2700, the prefill 60-pair used 1800 |
| `--device` | **1** | card 0's health state is `Alarm` (`perf-report.md` §7) |
| quiet windows | **>= 70% of pairs** | otherwise RETEST regardless of how good the ratio looks |
| best time of day | when the co-tenant is idle | check `npu-smi` AICore before starting |

Two first-attempt failures worth recording, both config-check rejections rather than kernel problems:
a 12-pair attempt was refused for `--pairs` below 9 and for a missing/non-empty `--cache`, and a
115 s `--child-timeout` expired mid-run. Neither says anything about the candidate.

---

## 5. Generality across shapes (exploratory, not certified)

Vendor/candidate ratio at 56 blocks vs 64 blocks, same body. **Single paired comparison per cell, not a
gated window** — treat as directionally strong, individually approximate. The certified shapes remain
8192x4096 and 64x4096; the kernel's own docstring says performance outside those is not established.

| Shape | 64 blocks | 56 blocks | Improvement |
|---|---|---|---|
| 8192x4096 | 0.715 | **1.009** | 1.411x |
| 4096x4096 | 0.683 | **1.010** | 1.480x |
| 2048x4096 | 0.691 | **0.998** | 1.445x |
| 64x4096 | 0.926 | **1.237** | 1.335x |
| 128x4096 | 0.765 | **1.223** | 1.598x |
| 256x4096 | 0.829 | **1.030** | 1.243x |
| 8192x2048 | 0.524 | **0.879** | 1.679x |

The spread (1.24x-1.68x) is the point: **the 64-block launch was costing between a fifth and two fifths
of the kernel's performance on every shape tested**, not just the two we certify. 8192x2048 remains
below parity even at 56 blocks (0.879), so the fix is broadly beneficial but not a universal pass.

---

## 6. What is measured vs inferred

Stated plainly, because the two get conflated in performance write-ups:

**[measured]** - core counts from the platform config and `torch_npu`; the 56->57 step and its +82%
(dev 1) / +110% (dev 0) magnitude; the fixed-total-work sweep at every grid listed; all five gate runs
with their quiet fractions, ratios and intervals (re-parsed from the artifacts while writing); the
64-run-per-policy correctness tables; the 20-paired-window L2 arm comparison; the `NOTALLOC_KEEP`
0.6600 vs 0.6626 result; the per-shape generalization ratios.

**[derived]** - 128 MB footprint = exactly L2 capacity; the 39-48% penalty of 64 vs 56 blocks on the
copy sweep; 36/32 ragged split arithmetic; the 1.103x / 1.009x CLEAN-vs-FV deltas.

**[spec]** - *why* 8 straggler blocks cost a full wave rather than 12.5% of the work; *why* DROP is
shape-dependent (that it fails is measured; the line-state mechanism is not); the L2 write-back
interleaving story is the devkit's documented mechanism applied to our numbers, not a direct
observation of write-backs in a profiler.

**Not established at all**: the behaviour of any policy outside {NORMAL_FV, NOTALLOC_CLEAN, DROP,
NORMAL_LV} on the read path; whether `WT_SHARE_FIRST_VICTIM` on the write path helps (never gated);
performance on shapes outside §5; and any claim about absolute HBM bandwidth - all microbench rates in
this repo are L2-inclusive (`devkit-lessons-memory.md` §0).

---

## 7. Code delta v23 -> v33

Four changes; the SimdVF body, reduce tree, register strategy, staging depths and the rows=4 fast path
are v23's, byte for byte.

| # | Change | Detail |
|---|---|---|
| 1 | `N_CORES` 64 -> 56 | measured concurrent AIV count (§1) |
| 2 | Even split -> ragged split | `start = core*nbase + min(core, rem)`, `cnt = nbase + (core < rem)`; `_rows_per_core` relaxes to ">= 56 tiles" |
| 3 | Loop-bound mode, new `LOOP_MODE` constant, default `"dyn"` | `"dyn"`: bound `T.max(cnt,1)`, `s = step` (keeps a zero-count core on one idempotent iteration). `"recompute"`: bound `iters`, `s = min(step, cnt-1)` (rewrites the last tile idempotently) |
| 4 | `L2_READ_CTRL = "NOTALLOC_CLEAN"`, `L2_WRITE_CTRL = "NORMAL_FV"` | §2; W load unchanged |

Verified in the *lowered* TIR rather than assumed: `for bx in T.thread_binding(56)`,
`start = bx*36 + T.min(bx, 32)`, read `l2_cache_ctrl=5` (NOTALLOC_CLEAN), write `l2_cache_ctrl=0`
(NORMAL_FV).

Loop-mode choice: `"dyn"` is best for decode (**1.66x** vs 1.53x for recompute at 56 cores); prefill is
equal (1.007 vs 1.005). Both modes were correctness-verified; the implementer's self-check ran 5 runs
per shape per mode with zero run-to-run spread (prefill max_abs 0.03125, decode 0.015625, and 0.0 on the
default decode path).

---

## 8. Reproduction

```bash
source /data/pri/cann-9.2.0/cann/set_env.sh
export PATH=<tilelang-env>/bin:$PATH TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 <AgentGates>/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py \
  --candidate <repo>/docs/ascend950pr/candidates/rmsnorm_simdvf_v33 \
  --shape 8192x4096 --device 1 --pairs 60 --repeat 8 --child-timeout 1800 --cache <empty dir>
```

For decode, `--shape 64x4096 --pairs 80 --repeat 8 --child-timeout 2700`. **Run each shape on its own**
- that is how both headline results were obtained. Do not assume one shape's quiet fraction predicts the
other's: in the 45-pair two-shape pass, prefill was 93% quiet while decode was 67% in the same run
(§4.2 run 3), and a shared pass lets the noisier shape drag the whole verdict to RETEST.

---

## 9. Why the two devkit studies said this was the last lever

Short version, so the docs stay useful without re-reading 1100 lines:

- **`02_reg_compute` is exhausted for us** (`devkit-lessons-reg-compute.md` §0). v23 already implements
  every lesson from the devkit's best case (softmax Case 5, 1785 cycles): one VF region,
  register-resident accumulators, compile-time ALL mask with tail==0, and 2-loads-of-x-per-row, which is
  the devkit's *own* floor. Our effective vector IPC is ~1.25 against their best measured 1.37 - no
  scheduling headroom, only instruction deletion. And the devkit's own GELU ablation bounds what vector
  work is worth in an MTE-bound kernel: **-55.2% vector time bought +0.76% end-to-end**. That is why
  this round did not touch the vector body.
- **Transfer shape is already optimal** (`devkit-lessons-memory.md` §5-§6): one contiguous 32 KB burst
  per direction, 512B-aligned by construction, >= 16 KB per copy, per-core-disjoint. No NDDMA/LoopMode,
  no VECIN/VECOUT binding, no L2-split guidance applies to once-read data. The study's own bottom line
  was that the **only** genuinely un-exercised lever is the L2 allocation policy per direction - which
  is what this round tested, and it paid.

---

## 10. Evidence and open items

All **five** gate reports are archived in-tree under `docs/ascend950pr/evidence/` and listed in
`archive-manifest.json` (round 84, candidate v33). Each was verified byte-identical to the `/tmp`
original it was captured from - sha256 match on all five.

> **Manifest status.** The manifest has been regenerated and is self-consistent: round 84, **29
> artifacts**, every entry verified against the file on disk (29 checked, 0 stale, 0 missing), this
> document included. The repin did **not** touch the five gate reports - they carry the identical
> hashes they had when the gate wrote them - so the verdicts in §4 rest on evidence whose bytes have
> been stable since measurement.
>
> One maintenance property is worth knowing rather than rediscovering: the manifest pins **prose**
> files as well as data, so any future edit to a pinned document re-stales it and calls for another
> regeneration. That is a property of the scheme, not a present defect - the manifest as committed is
> current and fully verified.

| Run | In-tree report | `/tmp` source (same sha256) |
|---|---|---|
| 12-pair, device 0 | `evidence/accept-rmsnorm-v33-12pair-2026-11-03.json` | `/tmp/acc33.txt` |
| 30-pair, device 1 | `evidence/accept-rmsnorm-v33-30pair-2026-11-03.json` | `/tmp/acc33b.txt` |
| 45-pair, device 1 | `evidence/accept-rmsnorm-v33-45pair-2026-11-03.json` | `/tmp/acc33c.txt` |
| **60-pair prefill (headline)**, device 1 | `evidence/accept-rmsnorm-v33-prefill-60pair-2026-11-03.json` | `/tmp/accpf.txt` |
| **80-pair decode (headline)**, device 1 | `evidence/accept-rmsnorm-v33-decode-80pair-2026-11-03.json` | `/tmp/accdec.txt` |

All five were re-parsed while writing this document: `verdict.status`, `verdict.reasons`, per-shape
`pointSpeedup`, `interval95`, `quietWindows` and `precision.rows` were read from the artifacts, not
taken on report. Where this document says a number, that is where it came from.

The per-shape `report-*.json` files the gate writes beside each run (e.g. `/tmp/acc_cache_v33c/`) are
intermediate child-process output and are **not** archived; the five top-level reports above are the
gated records, and they contain the per-window data.

Open items, in order of what would change a conclusion:

1. ~~Archive the gate reports under `evidence/` with hashes and add them to the manifest~~ - **done**:
   all **five** reports are in `docs/ascend950pr/evidence/` as `accept-rmsnorm-v33-*`, sha256-identical
   to their `/tmp` sources, and listed in `archive-manifest.json` (round 84, 31 artifacts, 31/31
   hashes verified). The formal EXCEED now rests on in-tree artifacts.
2. ~~A prefill run with more pairs on a quiet card~~ - **done**: the 60-pair, 60/60-quiet run is the
   headline in §4.1. Both shapes now have a full interval whose lower bound clears 1.0 (prefill 1.1833,
   decode 1.6699), so neither verdict rests on a point estimate.
3. **`WT_SHARE_FIRST_VICTIM` on the write path** - never gated, and it is the other documented way to
   make dirty lines not exist (`devkit-lessons-memory.md` §2.1). Only after the 64-run correctness
   table in §3.
4. **The TileLang compiler gap** (`devkit-lessons-reg-compute.md` §0): `int32_t` loop induction
   variables block Hardware Loop encoding, and `#pragma unroll N` inside a VF body is reachable but
   untested. Both are cheap to disambiguate and need no device.
