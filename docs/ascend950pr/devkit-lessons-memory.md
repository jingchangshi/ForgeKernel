# Devkit lessons: memory access / DMA / L2 / UB banks → RMSNorm prefill

Author: `devkit-mem-explorer` (read-only explorer). Task: `task-2`. Date: 2026-11-03.
Sources: `~/workspace/asc-devkit-master` (CANN examples + `docs/zh`), `~/workspace/tilelang`
(TileLang-Ascend dialect), `~/workspace/tilelang-mlir-ascend`.
Target: prefill 8192x4096 bf16, `docs/ascend950pr/candidates/rmsnorm_simdvf_v23/kernel.py`,
aggregate 1.97 TB/s read+write vs vendor 2.42 TB/s.

Labels: **[measured]** = number from a devkit table I read; **[doc]** = devkit states a rule without a
number; **[grep]** = TileLang exposure proven by the cited source line; **[spec]** = my reasoning, not
established by either repo.

---

## 0. Re-baseline the numbers we are comparing against

The devkit's own Ascend 950 GM figure is **~1.6 TB/s**, not 2.4-3.1:

- `05_best_practices/04_memory_access/data_copy/README_en.md:149` **[measured]** — "For Ascend 950
  Series, with GM bandwidth approximately 1.6TB/s, the theoretical duration is approximately 188.74us.
  In the large block scenario, GM to UB `aiv_mte2_time` is 185.66us". A pure single-pass GM->UB copy
  reaches ~peak HBM on 950.
- `data_copy/README_en.md:217` **[measured]** — "950PR has an L2Cache size of **128MB**".
- `docs/zh/guide/programming_guide/advanced_programming/memory_model/cache_coherence.md:18` **[doc]** —
  L2 is shared across cores, 128MB, line granularity "128B/512B".

Consequence: our microbench rates (read 2.35, write 3.08 TB/s) and the vendor's 2.42 TB/s are
**L2-inclusive**, not HBM. 3.08 > 1.6 means the write microbench was absorbing writes into a 128MB L2.
Our prefill footprint is 64MB X + 64MB Y = **128MB = exactly L2 capacity**, so the kernel runs at the
eviction boundary. The gap is therefore not "we lose HBM bandwidth" but "we thrash a 128MB L2 with a
128MB working set and pay dirty write-backs inside the read stream" — the mechanism the devkit
documents explicitly in §2.1.

---

## 1. Canonical data-movement idioms vs TileLang exposure

| Devkit idiom | Source + quote | TileLang-Ascend exposure | Effect on our prefill |
|---|---|---|---|
| One big contiguous burst instead of per-row loops | `memory_access/efficient_transfer_api.md:5` **[doc]**: a 2KB x 16-row `for` loop becomes one `DataCopyParams{blockCount,blockLen,srcStride,dstStride}` copy — "一次搬完…每次搬运 32KB" | **[grep]** automatic: `src/ascend/op/copy.cc:286-327` `PlanMTECopy` collapses contiguous regions to `burst_num=1, burst_len=plan_total_bytes` | **Already satisfied.** `T.copy(X[base:base+rows*d], x_ub)` (`kernel.py:52`) lowers to one 32KB burst. No gain available. |
| >=16KB per single transfer | `memory_access/transfer_larger_data_blocks.md:5` **[doc]**: "单次搬运数据长度16KB以上时，通常能较好地发挥出带宽的最佳性能" | **[grep]** not a knob; only region extent is user-controlled | **Already satisfied** (32KB burst). Explains r80's "DMA chunk size stops mattering once the L2 policy is set". |
| GM addresses 512B-aligned | `memory_access/gm_address_512b_alignment.md:5` **[measured]**: worst case "32B对齐场景只能达到512B对齐场景的70%" (both directions) | **[grep]** neither enforced nor exposed — `grep -n '512' src/ascend/op/copy.cc` = 0 hits; UB merge aligns to 32B only (`tilelang/ascend/pipeline.py:19` `_MERGE_UB_ALIGNMENT = 32`, `merge_ub_common.h:77-85` `AscendScopeAlignBytes` returns 0 for `shared`) | **Compliant by construction**: d=4096 bf16=8KB, rows*d=32KB, `base` a multiple of 32KB, and kernel args are 512B-aligned by the runtime (`avoid_same_address_access.md:19` **[doc]**). Verify once by dumping the lowered `.asc` and reading the emitted `asc_copy_gm2ub_align` byte args. |
| 128B alignment suffices on 950 (512B on A2/A3) | `data_copy/README_en.md:391` **[doc]** — "continuous transfer byte count satisfies Atlas A2/A3 Series 512B, **Ascend 950 Series 128B** alignment" | n/a | Satisfied (32KB bursts). |
| Stagger per-core traversal order | `avoid_same_address_access.md:17` **[doc]** — requests are processed at 512B granularity; concurrent requests in one 512B range "会被串行处理", worse with more cores. Fix at `:31` / `data_copy/data_copy_ub.h:106`: `curMBlockIdx = blockGroupStart + (mBlockIdx + blockIdx) % numBlocks` | **[grep]** pure indexing — `T.Kernel(N_CORES) as core` plus any `base` expression | **Not applicable; our layout is already the good one.** All 64 cores touch disjoint 32KB blocks. The devkit's own deltas settle it: 950 **UB** path only 342.29 -> 335.64us (+2%, `README_en.md:368-369`) because cores own disjoint data; the +97.5% win (369.99 -> 187.35us, `:378-379`) is the **L1/AIC** case where 32 cubes re-read one matrix. Already gated by us: `evidence/r54-partition-and-gate-fix.json` — contiguous beat interleaved by +8.9%. Dead lever. |
| `DataCopyPad` for unaligned tails | `data_copy/data_copy_ub.h:41-53`; `memory_access/reduce_invalid_data_transfer.md:11` **[doc]** Compact beats Normal for many sub-32B blocks | **[grep]** exposed as `pad_value` / `data_select` (right-pad to 32B): `tilelang/ascend/language/copy_op.py:239-287`, lowering `src/ascend/op/copy.cc:292-303` | Irrelevant: d is a multiple of 32B. |
| Loop-mode / NDDMA multi-dim strided DMA | `memory_access/reduce_non_contiguous_transfer.md:68-88` (`LoopModeParams` + `SetLoopModePara`), `:154-158` (`NdDmaLoopInfo<2>`, `asc_ndim_copy_gm2ub`, 950-only) | **[grep]** **absent**: `grep -rn 'NdDma\|NDDMA\|LoopMode\|ndim_copy' src/ascend tilelang/ascend` = 0 hits | Would matter for column-major/gather patterns; RMSNorm prefill is row-contiguous. **Will not transfer.** |
| VECIN/VECOUT binding to delete a UB->UB copy | `memory_access/vecin_vecout_reuse.md:5,94` **[measured]** — `TQueBind<VECIN,VECOUT,BUFFER_NUM>` makes `aiv_vec_time` "几乎缩减为0" | **[grep]** **absent**: `T.alloc_shared` (`allocate.py:15`) is the only UB allocator; `grep -rn 'VECIN\|VECOUT' src/ascend tilelang/ascend` = 0 hits | We have no UB->UB copy. Transferable residue: on 950 VECIN/VECOUT/VECCALC are the same physical UB (see §3.3). |

### Double buffering / multi-buffer depth

- `00_vector_compute/add_high_performance/README_en.md:560-564` **[doc]** — sizing rule: double buffer
  holds x/y/z x ping/pong = 6 blocks => per block ~= `UBSIZE/6`; 950: `256KB/6 = 42.67KB` -> 21840 half,
  rounded down to 512B-aligned `dataCopyLen=21760` ("85 x 512B"). Also `:263-265`: "dataCopyLen is not
  the bigger the better" (16384 -> 16512 was flat).
- `add.asc:309` `MAX_DATA_COPY_LEN=21840` (3510) vs `16384` (2201); `add.asc:417-426` per-arch case table
  (`largeDataCopyLen=21760`, `bankConflictDataCopyLen=21632` = 21760-128 = one 256B bank-group rotation,
  `multiCoreSplitM=64`).
- Manual ping-pong pattern: `add.asc:187-240` — 6 static `LocalTensor`s at explicit UB offsets, 4 flags
  pre-set, per iteration `WaitFlag<V_MTE2>` -> `DataCopy` in -> `SetFlag/WaitFlag<MTE2_V>` -> compute ->
  `SetFlag<V_MTE2>` -> `SetFlag/WaitFlag<V_MTE3>` -> `DataCopy` out, plus trailing drain waits.
  `bank_conflict_nd2nz.asc:58-71,87-89` is the same with VECIN/VECOUT tensors.
- Devkit's own verdict on depth **[measured]**: Case 3 -> Case 4 (double buffer) is only
  268.5 -> 264.02us (A2, **-1.7%**, `README_en.md:333`) and "double buffer benefits are limited,
  indicating the bottleneck is in transfer speed itself" (`:351-352`); 950: 251.684 -> 247.665us.
  Consistent with our r80 finding (stages 2/3/4/5/6 exhausted; stages=4 best).
- TileLang equivalents **[grep]**: `T.annotate_buffer_versions({buf: n})`
  (`tilelang/ascend/language/annotations.py:19-45`; `{buf:1}` opts out, `(n, mode)` fixes count and
  indexing mode `auto\|iteration\|counter`), `T.annotate_manual_multi_buffer` (`:114-141` — user writes
  the version index, pass only emits set/wait_flag), `T.Pipelined(iters, num_stages=, order=, stage=)`,
  `T.ascend_set_flag/ascend_wait_flag/ascend_pipe_barrier/ascend_get_buf/ascend_rls_buf`
  (`sync.py:8-202`), `T.Task(latency=, ii=)` / `T.Stage(stage=)` / `T.PerCoreTask()` (`schedule_hint.py`).
- Usage audit in our repo `[grep]` over `candidates/*/kernel.py`: `num_stages` 68 files, `order/stage`
  only v30-v32, and **0 uses** of `T.Task`, `T.Stage`, `T.PerCoreTask`, `assume_conflict`,
  `assume_no_conflict`, `annotate_manual_multi_buffer`, `ascend_set_flag/wait_flag`. r80 lists "manual
  Pipelined order/stage" as exhausted, but the explicit-latency form and manual multi-buffer annotation
  were never tried. **[spec]** Low expected value — our bottleneck is DRAM/L2, not issue order.

---

## 2. L2 cache modes — the one place the devkit's biggest measured win overlaps our gap

### 2.1 The mechanism (the devkit's documented version of our read/write-turnaround theory)

`docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md`,
【优先级】高 (`:3`):

- `:21` **[doc]** — "L2 Cache的带宽相比GM的带宽有数倍的提升": L2 BW is several x GM BW.
- `:33` **[doc]** — "当数据访问总量超出L2 Cache容量时…替换过程中**旧数据需要先写回GM（此过程中会占用GM带宽）**，
  旧数据写回后，新的数据才能进入L2 Cache."
- `:35` **[doc]** — "对于只访问一次的 Global Memory 数据设置其访问状态为不进入L2 Cache…避免一次性访问的数据替换有效数据."

That is exactly our shape: X is read **once**, Y is written **once**, footprint = L2 capacity. With both
DMA paths on `NORMAL_FV` (allocate + first-victim), every X read allocates an L2 line and evicts a dirty
Y line, forcing a GM write-back inside the read stream.

`SetL2CacheHint` mapping **[grep]**: `docs/zh/api/SIMD-API/basic_api/data_structures/GlobalTensor/SetL2CacheHint.md:82`
defines `CACHE_MODE_DISABLE = 0 // 不启用L2 Cache`, `CACHE_MODE_NORMAL = 1`, default NORMAL; and
`impl/basic_api/utils/kernel_utils_macros.h:233-244` encodes the hint in GM-address bits 60-62:
DISABLE = `0b100`, NORMAL = `0b000`, PERSISTENT = `0b010`. So basic-API `CACHE_MODE_DISABLE` ==
load-enum value **4 == NOTALLOC_KEEP**, and `CACHE_MODE_NORMAL` == **0 == NORMAL_FIRST_VICTIM**.
**This is the decisive reading: the devkit's Case 5 knob is `NOTALLOC_KEEP` on the read path — the exact
policy our v23 does not use.**

Enum semantics **[doc]** `docs/zh/api/SIMD-API/c_api/defs/enum/asc_load_l2_cache_mode.md`:
NORMAL_FIRST_VICTIM=0, NORMAL_LAST_VICTIM=1, NORMAL_PERSISTENT=2, NOTALLOC_KEEP=4 ("每次都直接从GM中读取，
并且保持已有Cache Line的状态不变"), NOTALLOC_CLEAN=5 (keeps the line but marks it **Clean**, so it is
dropped without write-back), NOTALLOC_DROP=6 (existing line discarded).
`asc_store_l2_cache_mode.md`: NORMAL_FIRST_VICTIM=0, NORMAL_LAST_VICTIM=1, NORMAL_PERSISTENT=2,
NOTALLOC_CLEAN=4, WT_SHARE_FIRST_VICTIM=12 — "采用write-through（直写）策略…写入数据时同步更新GM和L2 Cache，
Cache Line无需标记为Dirty，也无需等待Cache Line被替换或执行Clean操作后再回写GM", plus the Clean/Dirty
definition. **WTS_FV and NOTALLOC_CLEAN are the only documented ways to make dirty lines not exist.**

### 2.2 Measured wins

| Result | Number | Source |
|---|---|---|
| add Case 4 -> 5 (double buffer -> +L2 bypass), A2 | 264.02 -> **187.68us (-28.9%)**, MTE2 250.5 -> 176.0us (-29.8%) | `add_high_performance/README_en.md:384-388` |
| add Case 4 -> 5, Ascend 950 | 247.665 -> **182.04us**; mixed BW 1.712 -> **2.256 TB/s** | `:556-558` |
| add Case 6 (= Case 5 + bank-conflict fix) | 184.52us (-1.7% vs Case 5); `aiv_vec_time` 12.846 -> **6.954us (-45.9%)**; MTE3 -24.2%; mixed BW **2.3456 TB/s** | `:461-466`, `:499-505` |
| Why mixed BW exceeds GM BW | "z writes hit the L2 Cache, which provides high bandwidth" | `:544` |
| data_copy L2 sliced-reuse (4 rounds) | 741.58 -> 354.95us, hit rate 0.35% -> 66.64% | `data_copy/README_en.md:312-313` |

Code: `add.asc:242-247` `ProcessDoubleBufferL2Bypass()` = `xGm.SetL2CacheHint(CACHE_MODE_DISABLE);`
`yGm.SetL2CacheHint(CACHE_MODE_DISABLE);` then `ProcessDoubleBuffer()`. `data_copy_ub.h:93` sets the same
hint deliberately so that access order and cache policy are not mixed in the same-slice experiment.
Note the ordering in the devkit's own narrative: **L2 policy is worth ~29%, bank-conflict layout ~1.7%
end-to-end** (`README_en.md:350-352, 405-407, 464`).

### 2.3 TileLang exposure [grep]

- Full policy name table: `tilelang/ascend/language/copy_op.py:167-199` — load `NORMAL_FV=0, NORMAL_LV=1,
  NORMAL_PERS=2, NORMAL_PREF=3, NOTALLOC_KEEP=4, NOTALLOC_CLEAN=5, NOTALLOC_DROP=6, IDS_* =8-11,
  EXCLUSIV_* =12-15`; store `NORMAL_RED=3, NOTALLOC_CI=4, NOTALLOC_PW=5, NOTALLOC_PI=6, NOTALLOC_RED=7,
  WBH_FV..WBH_RED=8-11, WTS_FV..WTS_RED=12-15`. Names are suffix-matched (`_normalize_l2_cache_ctrl`,
  `:202-227`) and raw ints accepted, so **every hardware policy in the C API is reachable**, including
  `WTS_FV` — already used as the cache-flush idiom in `tilelang/profiler/msprof.py:35`.
- Lowering is straight-through, no filtering: `src/ascend/op/copy.cc:234,317-327` ->
  `ascend_copy_gm_to_ubuf` / `ascend_copy_ubuf_to_gm` -> `codegen_ascend.cc:1407-1417`
  (`asc_copy_gm2ub_align(..., static_cast<asc_load_l2_cache_mode>(...))`) and
  `:1425-1432` (`asc_copy_ub2gm_align(..., static_cast<asc_store_l2_cache_mode>(...))`).
- **Direction-dependent defaults** (explains r65): load default **0** = NORMAL_FV (`copy.cc:234`
  `L2CacheCtrlOr(0)`; header comment `src/ascend/op/copy.h:44-47` "GM loads default to 0, UB->GM stores to
  NOTALLOC_KEEP") vs store default `copy.cc:316` `constexpr int kStoreDefaultL2CacheCtrl = 4` = NOTALLOC_CI.
  Our unset-store microbench measured 1.138 TB/s and NOTALLOC_CI measured 1.134 — the same thing. The
  2.7x win we found was *undoing a bypass default with an allocating one*, and the devkit's Case 5 is the
  opposite move on the read path.
- L2 **way/policy partitioning, persistency allocation ranges**: not exposed (0 hits repo-wide for
  `SetL2BufferHint`, `L2CacheCrossCore`, `CACHE_MODE_ALLOCATE/STATIC/DYNAMIC`).
- `tilelang-mlir-ascend` (the other fork) exposes **no cache policy at all**: `grep -rIln 'l2_cache_ctrl'` =
  0 hits; `docs/Tilelang.language/内存操作/T.copy.md` lists only src/dst/`coalesced_width`/`size`, and
  states `coalesced_width` is "npuir 目标下传入会被忽略".

### 2.4 What we already tested and the confound that invalidates it

`evidence/r65-l2-breakthrough.json`: write sweep — unset 1.138, NORMAL_FV 3.079, NORMAL_RED 3.089,
NOTALLOC_CI 1.134, NOTALLOC_PW 1.132, WBH_FV 1.112, WTS_FV 1.154, WTS_LV 1.12. Read sweep — unset 2.104,
NORMAL_FV 2.12, NORMAL_LV 2.351, NORMAL_PREF 2.139, IDS_FV 1.819, EXCLUSIV_FV 2.104, NOTALLOC_KEEP 0.906.
Kernel arms ever gated (`grep` over `candidates/*/kernel.py`): read in {unset, NORMAL_FV, NORMAL_LV,
NORMAL_PREF}; write in {unset, NORMAL_FV, NORMAL_RED}. **NOTALLOC_*, WTS_*, WBH_*, IDS_*, EXCLUSIV_* were
ranked by microbenchmark only, never gated in the kernel.**

**[spec] Why that ranking cannot transfer:** the microbenchmarks re-read / re-write a small buffer, so
NORMAL_LV wins by hitting L2, while NOTALLOC_KEEP is forced to real GM and shows 0.906 TB/s — itself far
below the devkit's 1.6 TB/s, i.e. that microbench was not even reaching GM peak. Symmetrically, an
L2-resident write loop makes allocating NORMAL_FV look like 3.08 TB/s and write-through WTS_FV look slow
(1.154) precisely because WTS cannot hide in L2. In the real kernel the touched set equals L2 capacity and
each line is touched once, so both rankings are expected to invert or flatten. Our own r66 note says it:
"the microbenchmark winner on the read path (NORMAL_LV) made the kernel slower; only the gate can rank
policy choices." r80's `levers_exhausted` entry "L2 cache policy per direction" is therefore **overstated**:
only 4 of 12 load policies and 3 of 9 store policies were ever gated.

### 2.5 Actionable experiments (ranked; all one-line edits to `kernel.py:52/:155/:173/:196`)

1. **Read path -> `NOTALLOC_KEEP`, write stays `NORMAL_FV`** — literally the devkit Case 5 knob on a
   once-read workload. Expect either a Case-5-scale win or a clear regression; the sign is genuinely
   unknown because our only data point is the confounded microbench. Highest information per gate.
2. **Read path -> `NOTALLOC_CLEAN`** then **`NOTALLOC_DROP`** — CLEAN marks an existing same-address line
   Clean so eviction needs no write-back (`asc_load_l2_cache_mode.md`); if our cost is write-backs rather
   than allocations, CLEAN should beat KEEP.
3. **Write path -> `WTS_FV`** (write-through, store enum 12) with read NORMAL_FV — no dirty lines can ever
   exist, so no store write-back can interleave into the read stream; cost is losing L2 absorption of Y.
4. **Write path -> `NOTALLOC_PW`**, and the **both-bypass arm** (read NOTALLOC_KEEP + write NOTALLOC_PW /
   unset) — the honest test of whether our aggregate loss is L2-caused at all: if pure-streaming DMA beats
   1.97 TB/s aggregate, the whole 128MB-on-128MB story is confirmed.
5. **`NORMAL_PERS` (2) on the W-only copies** (`T.copy(W[:d], w_ub[:d])`, `kernel.py:45`) — the one tensor
   in this kernel that all 64 cores re-read, so the only tensor where the devkit says to keep L2 reuse
   (`l2_cache_mode_config.md:35`, `SetL2CacheHint.md`: persistent "计划于Ascend 950PR…提供支持" i.e.
   supported on 950 via the C-API enum even though the basic-API hint is still marked in development).
   Small absolute upside (8KB) but zero risk.
6. **Mixed grid** read {FV, LV, PREF, NOTALLOC_KEEP, NOTALLOC_CLEAN, NOTALLOC_DROP} x write {FV, WTS_FV,
   NOTALLOC_PW}: 18 arms, no dataflow change (correctness is preserved by construction), ~42s/window each;
   judge only on paired candidate-vs-vendor windows per `final-status.md` §10 (ratio is the stable metric;
   <20% single-window deltas are not evidence).

Explicit non-actions: **L2 Cache splitting / sliced repeated access will not help us** —
`memory_access/l2_cache_split.md:178` **[doc]**: "只读取一次的数据不存在跨Round复用，调整L2 Cache切分顺序
通常收益有限." Same for `data_copy` Optimization Point 3 (its 741->355us was a 4-round repeat),
`03_basic_api/10_cache_control/add_l2_cache_split`, and `04_advanced_api/00_matmul/matmul_l2cache`
(all assume repeated rounds).

**Caveat on policy names [grep]** `copy_op.py:167-199` is the *union* of two hardware spaces. The C API
we actually compile against defines only `asc_load_l2_cache_mode {0,1,2,4,5,6}` and
`asc_store_l2_cache_mode {0,1,2,4,12}` (`include/c_api/defs/enum.h:17-32`), while `codegen_pto.cc:2651-2659`
only bounds-checks to [0,16). So `NORMAL_PREF=3`, `NOTALLOC_RED=7`, `WBH_*=8..11`, `IDS_*=8..11`,
`EXCLUSIV_*=12..15`, `NOTALLOC_PW=5`, `NOTALLOC_PI=6`, `NOTALLOC_CI=4` are PTO-space names; several of the
values our r65 microbench swept (read NORMAL_PREF, IDS_FV, EXCLUSIV_FV; write NOTALLOC_PW/PI/RED, WBH_FV,
WTS_LV=13) are **outside the documented C-API enum** and their measurements should not be read as
"documented policies behaved badly". The arms that *are* documented and still ungated in the kernel are:
load **4 NOTALLOC_KEEP**, **5 NOTALLOC_CLEAN**, **6 NOTALLOC_DROP**, **2 NORMAL_PERSISTENT**, and store
**12 WT_SHARE_FIRST_VICTIM**. Those five are the trustworthy experiment set.

---

## 3. UB bank conflicts on 3510

### 3.1 Geometry and rules [doc, verified 3510]

`docs/.../memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md:11,16` and
`05_best_practices/04_memory_access/bank_conflict_3510/README_en.md:30-36,58-60`:

- 256KB UB = **8 bank groups x 2 banks**, bank = 16KB = 512 rows x 32B, **low-bit interleaved**.
  `UB_ADDR[17:0] = {BANK_DEPTH[8:0], BANK[0], BG[2:0], BANK_WIDTH[4:0]}` (9+1+3+5 bits).
- One beat = 256B = 8 DataBlocks of 32B; with `b = (base >> 5) & 0xF`, a beat's 8 blocks land on banks
  `b..b+7`, i.e. **BG 0-7 exactly once each** (`README_en.md:50-60`).
- Rules (`avoid_bank_conflict_npu_arch_3510.md:37-41`): a bank group per beat supports **2 reads, or 1 read
  + 1 write, and never 2 writes**. Hence: read-read = 2 reads on the same *bank* or >2 reads on the same
  *group*; write-write = >=2 writes on one group; read-write = same bank, or 2 reads + 1 write on one group.
- `bank_conflict_3510/README_en.md:110` **[doc]** — "In Ascend 950PR&950DT products, the scenario where
  src0/src1/dst starting addresses are distributed in different banks of different bank groups does not
  exist, and **the read-write conflict cannot be avoided in two-read-one-write scenarios**."

Measured magnitudes **[measured]** (`bank_conflict_3510/README_en.md:114-139`):

| Case | Situation | Cycles | Note |
|---|---|---|---|
| 1 | same-bank reads + RW conflict | 1,332,802 | LDU 40.9% / STU 17.6% |
| 2 | RW only | 1,470,520 | STU 27.1% |
| 6 | same bank for all three | 1,420,560 | STU 23.8% |
| **7** | same as 6 **with loop unrolling** | **1,330,688** | **+6.3% perf** (`:120`) |
| 8 | compute-heavy | 1,810,355,431 | conflicts become irrelevant |
| 11 | DataBlock-contiguous (row-wise) | 443,135 | 0 / 0 conflict |
| **12** | DataBlock-non-contiguous (column-wise, 256B gaps) | **1,510,497** | **3.4x slower, STU 53.5%** |
| 13/14 | repeated same-address reads | 440,911 / 440,847 | merged by HW, **no penalty** |
| 9/10 | gather indices dispersed vs same-subbank | 650,670 / 750,707 | -15.4% |

Documented fixes (`avoid_bank_conflict_npu_arch_3510.md:255-279`, `README_en.md:502`): (1) overlap two read
addresses so the HW merges them into one instruction; (2) for gather/scatter, disperse across subbanks of
different groups; (3) **iterate row-wise rather than column-wise so a repeat's 8 DataBlocks land in
different bank groups**; (4) for continuous aligned load/store, use loop unrolling.
Metrics: `aiv_vec_ldu_cflt_ratio`, `aiv_vec_stu_cflt_ratio` via `msopprof --aic-metrics=Default`
(`README_en.md:104-108`); `data_copy` uses `msopprof --ai-core=on --aic-metrics=L2Cache` and ships
`ResourceConflictRatio.csv`.

### 3.2 Applied to our kernel (rows=4, d=4096, bf16) — verdict

Per-instruction footprint **[grep]**: `T.simd.vld(..., dist="UNPK_B16")` reads `lanes//2` = 64 bf16 =
**128B = 4 DataBlocks** (`tilelang/ascend/language/simd.py:401-402`, matching devkit
`docs/zh/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/LoadAlign_continuous.md:147`
DIST_UNPACK_B16 = "按无符号整型u16加载VL/2数据量"). `T.simd.vsts(..., dist="PK_B32")` writes 64 bf16 = 128B.

1. **Reduce loop (`kernel.py:60-68`) is conflict-clean and already implements fix (3).** Four loads at
   `(i*4+k)*CH`, k=0..3 -> byte offsets 0,128,256,384 -> 16 DataBlocks = exactly **2 DataBlocks per bank
   group**, i.e. <=2 reads/group and never 2 on the same bank = the allowed combination. No exposure.
2. **Row pitch is a multiple of the bank-group period**: d = 8192B = 256 DataBlocks, and 256 mod 16 = 0, so
   `r0..r3` bases all map to the same bank. That is a cross-iteration effect, not a same-beat one, so no
   documented conflict rule fires. **[spec]**
3. **The scale/write loops (`kernel.py:135-154`) are a textbook two-read-one-write**, and per `README_en.md:110`
   that combination **cannot be laid out conflict-free on 950**: per iteration we read 128B from
   `x_ub[r + i*64]` (4 DataBlocks), read 256B from `wf[i*64]` (fp32, 64 elements = **8 DataBlocks, all 8
   groups**), and write 128B to `y_ub[r + i*64]`. Any concurrent read+write therefore shares a group with
   the `wf` read. Second-order risk: `x_ub` and `y_ub` bases come from `MergeUBAllocations` at 32B
   alignment (`pipeline.py:19`), so if their offsets are congruent mod 256B the x-read and y-write also land
   on the same groups.
4. **Expected magnitude: LOW.** Our own attribution (`final-status.md` §6) is ~40.5us vec of a 65us kernel,
   which is more vec-bound than the devkit's examples, so a vec-phase win is not automatically invisible —
   but the closest devkit analogue says otherwise: ND2NZ S1->S2 cut `aiv_vec_time` **45.25 -> 9.14us
   (4.95x)** while Task Duration moved only **141.691 -> 140.403us**, because `aiv_mte2_ratio` = 0.958-0.967
   — "**vec time not on the critical path**" (`bank_conflict_nd2nz/README_en.md:448-452,489,506`
   **[measured]**). And add Case 6's 45.9% vec cut was 1.7% end-to-end. A +6.3% VF-level gain on 40.5us of
   vec is ~2.5us of 65us, i.e. ~4%. **[spec]**
5. **Measure before dismissing**: `msopprof --aic-metrics=Default` -> `aiv_vec_ldu_cflt_ratio` /
   `aiv_vec_stu_cflt_ratio` for our kernel. If both are near 0, drop this line of work entirely; if STU
   conflict is in the 20-50% band, item 3 is real.

### 3.3 Bank-level experiments (cheap, kernel-only)

- **Rotate UB bases by 256B** (the devkit's own two mechanisms: `add.asc:332-337` `yAddrPingBC =
  BANK_CONFLICT_DATA_COPY_LEN*sizeof(half)` = a 256B offset, with `BANK_CONFLICT_DATA_COPY_LEN =
  MAX-128`; and `bank_conflict_nd2nz.asc:64-67` `kBank8ShiftBytes = 8*C0_ELEMS*sizeof(half)`,
  `nzBufBaseOffset = 2*ndBufBytes + kBank8ShiftBytes` so stores start at **bank 8** while loads use banks
  0-7). In TileLang this needs pad allocations (128 bf16 = 256B) plus, in practice, a
  `T.assume_no_conflict`/alias hint so `MergeUBAllocations` does not fold them; UB slack is ~40KB
  (x 32KB x4 + y 32KB x2 + w 8KB + wf 16KB ~= 216KB of 256KB), so pads are affordable. **[spec]**
  Reordering the `T.alloc_shared` statements is itself a knob — the pass lays out cliques in allocation
  order (`merge_ub_allocations.cc:300-320` with `AlignUp(cursor, 32)`).
- **Verify the actual offsets first**: `MergeUBAllocations(align_bytes=32, verbose=True)`
  (`tilelang/ascend/transform/__init__.py:176-189`) prints `byte_offsets`, and the lowered `.asc` shows the
  `__ubuf__` pointer arithmetic. Do not guess congruence mod 256B.
- **Interleave the four row-wise scale/write loops into one** — that would create **4 concurrent writes per
  beat window onto identical groups**, which is the forbidden write-write case. Listed as a
  **negative control**, not an optimization; keep rows sequential.
- **Explicit unroll** (case7's +6.3%): exposed as `T.unroll(start, stop, unroll_factor=N)` ->
  `"pragma_unroll_factor"` -> `#pragma unroll N` (`tilelang/ascend/language/loop.py:19-46`). **0 candidate
  uses.** Apply `unroll_factor=2`/`4` to the reduce loops at `kernel.py:60,80,100,120` (the scale/write
  loops are already 4x unrolled at Python level).
- **Split the two-read-one-write into one-read-one-write phases** (pre-multiply `x*wf` into a UB buffer,
  then scale): adds 32KB of UB->GM/GM->UB style traffic and a UB->UB copy — almost certainly worse. For
  completeness only; the devkit says the RW conflict is unavoidable anyway (`:110`).

---

## 4. Write patterns for DRAM page locality / read-write turnaround: **the devkit has no such knob**

Searched the whole devkit for page-locality / DRAM-turnaround guidance — "页本地", "行激活", "page policy",
"turnaround", "burst", "localit", DRAM bank — **nothing is documented** (only the UB bank material of §3
and the L2 line material of §2). The two address-locality rules the devkit does state are:

- request granularity **512B** for MTE2/MTE3/Scalar GM accesses, with same-512B-range concurrency
  serialized (`avoid_same_address_access.md:17`) — a *conflict* rule, not a locality-tuning rule;
- L2 line granularity **128B/512B**, L2 shared across cores (`cache_coherence.md:18`).

So the lead's "128 concurrent streams hurt page locality" theory has a documented **L2-level** analogue
(dirty write-back blocks allocation, `l2_cache_mode_config.md:33`) and **no documented DRAM-level
control**. The closest thing to an ordering lever is the L2 policy itself: read-bypass (NOTALLOC_*) stops
write-backs from interleaving into the read stream, and WT_SHARE_FIRST_VICTIM makes stores synchronous so
no write-back ever appears later. Both are already reachable through `l2_cache_ctrl` (§2.5).

Not exposed in TileLang (all 0 hits) and needed for a genuine page-locality experiment: L2 way/policy
partitioning, GM copy-engine selection (MTE vs NDDMA), device-scope accesses `asc_load_dev` /
`asc_store_dev` that bypass DCache, `asc_datacache_preload` / `asc_dcci`
(`docs/zh/api/SIMD-API/c_api/cache_ctrl/system_cache_overview.md`; the DCache-bypass pass exists only
internally, `src/ascend/transform/mark_scalar_dcache_bypass.cc`, and we have no scalar GM loops for it to
help). If the residual really is DRAM page thrashing, **no kernel-level change available to us can address
it**; it would need a lowering-level addition (copy-engine or GM layout control inside TileLang-Ascend).

---

## 5. What will NOT transfer (stated explicitly)

| Devkit material | Why it does not apply |
|---|---|
| `04_memory_access/bank_conflict_ub/` (8 scenarios, 2390 -> 4167 cycle deltas) | **Atlas A2/A3 only** (`dav-2201`: 48 banks, 16 groups, 3 banks/group, 4KB banks). Different geometry, different rules ("any 2 reads on one group conflict" is 2201-specific). The 950 equivalent is `bank_conflict_3510/` — the task description's list conflated them. |
| ND2NZ / NZ layout, `dstNzC0Stride = tileH + 1`, `make_ascend_compact_nz_layout`, `T.copy(..., transpose=True)`, `asc_set_gm2l1_nz_para` | Cube/L1 fractal-layout concerns. RMSNorm prefill is a pure-vector ND op; TileLang exposes the ND->NZ scatter only on UB->L1 (`copy_op.py:253-256`). |
| Gather/scatter subbank dispersion (case9/10, -15.4%) | No gather/scatter in RMSNorm. |
| `l2_cache_split`, `data_copy` Optimization Point 3, `add_l2_cache_split`, `matmul_l2cache` | All require the same data read over **multiple rounds**; ours is read once (`l2_cache_split.md:178` states that limitation itself). |
| `03_fusion_compute/ssbuf_aiv_aic_comm` (SSBuffer ring, `CrossCoreSetFlag<4,PIPE_S>`, 3KB, `alignas(128)`, volatile) | AIC<->AIV signalling only; **zero GM traffic**. |
| `03_basic_api/06_atomic/*` (`SetAtomicAdd`/`SetAtomicMax` + `DisableDmaAtomic`, `data_movement_with_atomic_operations.asc:37-46`) | No cross-core reduction in RMSNorm. |
| L1 / AIC examples (`data_copy_gm2l1`, `matmul_*`) | Wrong memory level for a vector op. |
| "more cores / even split" (Case 1 -> 2) | Already 64 AIV with a balanced contiguous partition. |

### The vendor RMSNorm sample is **not** a bandwidth-tuning reference (negative result worth recording)

`04_advanced_api/02_normalization/rmsnorm/rmsnorm.asc` **[grep/doc]**: `numBlocks = 1` (`:299`);
`TQue<VECIN,1>` x2 + `TQue<VECOUT,1>` (`:232-234`) — buffer depth **1**, i.e. no double buffering at all;
one whole-tile `DataCopy` in (`:191`), one out (`:221`); `PipeBarrier<PIPE_MTE2>` / `<PIPE_V>`
(`:209,212`); **no `SetL2CacheHint` anywhere**; example shape [4,8,64] float (`:248-250`). Tiling comes
from `GetRmsNormTilingInfo` (`:83-125`): `bsLength` is derived from a stack/temp-buffer byte budget,
capped at `MAX_REPEAT = 255` (`:27,111-113`), rows rounded to `ONE_BLK_FLOAT_NUM = 8` (`AlignToBlock`,
`:29-33`), and `memMode` 0/1/2/3 picks min/max/mid/available stack size (`:157-165`). Useful only if we
ever need to match vendor tiling arithmetic; it offers **no DMA or L2 tricks to steal**, and the operator
we must beat is `aclnnRmsNorm` on 950PR, not this sample. `layernorm` / `layernorm_v2` / `deepnorm` /
`welford` are the same high-level-API shape and were not read further (time-boxed by the lead).

---

## 6. Ranked conclusions for the prefill gap

| # | Technique | Devkit evidence | In v23? | Expected effect | Experiment |
|---|---|---|---|---|---|
| 1 | **L2 bypass on the once-read input**: load `NOTALLOC_KEEP` (= basic-API CACHE_MODE_DISABLE) | `add.asc:242-247`; A2 -28.9%; 950 mixed BW 1.712 -> 2.256 TB/s (`README_en.md:556-558`); mechanism `l2_cache_mode_config.md:33,35` | **No** (reads use NORMAL_FV) | **[spec]** largest ungated upside; sign unknown because only a confounded microbench exists | §2.5 arm 1, gated |
| 2 | **Policies that eliminate dirty lines**: store `WT_SHARE_FIRST_VICTIM`, load `NOTALLOC_CLEAN` | `asc_store_l2_cache_mode.md`, `asc_load_l2_cache_mode.md`, eviction rule `l2_cache_mode_config.md:33` | **No** (never gated) | **[spec]** directly attacks read/write interleaving | §2.5 arms 2-3 |
| 3 | **Both-direction bypass arm** (pure streaming) | same | No | **[spec]** diagnostic: confirms or refutes the 128MB-on-128MB-L2 story | §2.5 arm 4 |
| 4 | **Explicit VF unroll** | `bank_conflict_3510/README_en.md:120` case6 -> case7 **+6.3%** | No (`T.unroll` unused in all 36 candidates) | ~4% end-to-end ceiling **[spec]** | `unroll_factor=2/4` on `kernel.py:60,80,100,120` |
| 5 | **UB base rotation (256B / bank-8 shift)** | `add.asc:332-337`, `bank_conflict_nd2nz.asc:64-67`, Case 6 vec -45.9% | No | **[spec]** <=4-5%; ND2NZ shows a 4.95x vec win can be 0.9% end-to-end | pads + `verbose=True` offsets, then gate |
| 6 | **`T.assume_no_conflict` on versioned x_ub/y_ub** | `schedule_hint.py:196-243` (suppresses conservative cross-iteration deps) | No | **[spec]** unknown; could deepen overlap at zero DRAM cost | one arm with correctness gate |
| 7 | `NORMAL_PERS` on the weight copies only | `asc_load_l2_cache_mode.md`, `l2_cache_mode_config.md:35` | No | small **[spec]** | one arm |
| — | >=16KB bursts, 512B/128B alignment, no per-row DMA loops, contiguous large blocks | `transfer_larger_data_blocks.md:5`, `gm_address_512b_alignment.md:5`, `efficient_transfer_api.md:5`, `data_copy/README_en.md:158-160` (tiny -> large block: **+374.6%** on 950) | **Yes — already optimal** (one 32KB contiguous burst per direction) | 0 | none |
| — | Staggered multi-core access order | 950 **UB** path only +2% (`data_copy/README_en.md:368-369`); contiguous-vs-interleaved already gated +8.9% (`r54`) | n/a (disjoint per-core data) | 0 | none |
| — | Deeper pipelines / more buffer versions | Case 3 -> 4 = -1.7% (`README_en.md:333`); `:351` "the bottleneck is in transfer speed itself" | Yes (stages=4, x4/y2; stages=5 worse) | 0 | none |
| — | L2 split / sliced reuse, ND2NZ, NDDMA, TQueBind, SSBuffer, atomics | various | **Not applicable or not exposed** | 0 | none |

**Bottom line, plainly:** the `04_memory_access` material teaches us **nothing new about transfer shape** —
our DMA is already one large, aligned, contiguous, per-core-disjoint burst per direction, which is exactly
what those examples optimize (their 950 GM->UB best case is 185.66us against a 188.74us theoretical, so
"near-peak copy" is a state we already occupy). Bank-conflict work has a documented but small ceiling here
(+6.3% VF from unrolling; the closest analogue's 4.95x vec win moved end-to-end by 0.9%). The **one
genuinely un-exercised devkit lever is the L2 allocation policy per direction**: the devkit's largest
measured win in `add_high_performance` (Case 5) is the bypass family, and we rejected it based on
microbenchmark evidence alone. A 128MB working set on a 128MB L2 is precisely the regime where that ranking
is least trustworthy, and `final-status.md` §11's "TileLang exposes no control over DRAM page locality or
stream count" is true for *DRAM* but not for *L2 policy* — `NOTALLOC_KEEP`, `NOTALLOC_CLEAN`,
`NOTALLOC_DROP` and `WT_SHARE_FIRST_VICTIM` are all reachable today with a one-line `l2_cache_ctrl`
change and have never been gated.
