# asc-devkit `02_reg_compute` lessons → RMSNorm prefill (TileLang-Ascend SimdVF)

Author: devkit-reg-explorer (explorer, read-only). Date: 2026-11-03.
Task: shared task-1. Primary artifact: `asc-devkit-master/examples/01_simd_cpp_api/05_best_practices/02_reg_compute/softmax_high_performance/`
(gelu_high_performance + gelu_eltwise_high_performance as controls).
Target kernel: `docs/ascend950pr/candidates/rmsnorm_simdvf_v23/kernel.py` (prefill 8192×4096 = 0.771–0.850× vendor, mte2 86–93% busy,
1.97 TB/s mixed R+W vs vendor 2.42 TB/s, vector ≈40.5 µs of a ≈65 µs kernel).

Path abbreviations used below:
- `SOFT` = `/home/s00653124/workspace/asc-devkit-master/examples/01_simd_cpp_api/05_best_practices/02_reg_compute/softmax_high_performance`
- `REG` = `/home/s00653124/workspace/asc-devkit-master/examples/01_simd_cpp_api/05_best_practices/02_reg_compute`
- `DOCS` = `/home/s00653124/workspace/asc-devkit-master/docs/zh`
- `TL` = `/home/s00653124/workspace/tilelang`

---

## 0. Headline verdict (read this first)

The devkit's measured numbers are **vector-pipe** numbers. Every performance table in these examples ends with the same caveat: the
bandwidth-bound case is **MTE-bound and the vector work disappears into the DMA time** (`REG/gelu_high_performance/README_en.md:373` —
"This example is MTE2 bound, with data transfer ratio exceeding 90%, so end-to-end benefit is not obvious"). Our RMSNorm prefill is in
exactly that regime (mte2 86–93%). **Therefore almost nothing in `02_reg_compute` is a direct lever on our 19–23% prefill deficit.**
The one genuinely transferable structural idea is the *shape* of the pipeline (in-flight DMA depth, main/tail split, dual-issue-friendly
instruction mix), and the one genuinely transferable "register compute" win is **already implemented in v23** (per-row accumulators held
in registers, single VF region, no UB round-trip between reduce and scale).

The honest ranking of what this study bought us:

| # | Finding | Transferable? | Expected prefill effect |
|---|---------|---------------|-------------------------|
| 1 | RegBase/one-VF-region is *necessary but not sufficient* — devkit's own GELU data proves vector optimization gives **0.76%** end-to-end when MTE2 is 91% busy | Already done | ~0% (we are past this) |
| 2 | x *is* re-read (accumulate + apply), but so it is in the devkit's best case too — **2 loads/row is the proven floor, not a defect**. The real redundancy is `wf` re-read **once per row per chunk** (`kernel.py:137,142,147,152`) | Yes | −7.3% instructions, −38% UB read bytes/step; **≈0–1 µs end-to-end** (§7) |
| 2b | **v23's effective vector IPC ≈ 1.25** (2620 instr/step ÷ 66.8k cycles), vs the devkit's own best result of **1.37** and its stated "optimal" band of 1.4–1.5 | Diagnostic, not a lever | **No dual-issue headroom remains.** Any further vec win must come from deleting instructions, not scheduling them |
| 3 | Loop-fusion (Case 1→Case 2) is where the real 9.3× lives; sync placement (single `LocalMemBar` per phase boundary) | Yes | instruction-count only, see #2 |
| 4 | Outer-loop unroll ×2 for dual-issue (Case 2→3: 3413→2197) | v23 already does this manually (rows=4 unrolled) | already banked |
| 5 | Main-tail block split (Case 4/5) | Irrelevant to us | 0% — D=4096, CH=64 ⇒ tail=0 always |
| 6 | Cache-control / non-temporal stores | **NOT in these examples at all** | n/a — see §5 (the L2 sweep, A1) |
| 7 | Precision: `MulAddDst` single-instruction FMA, higher accuracy than Mul+Add | Yes | accuracy headroom; tiny perf |
| 8 | `DataCachePreload` (L2 prefetch) exists in the devkit API but not in these examples or in TileLang | Not exposed | unknown |
| 9 | **TileLang emits every VF loop induction variable as `int32_t`**; the devkit requires `uint16_t` for Hardware Loop encoding, so our VF loops are Software Loops — verified: **489/489** VF for-loops across 398 cached VF bodies, 0 `for (uint16_t` in 553 cached kernels | Real compiler gap | unknown — **no measurement, only the doc's rule** (§9.6) |
| 10 | `#pragma unroll N` inside a VF body (the devkit's single best vector lever, 3413→2197 cycles) is **reachable** in TileLang codegen but **exercised by 0 of 553 cached kernels** (173 bare pragmas, none with a factor, none inside a VF) — untested path | Yes (source-only) | unknown until someone reads the generated `.asc` (§9.6, A10) |

**Biggest single conclusion for the team:** the transferable-lesson search in `02_reg_compute` is **exhausted for kernel-level
changes** — v23 already implements everything the devkit's best case prescribes, and its effective vector IPC (≈1.25) is within 10% of the
devkit's own best measured result (1.37), so there is no scheduling headroom left. Remaining prefill gain lives in `04_memory_access`
(DataCopy pattern, UB bank conflicts on the DMA path, MTE2 scheduling) — which is task-2 (§7 gives the bandwidth argument).

**Second conclusion, and the one to escalate:** the study turned up a **TileLang compiler-frontend gap rather than a kernel gap**. TileLang
emits every VF loop induction variable as `int32_t` (verified: 489/489 loops across 398 cached VF bodies, 0 `uint16_t` in 553 generated
kernels), which the devkit states disqualifies Hardware Loop encoding (§9.6) — and `#pragma unroll N` inside a VF body, the devkit's single
biggest vector lever (1.55×, §9.7), is implemented in TileLang's codegen but exercised by zero tests and zero examples. I could not measure
either effect and say so plainly; both are cheap to *disambiguate* and neither needs a device.

---

## 1. Kernel structure in `softmax.asc`

### 1.1 It is deliberately NOT a pipeline example

`SOFT/README_en.md:84`:
> "This example focuses on VF performance optimization details within single-core Softmax. Factors such as multi-core partitioning,
> data movement, and DoubleBuffer are not considered here."

Confirmed in code: the kernel shell at `SOFT/softmax.asc:718-786` is a **fully serialized** 4-stage loop with zero overlap:

```cpp
// SOFT/softmax.asc:742-759 (abridged)
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
for (uint32_t i = 0; i < tiling.loopCount; i++) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);      // serialize OUT before IN
    for (uint32_t j = 0; j < tiling.aDim; j++) { ... DataCopyPad(xLocal[j*rDimAlign], xGm[...]) ... }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);          // wait load, then compute
    CustomSoftmax(yLocal, xLocal, expTensor, sumTensor, ...);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);          // wait compute, then store
    for (uint32_t j = 0; j < tiling.aDim; j++) { ... DataCopyPad(yGm[...], yLocal[j*rDimAlign]) ... }
```

Single-buffered: 1× `xLocal`, 1× `yLocal` (`SOFT/softmax.asc:731-732`), plus scratch `expTensor` and `sumTensor`. Buffer-count
optimization itself is case-dependent — Case 3-5 shrink `sumTensor` to 8 elements because the per-row sum is materialized only in
registers, while Case 0-2 keep a full UB copy (`SOFT/softmax.asc:734-741`):
```cpp
AscendC::LocalTensor<float> sumTensor;
if constexpr (scenarioNum >= 0 && scenarioNum <= 2) {
    sumTensor = ubAllocator.Alloc<float>(tiling.singleCoreAlignLength);
} else {
    sumTensor = ubAllocator.Alloc<float>(8);
}
```
That is a real register-compute signal: **the fused versions stop materializing intermediates in UB at all.**

### 1.2 Tiling is trivially chosen — whole problem fits UB

`SOFT/softmax.asc:798-808`: shape [128,128] fp32 = 64 KiB; `loopCount = 1`, `singleCoreHeight = kHeight = 128`. "Shape [128, 128] fits
entirely in UB on any NPU — use compile-time constants." So **there is no tiling policy to steal from this example.** The row-alignment
concept is the only transferable part: `rDimAlign` = ceil to 32 B (8 floats), `SOFT/softmax.asc:804-806`:
```cpp
uint32_t alignUnit = 32 / sizeof(DataType);
uint32_t widthAlign = (kWidth + alignUnit - 1) / alignUnit * alignUnit;
uint32_t singleCoreAlignLength = singleCoreHeight * widthAlign;
```
Rows are stored **32 B-aligned with padding** (`DataCopyPad` pad-params `{true, 0, rDimAlign - rDim, 0}`, `SOFT/softmax.asc:748-749`)
so that every `LoadAlign` inside the VF is 32 B-aligned. This is the "keep UB access aligned" discipline.
**RMSNorm relevance: we already satisfy it** — D=4096 bf16 = 8 KiB per row, already a multiple of 32 B and of the 256 B VL, and our CH=64
bf16 chunks are exactly one 128 B half-VL. Nothing to do; but note it as the reason `LoadAlign` (fast path) is legal at all.

### 1.3 Where the overlap structure lives: the GELU examples, not softmax

`REG/gelu_high_performance/gelu.asc:120-177` is the only place in `02_reg_compute` with a genuine ping-pong pipeline:

```cpp
// gelu.asc:131-141
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);        // pre-credit both buffers
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
    int32_t eventID = ((loopIdx & 1) == 0 ? EVENT_ID0 : EVENT_ID1);   // 2-phase ping-pong
    AscendC::LocalTensor<float>& xLocal = ((loopIdx & 1) == 0 ? xPing : xPong);
    ...
```
and `gelu.asc:179-183` lays out **four UB regions = depth-2 × {x, y}**:
```cpp
static constexpr uint32_t xAddrPing = 0;
static constexpr uint32_t xAddrPong = tileLen * sizeof(float);
static constexpr uint32_t yAddrPing = xAddrPong + tileLen * sizeof(float);
static constexpr uint32_t yAddrPong = yAddrPing + tileLen * sizeof(float);
```
The "pre-credit the flags before the loop" idiom is the trick that lets iteration *n*'s DMA start while iteration *n-1* is still in the
VF. TileLang's `T.Pipelined(num_stages=4)` + `annotate_buffer_versions` is the equivalent, auto-scheduled — v23 already uses depth 4 on
`x_ub`, which is **deeper than the devkit's own best practice.** No gain available from devkit pipeline depth.

Tile sizing in gelu: `gelu.asc:202-207` — `totalM=8192, totalN=8192, singleCoreM=256, singleCoreN=4096, tileLen = 2*singleCoreN`,
i.e. **2048 float per in-flight tile = 8 KiB… ×4 buffers = 32 KiB UB**. Note the comment `// tileLen 和 singleCoreN 满足整数倍关系`
(gelu.asc:206): tile length must be an exact multiple of the row segment. Our v23 uses `rows*d*2 B = 32 KiB` per x-buffer ×4 = 128 KiB
+ 64 KiB y + 16 KiB w + 16 KiB wf ≈ 224 KiB — close to the observed 256 KiB UB ceiling (`docs/ascend950pr/final-status.md`: rows=8 =
271360 B rejected). So **we cannot go deeper on `x_ub`; the devkit gives no mechanism to sidestep that.**

### 1.4 Two-phase structure with one barrier per boundary

The softmax VF body is: phase A (per-row reduce-max + exp + store), **one** `LocalMemBar`, phase B (per-row sum + div + store).
`SOFT/softmax.asc:674` (Case 5):
```cpp
AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_LOAD, AscendC::Reg::MemType::VEC_STORE>();
```
Case 1 (the anti-pattern, `SOFT/softmax.asc:150,171,193,215,232`…) inserts `LocalMemBar<VEC_ALL, VEC_ALL>` after **every** phase —
the README names this as a defect: `SOFT/README_en.md:150-151`: "`LocalMemBar` is inserted after each computation, introducing redundant
synchronization". Case 2+ coarsens to exactly one STORE→LOAD barrier per row-group boundary, and picks **directional** types
(`VEC_LOAD,` `VEC_STORE`) rather than `VEC_ALL, VEC_ALL`.

TileLang equivalent: `T.simd.mem_bar(mem_type)` — `TL/tilelang/ascend/language/simd.py:1486`. Real usage pattern exists in the repo:
`TL/examples/ascend/flash_attention/core.py:120,123,168` use `S.mem_bar("VST_VLD")`. **v23 emits no `mem_bar` at all** — it relies on
TileLang's auto-sync insertion. That is probably fine (v23 passes), but if we restructure into a fused two-pass form we will need an
explicit directional `mem_bar` at exactly the devkit's phase boundary — and v10's race history says this is where correctness dies.

---

## 2. Exact primitives used, per function

### 2.1 Load
| Purpose | Code | file:line |
|---|---|---|
| aligned contiguous load (fp32) | `AscendC::Reg::LoadAlign(dstReg, srcAddr + offset)` | `SOFT/softmax.asc:68` |
| bf16/fp16→widen load | `AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + offset)` | `SOFT/softmax.asc:65` |
| broadcast load (single elem) | `LoadAlign<float, LoadDist::DIST_BRC_B32>` | `DOCS/api/SIMD-API/basic_api/reg_vector_compute/reg_vector_compute.md:87` |
| unaligned | `LoadUnAlign` / `StoreUnAlign` | `DOCS/.../key_features/key_features.md:50-51` |
| post-update addressing | `PostLiteral::POST_MODE_UPDATE` | `DOCS/.../reg_vector_computation.md:90` |

**LoadAlign payload rule (important for bank-conflict reasoning)**: `DOCS/.../reg_data_load/LoadAlign_continuous.md:148` —
`DIST_UNPACK_B16` "按无符号整型u16加载VL/2数据量" = **loads VL/2 = 128 B per instruction when unpacking**, whereas `DIST_NORM`
loads the full VL (line 138). So the devkit's bf16 widen-load moves **half a vector register** of real data per instruction — the same
cost our v23 pays with `dist="UNPK_B16"`. Our `CH=64` bf16 = 128 B = exactly one UNPK_B16 payload. **v23's chunk size is already the
hardware-optimal one; there is no finer/coarser choice that reduces instruction count.** Verified: `TL/src/ascend/codegen/codegen_ascend.cc:2345`
maps `"UNPK_B16" → vlds_unpack`.

### 2.2 Store
| Purpose | Code | file:line |
|---|---|---|
| normal store | `AscendC::Reg::StoreAlign(dstAddr + offset, srcReg, mask)` | `SOFT/softmax.asc:83` |
| pack b32→b16 (fp32→bf16 out) | `StoreAlign<T, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstReg, mask)` | `SOFT/softmax.asc:81` |
| first-element only (scalar spill) | `StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(expAddr + i*rDimAlign, maxReg, maskOne)` | `SOFT/softmax.asc:169-170`, enum semantics `DOCS/.../StoreAlign_continuous.md:136` |

**DIST_FIRST_ELEMENT_B32 is the interesting one**: the devkit writes a cross-lane reduce result to UB using a 1-element store and a
`VL1` mask (4 B) instead of a full VL store. Our v23 uses the analogous `T.simd.vdupv(a0_6, full, pos="POS_LOWEST")` and keeps the
reduced scalar **entirely in registers** — strictly better than the devkit (no UB round-trip at all). `TL/tilelang/ascend/language/simd.py:933`
`def vdupv(src, mask=None, pos="POS_LOWEST", ...)`.

### 2.3 Reduce
| Purpose | Code | file:line |
|---|---|---|
| intra-register pairwise reduce (tree) | `AscendC::Reg::ReduceSum(sumReg, sumReg, maskFull)` | `SOFT/softmax.asc:684` |
| element-pair reduce | `AscendC::Reg::PairReduceElem<SUM>(dstReg, srcReg, mask)` | `DOCS/.../reduction_compute/PairReduceElem.md:40`, `:93-94` |
| accumulate across VLs | `Add<float, MaskMergeMode::MERGING>(sumReg, sumReg, srcReg, mask)` | `SOFT/softmax.asc:427` |

`DOCS/.../reduction_compute/Reduce.md:124` — "指令内累加顺序采用二叉树累加方式" (the reduce instruction uses **binary-tree
accumulation**), illustrated at `Reduce.md:156-167` for 128 elements. This is the fp32 accumulation-order guarantee: a single-instruction
tree reduce is *more* accurate than a serial add chain. Our v23 replaces `Reduce` with a 6-deep `vcadd` chain
(`kernel.py:70-71`, `TL`…`simd.py:957`) — the equivalent cross-lane tree, since TileLang exposes `vcadd`/`vcpadd`/`vcgadd`
(`TL/src/ascend/codegen/codegen_ascend.cc:2193-2207`) rather than the templated `Reduce`.

**Critical precision trap we already avoid:** `DOCS/.../reduction_compute/Reduce.md:83-94` lists the SUM type table and **`bfloat16_t`
is absent** — bf16→bf16 sum-reduce is not offered by hardware; `PairReduceElem.md:64` likewise supports only "half、float". Any
"reduce in bf16 to save bandwidth" idea is not just slow, it is unsupported → matches `final-status.md:53` ("bf16 `vmula` 无混合精度").

### 2.4 Reciprocal / sqrt
**There is no reciprocal instruction anywhere in these examples or in the Reg API doc tree.** I grepped:
`grep -rln "rsqrt|Reciprocal|倒数" DOCS/api/SIMD-API/basic_api/reg_vector_compute/` → **0 hits.**
Instead the devkit uses:
- `AscendC::Reg::Div(divReg, maxReg, sumReg, mask)` — a **full vector divide of every element** (`SOFT/softmax.asc:691`), and
- in the VF-fusion doc, `Duplicate(reg1, 1.0f, mask)` + `Div(reg2, reg1, reg0, mask)` to build 1/x (`DOCS/guide/operator_practice/simd_operator_optimization/vector_compute/vf_optimization/vf_fusion_optimization.md:47-48`).

Our v23 does the same: `T.simd.vdiv(vdup(1.0), vsqrt(...))` broadcast across all lanes (`kernel.py:74`). **Devkit parity confirmed: no
faster reciprocal exists to steal.** Precision control does exist though: `DOCS/.../basic_arithmetic/Sqrt.md:64` documents
`SqrtAlgo { INTRINSIC, FAST_INVERSE, PRECISION_1ULP_FTZ_TRUE, PRECISION_0ULP_FTZ_FALSE, PRECISION_1ULP_FTZ_FALSE }` — INTRINSIC is
max 1 ulp; FAST_INVERSE (float only) reaches **0 ulp**; and `Div.md:64,116` recommends `PRECISION_1ULP_FTZ_TRUE` or
`PRECISION_0ULP_FTZ_TRUE` "in general scenarios" for better performance. TileLang exposes this as `precision=` on `vdiv`
(`simd.py:792`), `vsqrt` (`simd.py:874`), `vexp` (`:860`), `vln` (`:867`) with codes 0=hw, 1=exact, 2=ftz_false
(`TL/src/ascend/codegen/sfu_precision.h:30-40`). v23 uses the default (exact fp32 division per `simd.py:5-10` module docstring).
See §6 P3 for why that matters.

### 2.5 Precision conversion

The devkit's canonical bf16-in / bf16-out pair — identical in shape to ours:
```cpp
// SOFT/softmax.asc:60-68 (LoadWithCastIfNeed)
static constexpr AscendC::Reg::CastTrait layoutHalf2Float = {
    Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, Reg::RoundMode::UNKNOWN};
AscendC::Reg::RegTensor<half> srcReg;
AscendC::Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + offset);
AscendC::Reg::Cast<float, T, layoutHalf2Float>(dstReg, srcReg, mask);
// SOFT/softmax.asc:75-81 (StoreWithCastIfNeed)
static constexpr AscendC::Reg::CastTrait layoutFloat2Half = {
    Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, Reg::RoundMode::CAST_RINT};
AscendC::Reg::Cast<T, float, layoutFloat2Half>(dstReg, srcReg, mask);
AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstAddr + offset, dstReg, mask);
```
Two details worth copying:
1. **`RegLayout::ZERO` + `MaskMergeMode::ZEROING` for widening** — the widened half occupies the low/high lanes per RegLayout; the
   devkit always pairs it with a `DIST_UNPACK_B16` load so exactly one half is used. This is exactly our `vcvt(..., "float32", part=0)`.
2. **`SatMode::NO_SAT` + `RoundMode::CAST_RINT` for the narrowing store** (`SOFT/softmax.asc:77-78`). `CAST_RINT` = round-to-nearest-even.
   TileLang's `vcvt` defaults to `round="ROUND_R"` (nearest, ties-to-even) and `sat=True` (`simd.py:1214-1218`): **our rounding mode
   already matches; our saturation differs (we saturate, devkit does not). For RMSNorm output that is safe — post-normalization values are
   bounded by |x|·|w|/rms — but flag it as an intentional, documented deviation.**
   Cross-check `DOCS/.../type_conversion/Cast.md:106,184`: `bfloat16_t ↔ float` is a listed pair and `Cast.md:184` enumerates valid
   round modes (`CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC`).
3. A 5th case in `SOFT/softmax.asc:116` (MemBase path) casts with `RoundMode::CAST_FLOOR` — the README does not defend it; using FLOOR
   on output would bias every element downward. Do not imitate.

### 2.6 "Register compute" tricks — the actual list

| Trick | Evidence | v23 status |
|---|---|---|
| Values resident in `RegTensor` across multiple ops inside ONE `__simd_vf__` body | `SOFT/softmax.asc:128-129`, `REG/gelu…/gelu.asc:86-108` | ✅ v23 is one `T.SimdVF()` region, `kernel.py:46-154` |
| Cross-lane reduce result broadcast back to all lanes **without touching UB** | devkit must store to UB then re-load (`softmax.asc:169-170` + `Duplicate` at `410`) | ✅ v23 uses `vdupv(..., POS_LOWEST)` — **strictly better than devkit** |
| Load source **once** and feed two passes (fuse reduce-pass with apply-pass) | `SOFT/README_en.md:163-185` — "Case 2 requires only 2 Loads … while Case 1 requires 4 Loads" | ⚠️ v23 loads x **twice** (pass1 `kernel.py:61-64`, pass4 `136-139`) = 2× per row; but total UB reads are 3×(x, x, w) not 2× — see §3.2 |
| Per-row independent register sets to break dependency chains | `SOFT/softmax.asc:641-645` (`srcReg1, maxReg1, expReg1, sumReg1, divReg1`) | ✅ v23 rows=4 unrolled, `kernel.py:56-134` |
| Reduce accumulator kept in reg across the whole row | `softmax.asc:678-685` | ✅ 4 accumulator chains/row, `kernel.py:56-59` |
| `fp32MinValue` union to materialize an exact bit pattern as a constexpr | `SOFT/softmax.asc:31-39` | n/a (we need 0.0 seed, already exact) |

Register budget is the binding constraint on all of this, and the devkit states the limits explicitly —
`DOCS/guide/operator_practice/simd_operator_optimization/vector_compute/vf_optimization/dual_issue_optimization.md:149-152`:
> "在同一个VF中，硬件可以同时处理的最大RegTensor寄存器个数为32，如果超出，编译器会进行数据的换入换出并加入同步指令，严重拖慢算子的执行效率。
> 在同一个VF中，MaskReg数量上限为8个… AddrReg数量上限为8个… UnalignRegForLoad、UnalignRegForStore的数量上限均为4个"

v23's rows=4 region holds 16 f32 accumulators + 4×4 transient v-registers + 4 scale registers + `full` mask = **~29-31 live vector
registers if the compiler does not coalesce them, against a hard 32**. `SOFT/README_en.md:313` confirms the same budget reading
("≤32 RegTensor, ≤8 MaskReg"). **This is the reason §3.2's proposed single-pass-fused variant will likely fail: holding 64 chunks × 1
f32 register each is impossible; you must keep only the *packed bf16* form, and the packed form of a row is 64 registers.** Labeling
explicitly: I did not verify the live-register count of the generated code, only the source-level count.

---

## 3. Measured ablations, reproduced verbatim

### 3.1 `02_reg_compute/softmax_high_performance` (vector-instruction cost = `aiv_total_cycle × aiv_vec_ratio`)

| Version | API | Characteristics | Cost (cycles) | Step gain |
|---|---|---|---|---|
| Case 0 | MemBase | single loop, all 7 ops | 44113 | baseline |
| Case 1 | RegBase | unfused, redundant Load/Store/sync | 31894 | **1.38× (README: "34% improvement")** |
| Case 2 | RegBase | UpdateMask + loop fusion + ExpSub | 3413 | **9.34×** (`SOFT/README.md:217`: "31894 → 3413 cycles（提升 9.3x）") |
| Case 3 | RegBase | UpdateMask + outer-loop unroll ×2 | 2197 | **1.55× / "提升 35.6%"** (`README.md:274`) |
| Case 4 | RegBase | main-tail block (no UpdateMask) | 3424 | 1.00× vs Case 2 — **main-tail alone buys nothing here** |
| Case 5 | RegBase | main-tail + unroll + ExpSub | **1785** | **27.2× over Case 0** (`README.md:318`) |

Source of the numbers: `SOFT/README_en.md:69-74` and identical `SOFT/README.md:69-74`. Case 2 vs Case 4 are within noise of each other
(3413 vs 3424) — the README explains why: `README.md:278` — under rDim=128 the tail is 0, so main-tail mode "达到收益上限" only removes
the `UpdateMask` scalar overhead, which at 2 iterations/row is tiny.

**Transfer to RMSNorm, honestly:**
- Case 0→1 (1.38×): N/A. We never used MemBase-style; `T.SimdVF` is RegBase by construction.
- Case 1→2 (9.34×): **This is the loop-fusion lesson and it is the only big one.** Our v23 already fuses pass1+pass2 within a row's
  accumulate, and pass4 re-reads. See §3.2.
- Case 2→3 (1.55×): already banked via manual rows=4 unroll.
- Case 4/5: **zero expected value** — our D=4096 with CH=64 chunks gives `repeatTimesMain = 64`, `tail = 0` structurally
  (`kernel.py:29-31` requires `d % 256 == 0`). Main-tail handling is dead code for us.
- Ceiling implied by this table: vector cost only, and the devkit itself says vector cost doesn't move MTE-bound kernels.

### 3.2 Applying "loop fusion" to v23 — the one concrete idea, with arithmetic

Current v23 per-row UB traffic (D=4096, 64 chunks):
| Pass | Code | vlds issued | UB bytes read |
|---|---|---|---|
| 1: sum-of-squares | `kernel.py:60-68` | 64 (`UNPK_B16`) | 8192 B (bf16) |
| 2: reduce/scale | `kernel.py:69-74` | 0 | 0 |
| 3-6: apply+store, **per row** | `kernel.py:135-154` | 64 (`UNPK_B16`) + 64 (`wf` NORM) | 8192 B (bf16 x) + 16384 B (f32 w) |
| store, **per row** | `kernel.py:139` | — | 8192 B (bf16 y) written |

x is therefore read from UB **twice per row**, while the fp32 weight vector is re-read **once per row**, i.e. 4× per step: 4 rows × 16 KiB
= 64 KiB of `wf` traffic against 64 KiB of x traffic per step. **The weight reload is the single largest avoidable UB read in the kernel**
(→ action item **A2**); the x reload is not avoidable (below). Two fusions are implied by the devkit's own principle:

**(a) Hoist `wf` re-reads across a whole row by keeping all 64 w-chunks in registers.** Not possible: 64 f32 registers > 32-register
budget. **But holding just ONE chunk while the 4 rows consume it is enough** (that is the chunk-major re-order, A2), and costs +1 register.
So this one is *not* rejected — it is the actionable item.

**(b) Fuse pass1 and pass3: keep x resident in bf16 registers between the accumulate and the scale.** Requires 64 registers per row;
over budget. Rejected.

**(c) The only feasible fusion: shrink the chunk loop so both passes for the *same chunk* run consecutively — impossible by construction,
the scale isn't known until all chunks are summed.** This is a real algorithmic barrier that softmax shares (hence the devkit *also*
re-reads: `SOFT/softmax.asc:653-658` then `663-666` reload the same `srcAddr`!). **So the devkit's "2 Loads not 4" is already our
floor: 2 loads of x per row, exactly what `softmax.asc` Case 5 does.** Confirmed by re-reading Case 5: the ReduceMax inner loop loads
`srcAddr` (line 654), and the ExpSub inner loop loads `srcAddr` again (line 664). The 9.3× gain from Case 1→2 was removing a *third and
fourth* re-load (the intermediate UB round-trip through `sumAddr`), which we never had.

**Verdict on §3.2: our v23 already sits at the devkit's best-practice load count. There is no fusion win left.** This is the most
important negative result in this report and it should stop the team from re-trying "single-pass fusion" variants.

### 3.3 `gelu_high_performance` — the MTE-bound reality check (numbers I most want the team to see)

`REG/gelu_high_performance/README_en.md:363-371`:
| Metric | Case 0 (MemBase + PipeBarrier) | Case 1 (RegBase VF fusion) | Improvement |
|---|---|---|---|
| Task Duration(μs) | 351.525 | 348.868 | **0.76%** |
| aiv_vec_time(μs) | 147.895 | 66.277 | **55.2%** |
| aiv_vec_ratio | 0.422 | 0.19 | — |
| aiv_scalar_time(μs) | 9.513 | 3.03 | 68.2% |
| aiv_mte2_time(μs) | 320.283 | 320.543 | — |
| aiv_mte3_time(μs) | 303.647 | 314.547 | — |

`README_en.md:373`: "This example is MTE2 bound, with data transfer ratio exceeding 90%, so end-to-end benefit is not obvious."
Case 2 (+`#pragma unroll 6`, `gelu.asc:93`): Task Duration 351.525 → **344.436 μs**, vec 66.277 → **63.655 μs**
(`README_en.md:455-457`). **A 55% vector-time reduction bought 0.76% end-to-end. A further 4% vector reduction bought 0.7% more.**

Our prefill: vec ≈ 40.5 µs of 65 µs, vendor 53 µs. If we cut vector work by the devkit's most aggressive ratio (Case 0→5 = 24.7×),
that does not make sense; a realistic 20-30% cut = 8-12 µs saved, **but only if it is not already overlapped**. `final-status.md:64`
says "若 DMA 与向量能完全重叠，理论上限约 40–45 µs" — i.e. the team already believes vec is *partly hidden*. **Conclusion: the devkit's
data says vector optimization in an MTE2-bound kernel yields well under 5% end-to-end. Our 19-23% gap is a DMA-efficiency gap.**

### 3.4 `gelu_eltwise_high_performance` — IPC ladder (vector-side ceilings)

`REG/gelu_eltwise_high_performance/README_en.md:521-526`:
| Case | Strategy | Vector Duration (μs) | IPC | vs Case 0 |
|---|---|---|---|---|
| 0 | single loop, parallelism **128 bytes** | 244.299 | 1.06 | 1× |
| 1 | single loop, parallelism **256 bytes** | 125.203 | 1.03 | 0.97× |
| 2 | loop split (Gelu + Element-wise) | 97.184 | **1.33** | 1.25× |
| 3 | loop split + unroll | 94.339 | **1.37** | 1.29× |

Two hard facts extracted:
- **VL choice matters more than anything else**: 128 B/iter → 256 B/iter halves vector time (244→125 μs) with *no* code change beyond the
  mask width (`README_en.md:101`: "Case 0 (scenarioNum == 0) uses a fixed VL32 mask (32 float elements, 128-byte parallelism)").
  Our v23 uses `T.simd.pset(32)` = full `PAT_ALL` on an fp32 vector = **64 lanes = 256 B**, correct. The bf16 loads use UNPK_B16
  (128 B payload, hardware-mandated). `README_en.md:397`: `{bytes/cycle} = 256/22 ≈ 11.64` — the 256 B is the standard-op lane width.
- **Nonlinear ops have a 4× lower lane rate**: `gelu_high_performance/README_en.md:511-512` — "6 linear operation instructions
  (Mul×2, Muls×2, Add×1, Adds×1): parallelism 256 bytes/cycle; 2 non-linear (Exp×1, Div×1): parallelism **64 bytes/cycle**". And
  `README_en.md:551`: "exp and div cannot simply be estimated as 128 bytes/cycle in dual-issue scenarios."
  **This is directly actionable for us: our per-row `vdiv`+`vsqrt` operate at 64 B/cycle. At 64 lanes f32 = 256 B, a Div costs 4
  issue-cycles and a Sqrt 4 more.** They run **once per row**, so 8 × 2 rows... small. But `vmul(vf, sc)` and `vmul(·, wv)` are 64×
  per row at 256 B/cycle = the true hot loop. Nothing here is a lever beyond what v23 does.
- Realistic IPC ceiling per `README_en.md:529`: "theoretical limit approaches 2.0… achieving 1.4~1.5 indicates relatively optimal".
  **Case 3's 1.37 is the devkit's own definition of "close to done". A v23 vec/DMA overlap analysis should target that, not 2.0.**

---

## 4. Scalar-vs-vector split, per-row vs per-tile, tail, bank conflicts

### 4.1 Per-row vs per-tile
The devkit is strictly **per-row** (`for i in aDim` outer, `for j in repeatTimes` inner), never per-tile, because softmax statistics
are per-row. RMSNorm likewise has per-row statistics. **v23's structure matches the devkit's best version exactly on this axis.**
One difference: the devkit splits the *outer loop over halves* and pairs rows i and i+halfA (`softmax.asc:655`: offset
`+ halfA * rDimAlign`) rather than pairing adjacent rows. Reason is not stated; **speculation:** with 2×128-row tiles this keeps the
two interleaved streams' UB addresses far apart, avoiding DRAM/L2 row conflicts at the DMA level, though within one UB buffer bank
conflicts are more plausibly the concern. Not transferable in any case (v23's rows are adjacent by design and already tested).

### 4.2 Scalar work
The devkit keeps *everything* vector except loop counters, and even then warns about the split:
`DOCS/…/vf_loop_optimization.md:108-119` shows the correct pattern — **copy struct/tiling members into `uint16_t` locals before the
loop** so the VF body doesn't reload struct fields per iteration:
```cpp
// 【正例】通过局部变量传递VF内访问的结构体或类对象成员变量
uint16_t srcK = tiling.srcK;  uint16_t srcM = tiling.srcM;
uint16_t reduceK = FLOAT_NUM_PER_BLK;  uint16_t originK = (uint16_t)originalSrcShape.k;
for (uint16_t i = 0; i < srcM; i++) { AscendC::ReduceMax(maxAddr + i*reduceK, srcAddr + i*srcK, workAddr, originK); }
```
`vf_loop_optimization.md:124-151` additionally: hoist loop-invariant instructions out (`Duplicate` outside the loop → "仅执行一次，
有效减少指令数"). v23 hoists `full = T.simd.pset(32)` (done) but **re-materializes `T.simd.vdup(T.float32(1.0), ...)` four times**
(`kernel.py:74,94,114,134`) and `T.float32(1.0/d)` inline in `vmuls`. Per the devkit's rule, `one = vdup(1.0)` should be allocated
once at the top of the `SimdVF` block and reused 4×. Expected: −3 vector instructions per step; scalar cost of `eps`/`1.0-d` handling
is already a kernel arg. **Marginal, but free and it is exactly the documented pattern.** Also note `vf_loop_optimization.md:157-160`:
loop-control scalar overhead grows in relative importance once vector time shrinks — which is our situation.

### 4.3 Tail handling — three documented modes

| Mode | Code | file:line | Verdict for us |
|---|---|---|---|
| `UpdateMask` per iteration | `mask = Reg::UpdateMask<float>(count); count decremented internally` | `SOFT/softmax.asc:405,414,425,433` | legal but costs a scalar mask-gen per iteration |
| Compile-time `CreateMask<ALL>` for the main block | `maskFull = Reg::CreateMask<T, MaskPattern::ALL>()` | `SOFT/softmax.asc:632,654` | **what v23 does** (`pset(32)`) |
| Explicit main/tail split | `tail = rDim % (256/sizeof(float)); repeatTimesMain = rDim / (256/sizeof(float)); hasTail = tail != 0` | `SOFT/softmax.asc:710-712` | **dead code for us** |
| `DataCopyPad` with pad-params for GM→UB alignment | `{true, 0, (uint8_t)(rDimAlign - rDim), 0}` | `SOFT/softmax.asc:748-749` | relevant if we ever take a non-multiple-of-32B row; our guard (`kernel.py:29-30`) forbids it |
| Last-row-of-batch special case | `if (tiling.lastADim != 0) { …DataCopyPad with pad… }` | `SOFT/softmax.asc:763-782` | relevant to **batch remainder** — v23 raises `ValueError` for `batch % 64 != 0` (`kernel.py:24`). **If acceptance ever needs ragged batches, the devkit's answer is per-row `DataCopyPad`, not a masked vector tail.** |

Also worth noting Case 4/5 compute `tail`/`hasTail` and then **never use them** (`softmax.asc:616-620`, `710-714` — passed as args,
unused in the body). The examples were only run at rDim=128. **Do not treat main-tail as validated code in the devkit — it is validated
only for the tail=0 case.** Flagging so the team doesn't over-trust it.

### 4.4 Bank conflicts

**`02_reg_compute` contains no bank-conflict avoidance code.** The only mention is a generic warning, `SOFT/README.md:103`:
"UB Bank冲突风险高：连续的UB读写访问极易触发Bank冲突" — attributed as a *MemBase* pathology, not something Case 5 addresses.
The concrete mitigations implied by the code (and all already satisfied by v23):
1. 32 B-aligned `LoadAlign`/`StoreAlign` only (`§1.2`, `key_features.md:48`).
2. `oneRepeatSize = GetVecLen()/sizeof(T)` — always stride by whole VL, never by arbitrary offsets (`softmax.asc:631`). v23 uses
   `CH=64` bf16 = exactly one UNPK_B16 payload and `i*CH` stepping — whole-vector strides. ✅
3. `rDimAlign` row padding to 32 B multiples so different rows don't alias (`softmax.asc:804-806`). v23 rows are 8192 B apart =
   many VLs, no aliasing. ✅
4. Sequential (`DIST_NORM`) accesses over a contiguous span — `vf_loop_optimization.md:194`: "当四层循环构成连续访问场景时，地址管理可被简化为
   一维模式，从而实现更高效的搬运优化，进一步提升数据访问的局部性". v23's `i*CH` loop is exactly this. ✅
5. Dual-register interleaved read of two halves of the same VL, i.e. `LoadAlign<DIST_DINTLV_B16>` returning a pair from one 2×VL span
   (`DOCS/…/LoadAlign_continuous.md:179`) — **TileLang has this: `T.simd.vld2(addr, dist="DINTLV_B16")` (`simd.py:417-428`):
   "DINTLV_B16: load 256xbf16/u16 -> two 128-lane regs"**. This is a **real, unused-by-us instruction-count halving**: one vld2 of
   256 bf16 elements replaces two vlds of 64. See §8 action item **A2b**.

Dedicated UB bank-conflict material is in `04_memory_access/{bank_conflict_ub,bank_conflict_3510,bank_conflict_nd2nz}` → task-2.

---

## 5. Cache control, non-temporal stores, write-combining, L2 policy

**Result of a hard grep: NONE.** `grep -rn "DataCachePreload|SetL2|CacheCleanAndInvalid|cache_ctrl|CacheMode|non_temporal|NonTemporal"
` over the whole `02_reg_compute` tree returns zero hits. These examples do not touch cache policy at all — they are single-UB-buffer,
whole-problem-in-UB kernels. `SOFT/README.md:84` explicitly disclaims "数据搬运" as out of scope.

The devkit *does* document cache control, just not here:
- `DOCS/api/SIMD-API/basic_api/cache_control/` → `DataCachePreload.md`, `DataCacheCleanAndInvalid.md`, `PRELOAD.md`,
  `ICachePreLoad_ISASI.md`, `system_cache_overview.md`.

**TileLang exposure today:**
- L2 hints on `T.copy`: yes — `TL/tilelang/ascend/language/copy_op.py:167-199` exposes the full LD/ST enum
  (`NORMAL_FV, NORMAL_LV, NORMAL_PERS, NORMAL_PREF, NOTALLOC_KEEP/CLEAN/DROP, EXCLUSIV_*, INVALID`; store-side
  `NORMAL_RED, NOTALLOC_CI/PW/PI/RED, WBH_*, WTS_*`). **`WTS_*` (Write-Through-Stream) and `NOTALLOC_*` (no L2 allocation) are exactly
  the "non-temporal / write-combining" family, and `TL` already accepts them as strings.**
- `DataCachePreload`: **not exposed** — `grep -rn "Preload|preload" TL/src/ascend TL/tilelang/ascend` → 0 hits.
- `mark_scalar_dcache_bypass.cc` (`TL/src/ascend/transform/mark_scalar_dcache_bypass.cc`) exists and auto-marks global buffers for
  dcache bypass on the scalar path — not user-controllable, and about L1/dcache not L2 policy.

**Status of experiments (from `docs/ascend950pr/candidates/*/kernel.py` policy census I ran):** only 4 distinct policies have ever been
tried — `NORMAL_FV` (57 uses), `NORMAL_LV` (3), `NORMAL_PREF` (2), `NORMAL_RED` (2). v23 = all-`NORMAL_FV`.
**Untested, and directly suggested by the traffic pattern (RMSNorm prefill reads each x element exactly once and writes each y element
exactly once → a pure streaming case where L2 allocation is waste):**
- Read path: `NOTALLOC_DROP` / `NOTALLOC_CLEAN` / `IDS_FV`
- Write path: `NOTALLOC_PW` (partial write) / `NOTALLOC_PI` / `WTS_FV` / `WTS_LV` / `WBH_FV` (write-back-hint)

**Expected effect:** the only lever with a plausible shot at 19-23%. `final-status.md:47-49` records two relevant prior results: (i) write
path without a policy = 1.13 TB/s → with `NORMAL_FV` = 3.08 TB/s (2.7×, decisive), and (ii) "微基准排名不能直接搬到内核（读路径微基准最优的
`NORMAL_LV` 在内核中反而更慢）". So microbenchmark rank does not transfer — but the fact that one policy gained 2.7× on the write path
proves the knob is *live* in the 1.97→2.42 TB/s range. **Action A1 for the implementer: sweep `{NOTALLOC_DROP, IDS_FV}` on the read
× `{NOTALLOC_PW, NOTALLOC_PI, WTS_FV, WTS_LV, WBH_FV, NORMAL_RED}` on the write, on the *kernel*, using the paired-window gate.**
No devkit evidence either way — labeled **hypothesis**, but it is a hypothesis about a knob TileLang already exposes that nobody has
turned, which is where the highest expected value lives.

---

## 6. bf16/fp32 mixed-precision techniques relevant to sum-of-squares

From `02_reg_compute` plus the API docs, ranked by applicability:

**(P1) Single-instruction fused multiply-add into a *wider-than-source* accumulator.**
`DOCS/…/composite_compute/overview.md:13`: `MulAddDst`: `dst_i = src0_i × src1_i + dst_i`, and `MulAddDst.md:104`:
> "对于非软仿实现的数据类型，在硬件层面单条指令完成乘加融合计算，**精度高于分开执行 Mul 和 Add**"

Type table `MulAddDst.md:84-94`: `bfloat16_t×bfloat16_t→bfloat16_t`, `float×float→float`, and critically **`half × half → float` is
NOT listed for MulAddDst** (all-same-type). But `MulDstAdd.md:72-77` lists `half,half→float` and `bfloat16_t,bfloat16_t→bfloat16_t`.
**TileLang status:** `T.simd.vmula(dst, src0, src1, mask)` (`TL/tilelang/ascend/language/simd.py:635-646`, "Fused multiply-add:
dst = dst + src0 * src1") — used by v23 (`kernel.py:65-68`) on **f32 triples**. `final-status.md:52` records "bf16 `vmula` 无混合精度",
consistent with the type tables: **there is no bf16-in / fp32-accumulate `vmula`, so the `vcvt` to fp32 before the multiply is mandatory,
not a choice.** Our current sequence (UNPK_B16 load → `vcvt` f32 → `vmula` f32) is the devkit-endorsed minimum. ✅ Nothing to change.
`T.simd.vmadd` exists (`simd.py:649`, `dst = dst*src0 + src1`) and `vaxpy` (`:663`) — unused, not needed.

**(P2) Binary-tree reduction order for the final fold.** `Reduce.md:124,156-167`. Our `vcadd` chain × 6 (2⁶=64 lanes) is a tree.
Error bound is better than the serial chain a naive `ReduceSum` over UB would give. ✅ Already correct, and the accuracy margin we hold
(≤3e-3 vs threshold 7.8e-3, `final-status.md:26`) is partly thanks to it.

**(P3) Deliberate FTZ/algo selection on the SFU ops.** `Sqrt.md:64,103-109` and `Div.md:106,116`: default `INTRINSIC` respects
`--cce-ftz` (default true → subnormals flushed). Our accumulator is fp32 with values ~1.0 scale, so FTZ is harmless; `eps=1e-6` is a
normal fp32. **No risk, no gain.** But note `Div.md:116` recommends `PRECISION_1ULP_FTZ_TRUE` "for performance" — TileLang's default
for `vdiv` fp32 is the **exact 0-ulp** path (`simd.py:8-10`: "the default exact FP32 division"; `sfu_precision.h:13` "fast_math for
vdiv" fallback). v23's `vdiv` (`kernel.py:74`) therefore runs the *expensive* division. **Action A3: try
`T.simd.vdiv(a, b, mask, precision="hw")` (or set `tl.enable_fast_math`, `TL/tilelang/transform/pass_config.py:83`) and re-measure
the accuracy gate.** Cost: 64-lane f32 Div at 64 B/cycle = 4 issue slots/row → 16 slots total per step ≈ negligible µs, so **do not
expect a win; do it only if the vec-time profile says SFU ops are the tail.** Label: low-priority, measurable-in-isolation-only.

**(P4) `MulsCast` — fused scale+narrow in one instruction.** `composite_compute/MulsCast.md`: `dst_i = cast_round_to_f16(src_i ×
scalarValue)`, table (from the doc's data-type section) `src=float, scalar=float, dst=half` — **fp32×scalar→fp16 in one op.** This is
*exactly* our "multiply row by scale then store as bf16" step, saving a `vcvt` per chunk (64 per row → 64 instructions saved per row,
14% of our per-row instruction count). **TileLang status: not exposed** — `TL/tilelang/ascend/language/simd.py` has `vmuls` (`:1462`)
and `vcvt` (`:1209`) separately, `grep -rn "mulscast|MulsCast" TL/src TL/tilelang` → 0 hits, and the codegen op list
(`codegen_ascend.cc`, extracted `tl.simd.*`) contains no such op. `vsts(dist="PK_B32")` is a *bit-truncation* pack, not a rounding
cast, so it is not a substitute for the rounding half.
**Workaround:** none clean in TileLang today. **What to add:** a `simd_mulscast` builtin → `asc_vf_call` body emitting
`Reg::MulsCast<float, half, RegLayout::ZERO, RoundMode::CAST_RINT>`; **for bf16 output it does not apply anyway** — the type table gives
only `dst=half`, and `Cast.md:106` shows bf16 narrowing comes from `float`. So P4 is **NOT transferable to BF16 output**; recorded so
the team does not spend a round on it. Labeling this a genuine dead end, not a "maybe".

**(P5) `ExpSub` (the softmax-specific fused op).** `softmax.asc:667`, `ExpSub.md:44-56,95-98` (src float→float; half→float with
internal widening, "精度提升", line 111). **TileLang has it: `T.simd.vexpdif(src0, src1, mask)` (`simd.py:1182-1195`) — but it
explicitly rejects non-fp32: "Widening float16 inputs to float32 requires separate even/odd results and is not represented by this API"
(`simd.py:1186-1188`).** Irrelevant to RMSNorm (no exp). Recorded only because the "keep the widening *inside* the fused op instead of a
separate `vcvt`" idea is what makes ExpSub fast, and the same idea would apply to a hypothetical fused bf16-square-accumulate — which
§6 P1 says the hardware does not provide.

---

## 7. Bandwidth argument: why `02_reg_compute` cannot close a 19-23% DMA gap

Reproducible arithmetic on our own numbers (prefill 8192×4096 bf16, 64 MiB read + 64 MiB write):
- Vendor: 128 MiB / 53 µs ≈ **2.53 TB/s** (`final-status.md:62` gives vendor ≈53 µs)
- v23: 128 MiB / 65 µs ≈ **2.06 TB/s**; device attribution reports **1.97 TB/s** mixed.
- Gap = 0.56 TB/s = **22%** — precisely the reported deficit.
- **Instruction budget per v23 step** (rows=4, D=4096, NCH=64; counted from `kernel.py:60-74,135-154`):
  accumulate 4×64×(1 vld UNPK_B16 + 1 vcvt + 1 vmula) = 768; merges 4×3 vadd = 12; butterfly 4×6 vcadd = 24;
  per-row scale 4×(vdupv, vdup, vmuls, vadds, vsqrt, vdiv) = 24; apply 4×64×(1 vld x + 1 vcvt + **1 vld w** + 2 vmul + 1 vcvt + 1 vsts) = 1792
  ⇒ **≈2620 vector instructions/step**, ×`iters = 8192/(64·4) = 32` ⇒ **≈83.8k instructions per core**.
- **The measured vec time is already explained by issue throughput**: 40.5 µs × 1650 MHz (`gelu…/README_en.md` frequency convention)
  = 66.8k cycles ⇒ **effective IPC ≈ 1.25** on the vector pipe. The devkit's own ceiling statement is
  `gelu_eltwise…/README_en.md:529`: theoretical ~2.0, "achieving 1.4~1.5 indicates relatively optimal", and their *best* case reached
  **1.37**. **v23 is at 1.25 — within 10% of the devkit's best-achievable result, so there is no dual-issue headroom left to harvest by
  re-arranging instructions: vec time can only fall by *deleting* instructions.** Caveats on this figure: it is an instruction *count*
  divided by cycles, not a profiled IPC — the true issued-IPC is lower if the compiler spills, and 1.25 (ours) vs 1.37 (their Case 3) spans
  two different kernels, so treat it as "v23's vector body is near the devkit's practical ceiling", not as a measurement of dual-issue rate.
- **Consequence:** vec time can only fall by *deleting instructions*, and the count exposes one pure redundancy:
  `wv = T.simd.vld(wf[i*CH])` is issued once **per row per chunk** (`kernel.py:137,142,147,152` — all four rows reload the same `wf`
  chunk) = 256 vlds/step. Re-ordering the apply pass chunk-major (load `wv` once, then apply to the 4 rows) removes 192 of them for
  **+1 live register**: 2620 → 2428 instructions/step (−7.3%) and UB reads 128 KiB → 80 KiB/step (x twice = 64 KiB; `wf` 64 KiB → 16 KiB).
  See **A2**. **Caveat, untested:** chunk-major addressing hits `x_ub[r0 + i*CH]`…`[r3 + i*CH]` with an 8192 B stride, so four 128 B
  (`UNPK_B16`) accesses can land in the same UB bank group — a possible bank conflict that may cancel the win. Measure, don't assume.
- **UB bandwidth is not the limiter.** Per core: 32 steps × (128 KiB read + 32 KiB written) ≈ 5 MiB of UB traffic inside the measured
  40.5 µs ⇒ **≈129 GB/s/core**, against a full-VL issue rate of 256 B/cycle × 1.65 GHz ≈ **422 GB/s** — we use ~30% of it. (If A2 lands,
  reads drop to 80 KiB/step and utilization to ~22%.) The limiter is **instruction issue**, which is why §9.6 (loop control) matters more
  than A2's 7%.
- The **1792-instruction apply pass lives in four non-unrolled `for i in T.serial(NCH)` loops** (`kernel.py:135,140,145,150`), whereas the
  accumulate pass is already 4-way split (`kernel.py:60,80,100,120`, `NCH//4`). Each apply loop runs the full `NCH` = 64 iterations,
  so a step executes **256 VF loop iterations** of compare/increment/jump plus scalar address arithmetic — the largest remaining
  *non-arithmetic* cost in the VF, and exactly what a Hardware Loop encoding would absorb. → **A10**, **A11** (§9.6, §9.7).
- And the devkit offers **no** overlap mechanism deeper than what we have: gelu's ping-pong is depth 2 vs our `stages=4`, its flags are
  the same `SetFlag/WaitFlag` pair TileLang's AutoSchedule emits, and it never pipelines more than one compute and one store in flight.

**Therefore: transferable content from task-1 for the prefill gap ≈ 0 through A1, small (A2/A10) through the rest.** v23 already has
everything the devkit's best case prescribes. A2 and A10 are the only two items I would actually run, and I would run them *before*
concluding that vector work is fully hidden — but neither is expected to close 22%.

---

## 8. Concrete action items (each with a TileLang-exposure verdict)

| ID | Item | Devkit source | TileLang today? | Expected prefill effect | If not exposed |
|---|---|---|---|---|---|
| **A1** | Sweep untried L2 policies: read ∈ {NORMAL_FV, NOTALLOC_DROP, IDS_FV}, write ∈ {NORMAL_FV, NOTALLOC_PW, NOTALLOC_PI, WTS_FV, WTS_LV, WBH_FV, NORMAL_RED} | **none** (no cache code in `02_reg_compute`) — motivated by traffic shape | **Yes**: `copy_op.py:167-199`, string or int, suffix-matched | **Only plausible shot at 19-23%.** 0 to +25%. Prior art: NORMAL_FV on writes = 2.7× (`final-status.md:47`) | n/a; just call it. Guard: verify microbenchmark rank ≠ kernel rank |
| **A2** | **Dedup the weight load**: merge the four `for i in T.serial(NCH)` apply loops (`kernel.py:135,140,145,150`) into one chunk-major loop that issues `wv = vld(wf[i*CH])` **once** and applies it to all 4 rows | devkit principle only (`vf_loop_optimization.md:124-151`, hoist loop-invariant work) — no direct devkit analogue, since softmax has no shared weight vector | **Yes** (pure source edit; +1 live register) | −192 `vld`s (the x loads stay, one per row; only `wf` is deduped) out of 2620 instrs = **−7.3%**, UB reads 128→80 KiB/step. Expected **0–1 µs**. **Risk: 8192 B stride across the 4 rows may create a UB bank conflict that cancels it — measure.** | n/a |
| **A2b** | Replace 2× `vld(dist="UNPK_B16")` of adjacent 64-elem chunks with 1× `vld2(dist="DINTLV_B16")` of 128 elements | `DOCS/…/LoadAlign_continuous.md:179`; `vf_loop_optimization.md:194` | **Yes**: `simd.py:417-428` → "load 256xbf16/u16 -> two 128-lane regs" | −32 of 768 accumulate loads/step (≈1.2% of total instrs). **≈0 end-to-end.** Only worth pairing with A2 | n/a |
| **A3** | `precision="hw"` on `vdiv`/`vsqrt`, or `tl.enable_fast_math` | `Div.md:116`, `Sqrt.md:64` | **Yes**: `simd.py:792,874` + `pass_config.py:83` | ≈0 (16 SFU ops/step). Re-run accuracy gate if tried | n/a |
| **A4** | Hoist `one = vdup(1.0)` out of the 4 row blocks | `vf_loop_optimization.md:124-151` | **Yes** (pure source edit) | −3 vec instrs/step. ≈0 | n/a |
| **A5** | 2-phase manual `order`/`stage` schedule to deepen copy/compute overlap | gelu ping-pong `gelu.asc:131-171` | **Yes**: `T.Pipelined(..., order=, stage=)` (`TL/tilelang/language/loop.py:112-132`); v30/v31/v32 already attempt "manual pipeline schedule" | **This is the actual target.** If overlap is the limiter, worth 5-10 µs | already being tried by the team — this report adds: devkit depth 2 < our depth 4, so **more buffering is not the answer; different ordering is** |
| **A6** | Fused fp32×scalar→narrow store (`MulsCast`, `dst_i = cast_round_to_f16(src_i × scalarValue)`, `composite_compute/MulsCast.md:47`) | `composite_compute/MulsCast.md:47` | **No** — 0 hits in `TL` | **NOT transferable**: type table is `float→half` only; our output is bf16 | Would need a new `simd_mulscast` builtin + `asc_vf_call` lowering; and even then dst=half only |
| **A7** | Main-tail block loop | `SOFT/softmax.asc:710-712` | `T.simd.update_mask` (`simd.py:125`) exists; `T.unroll(explicit=True)` + `unroll_factor` (`ascend/language/loop.py:21-44`) exists | **0%** — D=4096 ⇒ tail ≡ 0 | don't do it |
| **A8** | `DataCopyPad` for ragged batches (`batch % 64 ≠ 0`) | `SOFT/softmax.asc:763-782` | Partially: `T.copy(..., pad_value=)` (`copy_op.py:239`), `data_select`, `insert_oob_padding.cc` | 0% on certified shapes; **needed if acceptance widens shapes** | use `pad_value` + host-side masking; untested path |
| **A9** | `DataCachePreload` (L2 preload hint) | `DOCS/api/SIMD-API/basic_api/cache_control/DataCachePreload.md` | **No** — `grep -rn "Preload" TL/src/ascend TL/tilelang/ascend` = 0 | unknown; would need a new intrinsic + AscendC lowering | add builtin + `asc_preload` call emission |
| **A10** | **Apply the devkit's single best vector lever — `#pragma unroll N` inside the VF loop** (its Case 2→3 step, 3413→2197 cycles = 1.55×) to the apply pass: `for i in T.unroll(NCH, unroll_factor=4)`. Prerequisite check: emit the kernel and **read the generated `.asc`**, because `unroll_factor` is documented "honored by CUDA codegen only" (`TL/tilelang/language/loop.py:257`) even though `TL/src/ascend/codegen/codegen_ascend.cc:3042-3049, 2970-2976` implements it | `gelu.asc:92-94` (`#if SCENARIO_NUM==2 → #pragma unroll 6`); `dual_issue_optimization.md:89-96`; `gelu…/README_en.md:402` ("eliminating loop control overhead while allowing VF instructions from multiple iterations to be interleaved") | **Probably yes, but untested**: codegen path exists; `unroll_factor` appears in **0 of** the repo's tests/examples, and `#pragma unroll N` (with a factor) appears **0 times** across the 553 cached generated kernels (bare `#pragma unroll` appears 173×, never inside a VF body) | **Unknown until the emitted source is inspected.** If it works: removes loop control from the 1792-instruction apply pass — the largest non-arithmetic cost we have. If TileLang's `unroll_loop_skip_vf.cc:98` also strips the pragma inside VF, this needs a compiler-side fix | n/a |
| **A11** | Fix VF loop induction-variable type to `uint16_t` so the CANN compiler encodes Hardware Loops | `DOCS/…/reg_vector_compute/key_features/key_features.md:10`: "VF内所有Loop的迭代变量必须是uint16_t类型"; lines 5, 11-14 (0-start, +1 step, no in-loop branches, ≤4 nesting) | **No** — `CodeGenTileLangAscend::VisitStmt_(const ForNode*)` (`codegen_ascend.cc:2968-2977`) falls through to `CodeGenC::VisitStmt_(ForNode)`, which prints the loop var with its **TIR** dtype (int32 by default); nothing on the Ascend path coerces it to `uint16_t`. Verified empirically: **489/489** for-loops inside `__simd_vf__` bodies across 398 cached VF bodies are `for (int32_t …)`, and `grep "for (uint16_t"` = **0 hits in 553 cached kernels** | **Potentially the largest single compiler-side win** for any VF-heavy kernel, but **I have zero measurements** — the hardware-loop benefit is asserted by the doc only. TileLang cannot express it from Python today | add a dtype coercion in the AscendC codegen for VF-scope loop vars (must be provably non-negative, < 65536, step 1 — our `T.serial(NCH)` loops satisfy all three) |

---

## 9. Hardware-loop / codegen constraints the team must respect (from devkit docs, verified against TileLang codegen)

`DOCS/…/reg_vector_compute/key_features/key_features.md:5-15` (Hardware Loop encoding rules):
> max 4 nesting levels; **all VF loop induction variables must be `uint16_t`**; start at 0; step +1; **no jumps inside the loop**
> (no `if/else`, no ternary — the compiler does best-effort if-conversion but "不做完全性保证"); loop count immutable once started;
> to use an outer counter as an inner bound, copy it to a scalar first.

Consequences for us:
1. **v23's `if rows == 4:`/`else` split is done in Python** (`kernel.py:36`, `158`) → two separate compiled kernels, so no in-VF branch.
   ✅ correct approach; keep it.
2. The `if/else-free` requirement explains why `if (loopIdx == loopNum-1) ? tailRow : tileRow` in `gelu.asc:137` lives in
   `__aicore__` (scalar domain), **not** inside `__simd_vf__`. Our scalar conditionals must likewise stay outside `T.SimdVF`.
3. **Address registers**: `vf_loop_optimization.md:154-190` says the compiler auto-generates `AddrReg` when a loop's load address has
   the form `srcAddr + i*const1 + j*const2 + …` (≤4 dims), removing the scalar offset math — and warns (line 192): "若直接使用 AddrReg
   存储偏移量或其它底层接口，可能会限制编译器的全局优化能力" (hand-writing AddrReg usage *hurts*). **TileLang exposes `AddrReg`-style
   addressing only as `vld(..., post_inc=step)` with `make_ubuf_ptr` (`simd.py:324-356, 524`) — which is precisely the low-level form
   the doc says not to hand-write.** v23's `x_ub[r0 + (i*4+k)*CH]` is already the auto-recognized affine pattern. ✅ **Do not "optimize"
   it into `post_inc`.** Explicit anti-recommendation with a citation.
4. **Register ceilings** (§2.6, `dual_issue_optimization.md:149-152`): 32 RegTensor / 8 MaskReg / 8 AddrReg / 4 Unalign-regs per VF,
   and **exceeding them makes the compiler spill to UB *and insert synchronization* — "严重拖慢算子的执行效率"**. v23 is ~29-31 live
   registers (source-level count). **Any new idea that adds live registers (single-pass fusion, 8 acc chains — cf. v28 which was worse,
   `final-status.md:51`) risks crossing 32 and causing exactly the spilling pathology. This is the strongest argument I found against
   further register-level micro-optimization of v23.**
5. `dual_issue_optimization.md:57-58` (CAUTION box): a `StoreAlign` to UB followed by a `LoadAlign` from the **same** UB region inside
   one VF needs `LocalMemBar<VEC_STORE, VEC_LOAD>` — "同一RegTensor的寄存器保序适用于先读后写的场景，不能保证此处先写后读的UB访问顺序".
   v23 writes `y_ub` in-VF and reads it out-of-VF via `T.copy` — different pipes, handled by TileLang's sync insertion. But if we ever
   fuse the `wf` materialization (`kernel.py:48-49`, vsts) and its consumption (`kernel.py:137`, vld) into *one* `SimdVF` region
   (a tempting micro-opt), we would need that exact barrier — **and would reintroduce the v10-family race class.** Don't.
6. `vf_fusion_optimization.md:106-111`: prefer **continuous-computation** basic-API forms over high-dim split (repeat/stride) forms,
   because the compiler's VF-fusion legality check can't fuse across stride/mask params. TileLang note: **`--cce-simd-vf-fusion` is not
   plumbed anywhere in TileLang** (`grep -rn "vf-fusion|vf_fusion|cce-simd" TL/src TL/tilelang` → 0 hits), so our `T.SimdVF` regions are
   hand-fused — which is what `CMAKE_VF_MODE=false` in `SOFT/CMakeLists.txt:16,36` corresponds to. **The devkit's own best result
   (Case 5, 1785 cycles) was obtained with VF fusion DISABLED.** Our approach is aligned with their winner. ✅

### 9.6 Empirical check of rules 1-4 against TileLang's actual output (new, and the most actionable finding here)

I searched TileLang's own generated-kernel cache (`~/.tilelang/cache/0.1.15_cuda_git*/…/device_kernel.asc`) exhaustively — every
`.asc` file, every `__simd_vf__` body, every loop inside them (largest body 5618 chars, so the 8000-char extraction window truncates
nothing):

```
.asc files scanned:              553
__simd_vf__ bodies found:        398
for-loops inside VF bodies:      489
  induction-variable types:      { "int32_t": 489 }      <-- 489/489, no exceptions
'for (uint16_t' anywhere in the 553 files:               0 hits
'#pragma unroll' occurrences:    173, ALL bare; 0 with a factor, 0 inside a VF body
```

Read against `key_features.md:10` ("VF内所有Loop的迭代变量必须是uint16_t类型"), **every loop TileLang generates inside a
`__simd_vf__` is disqualified from Hardware Loop encoding and therefore runs as a Software Loop** (compare/increment/jump per iteration,
plus the scalar address math the hardware loop would have absorbed). Consequences, with honesty about what I do and don't know:

- **What is established:** the emitted C++ text is `int32_t`; the doc states `uint16_t` is mandatory; nothing in the TileLang frontend
  can request otherwise (`T.serial`/`T.unroll` loop vars come from TIR as int32). This is a **compiler-frontend gap, not a kernel gap** —
  it affects every VF-heavy TileLang kernel on 950PR, not just ours.
- **What is NOT established:** how much this costs. The devkit never quantifies "hardware loop vs software loop" on its own — it only
  lists it as a rule. `gelu…/README_en.md:402` attributes loop-control overhead generically to non-unrolled loops. **I did not run an
  experiment**, so treat the size of this as unknown. My §7 IPC estimate (1.25 vs their 1.37 best) is *consistent with* loop-control
  overhead being real and worth ~5-10% of vec time, but that is an inference from two different kernels, so label it **weak speculation**.
- **Why I flag it anyway:** it is the only finding in this whole study that points at a *new mechanism* rather than a knob we failed to
  turn, and the fix is small (coerce VF-scope loop vars to uint16 when extent is provably in [0, 65536) and step is 1). That is a
  `codegen_ascend.cc` change, not a language change — worth one issue to the TileLang owners.
- **Test for whoever picks this up (5 minutes, no device needed):** compile any VF kernel, grep the emitted `.asc` for
  `__simd_vf__` bodies, and check whether CANN's compiler reports hardware-loop usage (or simply diff cycle counts against a hand-written
  `uint16_t` version of the same loop in the devkit harness). Do **not** start by changing TileLang codegen.

### 9.7 `#pragma unroll N` inside a VF: implemented, never exercised

`TL/src/ascend/codegen/codegen_ascend.cc:3042-3049` reads the `"pragma_unroll_factor"` annotation into `unroll_factor_`, and
`:2970-2976` emits `#pragma unroll <N>` for a `ForKind::kUnrolled` loop when the map has an entry. `T.unroll(start, stop,
unroll_factor=N)` (`TL/tilelang/ascend/language/loop.py:15-45`) sets exactly that annotation. So the devkit's best lever
(`gelu.asc:92-94`, `#pragma unroll 6`, IPC 1.20→1.37) looks reachable — **but two cautions**:
1. `unroll_loop_skip_vf.cc:98` returns VF blocks untouched ("VF bodies are consumed by their dedicated lowering… also preserves explicit
   unroll loops nested anywhere in the body"), i.e. TileLang deliberately does *not* materialize unrolling inside VF and lets the pragma
   survive to the CANN compiler. That is the desired behaviour, and it is precisely what no test exercises.
2. `grep -rn "unroll_factor" TL/tilelang TL/testing TL/examples` shows it is **used by zero tests and zero examples**, and
   `TL/tilelang/language/loop.py:257` still documents it as "honored by CUDA codegen only" — a doc/codegen contradiction. The Ascend path
   may work, may be silently dropped by a later pass, or may hit an ICHECK. **Nobody knows.** Hence A10's "emit and read the `.asc`" step
   is the whole point: it is a 5-minute disambiguation of a lever the devkit measured at **1.55×** on the vector pipe.

---

## 10. Build/verify harness facts (for reproducing devkit numbers locally)

- `SOFT/CMakeLists.txt:14-17`: `CMAKE_ASC_RUN_MODE=npu|cpu|sim`, `CMAKE_ASC_ARCHITECTURES=dav-3510`, `CMAKE_VF_MODE` (default
  `false`), `SCENARIO_NUM` 0–5 → `-DSCENARIO_NUM=N` selects the case; `target_compile_options`: `--npu-arch=${CMAKE_ASC_ARCHITECTURES}`
  and `--cce-simd-vf-fusion=${CMAKE_VF_MODE}` (`CMakeLists.txt:34-37`). Links `tiling_api register platform m dl` (`39-44`).
- Shape: fixed **[128,128] fp32** ND in / ND out (`SOFT/README_en.md:42-62`); kernel name `softmax_custom`; `<<<kDim=1, 0, stream>>>`
  direct-launch, single core (`softmax.asc:828`).
- `scripts/gen_data.py:20`: `np.random.uniform(-5, 5, [128,128]).astype(np.float32)`; golden = standard max-subtracted softmax.
- `scripts/verify_result.py:19-21,41`: `RELATIVE_TOL=1e-4, ABSOLUTE_TOL=1e-5, ERROR_TOL=1e-4`; pass = fraction of mismatching elements
  ≤ 1e-4 (`error ratio: 0.0000` in `README_en.md:369`).
- Profiling recipe: `msopprof ./demo && cat ./OPPROF_*/PipeUtilization.csv` (`SOFT/README_en.md:387-389`); metric definitions in
  `SOFT/README_en.md:78-85` (`aiv_total_cycle`, `aiv_vec_ratio`) and `gelu…/README_en.md:157-168`
  (`aiv_mte2_time`=GM→UB, `aiv_mte3_time`=UB→GM). **Note: `docs/…/appendix/reg_vector_compute_interface_soft_simulation_summary.md`
  and the single/dual-issue interface summary referenced by `key_features.md:18` do NOT exist in this devkit copy**
  (`docs/zh/appendix/` absent) — so the per-instruction single-vs-dual-issue table cannot be checked here. Flagged as a gap.

---

## 11. What is NOT transferable (explicit, to stop wasted rounds)

1. **Any cache-policy / non-temporal / write-combining technique** — `02_reg_compute` contains none (§5). Do not mine these three
   examples for it.
2. **Any DoubleBuffer/multi-core/data-movement optimization** — explicitly disclaimed (`SOFT/README.md:84`); gelu's ping-pong is depth 2,
   shallower than our depth 4.
3. **Main-tail block splitting** — structurally no-op at D=4096, CH=64; and the devkit's own Case-4/5 tail code is present-but-unused
   (`softmax.asc:616-620, 710-714`), i.e. unvalidated at tail≠0.
4. **`ExpSub`/`vexpdif`, Welford, LayerNorm-style advanced APIs** — no exp in RMSNorm.
5. **`MulsCast`** — only `float→half`, and bf16 output isn't a target type (P4/A6); not exposed in TileLang either.
6. **A single-pass (one-load-per-row) RMSNorm** — impossible: the scale is not known until all 64 chunks accumulate, and keeping 64
   chunk registers exceeds the 32-register VF budget (§2.6). **The devkit itself re-loads x twice** (`softmax.asc:654` then `664`), so our
   2-loads-per-row is the floor, not a defect. **Do not schedule a round to "fix" this** — this was my own leading hypothesis at the start
   of the study and §3.2 disproves it. The residual redundancy that *does* exist is the `wf` weight reload, and it is A2, not fusion.
7. **Whole-problem-in-UB tiling** — devkit's [128,128] is 64 KiB; our 8192×4096 is 64 MiB. Their `loopCount=1` luxury is unavailable.
8. **Per-instruction dual-issue tables** — the referenced appendix docs are missing from this devkit copy; IPC claims in §3.4 are the
   READMEs' own numbers, not independently checkable here.

---

## 12. One-paragraph recommendation to the Lead

`02_reg_compute` has no remaining headroom for RMSNorm prefill at the kernel level: v23 already implements every structural lesson
the devkit's best case (Case 5, 1785 cycles) uses — one `__simd_vf__` region, register-resident accumulators, compile-time `ALL` mask
with tail≡0, 2-loads-per-row (the devkit's own floor, not a defect), row-pair unrolling for dual-issue — and it *exceeds* the devkit on
pipelining (4 stages vs 2) and on reduce-to-broadcast (register `vdupv` vs a UB round-trip). Counting the actual instruction budget
(≈2620 vector instructions/step against ≈66.8k cycles of measured vec time) puts **v23's effective vector IPC at ≈1.25**, i.e. within 10%
of the devkit's best measured 1.37 and inside their "relatively optimal" band; **the vector pipe has no scheduling headroom left.** The
devkit's GELU ablation independently quantifies what vector optimization is worth in an MTE-bound kernel: **0.76% end-to-end for a 55%
vector-time cut.** So redirect effort to (a) the untried L2 policy pairs on the *kernel* (**A1** — the knob that already bought 2.7× on the
write path once, and the only candidate large enough to explain a 22% deficit), (b) manual `order`/`stage` pipeline schedules (**A5**,
already in flight as v30-v32) — with the added data point that the answer is *ordering*, not depth, since devkit depth 2 < our depth 4 — and
(c) task-2's `04_memory_access` bank-conflict/DataCopy study, which is where the devkit actually addresses the pipe we are starved on.
Two cheap vector-side items are worth spending one window on *before* we retire the vector hypothesis entirely: **A2** (dedup the `wf`
weight load across the 4 rows: −7.3% instructions, −38% UB read bytes, 6-line edit, but watch the 8192 B-stride bank-conflict risk) and
**A10** (`#pragma unroll N` on the apply loop — the devkit's 1.55× lever; first confirm in the emitted `.asc` that TileLang actually keeps
the pragma inside the VF body, which no existing test does). Escalate **A11/§9.6** to the TileLang owners as a compiler bug or at least a
doc/codegen contradiction (`unroll_factor` documented "CUDA codegen only" while `codegen_ascend.cc` implements it for AscendC).
Keep v23's ~29-31 live-register count as a hard design constraint: the 32-RegTensor ceiling means exceeding it makes the compiler **spill
to UB and insert synchronization** ("严重拖慢算子的执行效率"), which is the failure mode most likely to masquerade as "an optimization that
made it slower" (cf. v28's 8-chain result).
