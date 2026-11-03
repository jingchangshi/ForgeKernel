# TLA capability on THIS machine, measured against what RMSNorm v33 needs

Investigator: `implementer-v33` (task-10), 2026-11-03. Write scope:
`docs/ascend950pr/tla/`. Probes: `probes/probe_compile_attempt.py`,
`probes/probe_aliased_import.py`, `probes/probe_tilelang_l2_emit.py`.
Nothing in the tilelang env, ForgeKernel source, or AgentGates was modified; no
package was installed.

Labels: **[measured]** = I ran it on this host and read the result.
**[source-only]** = read from source on this host, not executed.
**[inferred]** = my reasoning over verified facts. **[unknown]** = not determined.

---

## 0. Bottom line

**TLA cannot host v33 on this machine today**, for two independent reasons: there
is no installable artifact here that contains it, and the one lever our certified
prefill result depends on most (the non-allocating L2 read policy) is structurally
unreachable on the vector GM->UB path TLA exposes. Confidence: **high** for the
first, **high** for the second, and both are source- or measurement-backed rather
than name-plausible.

"Migrating from TileLang to TLA" is a **change of DSL family** — TileLang
(`T.*` over TVM/TIR) to Triton-with-a-`tla`-namespace — not a version bump inside
TileLang. Combined with the absence of any installable artifact, that framing is
the decision-relevant one.

---

## 1. Wheel situation: TLA is NOT installed here

| Fact | Label | Evidence |
|---|---|---|
| `import triton` works; it IS triton_ascend 3.6.0 | [measured] | `triton.__version__` 3.6.0, dist-info `triton_ascend-3.6.0+dev20260816011301` |
| `from triton.language.extra import tla` **fails** | [measured] | `ImportError: cannot import name 'tla'` |
| Installed `language/extra/` = {cann, cuda, hip, kernels, libdevice.py} — no `tla` | [measured] | directory listing |
| The wheel's RECORD has **zero** `tla` entries | [measured] | `grep -c 'extra/tla'` = 0 in all 5 triton_ascend wheels on the host |
| The source tree **does** have it: 19 .py files, 3977 lines | [measured] | `third_party/ascend/language/tla/` |
| TLA landed 2026-10-24; the installed wheel is dated 2026-08-16 | [measured] | `git log` first TLA commit `9e6fdfcfc`; wheel name `dev20260816011301` |

So TLA postdates every wheel on this host. **The missing artifact is not a
misconfigured path — it is a missing build.** Installing the adapter README's
pinned `dev20261030+git0c9d99ca` wheel is not possible here (it is not on this
host), and I did not install anything.

### 1.1 The source tree also cannot be aliased into place [measured]

The obvious workaround — point the hardcoded `triton.language.extra.tla` import
at the on-disk source tree — gets further and then fails at the native layer.
`probes/probe_aliased_import.py` does exactly that:

```
STEP alias:            OK   (source tree mounted as triton.language.extra.tla)
STEP import_tla:           OK
STEP jit_decorate:         OK
STEP compile_warmup:       FAIL AttributeError
  module 'triton._C.libtriton.ascend.ir' has no attribute 'tla_builder'
  ... in compiler.py:232  tla_builder = ascend_ir.tla_builder(context, arch or "")
```

The Python layer is pure source and imports fine; **the native builder it calls
does not exist in the compiled extension.** Confirmed directly [measured]:

```
libtriton.ir.builder .............. create_tla_load/store/alloc/buffer_type/
                                    set_flag/sync_all/subview/bind/simd_vfma/
                                    annotation_mark  -> ALL ABSENT
libtriton.ascend.ir ............... tla_builder           -> ABSENT
                                    any symbol matching 'tla' -> none (0 of 83)
```

For contrast, the source tree references 24 distinct `create_tla_*` and 26
`create_simd_*` native builders [source-only, `cpp/tla_builder.cc`]. None are in
the installed binary. So this is an ABI-level absence: populating
`language/extra/tla/` by hand would still leave TLA unable to compile anything.

**Consequence:** every row below is necessarily **[source-only]**; no TLA probe
could be run on device. The one thing I *did* measure on device was the TileLang
counterpart, to pin down what TLA would have to reproduce (section 3).

---

## 2. Capability table — v33 constructs vs TLA

Confidence is in the *existence* of a named, wired counterpart in the source
tree. Since nothing could be executed, "equivalent" here means "the API exists and
its docstring/lowering argues it does the same thing" — **not** "verified to
produce identical device behaviour".

| v33 construct | TLA counterpart | Verdict | Evidence / confidence |
|---|---|---|---|
| `T.Kernel(56)` grid launch | `@tla.jit` + `tl.program_id(0)`; launch grid is the Triton tuple | **equivalent** | `jit.py`; gate kernel launches `cast_back[(cores,)]` [source-only, high] |
| `T.alloc_shared` (w_ub/x_ub/y_ub) | `tla.alloc_buffer(shape, dtype, space="ub", versions=n)` | **equivalent** | `alloc.py:41`; `versions` is a first-class arg [source-only, high] |
| `T.annotate_buffer_versions({x:4,y:2})` | `tla.annotate_buffer_versions(buf, n)` and `alloc_buffer(versions=)` | **equivalent** | `aux_ops.py:57-74` — emits the same `hivm.multi_buffer` hint; `versions=1` opts out identically [source-only, high] |
| `T.Pipelined(n, num_stages=4)` | **no `Pipelined` equivalent**; `tl.range(..., num_stages=)` exists in the language but the TLA/HIVM lowering's use of it is unverified | **not found / unknown** | no `Pipelined`/`num_stages` in any of the 19 tla files [source-only, medium — see section 4] |
| **`T.copy(..., l2_cache_ctrl="NOTALLOC_CLEAN")`** | **NOT available on GM->UB.** `LoadGM2UBParams` exposes only `eviction_policy` in {EvictNormal, EvictFirst, EvictLast} | **NOT equivalent — 1.10x at risk** | `dma.py:134-141`, `:21-24`, `:245-249` [source-only, high — full chain in section 3] |
| `T.SimdVF()` register region | `with tla.scope.simd():` | **equivalent** | `scope.py`; used in the gate kernel [source-only, high] |
| `T.simd.pset(32)` (predicate) | `tla.simd.pge(n, lanes)` — prefix predicate | **equivalent in role** | `simd.py:1562`; v33's `pset(32)` is an all-32-lane-true mask, so the closer call is `pge(32,32)` [source-only, medium] |
| `T.simd.vld dist="UNPK_B16"` | `tla.simd.vlds(..., dist="UNPK_B16")` — same string in whitelist | **equivalent** | `simd.py:618`, `simd_semantic.py:58` [source-only, high] |
| `T.simd.vcvt part=0` | `tla.simd.vextf(v, f32, part="part_even")` | **equivalent** | `simd.py:684`; its docstring gives the BF16 chain `vlds(UNPK_B16) -> vextf(part_even) -> vmul`, i.e. v33's exact decode chain [source-only, high] |
| `T.simd.vmula` (fused mul-add) | `tla.simd.vfma(acc, lhs, rhs)` = `acc + lhs*rhs`, lowers to `ave.hir.vmula` | **equivalent, same native op** | `simd.py:1242` [source-only, high] |
| `vadd` / `vsub` / `vmul` | `vadd` / `vsub` / `vmul` | **equivalent** | `simd.py:789/1180/780` [source-only, high] |
| `vmuls(scalar)` / `vadds(scalar)` | `vsplat(scalar) + vmul`; `vadds` exists for add | **equivalent with a step** | no `vmuls`; `math.py:60-70` shows the idiom `S.vmul(a, S.vsplat(c, dtype))` [source-only, high] |
| `vdiv` | `tla.simd.vdiv(src0, src1, precision=)` | **equivalent** | `simd.py:881` [source-only, high] |
| `vsqrt` / `vdup` / `vdupv(pos="POS_LOWEST")` | `vsqrt` / `vdup` / `vdupv(..., pos=POS.LOWEST)` | **equivalent, same pos string** | `simd.py:1288/1087/1138`; `POS.LOWEST == "POS_LOWEST"` at `:238-250` [source-only, high] |
| `vsts dist="PK_B32"` | `tla.simd.vsts(..., dist="PK_B32")` | **equivalent** | `simd.py:648`, `simd_semantic.py:95` [source-only, high] |
| `vcadd` reduce tree, 6 levels | `tla.simd.vreduce_sum(src)` / `vreduce(src,"add")` | **equivalent outcome, different shape** | `simd.py:1470`, `:1300`. TLA gives one horizontal reduction op; v33 hand-builds 6 `vcadd` levels. Result lands in lane 0 both ways [source-only, high] |
| `T.simd.alloc_var` (register temp) | **no direct equivalent**; `vfma` takes an explicit acc register, other ops allocate implicitly | **partial / unknown** | no `alloc_var`/`alloc_local` in tla. v33 uses it for the 4x4 accumulators [source-only, medium — see section 6] |
| `T.max/T.min/T.if_then_else` for the ragged split | `tl.maximum`/`tl.minimum`/`tl.where` | **equivalent** | [measured] `hasattr(tl, ...)` is True for all three in the installed wheel |
| rows=4 explicit unrolled body | ordinary Python unrolling inside `tla.scope.simd()` | **equivalent** | gate kernel unrolls with `tl.static_range` [source-only, high] |
| `T.copy` Y write with `l2_cache_ctrl="NORMAL_FV"` | `store_ub_to_gm` + `StoreUB2GMParams(atomic_kind=...)` — **no cache field at all** | **NOT equivalent (harmless here)** | `dma.py:142-147`. Confirmed: the write path has only `atomic_kind`. v33's write is NORMAL_FV = hardware default 0, so its absence costs nothing [source-only, high] |

### 2.1 Constructs TLA has that v33 does not need
Not evidence of parity, just inventory: `cube` (mmad/L0C fixpipe), `sync`
(explicit FFTS flags, `sub_block_id`), `compensated_rstd` (a numerically
compensated rsqrt helper, `math.py:112`), and
`vintlv/vdintlv/vbitcast/vtruncf/vmod/vsstb/vci/membar`.

---

## 3. The L2 read policy — the single most decision-relevant row

**Verdict: TLA's GM->UB load cannot express `NOTALLOC_CLEAN`. Our 1.103x read-path
win is not reachable through the TLA vector path as it exists in this source tree.**
Confidence **high**. This is the row that decides whether a migration would
*regress* our certified result rather than improve it.

The chain, each link verified by me independently of the Lead's write-up in
`l2-policy-field-width.md` (which I cross-checked and agree with):

1. **Hardware field is wide** [source-only]: `asc_load_l2_cache_mode` is a
   `uint8_t` enum `{NORMAL_FIRST_VICTIM=0, NORMAL_LAST_VICTIM=1, NORMAL_PERSISTENT=2,
   NOTALLOC_KEEP=4, NOTALLOC_CLEAN=5, NOTALLOC_DROP=6}`
   (`cann-9.2.0/x86_64-linux/asc/include/c_api/utils/enum.h:17-24`). Note the gap at
   3 — `NOTALLOC_*` is different *instruction semantics*, not a lower priority.
2. **TLA's vector load exposes only eviction priority** [source-only]:
   `LoadGM2UBParams.eviction_policy: EvictionPolicy` (`dma.py:139`), enum
   `{EvictNormal, EvictFirst, EvictLast}` (`dma.py:21-24`), passed to
   `create_tla_load` as a string attr (`dma.py:245-249`).
3. **MLIR attr is 3-valued** [source-only]: `EvictNormal=1, EvictFirst=2,
   EvictLast=3` (`HIVMAttrs.td:1223-1231`).
4. **The vector copy narrows it to one bit** [source-only]:
   `uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);` in
   `Copy1D.cpp:95,179`, `Copy2D.cpp:161,272`, `Copy3D.cpp:144`, where the template
   enum is `enum class EvictionPolicy : uint32_t { EvictFirst = 0, EvictLast = 1 }`
   (`Template/include/RegBase/DMAUtils.h:39`). **Value 5 has nowhere to go.**
5. **Even `EvictNormal` is not value 2** [source-only]: the lowering maps
   `EvictFirst->0, EvictLast->1` and falls back to `EvictFirst` for anything else,
   warning "Only support EvictFirst & EvictLast" (`HIVMToStandard.cpp:68-71,
   1232-1236`). So TLA's `EvictNormal` would itself be silently coerced.
6. **`ConvertToHIVMOp.cpp:312-313` defaults every load to `EvictFirst`**, so a
   Triton/vector load is already evict-first unless told otherwise. **`EvictFirst`
   is thus not a new lever TLA offers — it is the existing default.** Any
   "try EvictFirst" experiment measures the status quo, not an improvement.

### 3.1 The trap: `l2_cache_ctrl` DOES exist — for the Cube path only

`LoadGM2L1Params.l2_cache_ctrl: int` (`dma.py:149-155`) is wired straight through
to `l2_cache_mode` (`dma.py:266-275`) and Cube's `ND2NZ.h:61` passes the real
value. So "TLA has `l2_cache_ctrl`" is **true and useless to us**: it is GM->L1
(Cube/ND2NZ), a path v33 never takes. Reading that as coverage of our lever is
exactly the misreading that would make a migration look safe. v33's read is GM->UB.

### 3.2 The vendor's own compiler docs agree, in nearly our words [source-only]

`HIVMDMAOps.td:381-387`:

> "`l2_cache_mode`: the raw value of the MTE2 descriptor's cache control field …
> LoadOp spells the same field as an EvictFirst/EvictLast enum, which is all the
> vector copies use, but the field is wider than that enum and a matmul needs the
> rest of it: **keeping a streamed-once operand out of L2 entirely is what stops it
> evicting the operand that is re-read, and no eviction priority can express that.**"

That is our R84 section 3 mechanism argument stated independently by the vendor. It
also means the gap is **structural to the compiler**, not a parameter TLA forgot —
so "it will be added upstream later" is not a well-founded hope unless LoadOp
gains the attribute or the vector copies are changed to take the wide field.

### 3.3 Why TileLang HAS it and TLA does not — measured, not assumed

This is the difference between the two DSLs, and it explains why our win cannot
be assumed to port.

TileLang does **not** route this through the HIVM enum at all. Its codegen emits
the **CANN C-API** call directly, casting to the *hardware* enum
(`tilelang/src/ascend/codegen/codegen_ascend.cc:1417-1433`, `EmitGmToUbufCopy_`):
`static_cast<asc_load_l2_cache_mode>(arg8)`. Because the cast target is the wide
hardware enum rather than HIVM's 2-value one, value 5 survives.

I confirmed the emitted device source for the **shipped v33 kernel** [measured,
`probes/probe_tilelang_l2_emit.py`]:

```
asc_load_l2_cache_mode  = 0    <- W tile read   (NORMAL_FV)
asc_load_l2_cache_mode  = 5    <- X tile read   (NOTALLOC_CLEAN)  <-- our win
asc_store_l2_cache_mode = 0    <- Y tile write  (NORMAL_FV)
```

Three DMA sites, matching v33's intent exactly. So the 1.103x lever is real, it is
in the shipped artifact as byte 5, and it is on a path TLA does not expose.
**[inferred, high]**: TLA cannot produce that byte 5 on GM->UB, therefore a TLA
port loses the lever and lands back near the `NORMAL_FV` prefill number
(vendor/cand about 1.017 rather than about 1.1185), i.e. **it would regress our
certified prefill by roughly the 1.10x the policy is worth**.

### 3.4 The one experiment that would upgrade this — NOT DONE

The Lead asked for `EvictFirst` vs hardware-default on a prefill-sized window
(128 MB working set vs 128 MB L2) to see whether it reaches about 1.10x. **NOT
DONE, and I did not simulate it.** Two verified reasons it cannot be run as asked:

1. There is no TLA artifact on this host (section 1) — a TLA kernel cannot be
   compiled at all, so there is nothing to time.
2. Per section 3 item 6, `EvictFirst` is already the compiler's **default** for
   vector loads, so an EvictFirst-vs-default comparison is default-vs-default and
   would measure noise. The meaningful contrast (default vs byte 5) is precisely
   what TLA cannot express.

So this row stays **[source-only]** by necessity, and I am explicitly *not*
claiming a measured number for TLA.

---

## 4. Pipeline versioning and `num_stages` — the second most important row

**Verdict: version annotation EXISTS and is first-class; the staged pipeline
schedule is NOT obviously reproducible, and `stages=(4,2)` is not transferable as
an argument.** Split verdict, confidence medium-high.

- **`annotate_buffer_versions` / `alloc_buffer(versions=)`: EXISTS**
  [source-only, high]. `aux_ops.py:57-74` raises the identical `hivm.multi_buffer`
  annotation TileLang emits, with the same "1 opts out" semantics, and
  `alloc.py:41` takes `versions` directly. The gate kernel (`test_tla.py:43-45`)
  uses `versions=VERSIONS` on all three buffers, so it is exercised, not merely
  declared.
- **`T.Pipelined(n, num_stages=k)`: NO COUNTERPART FOUND** [source-only, medium].
  No `Pipelined` and no `num_stages` appears anywhere in the 19 tla files. Triton's
  `tl.range` does accept `num_stages` [measured, in the installed wheel], but
  whether the TLA/HIVM lowering honours it as TileLang's software-pipeline
  directive is **[unknown]** — I could not compile a TLA kernel to inspect the
  schedule (section 1), and nothing in the source states the mapping.
- **Our `(4,2)` is not portable.** In v33 it is
  `stages,yver = (4,2) if rows>1 else (2,1)`, a buffer-version count derived from
  measurement. If TLA's staging is expressible only as `versions=`, we would be
  hand-rolling the schedule `T.Pipelined` currently derives, and the measured
  optimum would have to be re-searched on the new DSL rather than carried over.
  That is a real (unquantified) cost, not a mechanical translation.

---

## 5. The gate kernel as the working reference [source-only]

`/home/s00653124/workspace/agent_tla/test_tla.py` is the one kernel the gate runs.
It computes `out[t,c] = f32(x_fp8[t,c]) * scale[t, c//32]` with `@tla.jit`, three
`alloc_buffer(..., versions=2)`, a `tla.view` per pointer, one `tla.copy` in and one
out per row, and inside `tla.scope.simd()` a `pge(32,64)` group mask,
`vlds(dist="BRC_B32"/"UNPK4_B8")`, `vextf`, `vsel`, `vmul`, `vsts(dist="NORM_B32")`.

What this proves: the VF idiom, the buffer-version annotation, and fp8->f32
load/convert/store with a per-group scale all have working TLA expression.
What it does **not** prove: any cache-policy control (it passes no policy
anywhere), any multi-stage pipeline schedule (it has no `Pipelined` analogue), and
any reduction (it has no `vcadd`/`vreduce` use). Its own docstring is honest that
VF "still processes in 64-lane groups" with one MTE2 load and one store per row.

Its assertions are also a caution for expectations: `test_ir` hard-codes
`loads=2, stores=1, buffers=3` — the gate checks the DMA structure, not
performance and not a cache policy.

---

## 6. What I could not determine [unknown]

1. Whether TLA's `tl.range(num_stages=)` maps onto TileLang-style software
   pipelining in the HIVM pipeline. Needs a compiling TLA install.
2. Whether `tla.simd`'s implicit temporaries reproduce v33's explicit `alloc_var`
   register strategy, and whether the 4x4 accumulators would spill.
3. Whether TLA can match v33's measured numbers at all — unmeasurable here; no
   TLA artifact, so no device timing exists for any TLA kernel.
4. Whether the pinned `dev20261030+git0c9d99ca` wheel changes section 3. It
   postdates the source tree's last TLA commit, so it *might*; but the narrowing
   lives in `Template/` and `HIVMToStandard`, which are generated C++ rather than
   the Python wheel, so I would not expect it to. Not verifiable here —
   **[unknown]**, not a hope.
5. Whether `pset(32)` has an exact one-to-one. `pge` is prefix-true, and for an
   all-lanes-true 32-lane mask the honest equivalent is `pge(32,32)`. Semantic in
   role, unverified in codegen.

---

## 7. Answer to the question asked

> Can TLA express and run what v33 does, on this machine, today?

**Not today, and not on this machine.** Two independent blockers:

1. **No installable artifact** [measured, high confidence]. TLA is a source tree
   in `triton-ascend` (2026-10-24) while every wheel here is from August 2026;
   `language/extra/tla/` is absent from all of them, and hand-mounting the source
   fails because the 24 `create_tla_*` / 26 `create_simd_*` native builders are
   absent from the installed `libtriton`. The adapter README's pinned wheel is not
   on this host.
2. **The L2 lever does not carry over** [source-only, high confidence].
   `NOTALLOC_CLEAN` (byte 5) is unexpressible on TLA's GM->UB vector load, whose
   eviction enum is narrowed to {0,1} before reaching hardware, and whose default
   is already `EvictFirst`. The Cube-only `l2_cache_ctrl` is a decoy. Our measured
   1.103x read-path win would be forfeit, which by our own numbers moves prefill
   from vendor/cand about 1.12 back to about 1.02 — a **regression of the
   certified result**, not an improvement. The vendor's own `.td` comment says no
   eviction priority can express this, so it is structural rather than an omission
   to be patched casually.

Everything else in v33 — the VF register ops, the `dist` strings, buffer versions,
the ragged scalar arithmetic — has a plausible named counterpart in the TLA
source, mostly with the *same spelling*. So the gap is narrow and specific, which
is exactly why it is worth stating bluntly: **the migration is technically close,
and the one thing it would drop is the largest single win we have.**

---

## 8. Constraints honoured

- No package installed; tilelang env untouched.
- ForgeKernel source and AgentGates not modified. All writes are under
  `docs/ascend950pr/tla/`.
- Every device command stayed under 100 s (the longest was a 95 s `timeout`-bound
  compile probe that failed fast).
- Device 1 used throughout; `TORCH_DEVICE_BACKEND_AUTOLOAD=0` plus an explicit
  `import torch_npu` inside the probes, since autoload-off otherwise breaks
  `device="npu"`.

