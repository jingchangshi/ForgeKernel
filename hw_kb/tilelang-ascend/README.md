# TileLang (Ascend NPU) hardware knowledge base (hw_kb/tilelang-ascend)

ForgeKernel's **TileLang mode** (loop language `tilelang-ascend`: the student writes a single-file
Python `kernel.py` in TileLang's own DSL, built by tilelang's Ascend compiler) has its own hardware
knowledge base. It is **fully separate** from the CuTe DSL mode's `hw_kb/cutedsl/` and the CUDA mode's
`hw_kb/cuda/`: a `tilelang-ascend` run reads and writes only `hw_kb/tilelang-ascend/<arch>/`
(`src/forgekernel/loop/lang_mode.py`). The mechanism knowledge on all three sides can be the same, the
spelling differs — this tree teaches TileLang Python (and the TileLang / torch_npu probe surface), not
`cute.*` calls and not CUDA C++ — so the trees do not share contents.

## Invariants (same as `hw_kb/cutedsl/` and `hw_kb/cuda/`)

- **Hardware mechanisms and measured hardware facts only, no operator algorithms.** "How to size a UB
  tile / how to read the AIV pipe attribution" ✅; "how this operator's tiles are wired" ❌.
- Every entry is self-contained, minimal and runnable, and carries its own sources. Example code comes
  from **verifiable sources** (CANN / Ascend runtime API symbols, TileLang public API, this repo's
  Ascend950PR probes); nothing is made up.
- **The evidence chain is split in two layers by reader**: the student reads an entry's "Sources", which
  may cite only official docs / public-infra API symbols / this repo's probes; the reference
  implementation's file names, function names, line numbers, paraphrased content and measured footprints
  all go to `docs/kb-provenance/tilelang-ascend/<entry name>.md` (for human review, not in the agent read
  set). The mechanical gate is `src/forgekernel/loop/kb_lint.py`.
- **Only device-certified material enters**: a reading measured on an Ascend950PR with a probe in this
  repo (a resource ceiling / pipe census / timing sweep), or a `PROBE PASS` from such a probe. A reading
  taken on another architecture or another language tree is a hypothesis, not evidence, and is not
  carried over.

## Layout

```
hw_kb/tilelang-ascend/
  README.md            # rules, layout and fill strategy
  TEMPLATE.md          # entry template (examples are TileLang Python)
  ascend950pr/
    index.md           # Ascend 950PR list; the entry point is the "Look up by symptom first" table
    <interface>.md
```

**One tree per architecture.** `kb_arch` comes from `[loop] kb_arch` in `operators/<op>/loop.toml` (or
`forge loop --kb-arch`); the CLI's default is `sm90`, so a TileLang-on-Ascend operator must set
`[loop] kb_arch = "ascend950pr"`. The tree it points to must exist and have an `index.md`, otherwise the
run is refused at start (exit 14). Opening a tree for a new architecture = create the directory + write
`index.md` following the rules above; no code change needed.

## Initial contents

- **Measured hardware priors** transcribed from this repo's Ascend950PR bf16 probe evidence: the per-core
  UB ceiling and how to measure it; the AIV pipe attribution and the device-time basis; the row-tile /
  UB-budget trade-off and the effective-bandwidth arithmetic; and the shared-card contention bias. These
  are the priors an iteration should read instead of re-discovering each round.
- Later entries grow one by one from device certification in TileLang runs and from the finish-time
  harvest.

## Filling strategy

Demand-driven: whichever interface (or hardware fact) the student gets stuck on gets its entry first
(`meta/kb_recipe_prompt.md`); expensive probe wins and KEEPs that changed a contract axis are written up
as recipes in the finish round (`meta/kb_harvest_prompt.md`). No up-front enumeration.
