# <interface name> (e.g. UB capacity / AIV pipe attribution / row tiling)

> One file per hardware interface or measured hardware fact. Operator-agnostic; mechanism only.
> TileLang mode: examples are TileLang Python (`tilelang.ascend.language`); measurement recipes may use
> `tilelang.profiler.msprof` and the CANN / Ascend runtime surface.

## Purpose
What hardware-level problem this interface solves (one sentence).

## When needed (trigger)
Which student need / symptom calls for it (profiler symptom / dataflow shape / compiler or launch error text).

## Minimal runnable example
```python
# Minimal, self-contained example. From a verifiable source; nothing made up.
# Shows only "how to use / how to measure this interface"; embeds no specific operator's algorithm.
# A measurement recipe must run on the device (a compile alone does not enforce resource limits).
```

## Gotchas
- What error a misuse raises / where it goes silently wrong (e.g. resource ceilings, launch-time checks,
  pipe-share vs wall-clock, buffer budget accounting).

## Sources

**The student reads this section; only three kinds of citation are allowed** (a tool's symbols may be cited,
the reference solution's may not):

1. Official documentation links (CANN / Ascend runtime / TileLang docs);
2. **API symbols** of public infra (the TileLang / torch_npu / CANN interface surface, i.e. what the
   student's code has to call anyway);
3. Paths and verdicts of this repo's device-certified Ascend950PR probes (`PROBE PASS ...` or a measured
   reading).

**Forbidden**: the reference implementation's (the oracle's, open or closed source) file names, function
names, line numbers, paraphrases of its content, measured footprints, and the words "oracle / teacher /
reference implementation" — all such narrated provenance goes to
`docs/kb-provenance/tilelang-ascend/<this entry's name>.md` (for human review, in no agent read set); the
entry keeps only one line:

```
- Detailed provenance: docs/kb-provenance/tilelang-ascend/<entry>.md (for human review, not in the agent read set).
```

The mechanical gate `src/forgekernel/loop/kb_lint.py` scans for banned tokens at three points: loop start,
recipe certification into the KB, and publish.
