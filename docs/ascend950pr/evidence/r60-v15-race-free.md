# R60: race-free two-phase via explicit per-row registers

```json
{
  "round": "60",
  "date": "2026-11-02",
  "topic": "a race-free way to get the two-phase schedule: explicit per-row registers",
  "candidate": "docs/ascend950pr/candidates/rmsnorm_simdvf_v15/kernel.py",
  "design": "the rows=4 fast path is written out explicitly in the source (four blocks, four accumulator sets, four scale variables). Every row keeps its scale in its own register, so - unlike v10 - no shared buffer carries state between a reduce stage and a scale stage, and no Python for-loop inside the kernel is converted into a device loop.",
  "correctness": [
    {
      "batch": 64,
      "rows": 1,
      "rel": 0,
      "pass": true,
      "deterministic": true
    },
    {
      "batch": 128,
      "rows": 2,
      "rel": 0.000672,
      "pass": true,
      "deterministic": true
    },
    {
      "batch": 256,
      "rows": 4,
      "rel": 0.001309,
      "pass": true,
      "deterministic": true
    },
    {
      "batch": 1024,
      "rows": 4,
      "rel": 0.001136,
      "pass": true,
      "deterministic": true
    },
    {
      "batch": 8192,
      "rows": 4,
      "rel": 0.002058,
      "pass": true,
      "deterministic": true
    }
  ],
  "determinism_method": "five repeats per shape, outputs compared bit for bit",
  "gate": {
    "status": "RETEST",
    "quiet": 0,
    "note": "the device was occupied by external tenants and the child produced no measurement; authenticity and precision passed"
  },
  "note": "an earlier attempt at this file failed to compile with \"Immutable variable full is used outside its defining region\" because the generated phase blocks were written one indentation level outside the SimdVF region; that was a generator bug, fixed.",
  "status": "v15 is correctness-clean and race-free by construction; its performance is not yet measured"
}
```
