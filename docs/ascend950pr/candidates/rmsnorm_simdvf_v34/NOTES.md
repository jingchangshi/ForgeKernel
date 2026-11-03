# rmsnorm_simdvf_v34 — v33 with the redundant vcadd chains collapsed

Candidate: `docs/ascend950pr/candidates/rmsnorm_simdvf_v34/kernel.py`
Base: v33, **byte-identical except for five lines**. Author: lead, task-15, 2026-11-03.

## (a) The diff

v33 contains five chains of six chained `T.simd.vcadd` calls, 30 calls in total: four in the
`rows == 4` branch (`a0_*`..`a3_*`) and one in the decode branch (`a1..a6`). v34 replaces each
chain with a single `vcadd`, for 5 calls total. Nothing else changes: `N_CORES = 56`, the ragged
split, `LOOP_MODE`, `L2_READ_CTRL = NOTALLOC_CLEAN`, the staging depths `(4,2)`/`(2,1)`, every
accumulator and the whole load/store body are v33's.

## (b) Why collapsing is safe

`T.simd.vcadd` is a **full reduction into the lowest lane**, not a pairwise halving fold.
Measured on device 1 (evidence: `docs/ascend950pr/evidence/vcadd-semantics-probe.md`):

| input (64 lanes f32) | calls | lane0 | non-zero lanes |
|---|---|---|---|
| all 1.0 | 1 | 64.0 | 1 |
| ramp 0..63 | 1 | 2016.0 | 1 |

One call already yields the complete sum. The emitted instruction is
`simd_inst::vcadd(v0, full, MODE_ZEROING)`; `MODE_ZEROING` clears the lanes it does not write,
so a second call re-sums `{total, 0, ..., 0}` and returns the same total. **The chain is
idempotent**, which is why calls 2..6 were pure overhead rather than a correctness bug.

The pairwise operator in this DSL is `vcpadd` (32 non-zero lanes, then 16, then 8, needing six
calls to reach one lane). The source docstring at `simd.py:957` calls `vcadd` a "pairwise add
reduction", which is the upstream error that led the previous session to write the chain.

## (c) Verification [measured]

Bit-identical against v33 on **both** certified shape families, same inputs:

```
8192x4096: max_abs 0.000e+00  mismatches 0  bit_identical True
 64x4096: max_abs 0.000e+00  mismatches 0  bit_identical True
```

## (c2) Precision [measured]

rel_l2 against an independent float64 truth, three shapes; v34 equals v33 exactly because the
output is bit-identical:

```
8192x4096  rel_l2 v33=1.659178e-03  v34=1.659178e-03  equal=True
 64x4096  rel_l2 v33=1.657762e-03  v34=1.657762e-03  equal=True
256x4096  rel_l2 v33=1.657327e-03  v34=1.657327e-03  equal=True
```

About 1.66e-3 on every shape, which is the bf16 output floor. v33's certified precision result
therefore carries over without re-litigating the criterion.

## (d) Performance [measured, paired alternating A/B, 24 pairs, device 1]

| shape | v33 (30 vcadd) | v34 (5 vcadd) | per-pair median | interval | pairs agreeing |
|---|---|---|---|---|---|
| prefill 8192x4096 | 45.52 us | 45.20 us | 1.0066 | [1.0053, 1.0093] | 24/24 |
| decode 64x4096 | 6.74 us | 6.68 us | 1.0060 | [1.0029, 1.0146] | 18/24 |

Both shapes improve by about 0.6%, and the intervals exclude 1.0 on both. My earlier single
probe had suggested 1.6% for prefill; the paired measurement puts it at 0.66%, and the lower
figure is the one to quote.

## (e) Honest reading

**+0.6% may sit inside certification noise.** The case for v34 is not that it is a big win.
It is that the change is **provably output-preserving** (bit-identical on both shapes) and
strictly fewer instructions (25 fewer vector ops per tile in the prefill path, 5 in decode).
A change that cannot alter the answer and removes dead work is worth taking on its own merits;
the 0.6% is a bonus, and it should not be described as a discovery.

## (f) Reproduction

```bash
cd docs/ascend950pr/probes
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES=1
source /data/pri/cann-9.2.0/cann/set_env.sh
python3 v34_vs_v33.py
```