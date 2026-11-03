#!/usr/bin/env python3
"""Decide, by measurement, what T.simd.vcadd does on Ascend950PR.

Three hypotheses, separated by the SAME experiment:

  A  PAIRWISE    one call folds adjacent lanes and packs partial sums into the low
                 half: 64 lanes of 1.0 -> 2.0 in lanes 0..31, i.e. 32 non-zero lanes.
                 log2(64)=6 calls land the total in lane 0.
  B  FULL-LANE0  one call sums all active lanes into lane 0 and zeroes the rest:
                 64.0 in lane 0, 1 non-zero lane. The chain is IDEMPOTENT -- every
                 later call re-sums {total,0,...,0} = total, so 6 calls still give the
                 right answer and merely waste 5 instructions.
  C  FULL-BCAST  one call writes the total to EVERY lane: 64.0 in all 64 lanes. A chain
                 then multiplies: call k gives 64**(k-1) * total, astronomically wrong.

Lane 0 alone CANNOT decide: under A after 6 calls only lane 0 is non-zero, which is
indistinguishable from B. The discriminator is the NON-ZERO LANE COUNT after 1, 2, 3
calls: A gives 32 -> 16 -> 8, B gives 1 -> 1 -> 1, C gives 64 -> 64 -> 64.

Run:
  source /data/pri/cann-9.2.0/cann/set_env.sh
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES=1
  /home/s00653124/.conda/envs/s00653124_tilelang/bin/python3 vcadd_semantics_probe.py
"""
import json
import os
import sys

import torch
import tilelang

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _vcadd_chain_kernels as K  # noqa: E402

LANES = 64
OPS = ("vcadd", "vcpadd", "vcgadd")
CALLS = (1, 2, 3, 6)
TOL = 1e-6
OUT = "/tmp/vcadd_probe_results.json"


def run(op, ncalls, x):
    build = getattr(K, f"build_{op}_{ncalls}")
    k = tilelang.compile(build(), target="ascend", out_idx=[1])
    y = k(x)
    torch.npu.synchronize()
    return y.cpu().to(torch.float64).numpy(), k


def summarize(vec):
    return {
        "lane0": float(vec[0]),
        "nonzero_lanes": int((abs(vec) > TOL).sum()),
        "max": float(abs(vec).max()),
        "low8": [round(float(v), 4) for v in vec[:8]],
        "high4": [round(float(v), 4) for v in vec[60:]],
    }


def main():
    out = {"lanes": LANES, "tol": TOL, "results": {}, "emitted": {}}

    ones = torch.ones(LANES, dtype=torch.float32).npu()
    ramp = torch.arange(LANES, dtype=torch.float32).npu()
    # lane 0 = 0, lanes 1..63 = 1.0 -> true sum 63. Lane 0 starts at zero, so if a
    # second call CHANGES lane 0 the chain is not idempotent -- this isolates B from C
    # without relying on magnitudes.
    tail = torch.cat([torch.zeros(1), torch.ones(LANES - 1)]).to(torch.float32).npu()

    inputs = {"ones": ones, "ramp": ramp, "tail": tail}
    ref = {
        "ones": {"sum": float(LANES)},
        "ramp": {"sum": float(sum(range(LANES)))},
        "tail": {"sum": float(LANES - 1)},
    }
    out["reference"] = ref
    print(f"reference sums: ones={ref['ones']['sum']:.0f} "
          f"ramp={ref['ramp']['sum']:.0f} tail={ref['tail']['sum']:.0f}\n")

    for iname, x in inputs.items():
        out["results"][iname] = {}
        for op in OPS:
            out["results"][iname][op] = {}
            for n in CALLS:
                try:
                    vec, _ = run(op, n, x)
                    s = summarize(vec)
                    out["results"][iname][op][str(n)] = s
                    print(f"{iname:5s} {op:7s} calls={n}: lane0={s['lane0']:14.4f} "
                          f"nz={s['nonzero_lanes']:3d} low8={s['low8']}", flush=True)
                except Exception as exc:
                    out["results"][iname][op][str(n)] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"{iname:5s} {op:7s} calls={n}: ERROR {type(exc).__name__}: "
                          f"{str(exc)[:160]}", flush=True)

    # ---- what the DSL actually lowers to ---------------------------------------
    try:
        _, k = run("vcadd", 1, ones)
        src = k.get_kernel_source()
        hits = [ln.strip() for ln in src.splitlines() if "vcadd" in ln.lower()]
        out["emitted"] = {"lines": hits[:10], "count_vcadd": src.lower().count("vcadd")}
        print("\n=== emitted lines containing vcadd ===")
        for ln in hits[:10]:
            print("   ", ln[:200])
    except Exception as exc:
        out["emitted"] = {"error": f"{type(exc).__name__}: {exc}"}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
