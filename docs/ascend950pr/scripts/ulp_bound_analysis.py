"""Is 2^-7 a hole in the gate, or exactly the bf16 ulp bound? Settle it numerically.

If 2^-7 is one bf16 ulp relative to the leading bit, then ANY correctly-rounded bf16
implementation can sit exactly on it, and "zero margin" is a property of bf16 rather than
a defect in the tolerance. That distinction decides whether the gate should be loosened,
kept, or tightened - so measure it instead of arguing about it.
"""
import json, torch


def ulp_rel(v):
    """Relative size of one bf16 ulp at v, i.e. 2^-7 scaled by the binade exponent."""
    import math
    if v == 0:
        return 0.0
    e = math.floor(math.log2(abs(v)))
    return 2.0 ** (e - 7) / abs(v)


# 1) The bound: a value rounded to the nearest bf16 differs from the true value by at
#    most half an ulp; two values that are each within half an ulp differ by one ulp.
vals = [2.0 ** (e / 4) for e in range(-32, 16)]
worst = max(ulp_rel(v) for v in vals)

# 2) Empirically: round a wide spread of doubles to bf16 and take max relative shift.
torch.manual_seed(3)
t = torch.randn(2_000_000, dtype=torch.float64).abs().clamp_min(1e-6)
r = t.to(torch.bfloat16).double()
rel = ((r - t).abs() / t).max().item()

# 3) One-ulp differences: how large is the relative gap between neighbouring bf16 values?
base = torch.randn(200_000, dtype=torch.float32).abs().clamp_min(1e-3).to(torch.bfloat16)
import struct
b = base.view(torch.int16).to(torch.int32)
up = (b + 1).to(torch.int16).view(torch.bfloat16).float()
down = (b - 1).to(torch.int16).view(torch.bfloat16).float()
gaps = ((up - base.float()).abs() / base.float()).clamp_min(1e-12)

out = {
  "theoretical_one_ulp_bound": worst,
  "theoretical_two_ulp_bound": 2 * worst,
  "measured_rounding_shift_max_rel": rel,
  "measured_neighbour_gap_max_rel": float(gaps.max()),
  "measured_neighbour_gap_median_rel": float(gaps.median()),
  "gate_threshold_2^-7": 2 ** -7,
  "vendor_observed_max_rel": 0.0078125,
  "bf16_accum_observed_max_rel": [0.04316547, 0.02649007, 0.02919708],
}
out["interpretation"] = {
  "is_the_threshold_one_ulp": abs(worst - 2 ** -7) < 1e-9,
  "vendor_error_in_ulps": 0.0078125 / worst,
  "bf16_accum_errors_in_ulps": [round(v / worst, 2) for v in out["bf16_accum_observed_max_rel"]],
  "conclusion": (
    "2^-7 IS the one-ulp relative bound for bf16. A correctly rounded bf16 kernel can sit "
    "exactly on it, which is why a correct implementation shows zero margin - that is bf16, "
    "not a hole in the gate. The gate keeps its discriminating power because a bf16 "
    "accumulator lands 3.4-5.5 ulps out. Therefore the comparison must stay inclusive "
    "(<=), must NOT be tightened to <, and its power must be argued in ulps.")}
print("ULP_JSON " + json.dumps(out))