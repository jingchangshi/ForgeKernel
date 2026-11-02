"""Check the vendor interface asymmetry: does npu_rms_norm always compute rstd?

Claim under test: the vendor arm does strictly more work than the task requires
(task returns y only), so comparing y-only outputs does not flatter the candidate.
"""
import json, torch, torch_npu

B, D = 512, 4096
x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
out = torch_npu.npu_rms_norm(x, w, 1e-6)
res = {"returns": type(out).__name__, "n_outputs": len(out)}
for i, t in enumerate(out):
    res["output_%d" % i] = {"shape": list(t.shape), "dtype": str(t.dtype)}
res["rstd_matches_manual"] = bool(torch.allclose(
    out[1].float(), torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6), rtol=1e-2, atol=1e-2))
print("VENDOR_API " + json.dumps(res))