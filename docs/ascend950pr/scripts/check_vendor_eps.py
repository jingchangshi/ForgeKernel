import json, torch, torch_npu
B, D = 512, 4096
x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
y, rstd = torch_npu.npu_rms_norm(x, w, 1e-6)
manual = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
ratio = (rstd.float() / manual).flatten()
print(json.dumps({
  "rstd_shape": list(rstd.shape), "manual_shape": list(manual.shape),
  "median_ratio": round(float(ratio.median()), 6),
  "p1_ratio": round(float(ratio.quantile(0.01)), 6),
  "p99_ratio": round(float(ratio.quantile(0.99)), 6),
  "y_vs_manual_max_abs": round(float((y.float() - (x.float() * manual * w.float()).to(torch.bfloat16).float()).abs().max()), 6),
  "note": "ratio far from 1 means epsilon handling differs, not that the op is wrong"
}))