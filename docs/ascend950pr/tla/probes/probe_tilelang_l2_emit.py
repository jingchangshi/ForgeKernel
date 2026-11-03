import importlib.util, re
P = "/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v33/kernel.py"
spec = importlib.util.spec_from_file_location("k33", P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import tilelang
k = tilelang.compile(m._build(8192, 4096), target="ascend", out_idx=[2])
src = k.get_kernel_source()
print("=== every L2 cast in the emitted v33 kernel ===")
for mt in re.finditer(r"static_cast<(asc_(?:load|store)_l2_cache_mode)>\((\d+)\)", src):
    print("  ", mt.group(1), "=", mt.group(2))
print()
print("=== count of each DMA intrinsic ===")
for pat in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align", "asc_copy_gm2cbuf"):
    print(pat, src.count(pat))
