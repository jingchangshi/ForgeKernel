"""End-to-end proof that the patched ForgeKernel drives an Ascend job.

Builds a throwaway operator tree with the SAME contract a card uses (a git repo holding
gpu_entry.sh, submit.toml, eval/), then runs it through forgekernel.submit.direct with the
patched device plumbing. A real BF16 RMSNorm runs on the NPU inside the job and its device
time is measured with msprof - so this exercises the whole path a ForgeKernel student run
would take, not just an import.

Deliberately does NOT need the tilelang-ascend KB tree (that is P7): the job script imports
tilelang directly, which is the part the device plumbing is responsible for delivering.
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json, os, subprocess, sys, tempfile
from pathlib import Path

FK = Path(str(FORGEKERNEL) + "")
sys.path.insert(0, str(FK / "src"))

from forgekernel import device
from forgekernel.submit.config import SubmitConfig
from forgekernel.submit.direct import DirectRunner

out = {"steps": []}


def step(name, **kw):
    out["steps"].append({"step": name, **kw})


# The device decision must come from the task arch, exactly as a real run does it.
kind = device.kind_of_arch("ascend950pr")
visible = device.visible_env(kind)
lock = device.lock_name(kind, "0")
step("device_resolution", arch="ascend950pr", kind=kind, visible_env=visible, lock_name=lock)
assert kind == "npu", kind
assert visible == "ASCEND_RT_VISIBLE_DEVICES", visible

KERNEL = """
import json, torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof

B, D = 1024, 4096


def build(batch, d):
    N_CORES, rows = 64, 4
    TILE = tilelang.next_power_of_2(d)
    iters = batch // (N_CORES * rows)
    N = batch * d

    @T.prim_func
    def main(X: T.Tensor((N,), "bfloat16"), Y: T.Tensor((N,), "bfloat16"),
             W: T.Tensor((d,), "bfloat16"), eps: T.float32):
        with T.Kernel(N_CORES) as core_id:
            w_ub = T.alloc_shared((d,), "bfloat16")
            x_ub = T.alloc_shared(rows * d, "bfloat16")
            y_ub = T.alloc_shared(rows * d, "bfloat16")
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iters, num_stages=2):
                base = ((t * N_CORES + core_id) * rows) * d
                T.copy(X[base:base + rows * d], x_ub)
                with T.SimtVF(threads=256):
                    xf = T.alloc_fragment((rows, TILE), "float32")
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        xf[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    wf = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        wf[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    ss = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(ss)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(ss[r], xf[r, i] * xf[r, i])
                    res = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(ss, res)
                    for r in T.Parallel(rows):
                        res[r] = T.rsqrt(res[r] / d + eps)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.cast(xf[r, i] * res[r] * wf[i], "bfloat16")
                T.copy(y_ub, Y[base:base + rows * d])
    return main


x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
kern = tilelang.compile(build(B, D), target="ascend", out_idx=[1])
y = kern(x.flatten(), w, 1e-6).reshape(B, D)
rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
ref = (x.float() * rstd * w.float().unsqueeze(0)).to(torch.bfloat16)
cache = torch.empty(16 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")
ms = bench_with_msprof(lambda: kern(x.flatten(), w, 1e-6), cache, 20)
print("KERNEL_RESULT " + json.dumps({
    "max_abs_err": float((y.float() - ref.float()).abs().max()),
    "device_ms": round(ms, 6),
    "device_visible": __import__("os").environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    "npu_count": torch.npu.device_count(),
}))
"""

tmp = Path(tempfile.mkdtemp(prefix="fk-e2e-"))
op = tmp / "operators" / "rmsnorm"
op.mkdir(parents=True)
(op / "kernel_run.py").write_text(KERNEL)
ENTRY = Path(str(PLAN) + "/scripts/e2e_gpu_entry.sh")
(op / "gpu_entry.sh").write_text(ENTRY.read_text())
(op / "gpu_entry.sh").chmod(0o755)

subprocess.run(["git", "init", "-q", "-b", "main", str(op)], check=True)
for cmd in (["git", "-C", str(op), "config", "user.email", "e2e@local"],
            ["git", "-C", str(op), "config", "user.name", "e2e"]):
    subprocess.run(cmd, check=True)
subprocess.run(["git", "-C", str(op), "add", "-A"], check=True)
subprocess.run(["git", "-C", str(op), "commit", "-q", "-m", "e2e tree"], check=True)
step("operator_tree", path=str(op), branch="main")

cfg = SubmitConfig(op_dir=str(op), base_dir=str(tmp / "operators"),
                   exec_workdir=str(tmp / "work"), exec_gpu="0",
                   exec_kind=kind, env={"PYTHONUNBUFFERED": "1"}, timeout=300,
                   delivery="local", branch="main", entry_script="gpu_entry.sh")
env = __import__("forgekernel.submit.direct", fromlist=["_job_env"])._job_env(cfg)
step("job_env", visible_var=visible, value=env.get(visible), timeout=env.get("TASK_TIMEOUT"))
assert env.get(visible) == "0", env

runner = DirectRunner(cfg)
with open(tmp / "job.log", "w") as log_file:
    status = runner.run_mode("main", "all", log_file)
log = (tmp / "work" / op.parent.name / "rmsnorm" / "kernel_out.txt")
step("job_run", status=status, log_exists=log.exists())
joblog = tmp / "job.log"
if joblog.exists():
    step("job_log_tail", tail=joblog.read_text()[-800:])
if log.exists():
    text = log.read_text()
    line = [l for l in text.splitlines() if l.startswith("KERNEL_RESULT ")]
    if line:
        step("kernel_result", **json.loads(line[0][len("KERNEL_RESULT "):]))
    else:
        step("kernel_output_tail", tail=text[-600:])

print("E2E_JSON " + json.dumps(out))
