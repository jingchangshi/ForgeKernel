"""Real-card smoke: does the PATCHED ForgeKernel drive one TileLang compile+run on Ascend?

What this proves, and why it is not a unit test:

  * the device is selected through ``forgekernel.device`` (the module P2 added), so a job
    gets ``ASCEND_RT_VISIBLE_DEVICES`` and takes the ``npu<N>`` lock -- not a hard-coded
    ``CUDA_VISIBLE_DEVICES`` that would hand an NPU job nothing;
  * ``forgekernel.task`` accepts ``language = "tilelang-ascend"`` with
    ``arch = "ascend950pr"`` and resolves the TileLang KB root and kernel file through
    ``lang_mode`` -- the registry P3 fixed (an unknown language raises instead of
    silently reading the CuTe DSL KB);
  * ``forgekernel.loop.dsl_source`` locates the TileLang package a ``tilelang-ascend`` run
    would point its offline student at;
  * TileLang then really compiles and runs a BF16 RMSNorm on the NPU, the numbers match a
    torch reference, and ``msprof`` reports a device duration.

A smoke that only imported the modules would not prove the last point, and one that only
ran TileLang would not prove the first three. Every assertion below is on a real return
value, never on a print.

Run (the three env inits the briefing asks for, then this file):

    source /data/pri/cann-9.2.0/cann/set_env.sh
    export PATH="$CONDA_PREFIX/bin:$PATH"   # the env that has torch_npu and tilelang
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    python3 scripts/fk_ascend_smoke.py --device 1 --json out.json

``--device`` is a PHYSICAL Ascend id; the job sees device 0 in-process either way, which is
the distinction ``device.describe`` records. It runs under the same per-device lock the
harness takes, so it can share the box with other ForgeKernel runs.
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
# NOTE: no "from __future__ import annotations" here. TileLang's prim_func builder calls
# typing.get_type_hints on the kernel signature, and with PEP 563 the annotations become
# strings that are resolved in the MODULE globals -- where the closure's N / dtype / batch
# do not exist ("NameError: name 'N' is not defined"). Evaluated eagerly, they resolve in
# the enclosing scope at def time, which is what the shape parameters need.
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FK_SRC = Path(str(FORGEKERNEL) + "/src")
TILELANG_SRC = Path(str(TILELANG) + "")
CANN_SETUP = Path("/data/pri/cann-9.2.0/cann/set_env.sh")


def fail(why: str):
    print("SMOKE FAIL: " + why, file=sys.stderr)
    raise SystemExit(1)


def npu_smi() -> str:
    # Who else is on the box. Recorded, never a reason to skip: other runs on other
    # devices are normal, and the same device is handled by the lock below.
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60)
        return out.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return "(npu-smi failed: %r)" % (e,)


def check_toolchain() -> None:
    if not (TILELANG_SRC / "tilelang" / "__init__.py").is_file():
        fail("no TileLang checkout at %s" % TILELANG_SRC)
    if not CANN_SETUP.is_file():
        fail("no CANN setup at %s" % CANN_SETUP)


def check_fk_registry(device_id: str) -> dict:
    # The P2/P3 code paths, on their real return values.
    sys.path.insert(0, str(FK_SRC))
    from forgekernel import device, task, taskcard
    from forgekernel.loop import dsl_source, lang_mode

    kind = device.kind_of_arch("ascend950pr")
    assert kind == device.KIND_NPU, "kind_of_arch(ascend950pr) = %r, expected npu" % (kind,)
    assert device.visible_env(kind) == "ASCEND_RT_VISIBLE_DEVICES"
    assert device.lock_name(kind, device_id) == "npu" + device_id
    assert device.logical_name(kind) == "npu:0"

    # P3: the mode, the KB root and the kernel file of a tilelang-ascend run.
    assert lang_mode.mode_of("tilelang-ascend") == lang_mode.MODE_TILELANG
    assert lang_mode.kb_root_name("tilelang-ascend") == "hw_kb/tilelang-ascend"
    assert task.LOOP_LANGUAGE["tilelang-ascend"] == "tilelang-ascend"
    assert task.ARCH_RE["tilelang-ascend"].match("ascend950pr")
    assert taskcard.KERNEL_FILE["tilelang-ascend"] == "kernel.py"
    # The bug P3 fixed: an unknown language must refuse, not read the CuTe DSL KB.
    try:
        lang_mode.mode_of("tilelang")
    except lang_mode.UnknownLanguage:
        pass
    else:
        fail("mode_of(tilelang) did not raise")

    # dsl_source: the package a tilelang-ascend student would be pointed at.
    cfg = {"loop": {"tilelang_source_dirs": [str(TILELANG_SRC / "tilelang")]}}
    src = dsl_source.resolve(cfg, language="tilelang-ascend", submit_env={},
                             python=sys.executable, probe_env=dict(os.environ))
    assert src.dirs, "dsl_source resolved no TileLang directory"
    assert Path(src.dirs[0]).is_dir(), src.dirs

    return {
        "kind": kind,
        "visible_env": device.visible_env(kind),
        "lock": device.lock_name(kind, device_id),
        "logical": device.logical_name(kind),
        "mode": lang_mode.mode_of("tilelang-ascend"),
        "kb_root": lang_mode.kb_root_name("tilelang-ascend"),
        "dsl_source": str(src.dirs[0]),
        "describe": device.describe(kind, device_id),
    }


def rms_norm_fwd(batch: int, d: int, dtype: str):
    # The Ascend RMSNorm from tilelang/examples/ascend, BF16 in and out. The reduction
    # stays in fp32: bf16 has 8 mantissa bits, and accumulating a 4096-wide sum of squares
    # in bf16 would lose more than the tolerance this smoke checks.
    import tilelang
    import tilelang.ascend.language as T

    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)
    ub_budget = 250 * 1024  # measured UB ceiling is 256 KiB; keep margin for compiler static shared memory
    rows = 4
    while rows > 1 and (batch % (N_CORES * rows) or (2 * rows + 1) * d * 4 > ub_budget):
        rows //= 2
    reuse_input = (3 * rows + 1) * d * 4 > ub_budget
    if rows == 1:
        threads = 256 if TILE > 4096 else 128
    else:
        threads = 256 if rows == 4 else 512
    threads = min(threads, TILE)
    iterations = batch // (N_CORES * rows)
    bulk_rstd = iterations <= 128
    rstd_tiles = iterations if bulk_rstd else 1
    N = batch * d

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
        W: T.Tensor((d,), dtype),
        RSTD: T.Tensor((batch,), "float32"),
        eps: T.float32,
    ):
        with T.Kernel(N_CORES) as core_id:
            # Same dtype as the tensors: a DMA copy cannot cast, and the Ascend
            # checker refuses a bf16->fp32 T.copy outside a VF block. The widening
            # happens per element inside the VF below, which is what that block is for.
            w_ub = T.alloc_shared((d,), dtype)
            x_ub = T.alloc_shared(rows * d, dtype)
            y_ub = T.alloc_shared(rows * d, dtype)
            z_rstd_ub = T.alloc_shared((rstd_tiles, 8), "float32")
            rstd_view = T.reshape(RSTD, (iterations, N_CORES * rows))
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1,
                                        z_rstd_ub: 1 if bulk_rstd else 2})
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iterations, num_stages=2):
                row_id = (t * N_CORES + core_id) * rows
                base = row_id * d
                slot = t if bulk_rstd else 0
                T.copy(X[base : base + rows * d], x_ub)
                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((rows, TILE), "float32")
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    sum_sq = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(sum_sq)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(sum_sq[r], x_frag[r, i] * x_frag[r, i])
                    sum_result = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(sum_sq, sum_result)
                    for r in T.Parallel(rows):
                        sum_result[r] = T.rsqrt(sum_result[r] / d + eps)
                    w_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        w_frag[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            if reuse_input:
                                x_ub[r * d + i] = x_frag[r, i] * sum_result[r] * w_frag[i]
                            else:
                                y_ub[r * d + i] = x_frag[r, i] * sum_result[r] * w_frag[i]
                    T.copy(sum_result, z_rstd_ub[slot, :rows])
                if reuse_input:
                    T.copy(x_ub, Y[base : base + rows * d])
                else:
                    T.copy(y_ub, Y[base : base + rows * d])
                if not bulk_rstd:
                    T.copy(z_rstd_ub[0, :rows], RSTD[row_id : row_id + rows])
            if bulk_rstd:
                T.copy(z_rstd_ub[:, :rows],
                       rstd_view[:, core_id * rows : (core_id + 1) * rows])

    return main


def run_kernel(batch: int, d: int, eps: float) -> dict:
    import torch
    import torch_npu  # noqa: F401 -- registers the npu backend
    import tilelang

    dtype = "bfloat16"
    kernel = tilelang.compile(rms_norm_fwd(batch, d, dtype), target="ascend",
                              out_idx=[1, 3])

    torch.manual_seed(0)
    x = torch.randn(batch, d, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(d, device="npu", dtype=torch.bfloat16)

    y, rstd = kernel(x.flatten(), weight, eps)
    y = y.reshape(batch, d)

    # fp32 reference: same maths, more precision than the kernel, so a match means the
    # kernel is right and not that both lost the same bits.
    xf = x.float()
    ref_rstd = torch.rsqrt(xf.pow(2).mean(-1) + eps)
    ref_y = (xf * ref_rstd.unsqueeze(-1) * weight.float().unsqueeze(0)).to(torch.bfloat16)

    y_err = (y.float() - ref_y.float()).abs().max().item()
    rstd_err = (rstd.float() - ref_rstd).abs().max().item()
    # bf16 has ~3 decimal digits; 2e-2 absolute on a unit-scale tensor is a real check.
    assert y_err < 2e-2, "y max abs error %r too large" % (y_err,)
    assert rstd_err < 2e-2, "rstd max abs error %r too large" % (rstd_err,)
    assert not torch.isnan(y.float()).any(), "NaN in output"

    # NEGATIVE CONTROL. The errors above can be exactly 0.0 for a correct kernel (both
    # sides round the same fp32 result to bf16), which means they cannot distinguish a
    # real normalisation from a kernel that merely COPYIED its input through: a copy of x
    # is self-consistent, NaN-free and would pass a looser tolerance. So require that the
    # output actually moved away from the input, and that rstd is not the all-ones that a
    # skipped reduction would leave.
    moved = (y.float() - xf).abs().max().item()
    assert moved > 1e-2, "output is indistinguishable from the input (moved=%r)" % (moved,)
    assert float(rstd.float().min()) < 0.99 or float(rstd.float().max()) > 1.01, \
        "rstd looks like a skipped reduction"

    return {"y_max_abs_err": y_err, "rstd_max_abs_err": rstd_err,
            "y_moved_from_input": moved,
            "rstd_range": [float(rstd.float().min()), float(rstd.float().max())],
            "y_sample": float(y.float()[0, 0]), "ref_sample": float(ref_y.float()[0, 0])}


def timed_with_msprof(device_id: str, batch: int, d: int, eps: float) -> dict:
    # Device time from msprof: a subprocess so the profiler owns its own process, and the
    # only number reported is one msprof itself wrote.
    import re
    here = Path(__file__).resolve().parent
    out_dir = (here / ".." / "artifacts" / "smoke-msprof").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inner = [sys.executable, str(Path(__file__).resolve()), "--inner-timed",
             "--device", device_id, "--batch", str(batch), "--d", str(d),
             "--eps", repr(eps)]
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = device_id
    # msprof takes the application as a POSITIONAL after its own options ("[Note] This
    # option will be discarded in later versions. you can try to use: msprof [msprof
    # arguments] <app> [app arguments]"); --application as a single joined string is
    # rejected with "expected one argument".
    try:
        proc = subprocess.run(["msprof", "--output=" + str(out_dir), "--task-time=on",
                               *inner], capture_output=True, text=True,
                              timeout=110, env=env, cwd=str(here))
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"msprof_ok": False, "why": repr(e)}
    text = (proc.stdout or "") + (proc.stderr or "")
    (out_dir / "msprof-console.log").write_text(text, encoding="utf-8")
    if proc.returncode != 0:
        return {"msprof_ok": False, "why": "msprof rc=%d" % proc.returncode,
                "tail": text[-600:]}
    # The duration is NOT in the console: msprof writes it to op_summary_*.csv, one row
    # per device task, column "Task Duration(us)". Reading the console for a number only
    # finds timestamps, so the CSV is the evidence and this is where it is read.
    runs = sorted(out_dir.glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
    if not runs:
        return {"msprof_ok": False, "why": "msprof wrote no op_summary csv",
                "out_dir": str(out_dir)}
    import csv as _csv
    rows = []
    with runs[-1].open(encoding="utf-8", errors="replace") as fh:
        for row in _csv.DictReader(fh):
            name = (row.get("Op Name") or "").strip()
            raw = (row.get("Task Duration(us)") or "").strip()
            try:
                us = float(raw)
            except ValueError:
                continue
            rows.append({"op": name[:60], "us": us,
                         "type": (row.get("OP Type") or "").strip(),
                         "aicore_us": (row.get("aicore_time(us)") or "").strip()})
    # The total is the sum over device tasks: that is the kernel's device time.
    total = round(sum(r["us"] for r in rows), 3)
    assert rows, "op_summary csv had no task rows"
    return {"msprof_ok": True, "out_dir": str(out_dir), "csv": str(runs[-1]),
            "device_us_total": total, "task_count": len(rows),
            "longest": sorted(rows, key=lambda r: -r["us"])[:5]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="real-card TileLang-on-Ascend smoke")
    p.add_argument("--device", default="1", help="physical Ascend id (job sees device 0)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--d", type=int, default=4096)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--json", default="", help="write the result object here")
    p.add_argument("--inner-timed", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    check_toolchain()
    sys.path.insert(0, str(FK_SRC))
    from forgekernel import device

    # The physical id is the caller's; the in-process id is always 0. Set exactly the
    # variable device.py says this family uses -- the whole point of P2.
    os.environ[device.visible_env(device.KIND_NPU)] = args.device
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    started = time.time()
    result = {"npu_smi_before": npu_smi(), "device_id": args.device,
              "batch": args.batch, "d": args.d, "eps": args.eps}
    result["fk"] = check_fk_registry(args.device)

    # Serialise on the same per-device lock the harness uses, so this smoke and a real
    # run cannot be on one device at the same time.
    from forgekernel.submit import direct
    with direct.gpu_lock(args.device, kind=device.KIND_NPU):
        result["numeric"] = run_kernel(args.batch, args.d, args.eps)

    if not args.inner_timed:
        result["msprof"] = timed_with_msprof(args.device, args.batch, args.d, args.eps)
    result["wall_s"] = round(time.time() - started, 2)
    result["npu_smi_after"] = npu_smi()

    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("npu_smi_before", "npu_smi_after")}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    err = result["numeric"]["y_max_abs_err"]
    ms = result.get("msprof") or {}
    ours = [t for t in (ms.get("longest") or []) if "main_kernel" in (t.get("op") or "")]
    ok = err < 2e-2 and ms.get("msprof_ok") and bool(ours)
    # The kernel itself, not the harness's randn/assert_close aclnn ops around it.
    dev_us = ours[0]["us"] if ours else float("nan")
    print("SMOKE %s: bf16 rmsnorm on npu:%s | y_err=%.3e moved=%.3f "
          "| main_kernel=%.3f us | %d device tasks, %.3f us total"
          % ("PASS" if ok else "FAIL", args.device, err,
             result["numeric"]["y_moved_from_input"], dev_us,
             ms.get("task_count", 0), ms.get("device_us_total", float("nan"))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
