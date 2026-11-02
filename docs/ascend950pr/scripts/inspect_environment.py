#!/usr/bin/env python3
"""Read-only dependency snapshot. Does not compile kernels or initialize an NPU.
Run after sourcing the reviewed CANN setup with the configured Python.
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOTS = {
    "ForgeKernel": Path(str(FORGEKERNEL) + ""),
    "AgentGates": Path(str(AGENTGATES) + ""),
    "tilelang": Path(str(TILELANG) + ""),
}

def command(argv, cwd=None):
    p = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=20)
    return {"exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

report = {"python": sys.executable, "python_version": sys.version,
          "device_execution_performed": False, "repositories": {}, "imports": {}, "libraries": {}}
for name, root in ROOTS.items():
    report["repositories"][name] = {"path": str(root),
        "revision": command(["git", "rev-parse", "HEAD"], root),
        "status": command(["git", "status", "--porcelain"], root)}
for name in ("torch", "torch_npu", "tilelang", "tilelang.ascend.language"):
    try:
        module = importlib.import_module(name)
        report["imports"][name] = {"ok": True, "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None)}
    except Exception as exc:
        report["imports"][name] = {"ok": False, "error": str(exc)}
for name in ("bishengir-compile", "ccec", "npu-smi", "claude"):
    report.setdefault("executables", {})[name] = shutil.which(name)
for path in sorted((ROOTS["tilelang"] / "build" / "lib").glob("*.so")):
    real = path.resolve()
    report["libraries"][str(path)] = {"resolved": str(real), "sha256": sha256(real)}
report["ascend_home_path"] = os.environ.get("ASCEND_HOME_PATH")
print(json.dumps(report, ensure_ascii=False, indent=2))
sys.exit(0 if all(value["ok"] for value in report["imports"].values()) else 1)
