"""Show that static screening alone cannot separate real kernels from delegated ones.

Three source files are screened with a naive regex that looks for fusion-op names:
  1. a genuine TileLang kernel (no false positive should occur)
  2. a direct delegation (`torch_npu.npu_rms_norm(...)`)
  3. an aliased/indirect delegation (getattr + string join) that the same regex misses

This is the evidence for why patch P9 needs reachable-code inspection plus runtime
evidence, not a name-based deny list.
"""
import json, re

GENUINE = """
import torch, tilelang
import tilelang.ascend.language as T
def run(x, w, eps=1e-6):
    return compiled(x.flatten(), w, eps)
"""

DIRECT = """
import torch_npu
def run(x, w, eps=1e-6):
    return torch_npu.npu_rms_norm(x, w, eps)[0]
"""

ALIASED = """
import importlib
_m = importlib.import_module("torch" + "_npu")
_f = getattr(_m, "npu_rms" + "_norm")
def run(x, w, eps=1e-6):
    return _f(x, w, eps)[0]
"""

NAIVE = re.compile(r"\bnpu_rms_norm\b|\bnpu_layer_norm\b|\bnpu_add_rms_norm\b")

def screen(name, src):
    hits = NAIVE.findall(src)
    return {"case": name, "naive_regex_hits": hits, "naive_regex_flags": bool(hits)}

cases = [("genuine_tilelang", GENUINE), ("direct_delegation", DIRECT), ("aliased_delegation", ALIASED)]
rows = [screen(n, s) for n, s in cases]

# What reachable-call analysis would need to see instead: the resolved name is built at runtime.
extra = {"aliased_resolved_name": "torch_npu" + "." + "npu_rms" + "_norm",
         "why": "the fusion-op name never appears literally in aliased_delegation, so a name regex cannot flag it",
         "required_instead": ["AST/dataflow over the whole reachable module set",
                              "runtime trace of the tensors the archived kernel actually launches",
                              "artifact evidence: a compiled .aibin exists for the candidate"]}
print("SCREEN_JSON " + json.dumps({"screening": rows, **extra}))