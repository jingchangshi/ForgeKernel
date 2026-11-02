"""Independently verify the P3 registry fix on the live FK tree.

The original defect: `mode_of()` treated every non-CUDA language as cutedsl, so adding
`tilelang-ascend` would silently read the cutedsl knowledge base and never error. This
checks the three-way resolution and that an unknown language raises instead of defaulting.
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json, sys

sys.path.insert(0, str(FORGEKERNEL) + "/src")

out = {"checks": []}


def check(name, fn):
    try:
        out["checks"].append({"check": name, "ok": True, "value": fn()})
    except Exception as exc:
        out["checks"].append({"check": name, "ok": False, "error": type(exc).__name__ + ": " + str(exc)[:200]})


def resolve(lang):
    from forgekernel.loop.lang_mode import mode_of
    return mode_of(lang)


check("cutedsl_resolves", lambda: resolve("cutedsl"))
check("cuda_resolves", lambda: resolve("cuda-sol"))
check("tilelang_ascend_resolves", lambda: resolve("tilelang-ascend"))


def unknown_must_raise():
    try:
        value = resolve("totally-not-a-language")
    except Exception as exc:
        return {"raised": type(exc).__name__, "message": str(exc)[:160]}
    return {"raised": None, "value": value, "DEFECT": "unknown language silently resolved"}


check("unknown_language_raises", unknown_must_raise)


def kb_dir():
    from forgekernel.task import _kb_dir
    return str(_kb_dir(None, "tilelang-ascend", "ascend950pr"))


check("kb_dir_tilelang_ascend", kb_dir)


def languages_include():
    from forgekernel import task
    return {"LANGUAGES": sorted(task.LANGUAGES),
            "has_tilelang_ascend": "tilelang-ascend" in task.LANGUAGES}


check("languages_registry", languages_include)


def arch_accepts():
    from forgekernel import task
    per_language = task.ARCH_RE
    out = {}
    for lang in ("tilelang-ascend", "cutedsl", "cuda"):
        anchor = per_language.get(lang, per_language["default"])
        out[lang] = {"pattern": anchor.pattern,
                     "matches_ascend950pr": bool(anchor.match("ascend950pr")),
                     "matches_sm100": bool(anchor.match("sm100"))}
    return out


check("arch_regex", arch_accepts)

print("REGISTRY_JSON " + json.dumps(out))