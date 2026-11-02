"""Resolve workspace roots without hardcoding a user's home directory.

Every script in this directory used to embed an absolute path, which made the
recorded measurements reproducible on exactly one machine - defeating the point of
keeping them. Roots are resolved here instead: from the environment when provided,
otherwise by walking up to the enclosing checkout and taking its siblings.

This works whether the directory sits next to the checkouts in a workspace, or inside
the ForgeKernel checkout (as docs/ascend950pr), because it does not assume a fixed depth.

Override any of these with an environment variable when the layout differs:
    FORGEKERNEL_ROOT     path to the ForgeKernel checkout
    TILELANG_ROOT        path to the TileLang checkout
    AGENTGATES_ROOT      path to the AgentGates checkout
    WORK_AGENTGATES_ROOT path to the AgentGates worktree used for adapter runs
    DSH_CREDENTIALS      path to the harness credentials file, if a script needs it
"""
import os
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PLAN = SCRIPTS.parent


def _enclosing_checkout(start):
    """The nearest ancestor that is a git checkout, or start itself if there is none."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


# When this directory lives inside the ForgeKernel checkout, the enclosing checkout
# IS ForgeKernel. When it sits beside it in a workspace, the walk finds nothing and we
# fall back to a sibling of the plan directory.
_checkout = _enclosing_checkout(SCRIPTS)
_fallback_workspace = PLAN.parent

FORGEKERNEL = Path(os.environ.get(
    "FORGEKERNEL_ROOT",
    _checkout if (_checkout / ".git").exists() else _fallback_workspace / "ForgeKernel"))
_workspace = FORGEKERNEL.parent

TILELANG = Path(os.environ.get("TILELANG_ROOT", _workspace / "tilelang"))
AGENTGATES = Path(os.environ.get("AGENTGATES_ROOT", _workspace / "AgentGates"))
WORK_AGENTGATES = Path(os.environ.get(
    "WORK_AGENTGATES_ROOT", _workspace / "forgekernel-ascend950pr-work" / "sources" / "AgentGates"))
DSH_CREDENTIALS = Path(os.environ.get("DSH_CREDENTIALS", Path.home() / ".dsh" / ".credentials.yaml"))