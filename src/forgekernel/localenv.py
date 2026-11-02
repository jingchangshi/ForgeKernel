"""``local.env``: the machine a run uses -- never part of a task.

A gitignored file at the repository root, one ``KEY=VALUE`` per line (``#`` comments,
optional ``export`` prefix, optional quotes). The process environment wins over the file
for any key both set, so a one-off ``FK_DELIVERY=ssh forge run ...`` needs no edit.

Keys the harness reads:

    FK_DELIVERY      local | ssh                    (default: local)
    FK_GPU           device index to run on         (default: 0)
                     A CUDA device index (CUDA_VISIBLE_DEVICES) or an Ascend device id
                     (ASCEND_RT_VISIBLE_DEVICES), depending on the task's arch
                     (forgekernel.device). Either way the job sees device 0 in-process.
    REMOTE_HOST      ssh: host alias; must log in non-interactively
    REMOTE_WORKDIR   ssh: absolute directory on the host, dedicated to forgekernel
    REMOTE_ENV_FILE  ssh, optional: sourced on the host before every command

    FK_AGENT_SANDBOX strict | auto | off           (default: strict)
                     The OS sandbox (loop/agent_policy.py). ``strict``: on, must start,
                     and no command may run unsandboxed. ``auto``: on, but a machine
                     where it sometimes cannot start falls back to an unsandboxed
                     command with a warning instead of stopping the turn. ``off``:
                     deliberately not started -- for a machine that cannot start one at
                     all (bwrap needs unprivileged user namespaces, which
                     ``apparmor_restrict_unprivileged_userns=1`` refuses). Under
                     ``off`` the deny rules, the PreToolUse hook and the filesystem
                     lists still apply, but they are POLICY CHECKS, NOT OS ISOLATION,
                     and the agent notice and every refusal message say exactly that.
    FK_AGENT_NETWORK online | offline              (default: the role rules)
    FK_AGENT_CLUSTER allow | deny                   (default: the role rules)
                     Network and GPU-machine access are separate policies, not one:
                     by default the student is offline AND may not drive the GPU
                     machine (no ssh, no submits), while the teacher has both. An
                     explicit value here overrides the rule for EVERY role in the run,
                     so set one only to reconfigure the whole machine (a bare
                     ``FK_AGENT_NETWORK=offline`` would take the teacher offline too).

These three are the MACHINE layer of the three modes. Where a card's loop.toml
[loop] declares ``agent_sandbox`` / ``agent_network`` / ``agent_cluster``, the card
wins (``agent_policy.resolve_modes``); the file here is what applies when it does not,
including to the earliest policy a run builds (the judge, written before the card's
loop.toml exists) and to every resume. They are also carried as typed fields on
``Machine`` (``agent_sandbox`` etc.), so a caller that never entered
``launch._entered`` still sees them.

Every key in the file (credentials such as ``ANTHROPIC_API_KEY`` or
``CLAUDE_CODE_OAUTH_TOKEN`` included) is exported to the agents and the GPU jobs a run
starts; values are never printed.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.loop.agent_policy import (CLUSTER_MODES, NETWORK_MODES, SANDBOX_MODES,
                                          ENV_AGENT_CLUSTER, ENV_AGENT_NETWORK,
                                          ENV_AGENT_SANDBOX)
from forgekernel.paths import repo_root

LOCAL_ENV = "local.env"
DELIVERY_LOCAL = "local"
DELIVERY_SSH = "ssh"
DELIVERIES = (DELIVERY_LOCAL, DELIVERY_SSH)

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MachineError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{LOCAL_ENV}: the machine is not usable\n{lines}")


@dataclass(frozen=True)
class Machine:
    delivery: str
    gpu: str
    remote_host: str = ""
    remote_workdir: str = ""
    remote_env_file: str = ""
    env: dict = field(default_factory=dict)   # every KEY=VALUE the file sets (merged)
    # The three agent policy modes (loop/agent_policy.py), typed so a caller that did not
    # come through launch._entered still sees them instead of relying on the environment
    # that context manager happens to export. "" = not configured on this machine, which
    # the resolver reads as "use the next layer" (the same meaning as an absent key);
    # loop.toml still outranks these, so a card can override the machine.
    agent_sandbox: str = ""
    agent_network: str = ""
    agent_cluster: str = ""

    def policy_modes(self) -> dict:
        """The machine layer of ``agent_policy.resolve_modes``, as loop.toml keys.

        An empty value is left OUT rather than written as "": the resolver and
        ``declaration_problems`` both read "" as "not set", and a caller merging this
        dict into a [loop] table must not blank what the card declared.
        """
        from forgekernel.loop import agent_policy as ap
        pairs = ((ap.KEY_AGENT_SANDBOX, self.agent_sandbox),
                 (ap.KEY_AGENT_NETWORK, self.agent_network),
                 (ap.KEY_AGENT_CLUSTER, self.agent_cluster))
        return {k: v for k, v in pairs if v}

    def describe(self) -> str:
        if self.delivery == DELIVERY_SSH:
            return f"ssh {self.remote_host}:{self.remote_workdir} (device {self.gpu})"
        return f"this machine (device {self.gpu})"


def read_env_file(path: str | Path) -> dict:
    """KEY=VALUE pairs of ``path``; {} when it does not exist. A malformed line raises
    MachineError naming it (a credential silently dropped is worse than a refusal)."""
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict = {}
    bad: list[str] = []
    for n, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _KEY_RE.match(key):
            bad.append(f"line {n} is not KEY=VALUE")
            continue
        value = value.strip()
        if value[:1] in ("'", '"'):
            try:
                parts = shlex.split(value)
            except ValueError:
                bad.append(f"line {n}: unbalanced quotes")
                continue
            value = parts[0] if parts else ""
        out[key] = value
    if bad:
        raise MachineError([f"{p}: {b}" for b in bad])
    return out


def load_machine(root: str | Path | None = None, environ=None) -> Machine:
    """The machine ``local.env`` (under ``root``, the repository by default) describes,
    with the process environment taking precedence key by key."""
    environ = os.environ if environ is None else environ
    file_env = read_env_file(Path(root if root is not None else repo_root()) / LOCAL_ENV)
    merged = dict(file_env)
    for k in list(file_env) + ["FK_DELIVERY", "FK_GPU", "REMOTE_HOST", "REMOTE_WORKDIR",
                               "REMOTE_ENV_FILE", ENV_AGENT_SANDBOX, ENV_AGENT_NETWORK,
                               ENV_AGENT_CLUSTER]:
        if k in environ:
            merged[k] = environ[k]
    problems: list[str] = []
    # The agent policy modes, validated here so a typo is a machine error at load time
    # with every other problem listed, not a crash when the first policy is built.
    modes = {}
    for key, env_key, allowed in (("agent_sandbox", "FK_AGENT_SANDBOX", SANDBOX_MODES),
                                  ("agent_network", "FK_AGENT_NETWORK", NETWORK_MODES),
                                  ("agent_cluster", "FK_AGENT_CLUSTER", CLUSTER_MODES)):
        raw = str(merged.get(env_key, "") or "").strip().lower()
        if not raw:
            continue
        if raw not in allowed:
            problems.append(f"{env_key} must be one of {list(allowed)}, got {raw!r}")
            continue
        modes[key] = raw
    delivery = (merged.get("FK_DELIVERY") or DELIVERY_LOCAL).strip()
    if delivery not in DELIVERIES:
        problems.append(f"FK_DELIVERY must be one of {list(DELIVERIES)}, got {delivery!r}")
    gpu = (merged.get("FK_GPU") or "0").strip()
    # A device index, not a CUDA index: the same key selects an NVIDIA card on a CUDA
    # run and an Ascend device on an NPU one (forgekernel.device maps it to the family's
    # VISIBLE variable). Digits only, because it becomes a lock's name and a shell word.
    if not (gpu.isdigit() and gpu):
        problems.append(f"FK_GPU must be a numeric device index (a CUDA device on a CUDA "
                        f"run, an Ascend device id on an NPU run), got {gpu!r}")
    host = (merged.get("REMOTE_HOST") or "").strip()
    workdir = (merged.get("REMOTE_WORKDIR") or "").strip()
    env_file = (merged.get("REMOTE_ENV_FILE") or "").strip()
    if delivery == DELIVERY_SSH:
        if not host:
            problems.append("FK_DELIVERY=ssh needs REMOTE_HOST (an ssh alias that logs in "
                            "without a prompt)")
        if not workdir.startswith("/") or workdir.rstrip("/") in ("", "/root", "/home", "/tmp"):
            problems.append(f"FK_DELIVERY=ssh needs REMOTE_WORKDIR, an absolute directory on "
                            f"the host used only by forgekernel (got {workdir!r}); each run "
                            f"replaces trees under it")
    if problems:
        raise MachineError(problems)
    return Machine(delivery=delivery, gpu=gpu, remote_host=host, remote_workdir=workdir,
                   remote_env_file=env_file, env=merged, **modes)
