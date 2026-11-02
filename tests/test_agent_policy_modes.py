"""The agent-policy modes: the OS sandbox (strict / auto / off) and the two policies
that used to be welded to it (network, GPU-machine access).

Why this file exists at all: ``[loop] agent_sandbox`` has to reach the EARLIEST policy a
run builds. The judge is written before the card, so ``launch.write_judge`` runs with no
loop.toml on disk; the /goal session builds its own; the main loop builds a third. A
setting threaded into only one of them is the pitfall this suite is here to catch --
so each test below names the construction site it covers.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from forgekernel.loop import agent_policy as ap
from forgekernel.loop import cli as loop_cli


def _policy(tmp_path, cfg=None, environ=None):
    ref = tmp_path / "refs" / "RefLib"
    ref.mkdir(parents=True, exist_ok=True)
    base = {"loop": {"ref_source_paths": [str(ref)], "ref_source_urls": []}}
    for k, v in (cfg or {}).items():
        base["loop"][k] = v
    return ap.from_config(base, oracle_mode="blackbox", repo_root=tmp_path / "run",
                          teacher_trees=[], environ={} if environ is None else environ)


WS = Path("/tmp/run/operators/op-ws")


# -- the three modes, as rendered ---------------------------------------------------------

def test_strict_is_the_default_and_is_what_every_older_run_got(tmp_path):
    pol = _policy(tmp_path)
    assert pol.sandbox_mode == ap.DEFAULT_SANDBOX == ap.SANDBOX_STRICT
    s = pol.settings("student", WS)["sandbox"]
    assert s["enabled"] is True
    assert s["failIfUnavailable"] is True
    assert s["allowUnsandboxedCommands"] is False
    # the filesystem lists are there in every mode: they are the policy's statement of
    # what is denied, not a property of the OS layer
    assert s["filesystem"]["denyRead"] and s["filesystem"]["denyWrite"]


def test_auto_keeps_the_sandbox_on_but_degrades_instead_of_stopping(tmp_path):
    pol = _policy(tmp_path, {"agent_sandbox": "auto"})
    s = pol.settings("student", WS)["sandbox"]
    assert s["enabled"] is True
    # "on, but a sandbox that cannot start warns instead of stopping the turn" is only
    # coherent with an unsandboxed fallback allowed.
    assert s["failIfUnavailable"] is False
    assert s["allowUnsandboxedCommands"] is True


def test_off_turns_the_os_sandbox_off_and_says_so(tmp_path):
    pol = _policy(tmp_path, {"agent_sandbox": "off"})
    s = pol.settings("student", WS)["sandbox"]
    assert s["enabled"] is False
    # flags about a sandbox that is not expected are noise, and would read as "on"
    assert "failIfUnavailable" not in s and "allowUnsandboxedCommands" not in s
    # everything that is NOT the OS layer survives
    assert set(s["filesystem"]) == {"denyRead", "allowWrite", "denyWrite"}
    assert s["filesystem"]["denyRead"]


@pytest.mark.parametrize("mode", ["strict", "auto", "off"])
def test_every_mode_keeps_the_deny_rules_the_hook_and_the_filesystem_lists(tmp_path, mode):
    """off removes ONE thing (the OS sandbox). If it also quietly dropped the deny rules
    or the hook, "off" would be "unprotected", which is not what it means."""
    pol = _policy(tmp_path, {"agent_sandbox": mode})
    s = pol.settings("student", WS)
    deny = s["permissions"]["deny"]
    assert "WebFetch" in deny and "WebSearch" in deny and "WebFetch(domain:*)" in deny
    assert any(r.startswith("Read(/") for r in deny)
    assert s["hooks"]["PreToolUse"][0]["hooks"][0]["command"].count("agent_policy_hook.py")
    assert s["sandbox"]["filesystem"]["denyRead"]
    assert all(r.startswith(("Read(//", "Edit(//", "Bash(", "Web")) for r in deny)


def test_the_off_notice_does_not_promise_a_sandbox_that_is_not_running(tmp_path):
    off = _policy(tmp_path, {"agent_sandbox": "off"})
    notice = off.notice("student")
    assert "agent_sandbox = off" in notice
    assert "not OS isolation" in notice
    # the sentence the honest-wording requirement is about
    assert "the sandbox refuses it" not in notice.lower()
    assert "sandbox refuses the connection" not in notice.lower()
    # under strict the same sentence is true, and is still said
    strict = _policy(tmp_path)
    assert "sandbox" in strict.notice("student").lower()


def test_the_policy_document_carries_the_sandbox_mode_for_the_hook(tmp_path):
    """The hook cannot see loop.toml; the document is how it learns what is running."""
    for mode in ap.SANDBOX_MODES:
        pol = _policy(tmp_path, {"agent_sandbox": mode})
        assert pol.policy_document("student")["sandbox"] == mode
    # and a policy written before the key existed has no key, which the hook reads as
    # "not off" (those runs had the sandbox)
    assert "sandbox" not in {"role": "student"}


# -- network and GPU-machine access are separate ------------------------------------------

def test_the_network_and_the_gpu_machine_are_independent_switches(tmp_path):
    """The old behaviour welded them: student_online still forced no_cluster. A run that
    needs the web for one session must be able to have it and still not submit jobs."""
    online = _policy(tmp_path, {"agent_network": "online"})
    assert online.offline("student") is False and online.no_cluster("student") is True
    cluster = _policy(tmp_path, {"agent_cluster": "allow"})
    assert cluster.offline("student") is True and cluster.no_cluster("student") is False
    both = _policy(tmp_path, {"agent_network": "online", "agent_cluster": "allow"})
    assert both.offline("student") is False and both.no_cluster("student") is False
    # ... and the deny lists follow each switch independently. "ssh" is both a network
    # tool and a cluster tool, so the pair to compare is (network on, cluster on) against
    # (network on, cluster off): only the cluster switch decides there.
    on_off = dataclasses.replace(online, network_mode=ap.NETWORK_ONLINE)
    on_on = dataclasses.replace(online, network_mode=ap.NETWORK_ONLINE,
                               cluster_mode=ap.CLUSTER_ALLOW)
    net_deny = on_off.settings("student", WS)["permissions"]["deny"]
    assert "WebFetch" not in net_deny                     # network on
    assert "Bash(ssh:*)" in net_deny                      # cluster still off
    assert "Bash(python -m forgekernel.submit:*)" in net_deny
    cl_deny = on_on.settings("student", WS)["permissions"]["deny"]
    assert "WebFetch" not in cl_deny                      # network on
    assert "Bash(ssh:*)" not in cl_deny                   # cluster allowed
    assert "Bash(python -m forgekernel.submit:*)" not in cl_deny
    # offline, ssh is denied as a network tool whatever the cluster switch says: the
    # network rule is the stronger one and it must not weaken
    assert "Bash(ssh:*)" in cluster.settings("student", WS)["permissions"]["deny"]
    assert "WebFetch" in cluster.settings("student", WS)["permissions"]["deny"]


def test_a_role_default_is_what_applies_when_nothing_is_declared(tmp_path):
    pol = _policy(tmp_path)
    assert pol.offline("student") and pol.no_cluster("student")
    assert pol.offline("teacher") is False and pol.no_cluster("teacher") is False
    # the ref_from_goal session turns the student's network on without touching the
    # cluster rule
    online = dataclasses.replace(pol, student_online=True)
    assert online.offline("student") is False and online.no_cluster("student") is True


def test_an_explicit_policy_beats_the_ref_from_goal_session(tmp_path):
    """student_online is a per-session convenience; agent_network = offline is a decision
    about the run, and it must win (otherwise a blackbox run could not keep its session
    offline at all)."""
    pol = dataclasses.replace(_policy(tmp_path, {"agent_network": "offline"}),
                              student_online=True)
    assert pol.offline("student") is True


# -- resolution: loop.toml -> environment -> default ---------------------------------------

def test_the_config_layer_wins_then_the_machine_then_the_default():
    assert ap.resolve_modes({}) == (ap.SANDBOX_STRICT, "", "")
    assert ap.resolve_modes({"loop": {"agent_sandbox": "off"}}) == (ap.SANDBOX_OFF, "", "")
    env = {"FK_AGENT_SANDBOX": "auto", "FK_AGENT_NETWORK": "online",
           "FK_AGENT_CLUSTER": "allow"}
    assert ap.resolve_modes({}, env) == (ap.SANDBOX_AUTO, ap.NETWORK_ONLINE, ap.CLUSTER_ALLOW)
    # loop.toml beats the machine, key by key
    cfg = {"loop": {"agent_sandbox": "strict", "agent_network": "offline"}}
    assert ap.resolve_modes(cfg, env) == (ap.SANDBOX_STRICT, ap.NETWORK_OFFLINE,
                                          ap.CLUSTER_ALLOW)
    # an empty environment value is "not set", not "empty choice"
    assert ap.resolve_modes({}, {"FK_AGENT_SANDBOX": "  "})[0] == ap.SANDBOX_STRICT


def test_a_typo_in_a_mode_is_an_error_not_a_silent_default():
    for cfg, env, needle in (({"loop": {"agent_sandbox": "disabled"}}, None, "agent_sandbox"),
                             ({"loop": {"agent_network": "yes"}}, None, "agent_network"),
                             ({}, {"FK_AGENT_CLUSTER": "true"}, "agent_cluster")):
        with pytest.raises(ValueError, match=needle):
            ap.resolve_modes(cfg, env or {})
    # and it is reported with every other declaration problem at launch (exit 15)
    problems = ap.declaration_problems({"loop": {"agent_sandbox": "disabled"}}, "source")
    assert any("agent_sandbox" in p for p in problems)


def test_the_modes_survive_a_real_loop_toml_round_trip(tmp_path):
    """loop.toml is read straight into ``cfg["loop"]`` (forgekernel.loop.cli's own
    loader), so the modes have to survive a real file -- not just a dict literal."""
    import tomllib
    from forgekernel import taskcard

    p = tmp_path / "loop.toml"
    p.write_text("\n".join(["[loop]", 'language = "cutedsl"', 'kb_arch = "sm100"',
                             'agent_sandbox = "off"', 'agent_network = "online"',
                             'agent_cluster = "deny"']), encoding="utf-8")
    cfg = tomllib.loads(p.read_text())
    pol = ap.from_config(cfg, oracle_mode="source", repo_root=tmp_path, teacher_trees=[],
                         environ={})
    assert (pol.sandbox_mode, pol.network_mode, pol.cluster_mode) == (
        ap.SANDBOX_OFF, ap.NETWORK_ONLINE, ap.CLUSTER_DENY)
    # the file the harness generates uses the same spellings, so the two agree by test
    generated = tomllib.loads("\n".join(["[loop]", *taskcard._policy_mode_lines(
        {"FK_AGENT_SANDBOX": "off", "FK_AGENT_NETWORK": "online",
         "FK_AGENT_CLUSTER": "deny"})]))
    assert ap.resolve_modes(generated, {}) == (ap.SANDBOX_OFF, ap.NETWORK_ONLINE,
                                               ap.CLUSTER_DENY)


def test_env_overrides_reach_from_config(tmp_path):
    """from_config is the ONE door every construction site uses; the environment layer
    has to be visible through it, not only through resolve_modes."""
    pol = ap.from_config({"loop": {}}, oracle_mode="source", repo_root=tmp_path,
                         teacher_trees=[], environ={"FK_AGENT_SANDBOX": "off"})
    assert pol.sandbox_mode == ap.SANDBOX_OFF
    assert pol.settings("student", WS)["sandbox"]["enabled"] is False


# -- propagation to every construction site ------------------------------------------------

def test_the_judge_is_written_under_the_same_policy_modes(tmp_path, monkeypatch):
    """The EARLIEST policy: launch.write_judge builds it before the card (and therefore
    loop.toml) exists, so it can only see the machine layer. If it ignored it, every run
    would open its first teacher turn under a strict sandbox this machine cannot start."""
    from forgekernel import launch

    seen = {}

    class _Backend:
        def __init__(self, *a, **kw):
            seen["policy"] = kw.get("agent_policy")
            self.agent_cmd = None
            self.teacher_agent_cmd = kw.get("teacher_agent_cmd")

        def run_agent(self, prompt, role=""):
            return ""

    monkeypatch.setattr(loop_cli, "RealBackend", _Backend)
    monkeypatch.setattr(launch.judge_mod, "check_problems", lambda *a, **k: [])
    monkeypatch.setenv("FK_AGENT_SANDBOX", "off")
    monkeypatch.setenv("FK_AGENT_NETWORK", "online")
    monkeypatch.setenv("FK_AGENT_CLUSTER", "allow")
    task = _task_for(tmp_path)
    launch.write_judge(_Run(tmp_path), task, tmp_path / "check", [])
    pol = seen["policy"]
    assert pol.sandbox_mode == ap.SANDBOX_OFF
    assert pol.network_mode == ap.NETWORK_ONLINE
    assert pol.cluster_mode == ap.CLUSTER_ALLOW
    assert pol.settings("teacher", WS)["sandbox"]["enabled"] is False


def test_the_generated_loop_toml_carries_the_machine_policy_modes(tmp_path, monkeypatch):
    """A run started with FK_AGENT_SANDBOX=off must carry that in the loop.toml it writes:
    the file is regenerated on every launch, so a hand edit is lost, and a resume (or the
    /goal session) must read the same answer from the run's own tree."""
    from forgekernel import taskcard
    monkeypatch.setenv("FK_AGENT_SANDBOX", "off")
    monkeypatch.setenv("FK_AGENT_NETWORK", "online")
    lines = taskcard._policy_mode_lines({"FK_AGENT_SANDBOX": "off",
                                         "FK_AGENT_NETWORK": "online"})
    text = "\n".join(lines)
    assert 'agent_sandbox = "off"' in text and 'agent_network = "online"' in text
    assert "FK_AGENT_SANDBOX" in text   # says where it came from
    # nothing set: nothing written, so the loop.toml stays exactly what it was
    assert taskcard._policy_mode_lines({}) == []


def test_a_resume_reads_the_modes_from_the_runs_own_loop_toml(tmp_path):
    # The resume path is a different process, with a different environment (a later
    # shell, maybe no local.env exported yet). It must not fall back to the default
    # while the run was configured otherwise, so the answer travels in the run's
    # loop.toml, which every later stage reads back.
    import tomllib

    from forgekernel import taskcard
    lines = ["[loop]", *taskcard._policy_mode_lines({
        "FK_AGENT_SANDBOX": "off", "FK_AGENT_CLUSTER": "allow"})]
    written = tomllib.loads(chr(10).join(lines))
    # the resume process has nothing in its environment at all
    assert ap.resolve_modes(written, {}) == (ap.SANDBOX_OFF, "", ap.CLUSTER_ALLOW)
    # and a loop.toml that predates the keys still resolves to the old behaviour
    assert ap.resolve_modes({"loop": {"language": "cutedsl"}},
                            {}) == (ap.SANDBOX_STRICT, "", "")


def test_the_generated_loop_toml_is_valid_for_a_task(tmp_path):
    """The generated file must still parse as a loop.toml the task model accepts."""
    from forgekernel import taskcard
    import tomllib
    text = "\n".join(["[loop]", 'language = "cutedsl"', 'kb_arch = "sm100"',
                       *taskcard._policy_mode_lines({"FK_AGENT_CLUSTER": "deny"})])
    cfg = tomllib.loads(text)
    assert cfg["loop"]["agent_cluster"] == "deny"


# -- the hook ------------------------------------------------------------------------------

def _hook():
    import importlib.util
    from pathlib import Path as P
    repo = P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("agent_policy_hook",
                                                  repo / "scripts" / "agent_policy_hook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_hook_does_not_claim_a_sandbox_under_off():
    hook = _hook()
    event = {"tool_name": "Bash", "tool_input": {"command": "curl https://example.com"}}
    on = hook.decide({"role": "student", "offline": True, "sandbox": "strict"}, event)
    assert on and "sandbox refuses the connection as well" in on
    off = hook.decide({"role": "student", "offline": True, "sandbox": "off"}, event)
    assert off and "BLOCKED" in off
    assert "sandbox refuses" not in off
    assert "not OS isolation" in off
    # a document from before the key existed means "the sandbox was on"
    legacy = hook.decide({"role": "student", "offline": True}, event)
    assert "sandbox refuses the connection as well" in legacy


def test_the_hook_off_message_still_refuses_and_still_names_the_rule():
    hook = _hook()
    for event in ({"tool_name": "Bash", "tool_input": {"command": "pip install foo"}},
                  {"tool_name": "WebFetch", "tool_input": {"url": "https://example.com"}}):
        msg = hook.decide({"role": "student", "offline": True, "sandbox": "off"}, event)
        assert msg and "BLOCKED by the ForgeKernel agent policy" in msg
        # nothing that reads as a barrier which is not there
        assert "sandbox refuses" not in msg and "sandbox stops" not in msg


# -- fixtures ------------------------------------------------------------------------------

def _task_for(tmp_path):
    import json as _json

    from forgekernel.task import load_task
    root = tmp_path / "repo"
    (root / "hw_kb" / "cutedsl" / "sm100").mkdir(parents=True, exist_ok=True)
    d = tmp_path / "tasks" / "rmsnorm"
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n')
    (d / "description.md").write_text("out = x\n")
    (d / "shapes.json").write_text(_json.dumps([{"name": "one", "axes": {"m": 1}}]))
    return load_task(d, kb_root=root)


class _Run:
    def __init__(self, tmp_path):
        self.dir = tmp_path
        self.ws = tmp_path / "ws"
        self.log = tmp_path / "run.log"

    def record(self):
        return {}

    def update(self, **kw):
        return None


def test_the_machine_carries_the_policy_modes_as_typed_fields(tmp_path):
    # The machine layer used to exist only as an environment side effect of
    # launch._entered. A caller that loads local.env directly (an adapter, a harness,
    # a test) must see the same answer without that context manager.
    from forgekernel.localenv import load_machine

    m = load_machine(tmp_path, environ={})
    assert (m.agent_sandbox, m.agent_network, m.agent_cluster) == ("", "", "")
    assert m.policy_modes() == {}          # unset is absent, not an empty choice
    m2 = load_machine(tmp_path, environ={"FK_AGENT_SANDBOX": "OFF",
                                         "FK_AGENT_NETWORK": "online",
                                         "FK_AGENT_CLUSTER": "allow"})
    assert (m2.agent_sandbox, m2.agent_network, m2.agent_cluster) == ("off", "online", "allow")
    assert m2.policy_modes() == {"agent_sandbox": "off", "agent_network": "online",
                                "agent_cluster": "allow"}
    # local.env is the file layer; the process environment still wins per key
    (tmp_path / "local.env").write_text("FK_AGENT_SANDBOX=auto" + chr(10), encoding="utf-8")
    assert load_machine(tmp_path, environ={}).agent_sandbox == "auto"
    assert load_machine(tmp_path, environ={"FK_AGENT_SANDBOX": "off"}).agent_sandbox == "off"


def test_a_bad_mode_in_local_env_lists_every_problem_at_once(tmp_path):
    from forgekernel.localenv import MachineError, load_machine
    with pytest.raises(MachineError) as e:
        load_machine(tmp_path, environ={"FK_AGENT_SANDBOX": "disabled",
                                        "FK_DELIVERY": "carrier-pigeon"})
    text = " ".join(e.value.problems)
    assert "FK_AGENT_SANDBOX" in text and "FK_DELIVERY" in text


def test_the_machine_layer_folds_into_a_config_without_overriding_the_card(tmp_path):
    from forgekernel.localenv import load_machine

    m = load_machine(tmp_path, environ={"FK_AGENT_SANDBOX": "off",
                                        "FK_AGENT_NETWORK": "online"})
    merged = ap.with_machine({"loop": {"language": "cutedsl"}}, m)
    assert merged["loop"]["language"] == "cutedsl"
    assert ap.resolve_modes(merged, {}) == (ap.SANDBOX_OFF, ap.NETWORK_ONLINE, "")
    # the card wins where it declares something: loop.toml outranks the machine
    card = {"loop": {"agent_sandbox": "strict"}}
    assert ap.resolve_modes(ap.with_machine(card, m), {})[0] == ap.SANDBOX_STRICT
    # nothing configured on the machine: the cfg is handed back untouched
    plain = {"loop": {"language": "cutedsl"}}
    assert ap.with_machine(plain, load_machine(tmp_path, environ={})) is plain


def test_the_judge_honours_a_machine_passed_in_without_entered(tmp_path, monkeypatch):
    # write_judge is the earliest policy. Here there is NO environment at all -- only a
    # Machine -- which is exactly the caller that used to lose the modes.
    from forgekernel import launch
    from forgekernel.localenv import load_machine

    seen = {}

    class _Backend:
        def __init__(self, *a, **kw):
            seen["policy"] = kw.get("agent_policy")
            self.agent_cmd = None
            self.teacher_agent_cmd = kw.get("teacher_agent_cmd")

        def run_agent(self, prompt, role=""):
            return ""

    monkeypatch.setattr(loop_cli, "RealBackend", _Backend)
    monkeypatch.setattr(launch.judge_mod, "check_problems", lambda *a, **k: [])
    for k in ("FK_AGENT_SANDBOX", "FK_AGENT_NETWORK", "FK_AGENT_CLUSTER"):
        monkeypatch.delenv(k, raising=False)
    machine = load_machine(tmp_path, environ={"FK_AGENT_SANDBOX": "off",
                                              "FK_AGENT_CLUSTER": "allow"})
    launch.write_judge(_Run(tmp_path), _task_for(tmp_path), tmp_path / "check", [],
                       machine=machine)
    pol = seen["policy"]
    assert pol.sandbox_mode == ap.SANDBOX_OFF and pol.cluster_mode == ap.CLUSTER_ALLOW
    assert pol.settings("teacher", WS)["sandbox"]["enabled"] is False
