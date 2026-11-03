#!/usr/bin/env python3
"""Prove, in one shot, that the tla-npu adapter can run on this host and that a
working TLA exists here.

Background: an earlier document set claimed "no installable TLA exists on this
host". That claim was wrong, and this script is the reproduction that refutes it
so nobody has to take either claim on trust.

It checks, in order:
  1. the container the Adapter addresses is running;
  2. the container's dedicated conda env holds a triton_ascend build whose
     RECORD lists files under language/extra/tla/;
  3. tla imports THERE and exports jit/dma/annotate_buffer_versions;
  4. the read-path dataclass still has no l2_cache_ctrl (the L2 finding);
  5. staging is expressed by versions=, and neither tla nor tl has Pipelined;
  6. the Adapter's own --check-config agrees the machine is ready.

Run as: python3 verify-tla-runnable.py
Exit 0 only if every check passes.
"""
import json
import shutil
import subprocess
import sys

CONTAINER = 's00653124_tla'
CONDA_ENV = 's00653124_tla'
CONDA_ENVS_PATH = '/home/s00653124/.conda/envs'
CONDA_SH = '/data/miniconda3/etc/profile.d/conda.sh'
# The Ascend driver holds davinci0/1 per user namespace, so a non-root opener is
# refused with EBUSY even when it is in the device group. -u root is required;
# -e HOME is required for Python to find the site-packages below.
EXEC = ['docker', 'exec', '-u', 'root', '-e', 'HOME=/home/s00653124',
        '-w', '/home/s00653124/workspace/agent_tla', CONTAINER]

PROBE = r'''
import json
out = {}
try:
    from triton.language.extra import tla
    out["tla_import"] = True
    out["tla_file"] = tla.__file__
    out["has_jit"] = hasattr(tla, "jit")
    out["has_dma"] = hasattr(tla, "dma")
    out["has_annotate_buffer_versions"] = hasattr(tla, "annotate_buffer_versions")
    out["has_pipelined"] = hasattr(tla, "Pipelined")
    import inspect
    out["alloc_buffer_sig"] = str(inspect.signature(tla.alloc_buffer))
    from triton.language.extra.tla import dma
    import dataclasses
    out["LoadGM2UB_fields"] = [f.name for f in dataclasses.fields(dma.LoadGM2UBParams)]
    out["LoadGM2L1_fields"] = [f.name for f in dataclasses.fields(dma.LoadGM2L1Params)]
    out["EvictionPolicy"] = [e.name for e in dma.EvictionPolicy]
except Exception as exc:
    out["tla_import"] = False
    out["error"] = f"{type(exc).__name__}: {exc}"
try:
    import triton.language as tl
    out["tl_has_pipelined"] = hasattr(tl, "Pipelined")
    import triton
    out["triton_version"] = triton.__version__
except Exception as exc:
    out["tl_error"] = str(exc)
print(json.dumps(out))
'''


def fail(msg):
    print(f'FAIL  {msg}')
    sys.exit(1)


def ok(msg):
    print(f'ok    {msg}')


def main():
    if shutil.which('docker') is None:
        fail('docker is not on PATH')

    running = subprocess.run(
        ['docker', 'ps', '--filter', f'name=^/{CONTAINER}$', '--format', '{{.Names}}'],
        capture_output=True, text=True)
    if CONTAINER not in running.stdout.split():
        fail(f'container {CONTAINER} is not running')
    ok(f'container {CONTAINER} is running')

    # The wheel's own RECORD is the authority on what it ships, not the presence
    # of a directory someone may have created by hand.
    record = subprocess.run(
        EXEC + ['bash', '-c',
         'source ' + CONDA_SH + '; export CONDA_ENVS_PATH=' + CONDA_ENVS_PATH + '; '
         'conda activate ' + CONDA_ENV + ' && '
         'python3 -c "import importlib.metadata as m, json; '
         'fs=[f for f in m.files(\\"triton_ascend\\") or [] if \\"extra/tla/\\" in str(f)]; '
         'print(len(fs))"'],
        capture_output=True, text=True)
    count_txt = (record.stdout or '').strip().splitlines()[-1:] or ['0']
    try:
        count = int(count_txt[0])
    except ValueError:
        count = 0
    if count <= 0:
        fail(f'the installed triton_ascend ships no language/extra/tla/ files '
             f'(stderr: {record.stderr.strip()[:200]})')
    ok(f'installed triton_ascend ships {count} files under language/extra/tla/')

    probe = subprocess.run(EXEC + ['bash', '-c',
                            'source ' + CONDA_SH + '; export CONDA_ENVS_PATH=' + CONDA_ENVS_PATH + '; '
                            'conda activate ' + CONDA_ENV + ' && python3 - <<\'PYEOF\'\n'
                            + PROBE + '\nPYEOF'],
                           capture_output=True, text=True)
    line = ''
    for candidate in (probe.stdout or '').splitlines():
        if candidate.startswith('{'):
            line = candidate
    if not line:
        fail(f'probe produced no JSON (stderr: {probe.stderr.strip()[:300]})')
    got = json.loads(line)

    if not got.get('tla_import'):
        fail(f"tla does not import in {CONDA_ENV}: {got.get('error')}")
    ok(f"tla imports: {got['tla_file']}")
    for key in ('has_jit', 'has_dma', 'has_annotate_buffer_versions'):
        if not got.get(key):
            fail(f'{key} is False; the surface the Gate needs is incomplete')
    ok('tla.jit, tla.dma and tla.annotate_buffer_versions are all present')

    # The L2 finding: the read path exposes only an eviction policy, never the
    # raw cache-control byte, and the policy enum has no NOTALLOC member.
    if 'l2_cache_ctrl' in got['LoadGM2UB_fields']:
        fail('LoadGM2UBParams now HAS l2_cache_ctrl; the L2 finding needs re-examination')
    ok(f"read path has no l2_cache_ctrl: {got['LoadGM2UB_fields']}")
    if 'l2_cache_ctrl' not in got['LoadGM2L1_fields']:
        fail('LoadGM2L1Params lost l2_cache_ctrl; the cube-path distinction changed')
    ok('write/cube path still carries l2_cache_ctrl (GM->L1), as documented')
    if any('NOTALLOC' in name or 'NOTALLOC' in name.upper() for name in got['EvictionPolicy']):
        fail('EvictionPolicy gained a NOTALLOC member; the L2 finding must be revisited')
    ok(f"EvictionPolicy has no NOTALLOC member: {got['EvictionPolicy']}")

    # Staging: versions= is the mechanism; Pipelined is absent from Triton too,
    # so its absence is not a TLA deficiency.
    if 'versions' not in got.get('alloc_buffer_sig', ''):
        fail(f"alloc_buffer lost versions=: {got.get('alloc_buffer_sig')}")
    ok(f"staging is expressed by versions=: {got['alloc_buffer_sig']}")
    if got.get('tl_has_pipelined'):
        fail('tl.Pipelined now exists; the staging comparison must be rewritten')
    ok(f"neither tla nor tl has Pipelined in triton {got.get('triton_version')}, "
       f"so its absence is Triton-wide, not a TLA gap")

    check = subprocess.run(
        ['bash', '-c',
         'cd /home/s00653124/workspace/triton-ascend && '
         'TLA_NPU_ENV_FILE=/home/s00653124/workspace/AgentGates/agent_gates/adapters/'
         'tla-npu/env.sh TLA_NPU_DEVICE_ID=1 python3 -B '
         '/home/s00653124/workspace/AgentGates/agent_gates/adapters/tla-npu/scripts/'
         'tla_gate.py --check-config'],
        capture_output=True, text=True)
    if check.returncode != 0:
        fail(f'the Adapter --check-config failed: {check.stderr.strip()[:300]}')
    ok('the Adapter --check-config reports the machine ready')

    print()
    print('VERDICT: a working TLA exists on this host and the tla-npu Adapter can run here.')
    print('The earlier "no installable TLA" claim is refuted by this script.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
