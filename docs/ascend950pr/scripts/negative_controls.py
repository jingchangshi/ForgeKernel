"""Run the five negative controls end to end and report what actually happened.

The acceptance protocol requires that all five really fail. An assertion in a unit
test proves the judge function returns 'fail'; it does NOT prove the whole script
does, with real artifacts, real job evidence and real measured numbers. This
runner drives the actual accept script and prints its actual output, so the
report can quote observations rather than intentions.

Controls whose evidence is a recorded device measurement (#2, #5) use the numbers
from artifacts/precision-matrix.json and artifacts/probe/*.json, and say so: they
are real measurements of this machine, reused offline rather than re-measured,
because the card is shared and a control does not need a fresh window to be a
control. Controls #1, #3 and #4 need no device at all.

Usage:
    python3 negative_controls.py            # all controls that need no card
    python3 negative_controls.py --json     # one JSON object instead of the log
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PLAN = Path(__file__).resolve().parents[1]
ARTIFACTS = PLAN / 'artifacts'
ADAPTER = Path(str(WORK_AGENTGATES) + ''
               '/agent_gates/adapters/forgekernel-ascend-npu')
ACCEPT = ADAPTER / 'scripts/accept_rmsnorm.py'


def load_acceptance():
    spec = importlib.util.spec_from_file_location('accept_rmsnorm', ACCEPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_accept(arguments):
    """Run the real script as a subprocess and return its exit code and output."""
    done = subprocess.run([sys.executable, '-B', str(ACCEPT)] + arguments,
                          capture_output=True, text=True)
    payload = None
    if done.stdout.strip():
        try:
            payload = json.loads(done.stdout)
        except ValueError:
            payload = None
    return {'exitCode': done.returncode, 'report': payload,
            'stdout': done.stdout, 'stderr': done.stderr}


def control_one(temp):
    """#1: a candidate that delegates to the vendor operator is refused."""
    target = temp / 'delegating'
    shutil.copytree(ARTIFACTS / 'candidates' / 'delegating', target)
    result = run_accept(['--candidate', str(target), '--check-only'])
    verdict = (result['report'] or {}).get('verdict') or {}
    authenticity = (result['report'] or {}).get('authenticity') or {}
    return {
        'control': 1,
        'name': 'delegating candidate (importlib + getattr + string concatenation)',
        'expected': 'refused as inauthentic',
        'observed': {'exitCode': result['exitCode'],
                     'verdict': verdict.get('status'),
                     'stage': verdict.get('stage'),
                     'authenticity': authenticity.get('verdict'),
                     'delegations': authenticity.get('static', {}).get('delegations'),
                     'reasons': (verdict.get('reasons') or [])[:2]},
        'passed': verdict.get('status') == 'FAIL' and result['exitCode'] == 1,
    }


def control_two():
    """#2: a bf16-accumulating kernel is refused by the precision gate."""
    acceptance = load_acceptance()
    matrix = json.loads((ARTIFACTS / 'precision-matrix.json').read_text())
    rows, honest_rows = [], []
    for shape, block in matrix['shapes'].items():
        for family, row in block['families'].items():
            base = {'shape': shape, 'family': family, 'truth': 'float64'}
            rows.append({**base,
                         'vendor': {'rel_l2': row['vendor_rel_l2'],
                                    'max_rel': row['vendor_max_rel'],
                                    'mean_abs': row['vendor_mean_abs']},
                         'candidate': {'rel_l2': row['bf16_rel_l2'],
                                       'max_rel': row['bf16_max_rel'],
                                       'mean_abs': row['bf16_mean_abs']}})
            honest_rows.append({**base,
                                'vendor': {'rel_l2': row['vendor_rel_l2'],
                                           'max_rel': row['vendor_max_rel'],
                                           'mean_abs': row['vendor_mean_abs']},
                                'candidate': {'rel_l2': row['fp32_rel_l2'],
                                              'max_rel': row['fp32_max_rel'],
                                              'mean_abs': row['fp32_mean_abs']}})
    bf16 = acceptance.judge_precision(rows)
    honest = acceptance.judge_precision(honest_rows)
    return {
        'control': 2,
        'name': 'deliberate bf16 accumulation, measured on this machine',
        'expected': 'refused by the precision gate; the honest fp32 arm passes',
        'observed': {
            'bf16Arm': {'verdict': bf16['verdict'],
                        'failedFamilies': [item['shape'] + '/' + item['family']
                                           for item in bf16['rows']
                                           if item['status'] == 'fail'],
                        'ulp': [{'family': item['shape'] + '/' + item['family'],
                                 'candidate': item['ulp_cand'],
                                 'vendor': item['ulp_ref'],
                                 'limit': item['ulp_limit'],
                                 'candidateOverLimit': round(item['ulp_cand']
                                                             / item['ulp_limit'], 3)}
                                for item in bf16['rows']]},
            'honestArm': {'verdict': honest['verdict'],
                          'worstFractionOfLimit': max(item['ratio_to_limit']
                                                      for item in honest['rows'])},
        },
        'passed': bf16['verdict'] == 'fail' and honest['verdict'] == 'pass',
    }


def control_three(temp):
    """#3: a job that did not finish cleanly is FAIL, never PASS."""
    acceptance = load_acceptance()
    cases = {
        'missing marker': {'exitCode': 0, 'timedOut': False, 'interrupted': False,
                           'reason': None, 'completionMarker': False,
                           'collected': 2, 'skipped': 0},
        'non-zero exit': {'exitCode': 1, 'timedOut': False, 'interrupted': False,
                          'reason': None, 'completionMarker': True,
                          'collected': 2, 'skipped': 0},
        'timeout': {'exitCode': None, 'timedOut': True, 'interrupted': False,
                    'reason': None, 'completionMarker': True,
                    'collected': 2, 'skipped': 0},
        'everything skipped': {'exitCode': 0, 'timedOut': False, 'interrupted': False,
                               'reason': None, 'completionMarker': True,
                               'collected': 3, 'skipped': 3},
    }
    observed, all_passed = {}, True
    for name, record in cases.items():
        verdict = acceptance.judge_job_evidence(record, '{"ok": true}')
        observed[name] = {'verdict': verdict['verdict'],
                          'reasons': verdict['reasons'][:1]}
        all_passed = all_passed and verdict['verdict'] == 'fail'
    full = run_accept(['--candidate', str(ARTIFACTS / 'candidates' / 'genuine'),
                       '--check-only', '--job-evidence', str(temp / 'no-such-run')])
    observed['nonexistent evidence directory'] = {
        'exitCode': full['exitCode'],
        'stderr': full['stderr'].strip()[:160]}
    all_passed = all_passed and full['exitCode'] == 2
    return {'control': 3,
            'name': 'job evidence: missing marker / non-zero exit / timeout / all skipped',
            'expected': 'every one is FAIL; missing evidence is a usage error, not a pass',
            'observed': observed, 'passed': all_passed}


def control_four():
    """#4: a device baseline whose worst workload is host-bound is invalid."""
    acceptance = load_acceptance()
    workloads = [{'name': 'decode', 'hostBoundRatio': 0.88},
                 {'name': 'prefill', 'hostBoundRatio': 0.04}]
    audit = acceptance.audit_host_bound(workloads)
    result = {'authenticity': {'verdict': 'pass'},
              'hostBoundAudit': audit,
              'precision': {'verdict': 'pass'},
              'timing': {'8192x4096': {'status': 'MATCH', 'pointSpeedup': 1.01,
                                       'interval95': [1.0, 1.02]}}}
    verdict = acceptance.decide(result, check_only=False)
    return {
        'control': 4,
        'name': 'device baseline host_bound_ratio [0.04, 0.88]',
        'expected': ('the maximum 0.88 blocks the baseline; the ratio of geometric '
                     'means 0.1876 would have passed it'),
        'observed': {
            'maxRatio': audit['maxRatio'], 'verdictOnMax': audit['verdictOnMax'],
            'rederived': round(audit['rederivedRatioOfGeomeans'], 6),
            'geometricMeanOfRatios': round(math.sqrt(0.88 * 0.04), 6),
            'verdictOnRederived': audit['verdictOnRederived'],
            'verdictFlips': audit['verdictFlips'],
            'baselineValid': audit['valid'],
            'gateVerdict': verdict['status'], 'gateStage': verdict['stage'],
            'reason': (verdict['reasons'] or [''])[0][:200],
        },
        'passed': (not audit['valid'] and audit['verdictOnMax'] == 'invalid'
                   and audit['verdictOnRederived'] == 'valid'
                   and audit['verdictFlips']
                   and verdict['status'] == 'INVALID_EVIDENCE'),
    }


def control_five():
    """#5: a correct but slower candidate fails and keeps its artifact.

    The ratio is the one measured for the repository's example kernel against the
    vendor on 8192x4096 (0.1634 ms vs 0.0514 ms, 0.315x) - a real reading from
    artifacts/probe/, reused offline. Eight paired windows are synthesised around
    it with the +/-8 percent spread the paired protocol measured, so the failure
    is not an artifact of one number.
    """
    acceptance = load_acceptance()
    vendor_ms, candidate_ms = 0.0514, 0.1634
    # A shared drift that cancels exactly would give a zero-width interval, which
    # is not what a real window looks like and would make the control pass for the
    # wrong reason. Each arm gets the shared component (the machine got slower, so
    # both got slower) PLUS its own +/3 percent, so the paired ratio itself
    # wobbles the +/-8 percent the protocol measured.
    shared = [1.0 + 0.10 * ((index % 5) - 2) / 2.0 for index in range(12)]
    own = [1.0 + 0.03 * (((index * 7) % 5) - 2) / 2.0 for index in range(12)]
    windows = []
    for index in range(12):
        windows.append({
            'index': index,
            'order': ['candidate', 'vendor'] if index % 2 == 0 else ['vendor', 'candidate'],
            'vendor_ms': round(vendor_ms * shared[index] * own[index], 6),
            'candidate_ms': round(candidate_ms * shared[index] / own[index], 6),
            'speedup': None,
            'witness': {'aicoreUtilBefore': 0, 'aicoreUtilAfter': 0}})
    for window in windows:
        window['speedup'] = window['vendor_ms'] / window['candidate_ms']
    timing = acceptance.judge_timing(windows, '8192x4096')
    result = {'authenticity': {'verdict': 'pass'},
              'precision': {'verdict': 'pass'},
              'timing': {'8192x4096': timing},
              'candidate': {'root': str(ARTIFACTS / 'candidates' / 'genuine')}}
    verdict = acceptance.decide(result, check_only=False)
    return {
        'control': 5,
        'name': 'correct but 3.18x slower candidate (example kernel vs vendor)',
        'expected': 'FAIL, and the best correct artifact is preserved',
        'observed': {
            'pointSpeedup': timing['pointSpeedup'],
            'interval95': timing['interval95'],
            'status': timing['status'],
            'gateVerdict': verdict['status'], 'gateStage': verdict['stage'],
            'bestCorrectArtifact': verdict.get('bestCorrectArtifact'),
        },
        'passed': (timing['status'] == 'FAIL' and verdict['status'] == 'FAIL'
                   and verdict.get('bestCorrectArtifact')),
        'evidenceBasis': ('measurements from artifacts/probe/ (0.1634 ms example kernel '
                          'vs 0.0514 ms vendor, 8192x4096), reused offline'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    temp = Path(tempfile.mkdtemp(prefix='negative-controls-'))
    try:
        controls = [control_one(temp), control_two(), control_three(temp),
                    control_four(), control_five()]
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    report = {
        'date': __import__('time').strftime('%Y-%m-%dT%H:%M:%S%z'),
        'script': 'scripts/negative_controls.py',
        'acceptScript': str(ACCEPT),
        'acceptDigest': __import__('hashlib').sha256(ACCEPT.read_bytes()).hexdigest(),
        'controls': controls,
        'allPassed': all(item['passed'] for item in controls),
        'deviceUsed': False,
    }
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return 0 if report['allPassed'] else 1
    for item in controls:
        mark = 'REFUSED' if item['passed'] else 'NOT REFUSED'
        print('=' * 78)
        print('control %d  %s' % (item['control'], item['name']))
        print('  expected : %s' % item['expected'])
        print('  observed : %s' % mark)
        for key, value in item['observed'].items():
            text = json.dumps(value, ensure_ascii=False)
            print('    %-24s %s' % (key + ':', text[:220]))
    print('=' * 78)
    print('all five refused:', report['allPassed'])
    return 0 if report['allPassed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
