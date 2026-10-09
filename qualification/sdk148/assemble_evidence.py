"""Normalize fresh, retained SDK148 direct command captures without relabelling history."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve()
OUTPUT = ROOT / 'qualification/sdk148'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def descriptor(path: Path) -> dict:
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def proof(name: str, value: object) -> dict:
    path = OUTPUT / name
    write(path, value)
    return descriptor(path)


def retain(path: Path, name: str) -> dict:
    target = OUTPUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return descriptor(target)


def main() -> None:
    spec = importlib.util.spec_from_file_location('verify', ROOT / 'scripts/verify_local_qualification.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    policy = json.loads((BASE / 'policy-draft.json').read_text())
    current = verifier.runtime_inputs(ROOT, 'sdk')
    frozen = json.loads((BASE / 'runtime-inputs.json').read_text())
    source_sha = json.loads((BASE / 'plans/metadata-linux.json').read_text())['source_sha']
    receipt = {'schema': verifier.SCHEMA, 'project': 'sdk', 'repository': 'GBeurier/nirs4all', 'source_sha': source_sha, 'execution': {'location': 'local', 'github_actions': False, 'performance_policy': 'strict'}, 'input_fingerprints': current, 'hosts': [], 'runs': []}
    hosts = {}
    for gate in policy['gates']:
        family = gate['hosts'][0]
        plan = json.loads((BASE / 'plans' / (gate['id'] + '.json')).read_text())
        capture = BASE / 'captures' / ('native-quickstart-linux-v2' if gate['id'] == 'native-quickstart-linux' else gate['id'])
        terminal = json.loads((capture / 'terminal.json').read_text())
        identity = terminal['run_identity']
        assert identity['exit_code'] == 0, gate['id']
        gate['commands'][family] = verifier.canonical_command(identity['command'], identity['cwd'], gate, family)
        gate['required_environment'] = {key: identity['environment'][key].replace(identity['cwd'], '{root}') for key in gate['required_environment']}
        assert terminal['runtime_inputs_before'] == terminal['runtime_inputs_after'] == frozen
        before = terminal['observation_before']
        assert before == terminal['observation_after']
        assert before['host']['os_family'] == family
        hosts[before['host']['id']] = before['host']
        selected = verifier.gate_inputs(verifier.tracked_inputs(ROOT), gate)
        assert identity['input_fingerprints'] == selected
        folder = gate['id']
        raw = [retain(capture / filename, folder + '/' + filename) for filename in ('before.json', 'terminal.json', 'stdout.log', 'stderr.log')]
        stdout = (capture / 'stdout.log').read_text(encoding='utf-8').strip()
        facts = {}
        passed = 1
        kind = gate['id'].removesuffix('-' + family)
        if kind == 'metadata':
            data = json.loads(stdout)
            assert data['distributions'] == {name: before['versions'][name] for name in data['distributions']}
            facts = {'sdk_version': data['distributions']['nirs4all'], 'dag_version': data['distributions']['dag-ml'], 'core_version': data['distributions']['nirs4all-core'], 'methods_abi': data['methods_abi'], 'methods_sha256': data['methods_sha256']}
        elif kind == 'provider45':
            facts['xy_scores'] = ast.literal_eval(re.search(r'INSTALLED_XY_ASSEMBLY_OK (\[[^\n]+\])', stdout)[1])
            facts['cv_rmse'] = float(re.search(r'INSTALLED_CV_ARCHIVE_OK ([^ ]+)', stdout)[1])
            facts['hpo_best_value'] = float(re.search(r'INSTALLED_HPO_ARCHIVE_OK ([^ ]+)', stdout)[1])
            facts['pickle_bytes'] = int(re.search(r'HPO_RESULT_PICKLE_OK (\d+)', stdout)[1])
            events = ast.literal_eval(re.search(r'INSTALLED_PROGRESS_RESUME_OK ([^\n]+)', stdout)[1])
            facts['resume_trials'] = events[-1][1]
            facts['hpo_matrix_cases'] = int(re.search(r'INSTALLED_GENERATED_HPO_MATRIX_OK (\d+)', stdout)[1])
        elif kind == 'demo-replay':
            data = json.loads(stdout)
            assert data['status'] == 'INSTALLED_U07_OK'
            facts = {name: data[name] for name in ('trial_count', 'prediction_count', 'fit_on_replay')}
        elif kind in ('native-quickstart', 'studio-installed'):
            junit = BASE / (family + ('-quickstart-junit.xml' if kind == 'native-quickstart' else '-studio-junit.xml'))
            cases = list(ET.parse(junit).getroot().iter('testcase'))
            assert cases and all(case.find('failure') is None and case.find('error') is None and case.find('skipped') is None for case in cases)
            passed = len(cases)
            facts = {'actual_quickstart_tests' if kind == 'native-quickstart' else 'actual_studio_tests': passed, 'skips': 0}
            raw.append(retain(junit, folder + '/junit.xml'))
        elif kind == 'dependency-check':
            assert stdout == 'No broken requirements found.'
            facts = {'pip_check_ok': True}
        else:
            raise ValueError(kind)
        for rule in gate.get('facts', []):
            verifier.fact_check(facts, rule)
        run = {**identity, 'duration_seconds': terminal['duration_seconds'], 'timing_basis': 'direct-monotonic-duration', 'log': raw[2]}
        run['input_evidence'] = proof(folder + '/inputs.json', {'mode': 'captured', 'source_sha': source_sha, 'run_identity': identity, 'input_fingerprints': selected, 'whole_runtime_fingerprints_before': frozen, 'whole_runtime_fingerprints_after': terminal['runtime_inputs_after'], 'raw_reports': raw})
        provenance = {'tools': before['host']['tools'], 'source_artifacts': [], 'dependency_origins': []}
        gate['provenance_requirements'] = {}
        for field in ('source_artifacts', 'dependency_origins'):
            gate['provenance_requirements'][field] = before[field]
            for index, artifact in enumerate(before[field]):
                evidence = proof(folder + '/' + field + '-' + str(index) + '.json', {'artifact': artifact, 'run_identity': identity, 'source_sha': source_sha, 'log_sha256': run['log']['sha256'], 'observed_before': before[field][index], 'observed_after': terminal['observation_after'][field][index], 'raw_reports': raw})
                provenance[field].append({**artifact, 'evidence': evidence})
        run['report'] = proof(folder + '/report.json', {**run, 'summary': {'passed': passed, 'failed': 0, 'skipped': 0}, 'skips': [], 'facts': facts, 'provenance': provenance, 'raw_reports': raw})
        receipt['runs'].append(run)
    receipt['hosts'] = list(hosts.values())
    retain(BASE / 'capture_gate.py', 'capture_gate.py.txt')
    retain(BASE / 'assemble_evidence.py', 'assemble_evidence.py')
    retain(BASE / 'public-wheel-manifest.json', 'public-wheel-manifest.json')
    write(ROOT / 'qualification/policy.json', policy)
    receipt['policy_sha256'] = digest(ROOT / 'qualification/policy.json')
    write(ROOT / 'compat/local-qualification.json', receipt)
    print(json.dumps({'gates': len(receipt['runs']), 'hosts': len(receipt['hosts']), 'runtime_inputs': len(current), 'source_sha': source_sha}))


if __name__ == '__main__':
    main()
