#!/usr/bin/env python3
"""Run original SDK commands locally with the existing Core capture protocol."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

OWNER = Path('/mnt/c/Temp/n4a-downstream041-20261008/sdk147-local-prefix-v1')
ROOT = Path('/home/delete/nirs4all/_worktrees/2026-10-06-debt-extension/followup/sdk-current-cohort-docs')
TEMP = Path('/dev/shm/n4a-sdk147-20261008-private-posix')
FREEZE = OWNER / 'sdk147-before-local-source-freeze.json'
PYTHON = '/mnt/c/Temp/n4a-downstream041-20261008/sdk147-local-prefix-v1/linux-prefix/bin/python'
CARGO = '/home/delete/.cargo/bin/cargo'
METHODS = OWNER / 'linux-prefix/lib/python3.11/site-packages/n4m/lib/libn4m.so.2.17.0'
COMMANDS: dict[str, list[str]] = {}

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def source_capture():
    frozen = json.loads(FREEZE.read_text())
    result = {name: digest(ROOT/name) for name in frozen['all_tracked_plus_new_source_files']}
    for name, row in frozen['all_tracked_plus_new_source_files'].items():
        if result[name] != row['sha256']:
            raise RuntimeError('Source differs from exact reviewed candidate: '+name)
    return result

def owned_usage():
    total = 0
    for directory, _, files in os.walk(OWNER/'target'):
        for name in files:
            try:
                total += (Path(directory)/name).stat().st_size
            except FileNotFoundError:
                pass
    tmp_bytes = sum(p.stat().st_size for p in TEMP.rglob('*') if p.is_file())
    return {'tmpfs_bytes':tmp_bytes,'target_bytes':total,'c_free':shutil.disk_usage(OWNER).free,'root_free':shutil.disk_usage('/').free}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gate')
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()) if args.plan else {}
    if plan:
        if plan['gate'] != args.gate or not all(isinstance(v, str) for v in plan['command']):
            raise ValueError('Command plan must name the actual gate and argv')
        COMMANDS[args.gate] = plan['command']
    if args.gate not in COMMANDS:
        parser.error('Unknown local command')
    cwd = Path(plan.get('cwd', str(OWNER)))
    output = OWNER/'local-raw-v2'/args.gate
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('RUSTFLAGS', None)
    env.update({'TMPDIR':str(TEMP),'TMP':str(TEMP),'TEMP':str(TEMP),'PYTHONDONTWRITEBYTECODE':'1','N4A_ENGINE':'dag-ml','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','MPLCONFIGDIR':str(OWNER/'matplotlib'),'XDG_CACHE_HOME':str(OWNER/'cache')})
    env['PATH'] = '/mnt/c/Temp/n4a-core045-20261008/tooling:/home/delete/.nvm/versions/node/v22.21.1/bin:/home/delete/.cargo/bin:'+str(Path(PYTHON).parent)+':'+env.get('PATH','')
    env.update(plan.get('environment', {}))
    if plan.get('host','linux-wsl') not in ('linux-wsl','windows'):
        raise ValueError('Unknown actual local process host')
    global METHODS
    METHODS = Path(plan.get('methods_library', str(METHODS)))
    extra_paths = [Path(v) for v in plan.get('external_inputs', [])]
    extra_before = {str(p): {'sha256':digest(p),'bytes':p.stat().st_size} for p in extra_paths}
    inputs = source_capture()
    before = owned_usage()
    if before['root_free'] <= 1024**3 or before['c_free'] <= 16*1024**3 or shutil.disk_usage(TEMP).free < 1024**3:
        raise RuntimeError('Owned storage minimum unavailable before command')
    record = {'id':args.gate,'host':plan.get('host','linux-wsl'),'command':COMMANDS[args.gate],'cwd':str(cwd),'execution':{'location':'local','github_actions':False,'performance_policy':'strict'},'runner_sha256':digest(Path(__file__)),'source_tree':json.loads(FREEZE.read_text())['virtual_full_git_tree'],'captured_at':stamp(),'input_fingerprints':inputs,'environment':{name:env[name] for name in ('TMPDIR','TMP','TEMP','N4A_ENGINE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','MPLCONFIGDIR','XDG_CACHE_HOME')},'plan_sha256':digest(args.plan) if args.plan else None,'external_inputs_before':extra_before,'native_dependency_before':{'path':str(METHODS),'bytes':METHODS.stat().st_size,'sha256':digest(METHODS)},'storage_before':before}
    record['environment'].update({name:env[name] for name in plan.get('environment', {})})
    (output/'PRE.json').write_text(json.dumps(record,indent=2)+'\n')
    record['started_at'] = stamp()
    start = time.monotonic()
    with (output/'stdout.log').open('wb') as stdout, (output/'stderr.log').open('wb') as stderr:
        child = subprocess.Popen(COMMANDS[args.gate], cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        record['pid'] = child.pid
        (output/'active.json').write_text(json.dumps(record,indent=2)+'\n')
        stop_reason = None
        while child.poll() is None:
            time.sleep(5)
            usage = owned_usage()
            if usage['tmpfs_bytes'] > 1024**3 or usage['target_bytes'] > 20*1024**3 or usage['root_free'] < 1024**3 or usage['c_free'] < 16*1024**3:
                stop_reason = 'owned-storage-budget'
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                break
        code = child.wait()
    terminal = {'command':COMMANDS[args.gate],'direct_command_exit_code':code,'started_at':record['started_at'],'finished_at':stamp(),'seconds':time.monotonic()-start,'pid':child.pid}
    (output/'direct-terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    extra_after = {str(p): {'sha256':digest(p),'bytes':p.stat().st_size} for p in extra_paths}
    if extra_after != extra_before:
        stop_reason = 'external-input-bytes-changed'
    record.update(external_inputs_after=extra_after,direct_terminal=terminal,exit_code=code,stop_reason=stop_reason,input_fingerprints_after=source_capture(),native_dependency_sha256_after=digest(METHODS),storage_after=owned_usage(),logs={name:{'path':str(output/name),'bytes':(output/name).stat().st_size,'sha256':digest(output/name)} for name in ('stdout.log','stderr.log')})
    (output/'raw-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'id':args.gate,'exit_code':code,'stop_reason':stop_reason,'seconds':terminal['seconds'],'storage_after':record['storage_after']}),flush=True)
    if code or stop_reason:
        print((output/'stderr.log').read_text(errors='replace')[-4500:],flush=True)
    return code or int(stop_reason is not None)

if __name__ == '__main__':
    raise SystemExit(main())
