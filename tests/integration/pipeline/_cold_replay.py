"""Run cold bundle checks against the actual producer runtime or installed wheel."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import nirs4all


def run_cold_replay(script: str, arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Verify fitted replay payloads before loading an archive in a fresh process."""
    package = Path(nirs4all.__file__).resolve().parent
    sources = ("pipeline/execution/fitted.py", "pipeline/execution/preprocessing.py",
               "pipeline/bundle/loader.py", "controllers/data/concat_transform.py")
    installed = os.environ.get("NIRS4ALL_CV_WEIGHT_INSTALLED_PYTHON")
    expected = {"installed": bool(installed), "sources": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources}}
    environment = os.environ.copy()
    if installed:
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(package.parent), environment.get("PYTHONPATH"))))
    verification = r'''
import hashlib,json,pathlib,sys,nirs4all
expectation=json.loads(sys.argv.pop())
package=pathlib.Path(nirs4all.__file__).resolve().parent
if expectation['installed']:assert 'site-packages' in package.parts,package
for relative,digest in expectation['sources'].items():
    assert hashlib.sha256((package/relative).read_bytes()).hexdigest()==digest,relative
print('COLD_REPLAY_SDK '+json.dumps({'module':str(package),'installed':expectation['installed'],'verified_payloads':list(expectation['sources'])}))
'''
    return subprocess.run([installed or sys.executable, *( ["-I"] if installed else []), "-B", "-c", verification + script,
                           *arguments, json.dumps(expected)], cwd=cwd, env=environment, capture_output=True, text=True)
