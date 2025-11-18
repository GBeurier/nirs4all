"""Check manifest and binary files."""
import yaml
from pathlib import Path

workspace = Path("workspace/runs/2025-11-18_dataset")
manifests = sorted(workspace.glob('*/manifest.yaml'), key=lambda p: p.stat().st_mtime, reverse=True)

print(f"Most recent manifest: {manifests[0]}")

with open(manifests[0]) as f:
    manifest = yaml.safe_load(f)

print(f"\nPipeline UID: {manifest['uid']}")
print(f"Artifacts in manifest: {len(manifest.get('artifacts', []))}")

for art in manifest.get('artifacts', []):
    name = art.get('name')
    step = art.get('step')
    skipped = art.get('skipped', False)
    path = art.get('path', '')

    if not skipped and path:
        full_path = workspace / path
        exists = full_path.exists() if path else False
        print(f"  {name} (step {step}): path={path}, exists={exists}")
    else:
        print(f"  {name} (step {step}): SKIPPED or no path")

# Check _binaries directory
binaries_dir = workspace / "_binaries"
if binaries_dir.exists():
    print(f"\n_binaries directory contents:")
    for f in binaries_dir.iterdir():
        print(f"  {f.name} ({f.stat().st_size} bytes)")
else:
    print(f"\n_binaries directory does not exist!")
