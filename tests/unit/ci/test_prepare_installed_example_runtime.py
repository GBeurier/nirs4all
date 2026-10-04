"""Installed replay preparation rejects source drift and parent product hooks."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[3] / "scripts/prepare_installed_example_runtime.py"
SPEC = importlib.util.spec_from_file_location("installed_example_preparation", HELPER)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_dependency_projection_excludes_products_metadata_and_executable_hooks(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    for name in ["numpy", "nirs4all", "n4m", "dag_ml", "nirs4all_methods.libs"]:
        (parent / name).mkdir()
    for name in ["editable.pth", "project.egg-link", "__editable___finder.py", "sitecustomize.py"]:
        (parent / name).write_text("raise RuntimeError('parent hooks must never execute')\n")
    for name, distribution in [("numpy-2.4.6.dist-info", "numpy"), ("nirs4all-1.3.3.dist-info", "nirs4all")]:
        directory = parent / name
        directory.mkdir()
        (directory / "METADATA").write_text(f"Name: {distribution}\nVersion: 1\n")
    projection = tmp_path / "projection"
    inventory = helper.project_dependencies([parent], projection)
    assert set(inventory) == {"numpy", "numpy-2.4.6.dist-info"}
    assert (projection / "numpy").is_symlink()
    assert (projection / "numpy").resolve() == parent / "numpy"
    assert (parent / "editable.pth").read_text().startswith("raise RuntimeError")


def test_sdk_wheel_identity_and_source_payload_are_independently_checked(tmp_path):
    source = tmp_path / "source"
    (source / "nirs4all").mkdir(parents=True)
    (source / "nirs4all/__init__.py").write_bytes(b"__version__ = '1.4.0'\n")
    wheel = tmp_path / "sdk.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nirs4all/__init__.py", (source / "nirs4all/__init__.py").read_bytes())
        archive.writestr("nirs4all-1.4.0.dist-info/METADATA", "Name: nirs4all\nVersion: 1.4.0\n")
    assert helper.wheel_identity(wheel) == ("nirs4all", "1.4.0")
    assert helper.sdk_payload(wheel, source) == {"nirs4all/__init__.py": helper.sha256(source / "nirs4all/__init__.py")}
    (source / "nirs4all/__init__.py").write_bytes(b"__version__ = 'different runtime'\n")
    with pytest.raises(ValueError, match="wheel/source payload mismatch"):
        helper.sdk_payload(wheel, source)


def test_sdk_payload_refuses_a_wheel_without_runtime_members(tmp_path):
    wheel = tmp_path / "sdk.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nirs4all-1.4.0.dist-info/METADATA", "Name: nirs4all\nVersion: 1.4.0\n")
    with pytest.raises(ValueError, match="runtime payload absent"):
        helper.sdk_payload(wheel, tmp_path)
