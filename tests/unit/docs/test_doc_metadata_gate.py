"""Release citations stay tied to published metadata during development."""

from __future__ import annotations

import json
import sys

import pytest

from scripts import check_doc_metadata


@pytest.mark.parametrize(
    ("package_ver", "published_ver", "citation_ver", "valid"),
    [
        ("1.3.0", "1.3.0", "1.3.0", True),
        ("1.3.1.dev0", "1.3.0", "1.3.0", True),
        ("1.3.1.dev0", "1.3.1", "1.3.1", False),
        ("1.3.1.dev0", "1.3.1.dev0", "1.3.1.dev0", False),
        ("1.3.0", "1.2.9", "1.2.9", False),
        ("1.3.1.dev0", "1.3.0", "1.2.9", False),
    ],
)
def test_metadata_gate_tracks_published_release(
    tmp_path, monkeypatch, package_ver: str, published_ver: str, citation_ver: str, valid: bool
) -> None:
    (tmp_path / "nirs4all").mkdir()
    (tmp_path / "nirs4all" / "__init__.py").write_text(f'__version__ = "{package_ver}"\n')
    (tmp_path / ".zenodo.json").write_text(json.dumps({"version": published_ver, "title": "nirs4all"}))
    (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
    (tmp_path / "README.md").write_text(f"version = {{{citation_ver}}}\n")
    monkeypatch.setattr(check_doc_metadata, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_doc_metadata.py"])

    assert (check_doc_metadata.main() == 0) is valid
