"""Real Sphinx builds retain API text and resolve only published example sources."""

from __future__ import annotations

import io
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

Sphinx = pytest.importorskip("sphinx.application").Sphinx
pytest.importorskip("myst_parser")

EXTENSIONS = Path(__file__).resolve().parents[3] / "docs" / "source" / "_ext"


class HTMLInventory(HTMLParser):
    def __init__(self, path: Path):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[tuple[str, str]] = []
        self.text: list[str] = []
        self.problematic = False
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append((attributes["href"], attributes.get("class", "")))
        self.problematic |= "problematic" in attributes.get("class", "").split()

    def handle_data(self, data):
        self.text.append(data)


def build_fixture(tmp_path: Path, extra_link: str = "", *, parallel: int = 1, staged: bool = False) -> tuple[Path, str, str]:
    root = tmp_path / "repository"
    source = root / "docs" / "source"
    source.mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "examples" / "real_example.py").write_text("print('example')\n")
    (root / "fixture_module.py").write_text('class Model:\n    """An attribute n_components_ and absolute value |X|."""\n    def predict(self):\n        """Predict samples."""\n')
    for args in [("init", "-q"), ("add", "examples", "fixture_module.py"), ("-c", "user.name=Docs fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")]:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / "alias_module.py").write_text("from fixture_module import Model\n")
    (root / "examples" / "untracked").mkdir()
    (root / "examples" / "local.py").write_text("print('local')\n")
    if staged:
        (root / "examples" / "staged_dir").mkdir()
        (root / "examples" / "staged_dir" / "staged.py").write_text("print('staged')\n")
        subprocess.run(["git", "add", "examples/staged_dir"], cwd=root, check=True, capture_output=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(EXTENSIONS)!r})\nsys.path.insert(0, {str(root)!r})\n"
        "extensions=['myst_parser','sphinx.ext.autodoc','sphinx.ext.viewcode','legacy_links']\n"
        "suppress_warnings=['docutils']\nmaster_doc='index'\n"
    )
    (source / "index.md").write_text("# Index\n\n```{toctree}\n_generated_api/alias\n_generated_api/fixture\n```\n\n[Example](../../../examples/real_example.py)\n[Examples](../../../examples/)\n" + extra_link)
    generated = source / "_generated_api"
    generated.mkdir()
    alias_members = "   :members:\n" if parallel == 1 else ""
    if parallel > 1:
        for index in range(12):
            (generated / f"filler_{index}.md").write_text(f"# Filler {index}\n")
        (source / "index.md").write_text((source / "index.md").read_text().replace("_generated_api/alias", ":glob:\n_generated_api/filler_*\n_generated_api/alias"))
    (generated / "alias.md").write_text("# Alias API\n\n```{eval-rst}\n.. autoclass:: alias_module.Model\n" + alias_members + "   :no-index:\n```\n")
    (generated / "fixture.md").write_text("# Model API\n\n```{eval-rst}\n.. autoclass:: fixture_module.Model\n   :members:\n   :no-index:\n```\n")
    warning = io.StringIO()
    output = tmp_path / "html"
    if parallel > 1:
        result = subprocess.run([sys.executable, "-m", "sphinx", "-W", "-E", "-a", "-j", str(parallel), "-b", "html", str(source), str(output)], text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return output, result.stderr, revision
    app = Sphinx(str(source), str(source), str(output), str(tmp_path / "doctrees"), "html", status=io.StringIO(), warning=warning, freshenv=True)
    app.build(force_all=True)
    return output, warning.getvalue(), revision


def test_actual_html_source_links_and_no_index_backlinks(tmp_path: Path) -> None:
    output, warning, revision = build_fixture(tmp_path)
    assert not warning
    index = HTMLInventory(output / "index.html")
    assert any(href == f"https://github.com/GBeurier/nirs4all/blob/{revision}/examples/real_example.py" for href, _ in index.links)
    assert any(href == f"https://github.com/GBeurier/nirs4all/tree/{revision}/examples" for href, _ in index.links)
    api = HTMLInventory(output / "_generated_api" / "fixture.html")
    assert "fixture_module.Model" in api.ids
    assert "fixture_module.Model.predict" in api.ids
    assert "alias_module.Model" in api.ids
    assert "n_components_" in "".join(api.text) and "|X|" in "".join(api.text)
    assert not api.problematic
    viewcode = HTMLInventory(output / "_modules" / "fixture_module.html")
    backlinks = [href for href, classes in viewcode.links if "viewcode-back" in classes.split()]
    assert backlinks
    for href in backlinks:
        assert href.partition("#")[2] in api.ids


@pytest.mark.parametrize("target", ["../../../examples/missing.py", "../../../examples/../../private.txt", "../../../examples/untracked/", "../../../examples/local.py"])
def test_unpublished_or_escaping_targets_keep_diagnostics(tmp_path: Path, target: str) -> None:
    output, warning, _ = build_fixture(tmp_path, f"\n[Invalid]({target})\n")
    assert "cross-reference target not found" in warning
    html = HTMLInventory(output / "index.html")
    assert any(href == "#" + target for href, _ in html.links)


def test_parallel_alias_backlinks_use_merged_inventory(tmp_path: Path) -> None:
    output, warning, _ = build_fixture(tmp_path, parallel=2)
    assert not warning
    pages = {name: HTMLInventory(output / "_generated_api" / f"{name}.html") for name in ("alias", "fixture")}
    source = HTMLInventory(output / "_modules" / "fixture_module.html")
    backlinks = [href for href, classes in source.links if "viewcode-back" in classes.split()]
    assert backlinks
    for href in backlinks:
        path, _, anchor = href.partition("#")
        assert anchor in pages[Path(path).stem].ids


@pytest.mark.parametrize("target", ["../../../examples/staged_dir/staged.py", "../../../examples/staged_dir/"])
def test_staged_examples_are_not_in_pinned_revision(tmp_path: Path, target: str) -> None:
    output, warning, revision = build_fixture(tmp_path, f"\n[Staged]({target})\n", staged=True)
    assert "cross-reference target not found" in warning
    assert not any(f"/{revision}/examples/staged_dir" in href for href, _ in HTMLInventory(output / "index.html").links)


def test_incremental_build_refreshes_revision_and_removed_alias(tmp_path: Path) -> None:
    output, _, revision = build_fixture(tmp_path, parallel=2)
    root = tmp_path / "repository"
    source = root / "docs" / "source"
    (root / "examples" / "real_example.py").write_text("print('updated')\n")
    subprocess.run(["git", "add", "examples/real_example.py"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Docs fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "updated example"], cwd=root, check=True, capture_output=True)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (source / "_generated_api" / "alias.md").unlink()
    index = source / "index.md"
    index.write_text(index.read_text().replace("_generated_api/alias\n", ""))
    result = subprocess.run([sys.executable, "-m", "sphinx", "-W", "-j", "2", "-b", "html", str(source), str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    links = HTMLInventory(output / "index.html").links
    assert any(f"/blob/{current}/examples/real_example.py" in href for href, _ in links)
    assert not any(f"/blob/{revision}/" in href for href, _ in links)
    source_html = HTMLInventory(output / "_modules" / "fixture_module.html")
    api = HTMLInventory(output / "_generated_api" / "fixture.html")
    for href, classes in source_html.links:
        if "viewcode-back" in classes.split():
            assert "alias.html" not in href
            assert href.partition("#")[2] in api.ids
