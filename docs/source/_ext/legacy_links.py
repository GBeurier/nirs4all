"""Keep legacy source links and generated API anchors usable in built HTML."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import quote, urlsplit

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext.viewcode import viewcode_anchor
from sphinx.transforms.post_transforms import SphinxPostTransform


class _SourceLinksEnvironment(Protocol):
    legacy_source_links: dict[str, Any]


def prepare_source_inventory(app: Sphinx) -> None:
    """Bind source links to the checkout revision and its tracked example files."""
    root = Path(app.srcdir).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", revision, "--", "examples"], cwd=root, stderr=subprocess.DEVNULL).decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        # A source export cannot prove that local files are published. Keep
        # unresolved references visible to the normal Sphinx diagnostics.
        revision, tracked = "", []
    cast(_SourceLinksEnvironment, app.env).legacy_source_links = {"root": str(root), "revision": revision, "tracked": set(tracked)}


class RepositoryExampleLinks(SphinxPostTransform):
    """Resolve checked-in example references before MyST treats them as anchors."""

    default_priority = 8  # MyST resolves ordinary Markdown references at 9.

    def run(self, **kwargs: Any) -> None:
        inventory = cast(_SourceLinksEnvironment, self.env).legacy_source_links
        if not inventory["revision"]:
            return
        root = Path(inventory["root"])
        examples = (root / "examples").resolve()
        for node in list(self.document.findall(addnodes.pending_xref)):
            if node.get("reftype") != "myst":
                continue
            target = urlsplit(node.get("reftarget", ""))
            if target.scheme or target.netloc or target.query or target.fragment:
                continue
            parts = Path(target.path).parts
            if "examples" not in parts:
                continue
            start = parts.index("examples")
            if not parts[:start] or any(part != ".." for part in parts[:start]):
                continue
            relative = Path(*parts[start:])
            source = (root / relative).resolve()
            if not source.is_relative_to(examples):
                continue
            name = relative.as_posix()
            if source.is_file() and name in inventory["tracked"]:
                route = "blob"
            elif source.is_dir() and any(path.startswith(name + "/") for path in inventory["tracked"]):
                route = "tree"
            else:
                continue
            uri = f"https://github.com/GBeurier/nirs4all/{route}/{inventory['revision']}/{quote(name, safe='/')}"
            reference = nodes.reference("", "", *[child.deepcopy() for child in node.children], refuri=uri)
            node.replace_self(reference)


def preserve_generated_api_text(app: Sphinx, doctree: nodes.document) -> None:
    """Show malformed docstring markup literally rather than as error-page links.

    Undefined substitutions (for example absolute values ``|X|``), bare trailing
    underscores and unmatched code delimiters have no resolvable destination.
    Keep their original visible text; do not alter valid references or source
    docstrings, and leave the parser's existing diagnostics policy unchanged.
    """
    if not app.env.docname.startswith("_generated_api/"):
        return
    for node in list(doctree.findall(nodes.problematic)):
        text = node.astext()
        node.replace_self(nodes.literal(text, text))


def add_no_index_source_anchors(app: Sphinx, doctree: nodes.document) -> None:
    """Give real unindexed API descriptions anchors for viewcode's backlinks.

    ``:no-index:`` omits object registration as intended. It also removes the
    HTML IDs that viewcode nevertheless links back to. Restore only the local
    signature anchor; do not register duplicate Python-domain objects.
    """
    for node in doctree.findall(addnodes.desc_signature):
        if node.get("ids") or not node.get("module") or not node.get("fullname"):
            continue
        anchor = f"{node['module']}.{node['fullname']}"
        if anchor not in doctree.ids:
            node["ids"].append(anchor)
            doctree.note_explicit_target(node)


def add_imported_source_anchors(app: Sphinx, doctree: nodes.document) -> None:
    """Preserve viewcode backlinks when an object is described via two modules.

    Viewcode keeps one module-prefix per source file, while the destination
    description can come from an alias module. Its own source-link node proves
    which real signature it selected; add that exact backlink ID to the same
    description without dropping its existing IDs or changing object indexes.
    """
    modules = getattr(app.env, "_viewcode_modules", {})
    for signature in doctree.findall(addnodes.desc_signature):
        for link in signature.findall(viewcode_anchor):
            module = link["reftarget"].removeprefix("_modules/").replace("/", ".")
            entry = modules.get(module)
            if not entry:
                continue
            anchor = f"{entry[3]}.{link['refid']}"
            if anchor not in doctree.ids:
                signature["ids"].append(anchor)
                doctree.note_explicit_target(signature)


class ImportedSourceAnchors(SphinxPostTransform):
    """Use the merged viewcode inventory before its source nodes are converted."""

    default_priority = 99  # Viewcode converts its anchor nodes at 100.

    def run(self, **kwargs: Any) -> None:
        add_imported_source_anchors(self.app, self.document)


def reread_linked_documents(app: Sphinx, env: Any, added: set[str], changed: set[str], removed: set[str]) -> list[str]:
    """Refresh cross-document viewcode ownership and revision-pinned links.

    Viewcode's module prefix and backlink ownership depend on descriptions in
    other documents. Reread retained documents so incremental builds cannot
    retain a removed alias or a source URL pinned to an earlier checkout.
    """
    return sorted(env.found_docs - removed)


def refresh_viewcode_pages(app: Sphinx) -> list[Any]:
    """Regenerate source HTML because backlink ownership may change alone.

    Viewcode otherwise skips pages whenever the Python file is older than its
    HTML, even if a referring document was removed or renamed. Invalidate only
    its generated pages for modules in the current inventory, before viewcode
    renders them again; source files and API descriptions remain untouched.
    """
    if app.builder.format != "html":
        return []
    directory = (Path(app.outdir) / "_modules").resolve()
    suffix = getattr(app.builder, "out_suffix", ".html")
    for module in getattr(app.env, "_viewcode_modules", {}):
        page = (directory / (module.replace(".", "/") + suffix)).resolve()
        if page.is_relative_to(directory):
            page.unlink(missing_ok=True)
    return []


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("builder-inited", prepare_source_inventory)
    app.connect("doctree-read", preserve_generated_api_text, priority=400)
    app.connect("doctree-read", add_no_index_source_anchors, priority=400)
    app.connect("env-get-outdated", reread_linked_documents)
    app.add_post_transform(ImportedSourceAnchors)
    app.connect("html-collect-pages", refresh_viewcode_pages, priority=400)
    app.add_post_transform(RepositoryExampleLinks)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
