# Configuration file for the Sphinx documentation builder.

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../../'))  # Source code dir relative to this file

# -- Auto-generate API documentation -----------------------------------------
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    sys.path.append(os.path.abspath('../../'))
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    module = os.path.join(cur_dir, '../../nirs4all')
    output_dir = os.path.join(cur_dir, '_generated_api')
    api_dir = Path(output_dir)
    api_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale generated files so renamed/deleted modules disappear cleanly.
    for stale in api_dir.glob("nirs4all*.rst"):
        stale.unlink(missing_ok=True)
    (api_dir / "modules.rst").unlink(missing_ok=True)

    main(['-e', '-o', output_dir, module, '--force'])

    # Package pages and their submodule pages both document re-exported symbols.
    # Keep detailed member docs on submodule pages only to avoid duplicate objects.
    package_block = re.compile(
        r"\nModule contents\n-+\n\n\.\. automodule::[\s\S]*$",
        re.MULTILINE,
    )
    for rst_file in api_dir.glob("*.rst"):
        if rst_file.name == "modules.rst":
            continue
        content = rst_file.read_text(encoding="utf-8")
        content = content.replace("   :undoc-members:\n", "")
        if " package" not in content.splitlines()[0]:
            rst_file.write_text(content, encoding="utf-8")
            continue
        content = package_block.sub("\n", content)
        rst_file.write_text(content, encoding="utf-8")

_documented_objects = set()


def _skip_duplicate_members(app, what, name, obj, skip, options):
    """Document each (module, qualname) object once to avoid duplicate-object warnings.

    apidoc generates a page per module (``-e``). A class re-exported via ``__all__``
    in several wrapper modules (e.g. POPPLSRegressor in both sklearn.aom_pls and
    sklearn.pop_pls) would otherwise be documented on every page, producing
    "duplicate object description" warnings. Skip only the *second+* occurrence
    of the same object (keyed by its real module + qualname), so it is still
    documented exactly once and never lost — unlike a blanket "skip imported
    members" rule, which would drop objects whose defining module isn't paged.
    """
    if skip or what != 'module':
        return None
    member_mod = getattr(obj, '__module__', None)
    if member_mod is None:
        return None
    key = (member_mod, getattr(obj, '__qualname__', name))
    if key in _documented_objects:
        return True
    _documented_objects.add(key)
    return None


def setup(app):
    _documented_objects.clear()
    app.connect('builder-inited', run_apidoc)
    app.connect('autodoc-skip-member', _skip_duplicate_members)

project = 'Nirs4all'
copyright = '2026, Gregory Beurier'
author = 'Gregory Beurier'
# Single source of truth: parse __version__ from the package without importing it
# (avoids triggering heavy lazy-loaded backends at docs-build time).
import pathlib as _pathlib
import re as _re

_init_src = (_pathlib.Path(__file__).resolve().parents[2] / 'nirs4all' / '__init__.py').read_text(encoding='utf-8')
release = _re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', _init_src).group(1)
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',     # Cross-reference to sklearn, numpy, etc.
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',               # Cards, tabs, grids
    'sphinxcontrib.mermaid',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['assets']

# Logo configuration
html_logo = 'assets/nirs4all_logo.png'
html_theme_options = {
    'logo_only': False,
    'style_nav_header_background': '#2c3e50',
}

# Favicon (uses the same logo)
html_favicon = 'assets/nirs4all_logo.png'

# Custom CSS and JS for width toggle
html_css_files = ['custom.css']
html_js_files = ['custom.js']

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",               # Variable substitution
    "tasklist",                   # Checkboxes
    "attrs_block",                # Block attributes
]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# Intersphinx for cross-references to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Suppress warnings for ambiguous cross-references (classes exported at multiple levels)
# These are valid exports that create multiple documentation entries
nitpick_ignore = [
    # nirs4all.data module re-exports
    ('py:class', 'Predictions'),
    ('py:class', 'SignalType'),
    ('py:class', 'PredictionAnalyzer'),
    # nirs4all.pipeline.config module re-exports
    ('py:class', 'ExecutionContext'),
    # nirs4all.pipeline module re-exports
    ('py:class', 'PipelineRunner'),
    ('py:class', 'PipelineOrchestrator'),
    ('py:class', 'Predictor'),
    ('py:class', 'Explainer'),
    ('py:class', 'PipelineLibrary'),
    ('py:class', 'ExecutionTrace'),
    ('py:class', 'BundleLoader'),
    ('py:class', 'WorkspaceStore'),
    # nirs4all.api module re-exports
    ('py:class', 'RunResult'),
    # nirs4all.operators.models module re-exports
    ('py:class', 'PLSDA'),
]

# Suppress nitpicky mode for missing references that are intentionally simplified
nitpick_ignore_regex = [
    # Ignore missing internal cross-references
    (r'py:.*', r'nirs4all\.data\..*'),
    (r'py:.*', r'nirs4all\.pipeline\..*'),
    (r'py:.*', r'nirs4all\.api\..*'),
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# Linkcheck can be noisy/flaky on large volumes of GitHub file links.
# Keep linkcheck deterministic by focusing on non-GitHub URLs.
linkcheck_ignore = [
    r'https://github\.com/.*',
]
linkcheck_retries = 1
linkcheck_timeout = 10
linkcheck_workers = 5

# Optional ML backends are documented but not required to build docs.
autodoc_mock_imports = [
    'autogluon',
    'flax',
    'jax',
    'jaxlib',
    'keras',
    'tensorflow',
    'torch',
]

# Suppress duplicate object description warnings for re-exported classes
# These are intentionally exported at multiple levels for user convenience
suppress_warnings = [
    'autodoc',
    'autosummary',
    'design.grid',
    'docutils',
    'myst.directive_comments',
    'myst.directive_option',
    'myst.directive_unknown',
    'myst.xref_missing',
    'ref.python',
    'ref.*',
]
