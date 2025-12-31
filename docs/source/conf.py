# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Source code dir relative to this file

# -- Auto-generate API documentation -----------------------------------------
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    import sys
    sys.path.append(os.path.abspath('../../'))
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    module = os.path.join(cur_dir, '../../nirs4all')
    output_dir = os.path.join(cur_dir, 'api')
    main(['-e', '-o', output_dir, module, '--force'])

def setup(app):
    app.connect('builder-inited', run_apidoc)

project = 'Nirs4all'
copyright = '2025, Gregory Beurier'
author = 'Gregory Beurier'
release = '0.6.1'

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
napoleon_use_ivar = False
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
    ('py:class', 'ArtifactRegistry'),
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
