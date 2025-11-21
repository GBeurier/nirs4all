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
release = '0.4.2'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['assets']

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
]
myst_heading_anchors = 3
