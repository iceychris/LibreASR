# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
for x in os.walk("../libreasr"):
    sys.path.insert(0, x[0])


# -- Project information -----------------------------------------------------

project = "LibreASR"
copyright = "2020, Christian Bruckdorfer"
author = "Christian Bruckdorfer"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "recommonmark",
    "nbsphinx",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    True  # Remove 'view source code' from top of page (for html, not python)
)
html_sidebars = {
    '**': [
        'logo-text.html',
        'globaltoc.html',
        'localtoc.html',
        'searchbox.html',
    ]
}
autodoc_inherit_docstrings = True  # If no class summary, inherit base class summary

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# set material theme
html_theme = 'sphinx_material'

# Material theme options (see theme.conf for more information)
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'LibreASR',

    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXX',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://github.com/iceychris/LibreASR',

    # Set the color and the accent color
    'color_primary': 'blue',
    'color_accent': 'light-blue',

    # opts
    "html_minify": False,
    "html_prettify": True,
    "css_minify": True,

    # style
    "logo_icon": "&#xe029",

    # Set the repo location to get a badge with stats
    "repo_type": "github",
    'repo_url': 'https://github.com/iceychris/LibreASR',
    'repo_name': 'LibreASR',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 3,
    # If False, expand all TOC entries
    'globaltoc_collapse': False,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': False,
}


# -- Options for HTML output -------------------------------------------------

# on_rtd is whether on readthedocs.org, this line of code grabbed from docs.readthedocs.org...
"""
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
"""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
