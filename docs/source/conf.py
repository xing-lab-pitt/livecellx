# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from urllib.request import urlretrieve

module_path = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, os.path.abspath(module_path))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))

# import livecellx

# HERE = Path(__file__).parent
# sys.path[:0] = [str(HERE.parent)]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext"}
# bibtex_bibfiles = ["./notebooks/lap.bib", "./notebooks/livetracker_ref.bib"]
# bibtex_reference_style = "author_year"

master_doc = "index"
pygments_style = "sphinx"


github_org = "xing-lab-pitt"
github_code_repo = "livecellx"
github_ref = "master"
github_nb_repo = "livecellx-notebooks"

# Add notebooks prolog to Google Colab and nbviewer
# TODO
nbsphinx_prolog = r"""
"""
nbsphinx_execute = "never"  # never execute notebooks

# -- Project information -----------------------------------------------------

project = "livecellx"
copyright = "2022, Ke Ni, Dante Poe, Yajushi Khurana, Weikang Wang, and Jianhua Xing"
author = "Ke Ni, Dante Poe, Yajushi Khurana, Weikang Wang, and Jianhua Xing"

# The full version, including alpha/beta/rc tags
release = "0.0.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# specify sphinx version
needs_sphinx = "4"

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # Link to other project's documentation (see mapping below)
    "sphinx.ext.intersphinx",
    # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    # Automatically document param types (less noise in class signature)
    "sphinx_autodoc_typehints",
    # "sphinxcontrib.bibtex",
    "sphinx_gallery.load_style",
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "cycler": ("https://matplotlib.org/cycler/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "h5py": ("http://docs.h5py.org/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "louvain": ("https://louvain-igraph.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": (
        "https://networkx.org/documentation/stable/",
        None,
    ),
    "napari": ("https://napari.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = True
# If no class summary, inherit base class summary
autodoc_inherit_docstrings = True

autodoc_default_flags = [
    # Make sure that any autodoc declarations show the right members
    "members",
    "inherited-members",
    "private-members",
    "show-inheritance",
]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = "furo"
html_theme_options = dict(
    navigation_depth=4,
    logo_only=True,
)
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user=github_org,  # organization
    github_repo=github_code_repo,  # Repo name
    github_version="master",  # Version
    conf_py_path="/docs/source/",
)
html_show_sphinx = False
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# html_logo = "_static/logo.png"
html_logo = "_static/logo_with_word.png"


autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"


def setup(app):
    app.add_css_file("css/custom.css")


sphinx_enable_epub_build = False
sphinx_enable_pdf_build = False
