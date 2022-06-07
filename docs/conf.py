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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'qmlearn'
copyright = '2022, @PRG @Tuckerman Research Group'
author = 'Xuecheng Shao, Lukas Paetow, Md Rajib Khan Musa, Jessica A. Martinez B. and Michele Pavanello @ PRG at Rutgers University-Newark. Mark E Tuckerman @ Tuckerman Research Group at NYU'

# The full version, including alpha/beta/rc tags
release = '0.8'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.imgmath"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

#html_theme = "pydata_sphinx_theme"
#html_theme_options = {
#    "github_url": "https://gitlab.com/pavanello-research-group/qmlearn",
#    "show_prev_next": False,
#}
#html_sidebars = {
#    "**": [],
#}

html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_style = 'custom.css'
html_static_path = ['static']
html_last_updated_fmt = '%A, %d %b %Y %H:%M:%S'

html_theme_options = {
    'logo_only': True,
    'prev_next_buttons_location': 'both',
    # 'style_nav_header_background' : '#E67E22'
    # 'style_nav_header_background' : '#27AE60'
    'style_nav_header_background' : '#bdc3c7'
}

# latex_show_urls = 'inline'
latex_show_pagerefs = True
latex_documents = [('index', not True)]

graphviz_output_format = 'svg'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # ['_static']

# Output file base name for HTML help builder.

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
numpydoc_show_class_members = False


#Add external links to source code
def linkcode_resolve(domain, info):
    print('info module', info)
    if domain != 'py' or not info['module']:
        return None

    filename = info['module'].replace('.', '/')+'.py'
    return "https ://gitlab.com/pavanello-research-group/qmlearn/tree/master/%s" % filename

