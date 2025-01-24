# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pneumonia_detection'
copyright = '2025, Alexander Szabados'
author = 'Alexander Szabados'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
extensions = [
    'sphinx.ext.autodoc',  # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support Google & NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
]

autodoc_default_options = {
    "members": True,  # Include all members (functions & classes)
    "undoc-members": True,  # Include members without docstrings
    "show-inheritance": True,  # Show class inheritance
    "special-members": "__init__",  # Ensure __init__ docstrings appear
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
