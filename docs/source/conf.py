# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import subprocess
#install_script = os.path.abspath("../../install.py")
#if os.path.exists(install_script):
#    print(f"Running installation script: {install_script}")
#    subprocess.run(["python", install_script], check=True)
#else:
#    print("install.py not found, skipping installation.")

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath("../.."))  # Adjust path if needed

# Debugging
print("Updated PYTHONPATH:", sys.path)

# Check if 'code' is importable
try:
    import code.classifier
    print("Module 'code.classifier' imported successfully!")
except ModuleNotFoundError as e:
    print("Module import error:", e)

project = 'Pneumonia Detection'
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
    'sphinx.ext.viewcode',
    'nbsphinx'# Add links to source code
]
exclude_patterns = ['_build', '**.ipynb_checkpoints']
autodoc_default_options = {
    "members": True,  # Include all members (functions & classes)
    "undoc-members": True,  # Include members without docstrings
    "show-inheritance": True,  # Show class inheritance
    "special-members": "__init__",  # Ensure __init__ docstrings appear
}
nbsphinx_allow_errors = True  # Allows notebooks to be included even if errors occur
nbsphinx_toctree_depth = 1  # Ensures only the first level (title) appears in index
nbsphinx_execute = "never"  # Prevents execution of notebooks every time
nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput, .nboutput { display: block; }
        .nboutput .output_area { max-height: 500px; overflow-y: auto; }
    </style>
"""
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
