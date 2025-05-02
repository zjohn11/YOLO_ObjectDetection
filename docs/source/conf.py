# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Point Sphinx at your project root so it can import your code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -- Project information -----------------------------------------------------
project   = 'Yolo V8 Object Detection'
author    = 'Zach, Brenna, Saah, Nathan'
release   = '0.1.0'
copyright = f'2025, {author}'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',    # pull in your docstrings
    'sphinx.ext.napoleon',   # support Google & NumPy doc styles
    'rst2pdf.pdfbuilder',    # build PDF directly, no LaTeX
]

templates_path   = ['_templates']
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

# Language
language = 'en'

# -- Options for HTML output (not used, but safe defaults) -------------------
html_theme       = 'alabaster'
html_static_path = ['_static']

# -- rst2pdf PDF output ------------------------------------------------------
# Define the single PDF and its metadata:
pdf_documents = [
    ('index',      # source start file
     'Documentation',  # output PDF name (MyProject.pdf)
     'Yolo V8 Object Detection',
     author),
]

# Use a coverpage and shrink wide tables/figures to fit margins:
pdf_use_coverpage = True
pdf_fit_mode      = 'shrink'

# You can also customize style sheets:
# pdf_stylesheets = ['sphinx', 'kerning']
# pdf_break_level  = 1
# pdf_breakside   = 'any'
# pdf_use_index   = True
# pdf_hyperlinks  = True
