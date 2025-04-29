import os, sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'YOLO_OBJECTDETECTION'
author  = 'Zach, Brenna, Saah, Nathan'
release = '1.0'

# -- General configuration ---------------------------------------------------
master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
]

autosummary_generate = True
autodoc_mock_imports = [
    "torch", "fiftyone", "fiftyone.zoo", "albumentations",
    "ultralytics", "cv2", "icrawler", "tkinter"
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
templates_path   = ['_templates']

# -- HTML output -------------------------------------------------------------
html_theme = 'alabaster'

# -- LaTeX / PDF output ----------------------------------------------------

# Use pdflatex or xelatex (whichever you prefer / have installed)
latex_engine = 'pdflatex'

# Tell Sphinx to build one PDF called YOLO_OBJECTDETECTION
latex_documents = [
    ('index',                    # master document
     'YOLO_OBJECTDETECTION.tex', # output .tex file
     project,                    # document title
     author,                     # author list
     'manual'),                  # document class
]

# Removes the “start chapters on right-hand pages” behaviour
latex_elements = {
    'classoptions': 'oneside',
    # optionally tweak paper size, font size, etc:
    # 'papersize': 'letterpaper',
    # 'pointsize': '10pt',
}

