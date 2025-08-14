# -- Path setup --------------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.abspath(".."))

# -- Project info ------------------------------------------------------------
project = "raptorstats"
author = "Simon Gonzalez"
language = "en"

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# mock heavy deps if needed
autodoc_mock_imports = ["rasterio", "geopandas", "shapely", "rtree", "rasterstats"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
todo_include_todos = True
