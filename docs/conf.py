"""Sphinx configuration for Sampost."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project = "Sampost"
copyright = "2026, Sampost contributors"
author = "Sampost contributors"
release = "0.1.0"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
