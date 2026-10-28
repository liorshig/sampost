"""Lazy xarray diagnostics for SAM output."""

from .accessor import SampostAccessor
from .io import open_sam

__all__ = ["SampostAccessor", "open_sam"]
__version__ = "0.1.0"
