"""Lazy xarray diagnostics for SAM output."""

from .accessor import SampostAccessor
from .io import open_sam, open_sam_mfdataset

__all__ = ["SampostAccessor", "open_sam", "open_sam_mfdataset"]
__version__ = "0.1.0"
