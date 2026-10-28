"""Lazy xarray diagnostics for SAM output."""

from .accessor import SampostAccessor
from .io import open_sam, open_sam_mfdataset
from .turbulence import calculate_fluctuations, calculate_turbulent_fluxes

__all__ = [
    "SampostAccessor",
    "calculate_fluctuations",
    "calculate_turbulent_fluxes",
    "open_sam",
    "open_sam_mfdataset",
]
__version__ = "0.1.0"
