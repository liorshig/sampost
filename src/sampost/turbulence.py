"""Reynolds fluctuations and resolved turbulent fluxes."""

from __future__ import annotations

from collections.abc import Iterable

import xarray as xr

from .grid import align_w_to_z
from .thermo import virtual_potential_temperature

FLUX_FIELDS = {
    "uw": ("u", "w"),
    "vw": ("v", "w"),
    "w_temp": ("w", "temp"),
    "w_qv": ("w", "q_v"),
    "w_theta_v": ("w", "theta_v"),
}


def calculate_fluctuations(ds: xr.Dataset, mean: xr.Dataset) -> xr.Dataset:
    """Return explicit mean fields and fluctuations for every data variable.

    ``mean`` may be a profile or another broadcastable Dataset. Each field in
    ``ds`` must have a corresponding field in ``mean``; xarray performs the
    dimension alignment and broadcasting for the subtraction.
    """
    missing = [name for name in ds.data_vars if name not in mean]
    if missing:
        raise KeyError(f"Mean Dataset is missing variables: {', '.join(missing)}")

    result = xr.Dataset()
    for name in ds.data_vars:
        mean_field = mean[name]
        fluctuation = ds[name] - mean_field
        fluctuation.name = f"{name}_prime"
        fluctuation.attrs = dict(ds[name].attrs)
        if "long_name" in fluctuation.attrs:
            fluctuation.attrs["long_name"] = f"{fluctuation.attrs['long_name']} fluctuation"
        result[name] = mean_field
        result[f"{name}_prime"] = fluctuation
    return result


def _field_prime(ds: xr.Dataset, mean: xr.Dataset, name: str) -> xr.DataArray:
    """Return one fluctuation, including the derived virtual potential temperature."""
    if name == "theta_v":
        if "theta_v" in ds:
            source = ds["theta_v"]
        else:
            source = virtual_potential_temperature(ds["pressure"], ds["temp"], ds["q_v"])
        if "theta_v" in mean:
            mean_field = mean["theta_v"]
        else:
            mean_field = virtual_potential_temperature(
                mean["pressure"], mean["temp"], mean["q_v"]
            )
    else:
        if name not in ds or name not in mean:
            raise KeyError(f"Turbulent flux requires {name!r} in both data and mean Datasets")
        source, mean_field = ds[name], mean[name]
    return source - mean_field


def calculate_turbulent_fluxes(
    ds: xr.Dataset,
    mean: xr.Dataset,
    *,
    dims: Iterable[str] = ("x", "y"),
    fields: Iterable[str] = FLUX_FIELDS,
) -> xr.Dataset:
    """Calculate domain-averaged turbulent fluxes from explicit means.

    Parameters
    ----------
    ds, mean
        Instantaneous and mean fields. Mean fields may be reduced profiles
        that broadcast over the instantaneous dimensions.
    dims
        Dimensions over which the prime product is averaged.
    fields
        Flux keys selected from ``uw``, ``vw``, ``w_temp``, ``w_qv``, and
        ``w_theta_v``.
    """
    dimensions = tuple(dims)
    result = xr.Dataset()
    for field in fields:
        if field not in FLUX_FIELDS:
            valid = ", ".join(FLUX_FIELDS)
            raise ValueError(f"Unknown turbulent flux {field!r}; expected one of: {valid}")
        left, right = FLUX_FIELDS[field]
        left_prime = _field_prime(ds, mean, left)
        right_prime = _field_prime(ds, mean, right)
        if left == "w" and "zW" in left_prime.dims:
            left_prime = align_w_to_z(
                xr.Dataset({"w": left_prime}, coords={"z": ds["z"]})
            )
        if right == "w" and "zW" in right_prime.dims:
            right_prime = align_w_to_z(
                xr.Dataset({"w": right_prime}, coords={"z": ds["z"]})
            )
        flux = (left_prime * right_prime).mean(dimensions)
        flux.name = f"{field}_flux"
        flux.attrs = {
            "long_name": f"turbulent {left}'{right}' flux",
            "source_fields": f"{left}_prime, {right}_prime",
        }
        result[flux.name] = flux
    return result
