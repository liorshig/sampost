"""SAM NetCDF ingestion and canonical field naming."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import xarray as xr
from metpy.units import units

DEFAULT_VARIABLES: dict[str, str] = {
    "TABS": "temp",
    "T": "temp",
    "QV": "q_v",
    "QN": "q_n",
    "QP": "q_p",
    "W": "w",
    "U": "u",
    "V": "v",
    "RHO": "rho",
    "P": "p_base",
    "PRES": "pressure",
    "PP": "p_prime",
    "Z": "z",
    "ZW": "zW",
}

DEFAULT_DIMENSIONS = {"x": "x", "y": "y", "z": "z", "zW": "zW"}
SI_UNITS = {
    "temp": "K",
    "q_v": "dimensionless",
    "q_n": "dimensionless",
    "q_p": "dimensionless",
    "u": "m/s",
    "v": "m/s",
    "w": "m/s",
    "rho": "kg/m^3",
    "p_base": "Pa",
    "p_prime": "Pa",
    "pressure": "Pa",
    "x": "m",
    "y": "m",
    "z": "m",
    "zW": "m",
}


def _convert_to_si(data: xr.DataArray, target: str) -> xr.DataArray:
    source = data.attrs.get("units")
    if not source:
        raise ValueError(f"Variable {data.name!r} is missing required units; expected {target}")
    try:
        source_unit = units(str(source))
        target_unit = units(target)
        if source_unit.dimensionality != target_unit.dimensionality:
            raise ValueError(
                f"Variable {data.name!r} has units {source!r}, incompatible with {target!r}"
            )
    except Exception as exc:
        raise ValueError(
            f"Variable {data.name!r} has invalid or incompatible units {source!r}; "
            f"expected {target!r}"
        ) from exc

    converted = xr.apply_ufunc(
        lambda values: (values * source_unit).to(target_unit).magnitude,
        data,
        dask="parallelized",
        output_dtypes=[float],
    )
    converted.attrs = dict(data.attrs)
    converted.attrs["units"] = target
    converted.attrs["source_units"] = str(source)
    return converted


def open_sam(
    path: str | Path,
    *,
    variable_map: Mapping[str, str] | None = None,
    dimension_map: Mapping[str, str] | None = None,
    chunks: Any = "auto",
    **kwargs: Any,
) -> xr.Dataset:
    """Open a SAM NetCDF file using Sampost's canonical schema.

    Canonical variables include ``temp``, ``q_v``, ``q_n``, ``w``, ``rho``,
    ``pressure``, ``z``, and ``zW``. When both ``P`` and ``PP`` are present,
    they are retained as ``p_base`` and ``p_prime`` and combined into
    ``pressure``. Existing canonical names are preserved.
    Unit conversion is intentionally explicit: source variables must carry
    ``units`` attributes, or callers may provide canonical units through the
    returned Dataset before diagnostics are requested.
    """
    ds = xr.open_dataset(path, chunks=chunks, **kwargs)
    names = dict(DEFAULT_VARIABLES)
    if variable_map:
        names.update(variable_map)
    rename = {
        source: target
        for source, target in names.items()
        if source in ds and target not in ds
    }
    ds = ds.rename(rename)

    dims = dict(DEFAULT_DIMENSIONS)
    if dimension_map:
        dims.update(dimension_map)
    dim_rename = {
        source: target
        for target, source in dims.items()
        if source in ds.dims and target not in ds.dims
    }
    ds = ds.rename_dims(dim_rename)
    for name, target in SI_UNITS.items():
        if name in ds.data_vars:
            ds[name] = _convert_to_si(ds[name], target)
        elif name in ds.coords and "units" in ds[name].attrs:
            ds[name] = _convert_to_si(ds[name], target)
        elif name in ds.coords:
            # SAM grid coordinates are conventionally expressed in metres.
            ds[name].attrs["units"] = target
    if "p_base" in ds and "p_prime" in ds:
        base = ds["p_base"]
        perturbation = ds["p_prime"]
        if base.ndim != 1:
            raise ValueError(
                "SAM base pressure 'P' must be one-dimensional before combining it with 'PP'"
            )
        ds["pressure"] = base + perturbation
        ds["pressure"].attrs = dict(base.attrs)
        ds["pressure"].attrs.update(
            {
                "long_name": "absolute pressure",
                "standard_name": "air_pressure",
            }
        )
    return ds
