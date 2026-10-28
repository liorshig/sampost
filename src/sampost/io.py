"""SAM NetCDF ingestion and canonical field naming."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


def _normalize_unit_string(unit: str) -> str:
    """Normalize common SAM/meteorology unit aliases to Pint-compatible names."""
    text = str(unit).strip()
    if not text:
        return text
    aliases = {
        "mb": "hPa",           # Changed from "mbar"
        "mbar": "hPa",         # Changed from "mbar"
        "millibar": "hPa",
        "millibars": "hPa",
        "hpa": "hPa",
        "hpa.": "hPa",
        "hectopascal": "hPa",
        "hectopascals": "hPa",
        "pa": "Pa",
        "pascal": "Pa",
        "pascals": "Pa",
        "kgkg-1": "kg/kg",
        "kg/kg": "kg/kg",
        "kg kg-1": "kg/kg",
        "kg kg^-1": "kg/kg",
        "kgkg^-1": "kg/kg",
    }
    normalized = aliases.get(text.lower())
    if normalized is not None:
        return normalized
    return text


def _convert_to_si(data: xr.DataArray, target: str) -> xr.DataArray:
    """Convert a DataArray to an SI unit while preserving lazy execution."""
    source = data.attrs.get("units")
    if not source:
        raise ValueError(f"Variable {data.name!r} is missing required units; expected {target}")
    source = _normalize_unit_string(source)
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
        output_dtypes=[data.dtype],
    )
    converted.attrs = dict(data.attrs)
    converted.attrs["units"] = target
    converted.attrs["source_units"] = str(source)
    return converted


def _combine_pressure(ds: xr.Dataset) -> xr.Dataset:
    """Create absolute pressure from canonical ``p_base`` and ``p_prime``."""
    if "p_base" not in ds or "p_prime" not in ds:
        return ds
    base = ds["p_base"]
    perturbation = ds["p_prime"]
    if base.ndim != 1:
        raise ValueError(
            "SAM base pressure 'P' must be one-dimensional before combining it with 'PP'"
        )
    ds["pressure"] = base + perturbation
    ds["pressure"].attrs = dict(base.attrs)
    ds["pressure"].attrs.update(
        {"long_name": "absolute pressure", "standard_name": "air_pressure"}
    )
    return ds


def _standardize(
    ds: xr.Dataset,
    *,
    variable_map: Mapping[str, str] | None = None,
    dimension_map: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Rename SAM fields, convert SI units, and combine pressure fields."""
    names = dict(DEFAULT_VARIABLES)
    if variable_map:
        names.update(variable_map)
    for pressure_name in ("p", "P"):
        if pressure_name in ds and pressure_name not in names:
            if ds[pressure_name].ndim == 1 or list(ds[pressure_name].dims) == ["z"]:
                names[pressure_name] = "p_base"
            else:
                names[pressure_name] = "pressure"
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
                # Must use assign_coords for coordinate mutation
                ds = ds.assign_coords({name: _convert_to_si(ds[name], target)})
            elif name in ds.coords:
                ds.coords[name].attrs["units"] = target
    return _combine_pressure(ds)


def open_sam(
    path: str | Path | Sequence[str | Path],
    *,
    variable_map: Mapping[str, str] | None = None,
    dimension_map: Mapping[str, str] | None = None,
    chunks: Any = "auto",
    **kwargs: Any,
) -> xr.Dataset:
    """Open SAM NetCDF data using Sampost's canonical schema.

    Canonical variables include ``temp``, ``q_v``, ``q_n``, ``w``, ``rho``,
    ``pressure``, ``z``, and ``zW``. When both ``P`` and ``PP`` are present,
    they are retained as ``p_base`` and ``p_prime`` and combined into
    ``pressure``. Existing canonical names are preserved.
    Unit conversion is intentionally explicit: source variables must carry
    ``units`` attributes, or callers may provide canonical units through the
    returned Dataset before diagnostics are requested.

    Parameters
    ----------
    path
        A NetCDF path, glob-compatible path sequence, or file collection.
    variable_map, dimension_map
        Optional source-to-canonical name mappings.
    chunks
        Dask chunk specification passed to xarray. Use ``None`` to load
        eagerly and ``"auto"`` for automatic lazy chunks.
    **kwargs
        Additional options passed to :func:`xarray.open_dataset` or
        :func:`open_sam_mfdataset`.

    Returns
    -------
    xarray.Dataset
        A canonical, SI-normalized Dataset. Data remain lazy when chunks are
        enabled.
    """
    if isinstance(path, (str, Path)):
        ds = xr.open_dataset(path, chunks=chunks, **kwargs)
        return _standardize(
            ds, variable_map=variable_map, dimension_map=dimension_map
        )
    return open_sam_mfdataset(
        path,
        chunks=chunks,
        variable_map=variable_map,
        dimension_map=dimension_map,
        **kwargs,
    )


def open_sam_mfdataset(
    files_kinematic: str | Path | Sequence[str | Path],
    files_thermo: str | Path | Sequence[str | Path] | None = None,
    *,
    chunks: Any = "auto",
    parallel: bool = True,
    combine: str = "by_coords",
    data_vars: str = "minimal",
    variable_map: Mapping[str, str] | None = None,
    dimension_map: Mapping[str, str] | None = None,
    preprocess_kinematic: Callable[[xr.Dataset], xr.Dataset] | None = None,
    preprocess_thermo: Callable[[xr.Dataset], xr.Dataset] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Lazily load and combine SAM kinematic and thermodynamic file groups.

    This is the Sampost equivalent of opening two ``open_mfdataset`` groups
    and merging them. User preprocessors run before Sampost canonicalization
    and SI conversion. The returned Dataset remains lazy until ``compute`` or
    ``load`` is called.

    Parameters
    ----------
    files_kinematic, files_thermo
        Paths or path sequences for the kinematic and thermodynamic groups.
        Set ``files_thermo=None`` to load only the kinematic group.
    chunks
        Dask chunk specification passed to both xarray collections.
    parallel
        Whether xarray may open files in parallel.
    combine, data_vars
        Combination options passed to :func:`xarray.open_mfdataset`.
    variable_map, dimension_map
        Optional source-to-canonical name mappings.
    preprocess_kinematic, preprocess_thermo
        Optional callbacks applied to each source Dataset before standardizing.
    **kwargs
        Additional options passed to both ``open_mfdataset`` calls.

    Returns
    -------
    xarray.Dataset
        The merged canonical Dataset, lazy when chunks are enabled.
    """

    def preprocess(
        ds: xr.Dataset,
        callback: Callable[[xr.Dataset], xr.Dataset] | None,
    ) -> xr.Dataset:
        """Apply a user callback followed by Sampost standardization."""
        if callback is not None:
            ds = callback(ds)
        return _standardize(
            ds, variable_map=variable_map, dimension_map=dimension_map
        )

    kinematic = xr.open_mfdataset(
        files_kinematic,
        combine=combine,
        chunks=chunks,
        parallel=parallel,
        data_vars=data_vars,
        preprocess=lambda ds: preprocess(ds, preprocess_kinematic),
        **kwargs,
    )
    if files_thermo is None:
        return _combine_pressure(kinematic)
    thermo = xr.open_mfdataset(
        files_thermo,
        combine=combine,
        chunks=chunks,
        parallel=parallel,
        data_vars=data_vars,
        preprocess=lambda ds: preprocess(ds, preprocess_thermo),
        **kwargs,
    )
    return _combine_pressure(xr.merge([kinematic, thermo]))
