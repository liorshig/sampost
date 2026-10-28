"""Explicit helpers for SAM's staggered vertical grids."""

from __future__ import annotations

import xarray as xr


def interpolate_vertical(
    field: xr.DataArray,
    target: xr.DataArray | str,
    *,
    vertical_dim: str = "zW",
) -> xr.DataArray:
    """Interpolate a field onto a target vertical coordinate.

    Parameters
    ----------
    field
        DataArray containing the source vertical coordinate.
    target
        Coordinate DataArray or coordinate name on ``field``.
    vertical_dim
        Name of the source vertical dimension.

    Returns
    -------
    xarray.DataArray
        Interpolated field with the target coordinate values.
    """
    source = field[vertical_dim]
    if source.ndim != 1:
        raise ValueError(f"{vertical_dim!r} must be one-dimensional")
    values = source.values
    if values.size > 1 and not (
        (values[1:] >= values[:-1]).all() or (values[1:] <= values[:-1]).all()
    ):
        raise ValueError(f"{vertical_dim!r} must be monotonic")
    target_coord = field[target] if isinstance(target, str) else target
    return field.interp({vertical_dim: target_coord})


def align_w_to_z(ds: xr.Dataset) -> xr.DataArray:
    """Interpolate SAM vertical velocity from ``zW`` levels to ``z`` levels.

    Parameters
    ----------
    ds
        Dataset containing ``w`` on ``zW`` and a one-dimensional ``z``
        coordinate.

    Returns
    -------
    xarray.DataArray
        ``w`` on the thermodynamic ``z`` grid.
    """
    if "w" not in ds or "z" not in ds.coords or "zW" not in ds["w"].coords:
        raise ValueError("align_w_to_z requires w with zW and a z coordinate")
    result = ds["w"].interp({"zW": ds["z"]})
    return result.assign_coords(z=result["zW"]).drop_vars("zW")
