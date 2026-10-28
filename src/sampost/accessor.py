"""The public ``Dataset.sampost`` accessor."""

from __future__ import annotations

import xarray as xr

from .grid import align_w_to_z
from .thermo import (
    buoyancy,
    equivalent_potential_temperature,
    liquid_water_potential_temperature,
    potential_temperature,
    virtual_potential_temperature,
)


def _require(ds: xr.Dataset, *names: str) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Missing canonical SAM variables: {', '.join(missing)}")


def _condensate(ds: xr.Dataset) -> xr.DataArray:
    """Return SAM's combined liquid-and-ice condensate field."""
    if "q_n" in ds:
        return ds["q_n"]
    if "q_c" in ds:
        return ds["q_c"]
    raise KeyError("Missing canonical condensate variable: q_n")


@xr.register_dataset_accessor("sampost")
class SampostAccessor:
    """Lazy diagnostics over a canonical Sampost Dataset."""

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def potential_temperature(self) -> xr.DataArray:
        _require(self._ds, "pressure", "temp")
        return potential_temperature(self._ds["pressure"], self._ds["temp"])

    def virtual_potential_temperature(self) -> xr.DataArray:
        _require(self._ds, "pressure", "temp", "q_v")
        return virtual_potential_temperature(self._ds["pressure"], self._ds["temp"], self._ds["q_v"])

    def equivalent_potential_temperature(self) -> xr.DataArray:
        _require(self._ds, "pressure", "temp", "q_v")
        return equivalent_potential_temperature(self._ds["pressure"], self._ds["temp"], self._ds["q_v"])

    def liquid_water_potential_temperature(self) -> xr.DataArray:
        _require(self._ds, "pressure", "temp")
        return liquid_water_potential_temperature(
            self._ds["pressure"], self._ds["temp"], _condensate(self._ds)
        )

    def buoyancy(self, reference: xr.DataArray | None = None) -> xr.DataArray:
        """Calculate buoyancy relative to a supplied or horizontal-mean state."""
        theta_v = self.virtual_potential_temperature()
        if reference is None:
            reference = theta_v.mean(("x", "y")) if {"x", "y"} <= set(theta_v.dims) else theta_v
        return buoyancy(theta_v, reference)

    def cape_cin(self, vertical_dim: str = "z") -> xr.Dataset:
        """Integrate positive and negative buoyancy over height.

        This first-pass parcel diagnostic uses the local parcel virtual
        potential temperature and the horizontal-mean environment. Inputs must
        contain a height coordinate in metres; it is intentionally explicit
        about this parcel convention.
        """
        b = self.buoyancy()
        if vertical_dim not in b.coords:
            raise KeyError(f"Missing height coordinate: {vertical_dim}")
        cape = b.where(b > 0, 0).integrate(vertical_dim)
        cin = b.where(b < 0, 0).integrate(vertical_dim)
        return xr.Dataset({"cape": cape, "cin": cin})

    def cloud_mask(self, threshold: float = 1e-5) -> xr.DataArray:
        return (_condensate(self._ds) > threshold).rename("cloud_mask")

    def lwp(self, vertical_dim: str = "z", density_name: str = "rho") -> xr.DataArray:
        _require(self._ds, vertical_dim, density_name)
        result = (_condensate(self._ds) * self._ds[density_name]).integrate(vertical_dim)
        result.name = "lwp"
        result.attrs = {"units": "kg m-2", "long_name": "liquid water path"}
        return result

    def total_water_path(self, vertical_dim: str = "z", density_name: str = "rho") -> xr.DataArray:
        _require(self._ds, "q_v", density_name)
        result = ((self._ds["q_v"] + _condensate(self._ds)) * self._ds[density_name]).integrate(vertical_dim)
        result.name = "twp"
        result.attrs = {"units": "kg m-2", "long_name": "total water path"}
        return result

    def conditional_mean(self, mask: xr.DataArray, dims: tuple[str, str] = ("x", "y")) -> xr.Dataset:
        """Return masked horizontal means, retaining NaN for empty columns."""
        missing = [dim for dim in dims if dim not in self._ds.dims]
        if missing:
            raise KeyError(f"Missing horizontal dimensions: {', '.join(missing)}")
        return self._ds.where(mask).mean(dims, skipna=True)

    def horizontal_mean(self, dims: tuple[str, str] = ("x", "y")) -> xr.Dataset:
        missing = [dim for dim in dims if dim not in self._ds.dims]
        if missing:
            raise KeyError(f"Missing horizontal dimensions: {', '.join(missing)}")
        return self._ds.mean(dims)

    def reynolds_decomposition(self, dims: tuple[str, str] = ("x", "y")) -> xr.Dataset:
        return self._ds - self.horizontal_mean(dims)

    def resolved_fluxes(self, dims: tuple[str, str] = ("x", "y")) -> xr.Dataset:
        _require(self._ds, "w", "q_v")
        w = align_w_to_z(self._ds) if "zW" in self._ds["w"].dims else self._ds["w"]
        w_prime = w - w.mean(dims)
        q_v_prime = self._ds["q_v"] - self._ds["q_v"].mean(dims)
        result = xr.Dataset({"w_qv": (w_prime * q_v_prime).mean(dims)})
        if "temp" in self._ds:
            theta_v = self.virtual_potential_temperature()
            result["w_theta_v"] = (w_prime * (theta_v - theta_v.mean(dims))).mean(dims)
        return result

    def tke(self, dims: tuple[str, str] = ("x", "y")) -> xr.DataArray:
        _require(self._ds, "u", "v", "w")
        components = []
        for name in ("u", "v", "w"):
            field = align_w_to_z(self._ds) if name == "w" and "zW" in self._ds[name].dims else self._ds[name]
            components.append(((field - field.mean(dims)) ** 2).mean(dims))
        result = 0.5 * sum(components)
        result.name = "tke"
        result.attrs = {"units": "m2 s-2", "long_name": "resolved turbulent kinetic energy"}
        return result
