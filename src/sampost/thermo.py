"""MetPy-backed, Dask-safe thermodynamic calculations."""

from __future__ import annotations

import numpy as np
import xarray as xr
from metpy import calc
from metpy.constants import Cp_d, Lv
from metpy.units import units

GRAVITY = 9.80665


def _unit(data: xr.DataArray, default: str) -> str:
    return str(data.attrs.get("units", default))


def _magnitude(data: xr.DataArray, target: str) -> xr.DataArray:
    source = _unit(data, target)
    if source == target:
        return data
    return xr.apply_ufunc(
        lambda value: (value * units(source)).to(target).magnitude,
        data,
        dask="parallelized",
        output_dtypes=[float],
    )


def _metpy_ufunc(
    function: object,
    *inputs: xr.DataArray,
    input_units: tuple[str, ...],
    output_unit: str,
) -> xr.DataArray:
    magnitudes = [
        _magnitude(data, target) for data, target in zip(inputs, input_units, strict=True)
    ]

    def call(*values: np.ndarray) -> np.ndarray:
        quantities = [value * units(unit) for value, unit in zip(values, input_units, strict=True)]
        result = function(*quantities)
        return result.to(output_unit).magnitude

    return xr.apply_ufunc(
        call,
        *magnitudes,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )


def potential_temperature(pressure: xr.DataArray, temp: xr.DataArray) -> xr.DataArray:
    result = _metpy_ufunc(
        calc.potential_temperature,
        pressure,
        temp,
        input_units=("Pa", "K"),
        output_unit="K",
    )
    result.name, result.attrs = "theta", {"units": "K", "long_name": "potential temperature"}
    return result


def virtual_potential_temperature(
    pressure: xr.DataArray, temp: xr.DataArray, q_v: xr.DataArray
) -> xr.DataArray:
    result = _metpy_ufunc(
        calc.virtual_potential_temperature,
        pressure,
        temp,
        q_v,
        input_units=("Pa", "K", "dimensionless"),
        output_unit="K",
    )
    result.name, result.attrs = "theta_v", {"units": "K", "long_name": "virtual potential temperature"}
    return result


def equivalent_potential_temperature(
    pressure: xr.DataArray, temp: xr.DataArray, q_v: xr.DataArray
) -> xr.DataArray:
    def call(p: np.ndarray, t: np.ndarray, mixing: np.ndarray) -> np.ndarray:
        p_quantity = p * units.Pa
        t_quantity = t * units.K
        mixing_quantity = mixing * units.dimensionless
        dewpoint = calc.dewpoint(calc.vapor_pressure(p_quantity, mixing_quantity))
        return calc.equivalent_potential_temperature(
            p_quantity, t_quantity, dewpoint
        ).to("K").magnitude

    result = xr.apply_ufunc(
        call,
        _magnitude(pressure, "Pa"),
        _magnitude(temp, "K"),
        _magnitude(q_v, "dimensionless"),
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )
    result.name, result.attrs = "theta_e", {"units": "K", "long_name": "equivalent potential temperature"}
    return result


def liquid_water_potential_temperature(
    pressure: xr.DataArray,
    temp: xr.DataArray,
    q_c: xr.DataArray,
) -> xr.DataArray:
    theta = potential_temperature(pressure, temp)
    temperature = _magnitude(temp, "K")
    result = theta * np.exp(-Lv.to("J/kg").magnitude * q_c / (Cp_d.to("J/kg/K").magnitude * temperature))
    result.name, result.attrs = "theta_l", {
        "units": "K",
        "long_name": "liquid-water potential temperature",
    }
    return result


def buoyancy(
    theta_v: xr.DataArray, reference_theta_v: xr.DataArray, gravity: float = GRAVITY
) -> xr.DataArray:
    result = gravity * (theta_v - reference_theta_v) / reference_theta_v
    result.name, result.attrs = "buoyancy", {"units": "m s-2", "long_name": "buoyancy"}
    return result
