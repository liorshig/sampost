import dask.array as da
import numpy as np
import xarray as xr
from metpy.calc import potential_temperature as metpy_potential_temperature
from metpy.units import units

import sampost


def test_open_sam_combines_base_and_perturbation_pressure(tmp_path):
    source = xr.Dataset(
        {
            "P": (("z",), [1000.0, 900.0], {"units": "hPa"}),
            "QN": (("z",), [1e-4, 2e-4], {"units": "kg kg-1"}),
            "PP": (
                ("time", "z", "y", "x"),
                np.array([[[[-1.0, -2.0], [-3.0, -4.0]], [[0.5, 1.0], [1.5, 2.0]]]]),
                {"units": "hPa"},
            ),
        },
        coords={"z": [0.0, 1.0]},
    )
    path = tmp_path / "sam.nc"
    source.to_netcdf(path)

    result = sampost.open_sam(path, chunks=None)

    assert "p_base" in result
    assert "p_prime" in result
    assert "q_n" in result
    assert result.p_base.attrs["units"] == "Pa"
    assert result.p_prime.attrs["units"] == "Pa"
    np.testing.assert_allclose(
        result.pressure.isel(time=0, y=0, x=0),
        [99900.0, 90050.0],
    )
    assert result.pressure.attrs["standard_name"] == "air_pressure"


def test_open_sam_requires_units(tmp_path):
    source = xr.Dataset({"TABS": (("z",), [300.0, 301.0])}, coords={"z": [0.0, 1.0]})
    path = tmp_path / "missing_units.nc"
    source.to_netcdf(path)
    with np.testing.assert_raises_regex(ValueError, "missing required units"):
        sampost.open_sam(path, chunks=None)


def test_open_sam_lowercase_p_uses_dimension_aware_mapping(tmp_path):
    base = xr.Dataset(
        {"p": (("z",), [1000.0, 900.0], {"units": "mbar"})},
        coords={"z": [0.0, 1.0]},
    )
    base_path = tmp_path / "base.nc"
    base.to_netcdf(base_path)
    result = sampost.open_sam(base_path, chunks=None)
    assert "p_base" in result
    assert "pressure" not in result
    assert result["p_base"].attrs["units"] == "Pa"
    np.testing.assert_allclose(result["p_base"].values, [100000.0, 90000.0])

    perturb = xr.Dataset(
        {
            "p": (
                ("time", "z", "y", "x"),
                np.full((1, 2, 1, 1), 50.0),
                {"units": "mbar"},
            )
        },
        coords={"time": [0.0], "z": [0.0, 1.0], "y": [0.0], "x": [0.0]},
    )
    full_path = tmp_path / "full.nc"
    perturb.to_netcdf(full_path)
    result = sampost.open_sam(full_path, chunks=None)
    assert "pressure" in result
    assert "p_base" not in result
    assert result["pressure"].attrs["units"] == "Pa"
    np.testing.assert_allclose(result["pressure"].values, 5000.0)

    combined = xr.Dataset(
        {
            "P": (("z",), [1000.0, 900.0], {"units": "mbar"}),
            "PP": (
                ("time", "z", "y", "x"),
                np.array([[[[50.0], [10.0]], [[-50.0], [-10.0]]]]),
                {"units": "mbar"},
            ),
        },
        coords={"z": [0.0, 1.0], "time": [0.0], "y": [0.0, 1.0], "x": [0.0]},
    )
    combined_path = tmp_path / "combined.nc"
    combined.to_netcdf(combined_path)
    result = sampost.open_sam(combined_path, chunks=None)
    assert "p_base" in result
    assert "p_prime" in result
    assert "pressure" in result
    np.testing.assert_allclose(result["p_base"].values, [100000.0, 90000.0])
    np.testing.assert_allclose(result["pressure"].isel(time=0, y=0, x=0), [100050.0, 90000.0])


def test_open_sam_mfdataset_combines_groups_lazily(tmp_path):
    coords = {"time": [0.0], "z": [0.0, 1.0], "y": [0.0], "x": [0.0]}
    kinematic = xr.Dataset(
        {
            "U": (("time", "z", "y", "x"), np.ones((1, 2, 1, 1)), {"units": "m/s"}),
            "W": (("time", "z", "y", "x"), np.ones((1, 2, 1, 1)), {"units": "m/s"}),
        },
        coords=coords,
    )
    thermo = xr.Dataset(
        {
            "TABS": (
                ("time", "z", "y", "x"),
                np.full((1, 2, 1, 1), 300.0),
                {"units": "K"},
            ),
            "QV": (
                ("time", "z", "y", "x"),
                np.full((1, 2, 1, 1), 0.01),
                {"units": "kg/kg"},
            ),
        },
        coords=coords,
    )
    kin_path = tmp_path / "kin.nc"
    thermo_path = tmp_path / "thermo.nc"
    kinematic.to_netcdf(kin_path)
    thermo.to_netcdf(thermo_path)

    result = sampost.open_sam_mfdataset(
        [kin_path], [thermo_path], chunks={"time": 1}
    )

    assert set(result.data_vars) == {"u", "w", "temp", "q_v"}
    assert isinstance(result["u"].data, da.Array)
    assert isinstance(result["temp"].data, da.Array)


def dataset():
    z = np.arange(3.0)
    zw = np.arange(4.0) - 0.5
    shape = (2, 2, 3)
    return xr.Dataset(
        {
            "pressure": (("x", "y", "z"), np.full(shape, 100000.0)),
            "temp": (("x", "y", "z"), np.full(shape, 300.0), {"units": "K"}),
            "q_v": (("x", "y", "z"), np.full(shape, 0.01)),
            "q_n": (("x", "y", "z"), np.full(shape, 1e-4)),
            "rho": (("x", "y", "z"), np.full(shape, 1.0)),
            "w": (("x", "y", "zW"), np.ones((2, 2, 4))),
        },
        coords={"z": z, "zW": zw},
    )


def test_potential_temperature_and_cloud_mask():
    ds = dataset()
    theta = ds.sampost.potential_temperature()
    expected = metpy_potential_temperature(1000 * units.hPa, 300 * units.K).magnitude
    np.testing.assert_allclose(theta, expected)
    assert theta.attrs["units"] == "K"
    assert ds.sampost.cloud_mask().all()


def test_lwp_and_staggered_flux():
    ds = dataset()
    np.testing.assert_allclose(ds.sampost.lwp(), 2e-4)
    result = ds.sampost.resolved_fluxes()
    assert "w_qv" in result
    assert result.w_qv.sizes["z"] == 3
    np.testing.assert_allclose(ds.sampost.total_water_path(), 20.2e-3)


def test_buoyancy_and_cape():
    ds = dataset()
    diagnostics = ds.sampost.cape_cin()
    assert set(diagnostics.data_vars) == {"cape", "cin"}
    np.testing.assert_allclose(diagnostics.cape, 0)


def test_dask_is_lazy():
    ds = dataset().chunk({"x": 1, "y": 1})
    result = ds.sampost.virtual_potential_temperature()
    assert isinstance(result.data, da.Array)
    assert result.data.__dask_graph__() is not None
    assert result.data.npartitions == 4
