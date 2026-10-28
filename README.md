# Sampost

Sampost provides lazy `xarray` diagnostics for System for Atmospheric Modeling
(SAM) NetCDF output. It standardizes common SAM fields at ingestion and keeps
calculations compatible with Dask-backed arrays.

```python
import sampost

ds = sampost.open_sam("snapshot.nc")
theta_v = ds.sampost.virtual_potential_temperature()
cloud = ds.sampost.cloud_mask(threshold=1e-5)
fluxes = ds.sampost.resolved_fluxes()
cape_cin = ds.sampost.cape_cin()
```

`open_sam` is lazy when called with `chunks="auto"` (the default). For
separate SAM kinematic and thermodynamic file groups, use:

```python
ds = sampost.open_sam_mfdataset(
    files_kinematic,
    files_thermo,
    chunks=chunk_dict,
    parallel=True,
    data_vars="minimal",
)
```

This opens both groups lazily, applies canonical naming and SI conversion to
each input file, and merges them without loading the data into memory.

The canonical schema uses `temp`, `q_v`, `q_n`, `pressure`, `rho`, `w`, and
coordinates/dimensions `x`, `y`, `z`, and `zW`. `open_sam` maps common names
such as `TABL` and `QVBL`; custom maps can be supplied with `variable_map` and
`dimension_map`.

For SAM snapshots containing a one-dimensional base pressure `P` and a
four-dimensional pressure perturbation `PP` (the pressure fluctuation `p'`),
`open_sam` retains them as `p_base` and `p_prime` and creates absolute
`pressure = p_base + p_prime`. The two fields must use matching units.
`q_n` is SAM's combined non-vapor condensate mixing ratio: liquid plus ice.
Sampost uses this combined condensate for cloud masks, water paths, and the
first liquid-water-potential-temperature approximation.

`open_sam` converts canonical fields to SI units at ingestion: temperature to
K, pressure to Pa, velocity to m/s, density to kg/m³, mixing ratios to
dimensionless kg/kg, and declared spatial coordinates to metres. Source
variables must provide a valid `units` attribute; missing or incompatible units
raise an error instead of being guessed.

MetPy is a required runtime dependency for thermodynamic calculations. Sampost
passes Pint quantities to MetPy inside xarray/Dask-parallelized functions, so
elementwise diagnostics remain lazy. Profile-oriented MetPy routines must be
wrapped with an explicit vertical core-dimension policy.

## Documentation

The full documentation is built with Sphinx in the `uv`-managed environment:

```console
uv sync
uv run sampost-docs
```

Open `docs/_build/html/index.html` after the build. The examples describe how
to use MetPy's sample-data workflow while keeping the documentation build
offline and reproducible.

`lwp()` and `total_water_path()` integrate density-weighted mixing ratios over
`z`. `resolved_fluxes()` first interpolates `w` from `zW` to `z`, then computes
horizontal Reynolds fluxes. `cape_cin()` currently documents a local-parcel
virtual-potential-temperature convention and integrates buoyancy over height;
it requires a height coordinate and should be treated as an initial diagnostic
until a project-specific parcel specification is selected.
