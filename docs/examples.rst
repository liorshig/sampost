Examples
========

MetPy sample-data workflow
--------------------------

MetPy distributes meteorological sample files through its test-data helpers.
The documentation fixture follows the same unit-aware, xarray-oriented
workflow, but is deliberately small and local so that Sphinx builds are
offline and reproducible.

MetPy is installed as a required Sampost dependency. To inspect its sample-data
collection:

.. code-block:: console

   uv sync

The original MetPy sample files can then be accessed with MetPy's
``get_test_data`` helper. The exact file names are part of MetPy's test-data
collection and may vary by MetPy release; keep downloaded files outside the
Sampost source tree.

Thermodynamic diagnostics
-------------------------

.. code-block:: python

   theta = ds.sampost.potential_temperature()
   theta_v = ds.sampost.virtual_potential_temperature()
   theta_e = ds.sampost.equivalent_potential_temperature()
   theta_l = ds.sampost.liquid_water_potential_temperature()

Cloud fields and paths
----------------------

.. code-block:: python

   cloud = ds.sampost.cloud_mask(threshold=1e-5)
   lwp = ds.sampost.lwp()
   twp = ds.sampost.total_water_path()
   cloudy_profile = ds.sampost.conditional_mean(cloud)

Staggered velocity and turbulence
---------------------------------

SAM vertical velocity may be located on ``zW`` while thermodynamic fields are
on ``z``. Sampost interpolates ``w`` to ``z`` before calculating resolved
fluxes:

.. code-block:: python

   fluxes = ds.sampost.resolved_fluxes()
   print(fluxes["w_theta_v"])
   print(fluxes["w_qv"])

The turbulence diagnostics use horizontal slab means over ``x`` and ``y``:

.. code-block:: python

   mean_state = ds.sampost.horizontal_mean()
   fluctuations = ds.sampost.reynolds_decomposition()
   tke = ds.sampost.tke()
