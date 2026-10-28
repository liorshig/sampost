Getting started
===============

Environment management with uv
------------------------------

Create the uv environment and install Sampost with its runtime and
documentation dependencies:

.. code-block:: console

   uv sync

Run the test suite:

.. code-block:: console

   uv run --extra test pytest

Build the documentation with warnings treated as errors:

.. code-block:: console

   uv run sampost-docs

Opening SAM output
------------------

``open_sam`` maps common SAM names to Sampost's canonical schema. For example,
``TABL`` becomes ``temp``, ``QV`` becomes ``q_v``, and ``QN`` becomes
``q_n``. ``QN`` contains the combined liquid and ice condensate field.

.. code-block:: python

   import sampost

   ds = sampost.open_sam("snapshot.nc", chunks="auto")
   print(ds.data_vars)

The default dimensions are ``x``, ``y``, ``z``, and ``zW``. Use
``variable_map`` or ``dimension_map`` when a file uses different names.

SAM pressure perturbations
--------------------------

Some SAM outputs store a one-dimensional base pressure ``P`` and a
four-dimensional pressure perturbation ``PP`` (``p'``). ``open_sam`` maps
these to ``p_base`` and ``p_prime`` and creates the absolute pressure field:

.. code-block:: python

   ds = sampost.open_sam("snapshot.nc", chunks="auto")
   absolute_pressure = ds["pressure"]  # p_base + p_prime

The base pressure must be one-dimensional and the two fields must have
matching physical units. `open_sam` converts both to Pa before adding them.
More generally, canonical fields are converted to SI units at ingestion:
temperature to K, velocity to m/s, density to kg/m³, mixing ratios to kg/kg,
and spatial coordinates to metres. Missing or incompatible source units raise
an error.

Lazy execution
--------------

Chunked input produces Dask-backed results. Diagnostic construction only builds
a task graph:

.. code-block:: python

   theta_v = ds.sampost.virtual_potential_temperature()
   print(type(theta_v.data))
   print(len(theta_v.data.__dask_graph__()))

Execute explicitly when the result is needed:

.. code-block:: python

   theta_v = theta_v.compute()
