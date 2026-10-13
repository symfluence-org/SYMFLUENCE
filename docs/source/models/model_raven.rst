=========================================
Raven HBV-EC Model Guide
=========================================

.. warning::
   **EXPERIMENTAL MODULE** - The native Raven adapter is new and currently supports one lumped
   subbasin with elevation-band HRUs. The API may change without notice.

   **Known Limitations:**

   - Single subbasin (``ROUTE_NONE`` + triangular unit hydrograph); no in-Raven channel network
   - Glacier area is fixed; no firn accumulation or area evolution
   - Glacier ice is allocated to the highest elevation bands (``upper_ice_fractions``)
   - Requires the patched Raven build (finite glacier storage), installed by ``symfluence binary install raven``

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

SYMFLUENCE writes native `Raven <https://raven.uwaterloo.ca>`_ input files for an HBV-EC
structure on elevation-band HRUs, with optional explicit glacier HRUs and **finite** glacier ice
storage. Raven is built from ``CSHS-CWRA/RavenHydroFramework`` at a pinned commit with
``patches/finite_glacier_storage.patch``, which exposes Raven's finite-ice option as
``:FiniteGlacierStorage``. Stock Raven does not honour the initial ice storage and is rejected by the
run gate.

Each run is checked for: cumulative mass-balance error below 0.05 mm, initial basin ice equal to
the configured value, and non-negative ice storage in every HRU.

Configuration
=============

.. code-block:: yaml

   model:
     hydrological_model: RAVEN
     routing_model: none
     raven:
       hru_file: /path/to/HRUs_elevation.shp   # elevation-band HRU shapefile
       glacier_fraction: 0.113                 # basin fraction assigned to glacier HRUs
       glacier_ice_mm: 400000.0                # initial ice, mm w.e. per glacier HRU
       params_to_calibrate: RAINSNOW_TEMP,MELT_FACTOR,...,PRECIP_CORR,TEMP_CORR
       param_bounds:                           # optional overrides of the default bounds
         MELT_FACTOR: [2.5, 7]
       glacier_constraints: true               # adds glacier_score to calibration metrics
       glacier_mb_error_scale: 1.0             # m w.e.; glacier_score = 1 - RMSE / scale

Choose ``glacier_ice_mm`` large enough that ice cannot run out during the simulation unless
that is intended: an ice HRU that melts out changes the hydrograph abruptly.

Parameters
==========

=================== ============== =============================================================
Parameter           Default bounds Meaning
=================== ============== =============================================================
``RAINSNOW_TEMP``   -2 – 2 °C      Rain/snow partition temperature
``MELT_FACTOR``     2.3 – 9        Snow degree-day factor (mm/d/K)
``REFREEZE_FACTOR`` 0 – 3          Refreeze degree-day factor (mm/d/K)
``HBV_BETA``        0.5 – 6        HBV soil shape exponent
``SOIL_DEPTH``      0.1 – 3        Top soil depth (m)
``FAST_K``          0.01 – 0.5     Fast reservoir coefficient (1/d)
``SLOW_K``          0.0005 – 0.08  Slow reservoir coefficient (1/d)
``PERC``            0.1 – 10       Percolation to slow reservoir (mm/d)
``TIME_CONC``       0.5 – 15       Triangular unit hydrograph time of concentration (d)
``GLACIER_CORR``    1 – 2.5        Ice melt factor multiplier on ``MELT_FACTOR``
``GLACIER_K``       0.06 – 0.8     Glacier release coefficient (1/d)
``SNOW_CORR``       0.7 – 1.5      Snow-only gauge correction (on top of ``PRECIP_CORR``)
``PRECIP_CORR``     0.9 – 1.5      Rain and snow gauge correction
``TEMP_CORR``       -1 – 2.5 K     Additive gauge temperature correction
=================== ============== =============================================================

Defaults reproduce uncorrected forcing (``PRECIP_CORR`` 1, ``SNOW_CORR`` 1, ``TEMP_CORR`` 0).
``PRECIP_CORR`` and ``TEMP_CORR`` are effective corrections: report them, and do not read them as
estimates of forcing bias.

Glacier mass-balance constraint
===============================

With ``glacier_constraints: true`` the worker reads
``data/observations/cryosphere/glacier_timeseries.csv`` (``;``-separated, columns
``annual_net_MB`` in m w.e. and ``g_area_dyn`` in km², indexed by year, as in LamaH-Ice) and
reports ``glacier_score`` for the calibration period. Combine it with discharge skill through the
composite objective, for example:

.. code-block:: yaml

   optimization:
     metric: COMPOSITE
     composite_metric: {KGE: 1.0, glacier_score: 0.1}

Weights are normalised, so this ranks parameter sets like ``KGE - 0.1 * RMSE_MB``.
