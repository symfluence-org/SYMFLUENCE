=========================================
ParFlow-CLM Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

ParFlow-CLM is ParFlow (variably saturated 3-D subsurface flow with integrated overland flow) compiled with
the Common Land Model 3.0 (Dai et al., 2003) embedded as Fortran modules (Kollet & Maxwell, 2008). CLM
supplies the land-surface energy balance, snow and evapotranspiration; ParFlow receives CLM's net
infiltration each hour and routes water through the soil, the aquifer and over the land surface.

SYMFLUENCE runs it in two ways:

- **Default (lumped placeholder):** a small box domain with basin-averaged forcing, kept for workflow tests.
- **Basin mode (opt-in, ``CLMPARFLOW_BASIN_MODE: true``):** a regular grid over the real basin with D4
  overland slopes to the gauge, unit forcing mapped to cells, glacier cells, and chained water-year segments
  with spin-up. This is the configuration used for the Tungnaá ensemble member.

Installation
============

.. code-block:: bash

   symfluence binary install clmparflow --patched

The patched build checks out ParFlow ``v3.15.0``, applies the three SYMFLUENCE CLM patches, builds hypre
``v2.31.0`` (PFMG preconditioner) and ParFlow with ``-DPARFLOW_HAVE_CLM=ON`` (MPI ``mpi1`` layer when
``mpicc``/``mpif90`` exist). Basin mode needs it:

- **Glacier water.** Upstream ParFlow-CLM never passes water from land-ice tiles to ParFlow, so rain, snowmelt
  and ice melt on glaciers are lost. The patch passes them on, keeps the ice column full (unlimited ice, fixed
  glacier area) and logs melt per tile to ``glacier_ice.txt.<rank>``.
- **Restart read.** The v3.15.0 restart reader skips two snow-age fields the writer stores, so restarts fail.
- **Ice albedo.** The land-ice albedo (0.80/0.55 compiled in) can be set per run.

See ``src/symfluence/models/clmparflow/patches/README.md`` for the details and verification.

Basin mode
==========

Grid
----

The preprocessor rasterises the basin outline, the forcing units and the DEM onto a regular grid in a projected
CRS. Cells at least half inside the outline are active (restricted to the component connected to the outlet);
each takes the forcing unit covering most of it; a cell is glacier when at least half of it is glacier-unit
area. Slopes come from a D4 priority flood from the cell nearest the pour point, each cell sloping towards its
downstream neighbour, so ParFlow's overland flow follows the conditioned drainage network and leaves the domain
across the outlet face. The vertical column has 14 terrain-following layers (70 m): CLM's 10 soil layers (7 m)
over an aquifer. Lateral boundaries and the base are no-flow.

Run chain
---------

Each segment is one October–September water year, started from the previous segment's pressure field and CLM
restart. Spin-up repeats ``CLMPARFLOW_BASIN_SPINUP_YEARS`` (rewinding the calendar) until total storage changes
by less than ``CLMPARFLOW_BASIN_DRIFT_TOL`` of the cycle's precipitation; the production years then run
continuously. The chain is resumable: finished segments (``basin_runs/<segment>/summary.json``) are skipped.
Outlet discharge is ParFlow's overland outflow across the outlet face, written daily to
``simulations/<experiment>/CLMPARFLOW/clmparflow_basin_outlet.csv``.

Configuration
-------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: CLMPARFLOW
   CLMPARFLOW_BASIN_MODE: true
   CLMPARFLOW_NUM_PROCS: 2
   CLMPARFLOW_BASIN_CRS: EPSG:3057
   CLMPARFLOW_BASIN_DX: 1000
   CLMPARFLOW_BASIN_OUTLINE: /path/basin.gpkg          # default: shapefiles/river_basins
   CLMPARFLOW_BASIN_UNITS: /path/units.gpkg            # default: shapefiles/catchment
   CLMPARFLOW_BASIN_UNIT_FIELD: unit_id                # default: HRU_ID
   CLMPARFLOW_BASIN_ICE_FIELD: surface                 # glacier units: surface == ice
   CLMPARFLOW_BASIN_ICE_VALUE: ice
   CLMPARFLOW_BASIN_DEM: /path/dem.tif
   CLMPARFLOW_BASIN_POUR_POINT: /path/gauge.shp
   CLMPARFLOW_BASIN_FORCING: /path/unit_forcing.nc     # default: basin-averaged forcing (hruId, SUMMA names)
   CLMPARFLOW_BASIN_FORCING_UNIT_VARIABLE: unit_id
   CLMPARFLOW_BASIN_FORCING_VARIABLES: {DSWR: FSIN, DLWR: FLIN, APCP: PRE, Temp: TA, UGRD: UV, Press: PRES, SPFH: QA}
   CLMPARFLOW_BASIN_SPINUP_YEARS: [2010, 2011, 2012]
   CLMPARFLOW_BASIN_FIRST_WATER_YEAR: 2013
   CLMPARFLOW_BASIN_LAST_WATER_YEAR: 2023
   # hand-tuned Tungnaá values (WY2013-16)
   CLMPARFLOW_BASIN_PRECIP_SCALE: 1.15
   CLMPARFLOW_BASIN_ICE_ALBEDO: [0.20, 0.10]
   CLMPARFLOW_BASIN_GLACIER_BED_K: 0.001               # m/h, CLM layers of glacier cells
   CLMPARFLOW_BASIN_AQUIFER_K: 1.25                    # m/h
   CLMPARFLOW_BASIN_MANNING: 6.0e-6                    # h m^-1/3

Forcing must be hourly per unit (W m\ :sup:`-2`, mm s\ :sup:`-1`, K, m s\ :sup:`-1`, Pa, kg kg\ :sup:`-1`); wind
speed goes in ``UGRD``. ``CLMPARFLOW_BASIN_INITIAL_STATE`` starts the spin-up from a finished segment with the
same parameters instead of a cold start (water table ``CLMPARFLOW_BASIN_INITIAL_WATER_TABLE_DEPTH`` below the
surface).

Tungnaá at Maríufossar
----------------------

On a 1 km grid (1,140 cells, 139 glacier) with the values above, one spin-up cycle (drift −0.6 % of
precipitation) and 2 ranks at ~2 h per simulated year: KGE 0.867 (WY2013–16), 0.880 (WY2017–21), 0.680
(WY2022–23); glacier net balance r 0.82, RMSE 0.53 m w.e. against LamaH-Ice, under-melting in warm years.
The setting that mattered most is the near-impermeable glacier bed: without it, melt recharges the aquifer
and drains in autumn instead of reaching the river in summer.

Limitations
===========

- CLM 3.0, not CTSM: no frozen-soil impedance (ParFlow solves Richards' equation on liquid water only).
- Unlimited ice and fixed glacier area; glacier cells by majority at the grid resolution.
- Soil and aquifer structure is uniform apart from the soil/aquifer split and the glacier bed.
- No automatic calibration in basin mode; each simulated year costs on the order of an hour of computation.
