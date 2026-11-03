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

Optional structure
------------------

Three options, all off by default (unset, every input file is byte-identical to before). Geometry is applied by
the preprocessing and stored with the grid; the factors are run parameters.

.. code-block:: yaml

   CLMPARFLOW_BASIN_GRID_DIR: /path/grid_dir        # keep an option grid apart from settings/CLMPARFLOW/basin
   # (a) aquifer anisotropy: multipliers of the aquifer K along grid x, grid y and vertical
   CLMPARFLOW_BASIN_K_TENSOR: [1.0, 0.033, 0.1]
   CLMPARFLOW_BASIN_GRID_ROTATION: 45                # grid x axis 45 deg counter-clockwise from east
   # (b) permeability zone
   CLMPARFLOW_BASIN_K_ZONE: /path/zone.gpkg
   CLMPARFLOW_BASIN_K_ZONE_FACTOR: 0.135             # aquifer K x factor in the zone
   CLMPARFLOW_BASIN_K_ZONE_SOIL: false               # true: also the soil K of non-glacier cells
   # (c) lake
   CLMPARFLOW_BASIN_LAKE: /path/lake.gpkg
   CLMPARFLOW_BASIN_LAKE_MANNING: 3.0e-4             # h m^-1/3 on lake cells

- **Anisotropy** writes cell-wise multipliers ``kx.pfb``, ``ky.pfb``, ``kz.pfb`` (``Perm.TensorType
  TensorByFile``): ``K_TENSOR`` in the aquifer layers, 1 in the CLM soil layers. ParFlow's tensor is diagonal in
  the grid axes. A horizontal anisotropy whose major axis is at an angle θ to grid x has the off-diagonal term
  (K1 − K2) sin θ cos θ, which ParFlow cannot hold; the best grid-aligned approximation keeps the diagonal,
  kx = K1 cos²θ + K2 sin²θ and ky = K1 sin²θ + K2 cos²θ (at 45°, kx = ky = (K1 + K2)/2: the horizontal
  anisotropy disappears). ``CLMPARFLOW_BASIN_GRID_ROTATION`` instead rotates the whole grid (about the basin
  centroid) so grid x follows the strike; the outline, units, DEM, pour point and polygons are sampled with the
  rotated transform, so ParFlow still sees a regular grid. A rotated grid cannot restart from an unrotated run.
- **K zone**: cells at least half inside the polygon; their aquifer K (and with ``K_ZONE_SOIL`` the soil K of
  non-glacier cells) is multiplied by the factor in ``perm.pfb``.
- **Lake**: cells at least half inside the polygon are levelled to the lake cells' median DEM elevation before the
  priority flood (so they slope at ``CLMPARFLOW_BASIN_MIN_SLOPE`` towards the lake outlet) and get their own
  Manning's n (``Mannings.Type PFBFile``). ParFlow's kinematic ``OverlandFlow`` boundary moves ponded water only
  down each cell's prescribed slope (q = √S / n · h\ :sup:`5/3` per unit width), so an unfilled depression is a
  sink that never spills. A rougher lake surface instead holds the ponded depth that conveys the lake's
  throughput: a cascade of nonlinear reservoirs whose storage and delay grow with n. It is not a level pool
  (depth grows towards the lake outlet, exponent 5/3) and has no open-water evaporation.

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
- Soil and aquifer structure is uniform apart from the soil/aquifer split, the glacier bed and the optional K
  zone and grid-aligned anisotropy.
- No automatic calibration in basin mode; each simulated year costs on the order of an hour of computation.
