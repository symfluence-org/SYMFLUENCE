.. _forcing_correction:

Shared Forcing Correction
=========================

Several models carry their own precipitation (and sometimes temperature)
correction as an ordinary calibration parameter, each with different bounds:
GR ``PCORR`` (0.5–2.0) and ``TCORR`` (±3 K), MESH ``PRECIP_MULT`` (0.8–1.6),
Wflow ``PCORR`` (0.8–1.6), SUMMA ``precip_mult`` (0.7–1.6), mHM
``precipCorrectionFactor`` (0.8–1.8), HYPE ``pcurain``/``pcusnow``. Calibrated
together with the model's own parameters, such a multiplier also absorbs
structural error (GR's ``PCORR`` trades off with the ``X2`` exchange term, for
example), so the members of a multi-model ensemble end up driven by different
forcing.

The ``FORCING_CORRECTION_*`` section replaces these with one definition that
every model uses.

.. note::

   Without ``FORCING_CORRECTION_MODE`` nothing changes: each model keeps its
   own correction parameters exactly as before.

Modes
-----

``FORCING_CORRECTION_MODE``:

``off``
   No correction. The models' own precipitation/temperature correction
   parameters are removed from calibration.

``fixed``
   The same values for every model. Give them in the config, or let the
   precipitation multiplier be estimated once from the water balance
   (see below).

``calibrated``
   One common parameter per quantity, with common bounds, calibrated in each
   model's optimization: ``FC_PRECIP_MULT`` (total precipitation),
   ``FC_SNOW_MULT`` (extra snowfall factor on top of it) and ``FC_TEMP_OFFSET``
   (K). ``FORCING_CORRECTION_QUANTITIES`` picks which ones (default
   ``[precip]``). An optional prior penalty keeps the multiplier near a
   plausible value.

In every mode the model's own precipitation and temperature correction
parameters (``PCORR``, ``TCORR``, ``precip_mult``, ``PRECIP_MULT``,
``TEMP_OFFSET``, ``precipCorrectionFactor``, ``pcurain``/``pcusnow``,
``PRECIP_CORR``/``TEMP_CORR``, PRMS/GSFLOW ``rain_adj``/``snow_adj``/``tmax_adj``/``tmin_adj``
and their ``*_cbh_adj`` forms) are superseded: if a
model's ``*_PARAMS_TO_CALIBRATE`` still lists one, it is dropped from the
calibrated set and a warning names it. A snowfall parameter that is also a
model parameter in its own right (SUMMA's ``frozenPrecipMultip``, Raven's
``SNOW_CORR``) is only
superseded when the section sets the snow quantity. ``MESH_PRECIP_MULTIPLIER``
(applied once at MESH preprocessing) also triggers the warning, since it would
stack with the shared correction. Superseding also overrides values set through
``PRMS_PARAMETER_OVERRIDES`` / ``HYPE_EXTRA_PARAMETERS`` for those blocks
(e.g. a glacier ``snow_adj``).

Configuration
-------------

.. code-block:: yaml

   # Calibrate one common precipitation multiplier with a weak prior
   FORCING_CORRECTION_MODE: calibrated
   FORCING_CORRECTION_QUANTITIES: [precip]
   FORCING_CORRECTION_PRECIP_BOUNDS: [0.7, 1.6]
   FORCING_CORRECTION_PRIOR_WEIGHT: 0.05
   FORCING_CORRECTION_PRIOR_PRECIP_MEDIAN: 1.0
   FORCING_CORRECTION_PRIOR_PRECIP_SIGMA: 0.2

.. code-block:: yaml

   # The same fixed multiplier for every model, from the water balance
   FORCING_CORRECTION_MODE: fixed
   FORCING_CORRECTION_PRECIP_MULTIPLIER: water_balance
   FORCING_CORRECTION_PET_METHOD: oudin      # or hamon
   # FORCING_CORRECTION_ET_MM_PER_YEAR: 350  # known actual ET instead of Budyko

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Meaning
   * - ``FORCING_CORRECTION_MODE``
     - unset
     - ``off`` | ``fixed`` | ``calibrated``; unset = per-model behaviour
   * - ``FORCING_CORRECTION_PRECIP_MULTIPLIER``
     - unset
     - Fixed precipitation multiplier, or ``water_balance``. Unset in
       ``fixed`` mode means 1.0.
   * - ``FORCING_CORRECTION_SNOW_MULTIPLIER``
     - unset
     - Fixed snowfall factor on top of the precipitation multiplier
   * - ``FORCING_CORRECTION_TEMP_OFFSET``
     - unset
     - Fixed additive temperature offset (K)
   * - ``FORCING_CORRECTION_QUANTITIES``
     - ``[precip]``
     - Calibrated quantities (``precip``, ``snow``, ``temp``). A quantity not
       calibrated but given a fixed value above is applied as fixed.
   * - ``FORCING_CORRECTION_PRECIP_BOUNDS``
     - ``[0.7, 1.6]``
     - Bounds of ``FC_PRECIP_MULT``
   * - ``FORCING_CORRECTION_SNOW_BOUNDS``
     - ``[0.7, 1.6]``
     - Bounds of ``FC_SNOW_MULT``
   * - ``FORCING_CORRECTION_TEMP_BOUNDS``
     - ``[-3, 3]``
     - Bounds of ``FC_TEMP_OFFSET`` (K)
   * - ``FORCING_CORRECTION_PRIOR_WEIGHT``
     - 0
     - Weight of the prior penalty (0 = no prior)
   * - ``FORCING_CORRECTION_PRIOR_PRECIP_MEDIAN``
     - 1.0
     - Median of the log-normal prior on ``FC_PRECIP_MULT``
   * - ``FORCING_CORRECTION_PRIOR_PRECIP_SIGMA``
     - 0.2
     - Standard deviation of ln(``FC_PRECIP_MULT``)
   * - ``FORCING_CORRECTION_PRIOR_SNOW_SIGMA``
     - 0.2
     - Standard deviation of ln(``FC_SNOW_MULT``) (median 1)
   * - ``FORCING_CORRECTION_PRIOR_TEMP_SIGMA``
     - 1.0
     - Standard deviation (K) of the normal prior on ``FC_TEMP_OFFSET`` (mean 0)
   * - ``FORCING_CORRECTION_ET_MM_PER_YEAR``
     - unset
     - Actual ET for the water-balance estimate
   * - ``FORCING_CORRECTION_PET_METHOD``
     - ``oudin``
     - PET formula for the water-balance estimate (``oudin`` | ``hamon``)

Prior penalty
-------------

In ``calibrated`` mode with ``FORCING_CORRECTION_PRIOR_WEIGHT`` > 0 the
objective (in its maximization form, e.g. KGE) is reduced by

.. math::

   w \cdot \tfrac12 \sum z_i^2, \qquad
   z_P = \frac{\ln(m_P / \tilde m_P)}{\sigma_P}, \quad
   z_S = \frac{\ln m_S}{\sigma_S}, \quad
   z_T = \frac{\Delta T}{\sigma_T}

i.e. a negative log prior (log-normal on the multipliers, normal on the
offset) scaled by ``w``. With ``w = 0.05`` and ``σ_P = 0.2`` a multiplier of
1.3 costs 0.043 KGE. The penalty is applied in the calibration worker; the
model's metrics (``KGE`` etc.) stay unpenalized and the penalty is reported
under ``FORCING_CORRECTION_PRIOR_PENALTY``. It enters single-objective
scores only; multi-objective (NSGA-II, MOEA/D) objectives and native-gradient
(Adam/L-BFGS with autodiff workers) losses do not include it.

Water-balance estimate
----------------------

``FORCING_CORRECTION_PRECIP_MULTIPLIER: water_balance`` estimates the
multiplier once, when calibration starts, over ``CALIBRATION_PERIOD``:

* P and air temperature from the basin-averaged forcing (``pptrate``,
  ``airtemp``; mean over HRUs), daily;
* Q from the processed streamflow observations divided by the catchment area
  (``FIXED_CATCHMENT_AREA`` or the basin shapefile);
* with ``FORCING_CORRECTION_ET_MM_PER_YEAR``: :math:`m = (\bar Q + ET)/\bar P`;
* otherwise actual ET follows the Budyko (1974) curve of the corrected
  precipitation and PET (Oudin or Hamon, as used elsewhere in the code base),
  and :math:`m` solves :math:`m\bar P - E_{Budyko}(m\bar P, \overline{PET}) = \bar Q`.

Storage change, glacier mass loss and inter-basin groundwater exchange are
neglected, so use several years and, in glacierized basins, give an ET or a
fixed value that accounts for the glacier contribution. The inputs and result
are written to ``forcing_correction.json``. Every model reads the same data
and gets the same value.

Supported models
----------------

The common values are translated into each model's own mechanism when
parameter sets are denormalized, so model workers and final evaluations need
no change. Where a model has no correction parameter, the common names pass
through to its worker, which applies them to a private copy of the forcing it
reads (shared forcing is never written).

.. list-table::
   :header-rows: 1
   :widths: 13 22 18 18 29

   * - Model
     - Precipitation
     - Snowfall
     - Temperature
     - Mechanism
   * - GR
     - ``PCORR``
     - –
     - ``TCORR``
     - Scales/offsets ``BasinObs`` before the R run
   * - SUMMA
     - ``precip_mult``
     - ``frozenPrecipMultip``
     - –
     - Rescales ``pptrate`` in copied forcing files (file manager redirected);
       SUMMA's snowfall multiplier in ``trialParams``
   * - MESH
     - ``PRECIP_MULT``
     - –
     - ``TEMP_OFFSET``
     - Private copy of ``MESH_forcing_safe.nc`` from its recorded unscaled source
   * - Wflow
     - ``PCORR``
     - –
     - –
     - Precipitation ``scale`` in the TOML
   * - mHM
     - ``precipCorrectionFactor``
     - –
     - –
     - Corrected ``pre.nc`` in ``forcing_adjusted`` (glacier ice factor kept)
   * - HYPE
     - ``pcurain = m_P − 1``, ``pcusnow = m_P·m_S − 1``
     - ``pcusnow``
     - –
     - ``par.txt``; both entries must exist (``HYPE_EXTRA_PARAMETERS``)
   * - PRMS
     - ``rain_adj`` + ``snow_adj`` (station) or ``rain_cbh_adj`` + ``snow_cbh_adj`` (climate_hru)
     - snow adj = m_P·m_S
     - ``tmax_adj`` + ``tmin_adj`` (or ``*_cbh_adj``)
     - Blocks in ``params.dat``, picked by forcing mode
   * - GSFLOW
     - as PRMS
     - as PRMS
     - as PRMS
     - PRMS blocks in the GSFLOW ``params.dat``
   * - Raven
     - ``PRECIP_CORR``
     - ``SNOW_CORR``
     - ``TEMP_CORR``
     - Gauge ``:RainCorrection`` / ``:SnowCorrection`` (= precip × snow) /
       ``:TemperatureCorrection`` in ``model.rvt``
   * - FUSE
     - forcing copy
     - –
     - forcing copy
     - ``precip`` × m, ``temp`` + ΔT in a copy (per-trial ``sim_input.nc``;
       ``forcing/FUSE_input_corrected`` in forward runs). PET is not
       recomputed. ``RFERR_MLT`` is not used: it only acts under one
       ``RFERR`` decision and is pinned by the constraints file.
   * - CLM
     - forcing copy
     - –
     - forcing copy
     - Corrected copies of the DATM ``clmforc`` files (``PRECTmms`` × m,
       ``TBOT`` + ΔT); ``datm.streams.xml`` repointed
   * - ParFlow-CLM
     - forcing copy / ``precip_scale``
     - –
     - forcing copy / ``temp_offset``
     - Lumped: run-directory copy of ``forcing.1d`` and the Snow-17
       ``z_upper`` flux recomputed from a corrected cache. Basin mode:
       ``precip_scale`` (supersedes ``CLMPARFLOW_BASIN_PRECIP_SCALE``) and a
       new ``temp_offset``; existing chain segments are reused, so change the
       correction with a fresh ``basin_runs`` directory.
   * - HYPE-MODFLOW (``COUPLED_GW``)
     - the land model's (e.g. HYPE ``pcurain``/``pcusnow``)
     - the land model's
     - the land model's
     - Calibration uses the land model's adapter; MODFLOW receives the
       correction through recharge (``MODFLOW`` itself is pass-through)

CLM and ParFlow-CLM have no snowfall multiplier (CLM partitions rain and snow
itself), FUSE has none either. Other models (e.g. Noah-MP, VIC, ngen) have no
correction mechanism: ``off`` and identity values work, anything else raises a
configuration error rather than silently giving that ensemble member
different forcing.

Forward runs
------------

The correction also applies when models are simply run (``workflow run``
without calibration, the run-models step, forecast runs). Before each model
runs, ``symfluence.core.calibration.forcing_correction.apply_forward_forcing_correction``
resolves the values and hands them to the model's ``forcing_correction.apply_forward``
(a module next to its runner), which writes them idempotently into the run
settings or a corrected forcing copy, exactly as a calibration's final
evaluation does:

``fixed``
   The configured (or water-balance) values.
``calibrated``
   The values applied at the best parameter set of the newest calibration of
   that model and experiment (``optimization/<MODEL>/*_<EXPERIMENT_ID>/forcing_correction.json``),
   i.e. what the final evaluation ran with. Without one, a warning is logged
   and calibrated quantities stay uncorrected.
``off``
   Identity, which also resets values an earlier corrected run wrote into the
   settings.

The applied values are logged and written to
``simulations/<EXPERIMENT_ID>/<MODEL>/forcing_correction.json``. GR applies
them inside its runner. Scripts that construct runners directly should call
``apply_forward_forcing_correction(model, config, logger, project_dir, runner_class=...)``
first. With the section absent nothing is touched; settings a previous
corrected run wrote stay until the section is set again (``off`` resets them).

Outputs
-------

``forcing_correction.json`` in the calibration results directory
(``optimization/<MODEL>/<algorithm>_<experiment>/``) records the mode, the
native parameters used, the superseded model parameters, fixed values,
bounds, prior settings, the water-balance inputs (when estimated) and, after
the optimization, the values applied at the best parameter set (``applied``
by common name, ``applied_native`` by model parameter, and
``prior_penalty_at_best``). The best-parameter files carry the native names
(e.g. ``PCORR``) as before.

Relation to source-to-reference forcing corrections
---------------------------------------------------

Correcting a forcing *product* towards a reference product (e.g. NWP
forecasts towards the reanalysis a model was calibrated on) is a separate,
earlier step applied to the forcing files. The two compose: first map the
forcing onto the reference product, then this hydrological (undercatch /
water-balance) correction applies on top, identically for every model.
