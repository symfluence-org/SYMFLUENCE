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
``precipCorrectionFactor``, ``pcurain``/``pcusnow``) are superseded: if a
model's ``*_PARAMS_TO_CALIBRATE`` still lists one, it is dropped from the
calibrated set and a warning names it. A snowfall parameter that is also a
model parameter in its own right (SUMMA's ``frozenPrecipMultip``) is only
superseded when the section sets the snow quantity. ``MESH_PRECIP_MULTIPLIER``
(applied once at MESH preprocessing) also triggers the warning, since it would
stack with the shared correction.

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
parameter sets are denormalized, so model workers, final evaluations and
model files are unchanged.

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 12 32

   * - Model
     - Precipitation
     - Snowfall
     - Temperature
     - How
   * - GR
     - ``PCORR``
     - –
     - ``TCORR``
     - Scales/offsets ``BasinObs`` before the R run
   * - SUMMA
     - ``precip_mult``
     - ``frozenPrecipMultip``
     - –
     - Rescales ``pptrate`` in copied forcing files; SUMMA's snowfall multiplier
   * - MESH
     - ``PRECIP_MULT``
     - –
     - –
     - Rescales ``PRE`` in the run forcing file
   * - Wflow
     - ``PCORR``
     - –
     - –
     - Precipitation ``scale`` in the TOML
   * - mHM
     - ``precipCorrectionFactor``
     - –
     - –
     - Rescales ``pre.nc`` in the worker's adjusted forcing
   * - HYPE
     - ``pcurain = m_P − 1``, ``pcusnow = m_P·m_S − 1``
     - ``pcusnow``
     - –
     - ``par.txt``; both entries must exist (``HYPE_EXTRA_PARAMETERS``)

Other models have no correction mechanism the calibration can drive. For
them ``off`` (and identity values) work, and any other value raises a
configuration error rather than silently giving that ensemble member
different forcing. Run such members with ``FORCING_CORRECTION_MODE: off`` or
apply the correction to their forcing beforehand.

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
