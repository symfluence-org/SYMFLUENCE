.. _multivariable_evaluation:

Multi-variable evaluation
=========================

Streamflow alone cannot tell glacier melt from precipitation: a model that
melts too much ice and under-catches precipitation can reproduce the
hydrograph as well as one that gets both right. Checking that compensation
needs the same glacier, snow and evapotranspiration scores for every member
of an ensemble. SYMFLUENCE provides them through two layers:

1. **Output-variable adapters** — each model declares how to read a small set
   of standard variables from its own output files, as basin (area-weighted)
   time series in standard units.
2. **Model-agnostic evaluators** — ``GLACIER_MB``, ``BASIN_SWE``,
   ``BASIN_SCA`` and ``BASIN_ET`` read model output only through the adapters
   and score it with bias-sensitive metrics.

The SUMMA-oriented ``SNOW``, ``ET`` and ``TWS`` evaluators are unchanged. The
model-specific LamaH-Ice cryosphere constraints
(``<MODEL>_CRYOSPHERE_CONSTRAINTS``) are unchanged too.

.. contents::
   :local:
   :depth: 2

Standard variables
------------------

========================= ================================ ==========================================
Variable                  Unit                             Meaning
========================= ================================ ==========================================
``glacier_storage``       m³ w.e.                          total glacier water-equivalent storage
``glacier_area``          km²                              glacier area in the basin
``glacier_mass_balance``  mm w.e. (cumulative, specific)   ``Σ ΔV_t / A_t``; derived from the two
                                                           above, or reported directly (GR)
``swe``                   mm                               basin-mean snow water equivalent
``snow_covered_fraction`` 0–1                              of the ice-free area where the model
                                                           separates ice, otherwise of the basin
``et``                    mm d⁻¹                           basin-mean actual ET (positive = loss)
``discharge``             m³ s⁻¹                           served by the streamflow evaluator and
                                                           ``R.result_extractors``, not adapters
========================= ================================ ==========================================

Support matrix
--------------

``✓`` = provided; ``–`` = the adapter raises ``UnsupportedVariableError``.
``glacier_mass_balance`` is available wherever storage and area are.

=========== ========= ======== ====== ===== ===== ====================================================
Model       glac. MB  glac. A  SWE    SCA   ET    Source (and caveats)
=========== ========= ======== ====== ===== ===== ====================================================
SUMMA       ✓         ✓        ✓      ✓     ✓     ``*_day.nc``: ``basin__GlacierStorage`` (Gt; ice +
                                                  firn + snow on glacier), ``basin__GlacierArea``,
                                                  ``scalarSWE`` (HRU-area weighted if ``HRUarea`` is
                                                  written), ``scalarGroundSnowFraction`` (binary per
                                                  HRU in some builds), ``scalarTotalET``
HYPE        ✓         ✓        ✓      ✓     ✓     ``timeGLCV/GLCA/SNOW/CFSC/EVAP.txt`` + ``GeoData.txt``
                                                  areas; GLCV is bare ice (×0.85); CFSC
                                                  re-normalised to the ice-free area
GR          ✓         ✓        –      –     –     ``GR_glacier_diagnostics.json``: annual (Oct–Sep)
                                                  balances, constant outline area; CemaNeige SWE stays
                                                  in the ``.Rdata``
PRMS        ✓         ✓        ✓      ✓     ✓     ``statvar``: ``basin_gl_storage``, ``basin_pweqv``,
                                                  ``basin_actet``, ``snowcov_area_i``; areas and ice
                                                  HRUs from ``params.dat`` (fixed geometry)
GSFLOW      ✓         ✓        ✓      ✓     ✓     as PRMS (``GSFLOW_PARAMETER_FILE``)
Raven       ✓         ✓        ✓      –     –     ``*_Daily_Average_ByHRU.csv`` (GLACIER_ICE, GLACIER,
                                                  SNOW, SNOW_LIQ) + ``geometry.json``
Wflow       ✓         ✓        ✓      –     –     ``output.csv`` ``SWE`` / ``Ice`` (bare ice; stamps
                                                  shifted −1 day) + ``wflow_staticmaps.nc``
mHM         –         –        ✓      –     ✓     ``mHM_Fluxes_States*.nc`` ``snowpack`` / ``aET`` —
                                                  only when enabled in ``mhm_outputs.nml`` (default
                                                  writes runoff only)
MESH        –         –        –      –     –     not yet (per-unit ``*_D.csv`` + glacier ledger exist;
                                                  see ``mesh/cryosphere.py``)
ParFlow-CLM –         –        –      –     –     not yet (ET is layer 3 of the CLM ``.pfb`` output)
others      –         –        –      –     –     no adapter registered
=========== ========= ======== ====== ===== ===== ====================================================

``symfluence.core.modeling.output_variables.support_matrix()`` returns the
live matrix of registered adapters.

What a model counts as *glacier storage* differs (bare ice only vs ice + firn
+ snow on the glacier). For multi-year geodetic periods the seasonal snow on
the glacier largely cancels; for annual or seasonal balances it does not, so
compare like with like.

Glacier mass balance (``GLACIER_MB``)
-------------------------------------

Observations are periods with a mean specific rate, e.g. the geodetic
``dmdtda`` of Hugonnet et al. (2021) or annual/seasonal glaciological
balances. ``GLACIER_MB_OBS_PATH`` (CSV with any delimiter, or NetCDF) may
use:

* ``start_date`` / ``end_date`` columns, or a Hugonnet ``period`` column
  (``2000-01-01_2020-01-01``), or a balance ``year`` (30 Sep of the previous
  year to 30 Sep, as in LamaH-Ice ``glacier_timeseries.csv``; an unnamed
  first column of years is accepted);
* a rate column: ``dmdtda``, ``mb_rate``, ``annual_net_MB``, ``mass_balance``…;
* optionally an uncertainty (``err_dmdtda``, ``uncertainty``…) and a glacier
  area in km² (``area_km2``, ``g_area_dyn``).

``GLACIER_MB_OBS_UNITS`` is ``m_we_per_yr`` (default) or ``mm_we_per_yr``.

For each period the simulated rate is ``(S(end) − S(start)) / years``, with
the latest simulated state at or before each boundary, no further than
``GLACIER_MB_ENDPOINT_TOLERANCE_DAYS`` (default 1; raise it for GR's annual
stamps). When the row gives an area and the model reports glacier storage,
the storage change is divided by the *observed* area (the operator the
model-specific LamaH-Ice constraints use); otherwise the model's own
cumulative specific balance is differenced. Only periods inside
``CALIBRATION_PERIOD`` (``Calib_*``) and, for final evaluation,
``EVALUATION_PERIOD`` (``Eval_*``) are scored.

Metrics (rates in mm w.e. yr⁻¹):

``GMB_BIAS``, ``GMB_ABS_BIAS``, ``GMB_REL_BIAS``
   mean (simulated − observed) rate, its absolute value, and relative to
   ``|mean observed|``.
``GMB_RMSE``
   over the scored periods.
``GMB_SCORE``
   ``1 − sqrt(mean(((sim − obs) / σ)²))`` with σ the observation uncertainty
   of each period, or ``GLACIER_MB_SCALE`` (default 1000 mm = 1 m w.e. yr⁻¹,
   the scale the model-specific constraints use) when none is given. 1 is
   perfect; 0 means an error of one σ. Maximised.
``GMB_WITHIN_UNCERTAINTY``
   fraction of periods whose error is within σ (when σ is given).

SWE, SCA and ET (``BASIN_SWE``, ``BASIN_SCA``, ``BASIN_ET``)
------------------------------------------------------------

Observed series go in ``BASIN_SWE_OBS_PATH``, ``BASIN_SCA_OBS_PATH`` and
``BASIN_ET_OBS_PATH`` (CSV with a date column, or plain NetCDF averaged over
non-time dimensions). SCA in percent (maximum > 1.5) is divided by 100. The
evaluators return the standard set: ``KGE`` with its components ``r``,
``alpha`` (variability ratio) and ``beta`` (bias ratio), ``NSE``, ``PBIAS``,
``RMSE``, ``MAE`` and ``correlation``.

Use KGE (or its ``beta`` together with ``r``), not correlation, as the
objective: correlation is unchanged by a 50 % bias, so it cannot constrain the
water balance.

When ``HYDROLOGICAL_MODEL`` lists several models, set
``OUTPUT_ADAPTER_MODEL`` to pick the adapter.

Calibrating with several variables
----------------------------------

**Multivariate objective.** ``OPTIMIZATION_TARGET: multivariate`` with
per-variable weights; the default metric of each variable is bias-sensitive
(KGE for streamflow, SWE, SCA and ET; NSE for TWS; ``GMB_SCORE`` for
``GLACIER_MB``). Each term contributes ``w (1 − metric)`` for maximised
metrics and ``w |metric|`` for minimised ones (``GMB_ABS_BIAS``, RMSE,
PBIAS):

.. code-block:: yaml

   OPTIMIZATION_TARGET: multivariate
   OBJECTIVE_WEIGHTS: {STREAMFLOW: 0.6, GLACIER_MB: 0.25, BASIN_SCA: 0.15}
   GLACIER_MB_OBS_PATH: data/observations/glacier/hugonnet_2000_2020.csv

**Composite metric.** Any worker can add a glacier term to the streamflow
metric: workers compute ``GMB_*`` through the ``GLACIER_MB`` evaluator when a
configured objective asks for one, and score the run as failed if it cannot
be computed (no silent fallback to KGE):

.. code-block:: yaml

   OPTIMIZATION_METRIC: COMPOSITE
   COMPOSITE_METRIC: {KGE: 0.7, GMB_SCORE: 0.3}

**NSGA-II front of streamflow vs glacier mass balance.**

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: NSGA-II
   NSGA2_PRIMARY_METRIC: KGE
   NSGA2_SECONDARY_METRIC: GMB_SCORE      # or GMB_ABS_BIAS (minimised)

For SUMMA, whose worker scores objectives per target, use the multi-target
form instead:

.. code-block:: yaml

   NSGA2_MULTI_TARGET: true
   NSGA2_PRIMARY_TARGET: streamflow
   NSGA2_SECONDARY_TARGET: glacier_mb
   NSGA2_SECONDARY_METRIC: GMB_SCORE      # maximised; SUMMA does not transform minimised metrics

Writing an adapter
------------------

Subclass ``OutputVariableAdapter``, list the variables in ``SUPPORTED`` and
implement ``_<variable>(sim_dir) -> pd.Series`` for each; register it with
``@R.output_adapters.add('<MODEL>')`` and lazily in the model's
``register()`` (``R.output_adapters.add_lazy``). Return standard units;
raise ``EvaluationError`` for unreadable output. ``extract()`` handles
validation, fill values, de-duplication and the derived
``glacier_mass_balance``.

References
----------

* Hugonnet, R. et al. (2021). Accelerated global glacier mass loss in the
  early twenty-first century. *Nature*, 592, 726–731.
* Gupta, H.V. et al. (2009). Decomposition of the mean squared error and NSE
  performance criteria. *Journal of Hydrology*, 377, 80–91.
