.. _data_assimilation:

Data Assimilation
=================

SYMFLUENCE can update the states of an in-process conceptual model with
streamflow observations using an Ensemble Kalman Filter (EnKF), and verify
whether doing so adds value. This page covers the design, the assumptions
behind each piece, and how to read the verification output before making
any claim about forecast improvement.

.. contents::
   :local:
   :depth: 2

Running it
----------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: HBV            # or XINANJIANG (stand-alone, daily)
   OPTIMIZATION_METHODS: [data_assimilation]
   RANDOM_SEED: 42

   ENKF_ENSEMBLE_SIZE: 50
   ENKF_PRECIP_PERTURBATION_STD: 0.3      # lognormal, mean-one multiplier
   ENKF_TEMP_PERTURBATION_STD: 1.0        # K, additive
   ENKF_FORCING_CORRELATION_HOURS: 24     # AR(1) e-folding time, 0 = white
   ENKF_OBS_ERROR_TYPE: heteroscedastic   # sigma = a + b * Q_obs
   ENKF_OBS_ERROR_STD: 0.1                # b (relative)
   ENKF_OBS_ERROR_ABS: null               # a in m3/s; null = 1 % of mean Q_obs
   ENKF_OPEN_LOOP: true                   # baseline run alongside the filter
   ENKF_FORECAST_LEAD_DAYS: 10            # 0 = no lead-time forecasts
   ENKF_FORECAST_ISSUE_INTERVAL: 1        # issue every N timesteps
   ENKF_PRECIP_BIAS_STATE: false          # estimate a precipitation multiplier

Observations come from ``observations/streamflow/preprocessed/<domain>_streamflow_processed.csv``
(run ``process_observed_data`` first). Outputs go to
``simulations/<experiment>/data_assimilation/``:

``<experiment>_enkf_results.nc``
   Filter priors per member (``ensemble_streamflow``), the open loop
   (``open_loop_streamflow``), observations, the analysis mask, lead-time
   forecasts (``forecast_streamflow`` on ``issue`` x ``lead`` x ``member``)
   and, if enabled, the precipitation-bias multiplier per member.
``<experiment>_enkf_results_verification.json``
   The added-value summary (below).
``<experiment>_enkf_results_lead_scores.csv``
   Scores by forecast lead time (when ``ENKF_FORECAST_LEAD_DAYS > 0``).

All streamflow values in these files are in the model's units
(mm per timestep); observations are converted with the catchment area.

Design
------

Cycle
~~~~~

At every timestep the ensemble is advanced with perturbed forcing. If an
observation exists for that step (matched by timestamp, never by position),
warm-up is over and the step is an assimilation step, the EnKF updates the
member states. The state vector is augmented with the predicted discharge so
the observation operator is a selection, and the update reaches every
storage through the ensemble cross-covariances. Stochastic (perturbed
observations) and deterministic (DEnKF, Sakov and Oke 2008) variants are
available (``ENKF_FILTER_VARIANT``); multiplicative covariance inflation
through ``ENKF_INFLATION_FACTOR``.

Three things are produced in the same pass, on the same model object:

* the **filter** ensemble; its recorded predictions are *priors*, made
  before the observation at that step is assimilated, so they are honest
  one-step forecasts;
* the **open loop**: the *same* members, the *same* parameter draws and the
  *same* forcing perturbations, never updated. Any difference between the
  filter and the open loop is due to the analyses alone;
* optional **lead-time forecasts**: from the analysis at each issue step the
  ensemble is run ``ENKF_FORECAST_LEAD_DAYS`` ahead with no further updates.

Model-agnostic state interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The driver needs only what :class:`~symfluence.data_assimilation.enkf.ensemble_manager.DAModel`
declares:

``initialize_members()``
   create the members (parameters perturbed by ``ENKF_PARAM_PERTURBATION_STD``
   times each parameter's range);
``get_state()`` / ``set_state(X)``
   the complete prognostic state of all members as an
   ``(n_members, n_state)`` matrix laid out by ``state_specs`` (with
   physical bounds). *Complete* matters: the open loop and the forecasts
   are run by swapping state matrices in and out;
``step(t, forcing)``
   advance all members over step ``t`` with per-member forcing and return
   the predicted discharge, ``(n_members,)`` or ``(n_members, n_gauges)``;
``perturbable_forcing``
   which forcing variables the driver may perturb, multiplicatively or
   additively.

HBV-96 (``jhbv``) and stand-alone Xinanjiang (``jxaj``, JAX) implement it;
with JAX the members are advanced together by one compiled ``vmap`` of the
model's step. Any other model whose state can be read and written in
memory can be added the same way; models that run as external executables
would need restart-file round trips and are not supported.

Forcing perturbations
~~~~~~~~~~~~~~~~~~~~~

Forcing errors are persistent: a storm that is under-caught is under-caught
for its whole duration. Perturbations are therefore AR(1) in time,

.. math::

   z_t = \phi z_{t-1} + \sqrt{1-\phi^2}\,\varepsilon_t,\quad
   \phi = e^{-\Delta t / \tau},\quad z_0 \sim N(0, 1)

with :math:`\tau` = ``ENKF_FORCING_CORRELATION_HOURS``. :math:`z_t` is
standard normal at every step, so the marginals are those of white noise.
Precipitation is multiplied by :math:`\exp(\sigma z_t - \sigma^2/2)`, a
lognormal with expectation one (no wet bias from the perturbation itself);
temperature gets :math:`\sigma_T z_t` added. The whole period's
perturbations are drawn up front and shared by the filter, the open loop and
the forecasts. ``ENKF_FORCING_CORRELATION_HOURS: 0`` restores white noise.

Precipitation bias state
~~~~~~~~~~~~~~~~~~~~~~~~

With ``ENKF_PRECIP_BIAS_STATE: true`` each member carries a log-space
precipitation multiplier :math:`\beta` (initial
:math:`N(-s^2/2, s^2)`, :math:`s` = ``ENKF_PRECIP_BIAS_STD``, so
:math:`E[e^\beta] = 1`) that follows a small random walk
(``ENKF_PRECIP_BIAS_WALK_STD`` per step) and is part of the analysed
state. Persistent under- or over-estimation of discharge then moves the
multiplier instead of being absorbed entirely by the storages. The open
loop keeps the initial draws and the same walk, without updates. Watch the
``precip_bias`` trace: a multiplier that keeps drifting is compensating for
something other than precipitation (evaporation, snow, glaciers, a rating
change).

Observation error
~~~~~~~~~~~~~~~~~

Discharge errors grow with discharge (rating-curve extrapolation, stage
error times rating slope). The default model is heteroscedastic,

.. math::

   \sigma(Q_{obs}) = a + b\,Q_{obs},\qquad R = \mathrm{diag}(\sigma^2)

with :math:`b` = ``ENKF_OBS_ERROR_STD`` (default 0.1, i.e. 10 %) and
:math:`a` = ``ENKF_OBS_ERROR_ABS`` in m3/s (default 1 % of the mean observed
discharge), which keeps zero or near-zero flows from being trusted
absolutely. 10 % is a typical figure for a good rated section in open water;
ice-affected winter flows, out-of-bank floods and recently shifted controls
deserve much larger values. ``relative`` (:math:`\max(bQ, \text{floor})`)
and ``absolute`` (constant) are kept for older configurations. Several
gauges form a vector observation with a diagonal R (independent errors);
the cycle supports it for models that return one prediction per gauge.

What it assumes
---------------

* **Gaussian, linear update.** The EnKF update is optimal for Gaussian
  errors and linear dynamics. Storages are bounded at zero after the update
  and thresholds (field capacity, snow) make the response non-linear, so the
  update is an approximation that can create or destroy water.
* **The ensemble spread is the forecast error.** If the perturbations are too
  small the filter ignores observations; too large and it chases noise.
  Check the spread-error ratio and rank histogram, not only the mean error.
* **Errors are only forcing, parameter and initial-state errors.** Structural
  model error is not represented; with a biased model the filter keeps
  correcting the same error and the corrections decay at the model's
  recession rate.
* **Observation errors are independent in time and between gauges.**
* **Forecasts use observed forcing.** Lead-time forecasts reuse the observed
  forcing (with the ensemble's perturbations) beyond the issue time. They
  measure how long the improved initial state helps; they do not include
  meteorological forecast error. Operational skill with NWP forcing is lower
  and must be verified with the forecast forcing actually available at issue
  time.

How to judge added value
------------------------

Score on a common sample, after warm-up, and against baselines that are
known at issue time. The verification JSON contains, on one sample:

``enkf``
   the filter's one-step priors: CRPS, spread-error ratio, rank histogram,
   RMSE, NSE and KGE of the ensemble mean;
``open_loop``
   the same scores for the never-updated ensemble;
``persistence``
   the last observation held one step (CRPS of a single value is its MAE);
``crpss_vs_open_loop``, ``crpss_vs_persistence``
   :math:`1 - \mathrm{CRPS}/\mathrm{CRPS}_{ref}`; positive means the filter
   is better.

The CRPS is the "fair" ensemble estimator (unbiased for the distribution the
members are drawn from). The spread-error ratio uses
:math:`\sqrt{\overline{\sigma^2}(m+1)/m}` over the RMSE of the ensemble mean,
so a reliable ensemble scores 1; below 1 means under-dispersion, the usual
failure of a filter that has collapsed.

The lead-time CSV has one row per lead: CRPS of the forecast, of the open loop
and MAE of persistence at the same target steps, the two CRPSS, spread-error
ratio, KGE, and bias/MAE/RMSE/NSE plus MSE skill vs persistence and open loop
of the ensemble mean. (The ensemble-mean scores come from
``symfluence.evaluation.forecast_verification.scores``, the same function the
deterministic forecast verification uses; ``diagnostics.point_scores`` is a
thin wrapper around it.)

A claim that assimilation improves forecasts needs, at minimum:

1. **CRPSS vs open loop > 0** at the leads you care about. This isolates the
   value of the updates; the open loop is the fair reference because it
   shares every perturbation.
2. **CRPSS vs persistence > 0.** At short leads on slowly varying rivers,
   persistence is hard to beat; if the filter does not beat it, a simple
   error-update would serve as well.
3. **A spread-error ratio near 1 and a flat rank histogram.** A sharp but
   unreliable ensemble wins on RMSE and loses on CRPS; a U-shaped histogram
   means the ensemble is too narrow (raise the perturbations or inflation).
4. **Skill that decays with lead.** Gains from a better initial state
   should fade at about the recession rate of the updated storage. Skill that
   does not decay usually means bias correction through the state, which
   belongs in the model or the bias state.
5. **Independent periods.** Tune perturbation sizes and observation error on
   one period and report on another; the filter has free parameters too.
6. **Scores in observation space only.** An updated state that fits the gauge
   can still be unphysical (soil moisture pinned at a bound, snow created in
   summer). Inspect the state trajectories as well.

References
----------

* Evensen, G. (2003). The Ensemble Kalman Filter: theoretical formulation and
  practical implementation. *Ocean Dynamics* 53, 343-367.
* Sakov, P. and Oke, P. R. (2008). A deterministic formulation of the ensemble
  Kalman filter. *Tellus A* 60, 361-371.
* Clark, M. P. et al. (2008). Hydrological data assimilation with the
  ensemble Kalman filter: use of streamflow observations to update states in a
  distributed hydrological model. *Advances in Water Resources* 31, 1309-1324.
* Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Q. J. R.
  Meteorol. Soc.* 140, 1917-1923.
* Fortin, V., Abaza, M., Anctil, F. and Turcotte, R. (2014). Why should
  ensemble spread match the RMSE of the ensemble mean? *J. Hydrometeorology*
  15, 1708-1713.
