.. _multi_model_comparison:

Multi-model comparison
======================

Comparing hydrological models is only meaningful when every member is given
the same chance and every run can be traced and regenerated. SYMFLUENCE
supports this with four pieces: recorded seeds, a resolved-config and input
fingerprint in the run manifest, an optional per-parameter calibration budget,
and the ``symfluence ensemble`` command that combines and scores members and
tabulates the conditions each one ran under.

Reproducible seeds
------------------

When ``RANDOM_SEED`` is set, calibration behaves exactly as before. When it is
unset, SYMFLUENCE now **draws a seed, logs it and records it** instead of
using unrecorded entropy::

   RANDOM_SEED not set; drew seed 1460593047 for this calibration. Set RANDOM_SEED: 1460593047 to regenerate it.

The seed seeds the optimizer's generator, Python's and numpy's global
generators, and the per-task seeds of parallel evaluations. It is written to

* ``<experiment>_<algorithm>_calibration_conditions.json`` (``random_seed``,
  ``random_seed_source: drawn|config``) in the calibration results directory,
* ``<experiment>_<algorithm>_best_params.json`` (``random_seed``),
* the run manifest ``_workLog_<domain>/run_manifest.json`` (``calibration``).

Setting ``RANDOM_SEED`` to the recorded value regenerates the run (same
machine, same code and inputs).

Run provenance
--------------

Every workflow run writes, next to ``run_manifest.json`` in
``_workLog_<domain>/``:

``resolved_config.yaml``
   The validated configuration with every default filled in (credential-like
   keys masked). It can be passed back to ``symfluence`` to rerun.

``run_manifest.json`` → ``resolved_config.sha256``
   SHA-256 of the canonical JSON of those values (sorted keys). It depends only
   on the resolved values: key order, YAML layout and spelling out a default do
   not change it, so two runs used the same configuration iff the hashes match.

``run_manifest.json`` → ``inputs.forcing`` / ``inputs.observations``
   A fingerprint of ``forcing/basin_averaged_data`` and the processed
   observations: file count, total bytes and a SHA-256. ``PROVENANCE_INPUT_HASH``
   chooses how:

   * ``listing`` (default) hashes relative path, size and modification time of
     each file -- cheap even for decades of gridded forcing;
   * ``content`` hashes every byte -- immune to copies that change mtimes, but
     reads all the data;
   * ``none`` skips it.

The DEM is hashed by content as before.

Equal calibration budgets
-------------------------

``NUMBER_OF_ITERATIONS`` is one number for all models, so a 4-parameter GR
model and a SUMMA set-up with dozens of parameters get the same budget -- far
fewer evaluations per dimension for the larger model. For a comparison, give
every member a budget proportional to its number of calibrated parameters:

.. code-block:: yaml

   CALIBRATION_BUDGET_MODE: per_parameter     # default: fixed (= NUMBER_OF_ITERATIONS)
   EVALUATIONS_PER_PARAMETER: 100
   CALIBRATION_BUDGET_MIN_EVALUATIONS: 500
   CALIBRATION_BUDGET_MAX_EVALUATIONS: 5000   # optional

The budget is ``clip(EVALUATIONS_PER_PARAMETER x n_params, MIN, MAX)``
evaluations, converted to the algorithm's iteration unit: one evaluation per
iteration for single-point searches (DDS, simulated annealing, Nelder-Mead,
basin hopping, Bayesian optimisation), about ``POPULATION_SIZE`` per
generation for population algorithms (PSO, DE, GA, NSGA-II, MOEA/D, ABC,
DREAM, GLUE, CMA-ES, SCE-UA -- an approximation for SCE-UA, whose evaluations
per iteration vary), and Adam/L-BFGS steps for gradient methods. The resolved
budget and the evaluations actually used are recorded in
``*_calibration_conditions.json``.

Combining and comparing members: ``symfluence ensemble``
--------------------------------------------------------

``symfluence ensemble`` (or :func:`symfluence.evaluation.multimodel_ensemble.run_ensemble`)
takes N members' simulated discharge and an observed series and produces:

``equal_weight``
   The member mean.
``skill_weighted``
   A weighted mean whose weights are fitted on the **calibration period
   only**: ``--weighting kge`` (default) uses ``w_i ∝ max(KGE_i, 0)`` (members
   no better than KGE 0 get no weight); ``--weighting inverse_mse`` uses
   ``w_i ∝ 1 / MSE_i``. Weights sum to one; if no member has positive skill they
   fall back to equal weights. Evaluation- and test-period data never touch the
   weights, so the combination's evaluation and test scores are out of sample.
``median`` (``--median``)
   The member median.
``vote_weighted``
   A weighted mean with fixed integer ``votes`` per member, given in the
   manifest (only produced when a member has ``votes``).

Every member and combination is scored (KGE, NSE, RMSE, PBIAS) on the
calibration, evaluation and, when set, test period (``TEST_PERIOD``).

Quick use with files on the command line:

.. code-block:: bash

   symfluence ensemble \
       --member GR=gr/q.csv --member HYPE=hype/q.csv --member SUMMA=summa/q.csv \
       --obs obs.csv --calibration-period "2000-01-01, 2009-12-31" \
       --evaluation-period "2010-01-01, 2015-12-31" --median -o comparison/

or take the periods from a SYMFLUENCE config with ``--period-config config.yaml``.
For conditions tables, use a manifest:

.. code-block:: yaml

   # ensemble.yaml -- paths relative to this file
   observations: {file: obs.csv, column: discharge_cms}
   periods:
     calibration: 2000-01-01, 2009-12-31
     evaluation: 2010-01-01, 2015-12-31
     test: 2016-01-01, 2020-12-31          # optional
   options: {weighting: kge, median: true, resample: D}
   members:
     - name: GR
       simulated: gr/q.csv                 # CSV (time index + discharge column) or NetCDF
       column: q_sim                       # optional; auto-detected
       results_dir: domain_X/optimization/GR/dds_run1
       votes: 2                            # optional
     - name: SUMMA
       simulated: summa/out.nc
       variable: averageRoutedRunoff
       select: {gru: 0}
       config: summa/config.yaml
       conditions: {n_hrus: 31}            # explicit values win

.. code-block:: bash

   symfluence ensemble --manifest ensemble.yaml -o comparison/

Outputs in the output directory:

* ``ensemble_scores.csv`` -- one row per member/combination, columns
  ``<period>_<metric>``;
* ``comparison_conditions.csv`` -- per member: model, forcing dataset, forcing
  correction, discretisation, number of HRUs, calibration algorithm, budget
  mode, evaluations used, number of parameters, seed and config hash;
* ``ensemble_comparison.json`` -- periods, weights (with
  ``weights_fitted_on: calibration``), scores and conditions;
* ``ensemble_series.csv`` -- observed, members and combinations on the common
  time steps;
* ``ensemble_comparison.png`` -- KGE per member/combination and period.

The conditions are read from the member's ``results_dir`` (one level deep):
``*_calibration_conditions.json`` (algorithm, seed, evaluations, parameters,
budget), ``resolved_config.yaml`` or the ``config`` file (forcing,
discretisation, ``FORCING_CORRECTION_*``), ``run_manifest.json`` (config hash),
``forcing_correction.json`` (the applied shared forcing correction) and a
per-run ``metrics.json`` when present. Anything not recorded is reported as
``unknown`` -- an honest table is the point.

Checklist for a fair comparison
-------------------------------

* Same forcing (or the same shared ``FORCING_CORRECTION``), same observation
  file, same periods and spin-up for every member.
* ``CALIBRATION_BUDGET_MODE: per_parameter`` and the same algorithm.
* Fit any ensemble weights on the calibration period only, and judge members
  and combinations on the evaluation/test periods.
* Keep each member's ``run_manifest.json``, ``resolved_config.yaml`` and
  ``*_calibration_conditions.json`` with its results; compare config hashes and
  input fingerprints before comparing scores.
