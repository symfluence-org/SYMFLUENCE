Getting Started
===============

This guide takes you from a fresh install to a complete simulation: the Bow
River at Banff, Alberta, modeled with SUMMA and ERA5 forcing.

Quick Start
-----------

.. code-block:: bash

   # Create a configuration from a built-in preset
   symfluence project init bow-river

   # Validate it
   symfluence config validate --config config_Bow_at_Banff.yaml

   # Run the full workflow
   symfluence workflow run --config config_Bow_at_Banff.yaml

``symfluence project list-presets`` lists the built-in presets. All data and
results land under the ``SYMFLUENCE_DATA_DIR`` set in your configuration
(default: ``~/symfluence_data``), in ``domain/<DOMAIN_NAME>/``.

The Workflow
------------

Every SYMFLUENCE run follows the same pipeline:

1. **Domain definition** — delineate the watershed and discretize it into HRUs
2. **Data acquisition** — download attributes (DEM, soil, land cover) and forcing data
3. **Preprocessing** — remap forcing to the domain and build model input files
4. **Simulation** — run the hydrological model, with routing where required
5. **Evaluation** — extract streamflow, compute metrics, generate plots

``symfluence workflow run`` executes the whole pipeline. Each stage can also be
run on its own with ``symfluence workflow step <step_name>``;
``symfluence workflow list-steps`` shows all steps and their aliases.

Walkthrough: Bow River at Banff
-------------------------------

The same run as the Quick Start, one stage at a time — useful for
understanding what each stage produces, and for restarting one that fails.

Prerequisites:

- SYMFLUENCE and its model binaries installed (see :doc:`installation`);
  verify with ``symfluence binary doctor``
- ~10 GB free disk space and an internet connection
- 1.5–3 hours end to end, most of it data download

**1. Create and check the configuration**

.. code-block:: bash

   symfluence project init bow-river

This writes ``config_Bow_at_Banff.yaml`` to the current directory. The preset
pins the essentials — domain, gauge, model, forcing, a one-year test period:

.. code-block:: yaml

   DOMAIN_NAME: "Bow_at_Banff"
   POUR_POINT_COORDS: "51.1722/-115.5717"
   HYDROLOGICAL_MODEL: "SUMMA"
   FORCING_DATASET: "ERA5"

Adjust anything you like, then validate:

.. code-block:: bash

   symfluence config validate --config config_Bow_at_Banff.yaml

**2. Set up the project and domain**

.. code-block:: bash

   symfluence workflow step setup_project --config config_Bow_at_Banff.yaml
   symfluence workflow step define_domain --config config_Bow_at_Banff.yaml
   symfluence workflow step discretize_domain --config config_Bow_at_Banff.yaml

``setup_project`` creates the directory tree under
``$SYMFLUENCE_DATA_DIR/domain/Bow_at_Banff/``. ``define_domain`` downloads a
DEM and delineates the catchment from the pour point (5–15 minutes);
``discretize_domain`` splits it into HRUs. Both write shapefiles to
``shapefiles/catchment/``.

**3. Acquire data**

.. code-block:: bash

   symfluence workflow step acquire_attributes --config config_Bow_at_Banff.yaml
   symfluence workflow step acquire_forcings --config config_Bow_at_Banff.yaml

Attributes (elevation, land cover, soil) go to ``attributes/``; forcing data
goes to ``forcing/raw_data/``. This is the long part — 30–90 minutes depending
on network and archive queues. ERA5 needs no credentials; CARRA/CERRA require
CDS API credentials (see :doc:`installation`).

**4. Preprocess**

.. code-block:: bash

   symfluence workflow step model_agnostic_preprocessing --config config_Bow_at_Banff.yaml
   symfluence workflow step model_specific_preprocessing --config config_Bow_at_Banff.yaml

The first step remaps forcing onto the catchment
(``forcing/basin_averaged_forcing/``); the second builds SUMMA's input and
settings files (``forcing/SUMMA_input/``, ``settings/SUMMA/``).

**5. Run and evaluate**

.. code-block:: bash

   symfluence workflow step run_model --config config_Bow_at_Banff.yaml
   symfluence workflow step postprocess_results --config config_Bow_at_Banff.yaml

``run_model`` executes SUMMA and routes the output with mizuRoute
(10–30 minutes for this domain). ``postprocess_results`` extracts streamflow to
``results/``, scores it against the gauge observations and writes the run
summary described in `Checking results`_. A low baseline score is normal — see
:doc:`calibration` for parameter optimization.

If a step fails, check the run log in
``domain/Bow_at_Banff/_workLog_Bow_at_Banff/`` and see :doc:`troubleshooting`.
``symfluence workflow status`` shows how far a run got, and
``symfluence workflow resume`` continues from the last completed step.

Checking results
----------------

Every run that produces streamflow answers "how good is my model?" in
``domain/<DOMAIN_NAME>/results/<EXPERIMENT_ID>/``:

``metrics.json``
   Machine-readable scores, written after ``postprocess_results`` and again
   after the calibration final evaluation (``"stage"`` says which; the latest
   wins, so after calibration it describes the calibrated model).
``metrics.csv``
   The same scores, one row per model and period.
``summary.png``
   One page: hydrograph with the calibration / evaluation / test periods
   shaded (spin-up greyed), flow duration curve and the metrics table.
   Written after postprocessing; ``SUMMARY_REPORT: false`` turns it off.

``symfluence workflow run`` ends with the same scores as a table::

   Run performance (postprocess): Bow_at_Banff / run_1
   Model  Period       KGE    r      alpha  beta   NSE    PBIAS %  n
   SUMMA  calibration  0.612  0.851  1.204  1.081  0.583  +8.1     1826
   SUMMA  evaluation   0.574  0.822  1.251  1.103  0.540  +10.3    1461
   SUMMA  full         0.598  0.840  1.222  1.090  0.566  +9.0     3652

Each configured period (``CALIBRATION_PERIOD``, ``EVALUATION_PERIOD`` and,
where supported, ``TEST_PERIOD``) is scored, plus ``full`` — the whole
overlapping record after spin-up (``SPINUP_PERIOD`` /
``EVALUATION_SPINUP_YEARS``). Daily observations are compared with daily-mean
simulation; when both are sub-daily they are matched at their native step.

The ``metrics.json`` schema (``schema_version: 1``; keys are only added
within a version):

.. code-block:: text

   schema_version, stage ("postprocess" | "calibration"), generated_at,
   domain, experiment_id,
   target:     {variable, station_id, observations_path, observations_found}
   periods:    {spinup, calibration, evaluation, test}   # [start, end] or null
   resolution: "daily" | "native"
   models:
     <MODEL>:
       source:     where the simulation (or final evaluation) came from
       metrics:    {<period>: {KGE, r, alpha, beta, NSE, logNSE, PBIAS,
                               RMSE, n_obs, start, end}}
       benchmarks: {<period>: {<benchmark>: {KGE, KGE_skill}}}
   warnings:   [messages]
   provenance: {symfluence_version, results_csv, final_evaluation,
                run_manifest, run_id}

Missing values are ``null``. Benchmark scores (mean flow, day-of-year
climatology, persistence; fitted on the calibration period) appear when the
benchmark module is available and a calibration period is set.

If the observations are missing, unreadable or do not overlap the
simulation, the run still succeeds but logs a WARNING naming the expected
file (by default
``data/observations/streamflow/preprocessed/<DOMAIN_NAME>_streamflow_processed.csv``,
or ``OBSERVATIONS_PATH``), prints it under the table, and records it in
``warnings``. Run ``process_observed_data`` or set ``OBSERVATIONS_PATH`` and
rerun ``postprocess_results``.

From Python, the same information comes back from the run itself:

.. code-block:: python

   from symfluence import SYMFLUENCE

   result = SYMFLUENCE("config_Bow_at_Banff.yaml").run()
   result.status                       # 'completed', 'partial' or 'failed'
   result.metrics["models"]["SUMMA"]["metrics"]["evaluation"]["KGE"]
   result.outputs["summary_figure"]    # Path to summary.png

Next Steps
----------

- :doc:`calibration` — calibrate the model you just ran
- :doc:`configuration` — the full YAML reference; try other models
  (``HYDROLOGICAL_MODEL: FUSE``, ``GR``, ``HYPE``, ...)
- :doc:`examples` — progressive tutorials from point scale to continental
- :doc:`agent_guide` — drive all of the above with an AI agent
