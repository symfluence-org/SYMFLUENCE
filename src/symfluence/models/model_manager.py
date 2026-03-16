# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model Manager

Lightweight facade that resolves model workflows, runs preprocess/execute/
postprocess/visualize steps, and delegates component lookups to
``ModelRegistry``. Detailed behaviour now lives in the docs (see
``docs/source/architecture`` and ``docs/source/models/*``).
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.core.base_manager import BaseManager
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.registries import R
from symfluence.models.base.protocols import ModelPostProcessor, ModelPreProcessor, ModelRunner
from symfluence.models.utilities.routing_decider import RoutingDecider


class ModelManager(BaseManager):
    """Facade that turns a hydrological model list into an ordered workflow.

    Resolves routing dependencies, then runs preprocess → execute →
    postprocess → visualize phases using components pulled from
    ``ModelRegistry``. Full behaviour is covered in the docs; keeping this
    concise speeds imports.
    """

    # Shared routing decision logic (class-level for efficiency)
    _routing_decider = RoutingDecider()

    def _resolve_model_workflow(self) -> List[str]:
        """Resolve execution order and add implicit routing dependencies."""

        models_str = self.config.model.hydrological_model or ''
        configured_models = [m.strip() for m in str(models_str).split(',') if m.strip()]
        execution_list = []

        # Models that support routing via mizuRoute or dRoute
        # Note: MESH, HYPE, and NGEN have internal routing, so don't need external routing
        routable_models = {'SUMMA', 'FUSE', 'GR'}

        # Determine which routing model to use
        routing_model = self._get_config_value(
            lambda: self.config.model.routing_model,
            default=None
        )
        routing_upper = str(routing_model).upper() if routing_model else ''
        use_droute = routing_upper == 'DROUTE'
        use_troute = routing_upper in ('TROUTE', 'T-ROUTE', 'T_ROUTE')

        # Add groundwater model if configured (e.g. GROUNDWATER_MODEL: MODFLOW)
        gw_model = self._get_config_value(
            lambda: self.config.model.groundwater_model,
            default=None,
        )
        if gw_model:
            gw_upper = str(gw_model).upper()
            if gw_upper not in configured_models:
                configured_models.append(gw_upper)
                self.logger.info(
                    f"Adding groundwater model {gw_upper} to workflow "
                    f"(from GROUNDWATER_MODEL config)"
                )

        for model in configured_models:
            if model not in execution_list:
                execution_list.append(model)

            # Check implicit dependencies (routing) using shared routing decider
            if model in routable_models:
                if self._routing_decider.needs_routing(self.config_dict, model):
                    if use_droute:
                        self._ensure_droute_in_workflow(execution_list, source_model=model)
                    elif use_troute:
                        self._ensure_troute_in_workflow(execution_list, source_model=model)
                    else:
                        self._ensure_mizuroute_in_workflow(execution_list, source_model=model)

        # Ensure MODFLOW runs after its coupling source (e.g. SUMMA)
        if 'MODFLOW' in execution_list:
            coupling_source = self._get_config_value(
                lambda: self.config.model.modflow.coupling_source if self.config.model.modflow else None,
                default=None,
            )
            if coupling_source and coupling_source in execution_list:
                mf_idx = execution_list.index('MODFLOW')
                src_idx = execution_list.index(coupling_source)
                if mf_idx < src_idx:
                    execution_list.remove('MODFLOW')
                    execution_list.insert(src_idx + 1, 'MODFLOW')

            # Ensure routing comes after MODFLOW
            for rt in ('MIZUROUTE', 'DROUTE', 'TROUTE'):
                if rt in execution_list:
                    rt_idx = execution_list.index(rt)
                    mf_idx = execution_list.index('MODFLOW')
                    if rt_idx < mf_idx:
                        execution_list.remove(rt)
                        execution_list.insert(mf_idx + 1, rt)

        # Ensure PARFLOW runs after its coupling source (e.g. SUMMA)
        if 'PARFLOW' in execution_list:
            coupling_source = self._get_config_value(
                lambda: self.config.model.parflow.coupling_source if self.config.model.parflow else None,
                default=None,
            )
            if coupling_source and coupling_source in execution_list:
                pf_idx = execution_list.index('PARFLOW')
                src_idx = execution_list.index(coupling_source)
                if pf_idx < src_idx:
                    execution_list.remove('PARFLOW')
                    execution_list.insert(src_idx + 1, 'PARFLOW')

            # Ensure routing comes after PARFLOW
            for rt in ('MIZUROUTE', 'DROUTE', 'TROUTE'):
                if rt in execution_list:
                    rt_idx = execution_list.index(rt)
                    pf_idx = execution_list.index('PARFLOW')
                    if rt_idx < pf_idx:
                        execution_list.remove(rt)
                        execution_list.insert(pf_idx + 1, rt)

        return execution_list

    def _ensure_mizuroute_in_workflow(self, execution_list: List[str], source_model: str):
        """
        Add mizuRoute to workflow and log routing context.

        Args:
            execution_list: Current list of models to execute (modified in-place)
            source_model: Name of the model that requires routing (e.g., 'SUMMA')
        """
        if 'MIZUROUTE' not in execution_list:
            execution_list.append('MIZUROUTE')
            self.logger.info(f"Automatically adding MIZUROUTE to workflow (dependency of {source_model})")

        # Check if MIZU_FROM_MODEL is set in config
        mizu_from = self._get_config_value(
            lambda: self.config.model.mizuroute.from_model if self.config.model.mizuroute else None,
            default=None
        )
        if not mizu_from:
            # Log the source model (config is immutable, so we can't update it)
            self.logger.info(f"MIZU_FROM_MODEL not set, using {source_model} as source")

    def _ensure_droute_in_workflow(self, execution_list: List[str], source_model: str):
        """
        Add dRoute to workflow and log routing context.

        dRoute is an experimental routing model with AD support for gradient-based
        calibration. It uses mizuRoute-compatible network topology format.

        Args:
            execution_list: Current list of models to execute (modified in-place)
            source_model: Name of the model that requires routing (e.g., 'SUMMA')
        """
        if 'DROUTE' not in execution_list:
            execution_list.append('DROUTE')
            self.logger.info(
                f"Automatically adding DROUTE to workflow (dependency of {source_model}). "
                "Note: dRoute is EXPERIMENTAL."
            )

        # Check if DROUTE_FROM_MODEL is set in config
        droute_from = self._get_config_value(
            lambda: self.config.model.droute.from_model if self.config.model.droute else None,
            default=None
        )
        if not droute_from or droute_from == 'default':
            self.logger.info(f"DROUTE_FROM_MODEL not set, using {source_model} as source")

    def _ensure_troute_in_workflow(self, execution_list: List[str], source_model: str):
        """Add t-route to workflow and log routing context.

        Args:
            execution_list: Current list of models to execute (modified in-place)
            source_model: Name of the model that requires routing (e.g., 'SUMMA')
        """
        if 'TROUTE' not in execution_list:
            execution_list.append('TROUTE')
            self.logger.info(
                f"Automatically adding TROUTE to workflow (dependency of {source_model})"
            )

        troute_from = self._get_config_value(
            lambda: self.config.model.troute.from_model if self.config.model.troute else None,
            default=None
        )
        if not troute_from or troute_from == 'default':
            self.logger.info(f"TROUTE_FROM_MODEL not set, using {source_model} as source")

    def preprocess_models(self, params: Optional[Dict[str, Any]] = None):
        """Preprocess forcing data into model-specific input formats.

        Transforms generic forcing data (from data acquisition) into model-specific
        input formats. Invokes registered preprocessor for each model in resolved
        workflow. Preprocessors handle all model-specific input requirements.

        Preprocessing Workflow:
            1. Resolve model workflow (includes implicit dependencies)
            2. For each model in workflow:
               a. Create model input directory (project_dir/forcing/{MODEL}_input/)
               b. Retrieve preprocessor class from ModelRegistry
               c. Instantiate preprocessor with config, logger, and params
               d. Run preprocessor.run_preprocessing()
            3. Preprocessor outputs go to model-specific input directories

        Model-Specific Preprocessing Examples:
            SUMMA:
            - ERA5 NetCDF → SUMMA forcing file format
            - Time step interpolation/aggregation
            - Unit conversion (SI → SUMMA units)
            - Spatial interpolation to model grid

            FUSE:
            - Catchment-averaged forcing extraction
            - Temporal aggregation to model timestep
            - Unit conversion

            GR (Rainfall-Runoff):
            - Daily precipitation, temperature aggregation
            - Missing value handling

            mizuRoute:
            - Basin delineation and network structure
            - Unit hydrograph parameters
            - Routing network initialization

        Parameter Usage:
            params dict passed to preprocessor for calibration scenarios:
            - Preprocessor may use params to adjust input processing
            - Example: Parameter-dependent unit conversion or scaling
            - If preprocessor doesn't accept params, they're ignored

        Args:
            params: Optional Dict[str, Any] with parameter values
                - Example: {'SAI_SV': 0.3, 'snowCriticalTemp': -1.5}
                - Used for calibration (different param values → different inputs)
                - If None, uses default parameter values from config
                - Preprocessor determines if params are needed (introspection)

        Raises:
            Exception: If preprocessing fails for any model (logged and re-raised)
                - Caught internally with full traceback logged
                - Enables debugging of preprocessing issues

        Side Effects:
            - Creates project_dir/forcing/{MODEL}_input/ directories
            - Generates model-specific input files
            - Logs preprocessing progress and errors to logger

        Examples:
            >>> # Standard preprocessing with default parameters
            >>> manager.preprocess_models()

            >>> # Preprocessing with parameter variations (calibration)
            >>> params = {'param1': 0.5, 'param2': 100.0}
            >>> manager.preprocess_models(params=params)

        Notes:
            - LSTM and similar data-driven models skip preprocessing
            - Registry lookup enables new models without modifying this method
            - Parameter introspection (inspect.signature) handles optional params
            - Errors in preprocessing halt workflow and raise exception

        See Also:
            ModelRegistry.get_preprocessor(): Retrieve preprocessor class
            run_models(): Execute preprocessed models
            postprocess_results(): Extract and standardize results
        """
        self.logger.debug("Starting model-specific preprocessing")

        workflow = self._resolve_model_workflow()
        self.logger.debug(f"Preprocessing workflow order: {workflow}")

        for model in workflow:
            with symfluence_error_handler(
                f"preprocessing model {model}",
                self.logger,
                error_type=ModelExecutionError
            ):
                # Create model input directory
                model_input_dir = self.project_forcing_dir / f"{model}_input"
                model_input_dir.mkdir(parents=True, exist_ok=True)

                # Select preprocessor for this model from registry
                preprocessor_class = R.preprocessors.get(model)

                if preprocessor_class is None:
                    # Models that truly don't need preprocessing (e.g., LSTM)
                    if model in ['LSTM']:
                        self.logger.debug(f"Model {model} doesn't require preprocessing")
                    else:
                        # Only warn if it's a primary model, not a utility like MIZUROUTE which definitely has one
                        self.logger.debug(f"No preprocessor registered for {model} (or not required).")
                    continue

                # Run model-specific preprocessing
                self.logger.debug(f"Running preprocessor for {model}")

                # Check preprocessor signature to determine what arguments to pass
                import inspect
                sig = inspect.signature(preprocessor_class.__init__)
                kwargs = {}

                # Add optional params if supported
                if 'params' in sig.parameters:
                    kwargs['params'] = params

                # Add project_dir and device if preprocessor needs them
                if 'project_dir' in sig.parameters:
                    kwargs['project_dir'] = self.project_dir
                if 'device' in sig.parameters:
                    # Determine device - prefer GPU if available
                    try:
                        import torch
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    except ImportError:
                        device = None
                    kwargs['device'] = device

                preprocessor = preprocessor_class(self.config, self.logger, **kwargs)

                # Call appropriate preprocessing method
                if isinstance(preprocessor, ModelPreProcessor):
                    preprocessor.run_preprocessing()
                else:
                    # Some models like LSTM don't need preprocessing
                    self.logger.debug(f"No run_preprocessing method for {model} preprocessor")

            # Generate preprocessing diagnostics
            if self.reporting_manager:
                with symfluence_error_handler(
                    f"generating preprocessing diagnostics for {model}",
                    self.logger,
                    reraise=False,
                    error_type=ModelExecutionError
                ):
                    self.reporting_manager.diagnostic_model_preprocessing(
                        input_dir=model_input_dir,
                        model_name=model
                    )

        self.logger.info("Model-specific preprocessing completed")



    def run_models(self):
        """Execute models in resolved workflow order.

        Prefers dCoupler graph-based execution when available for multi-model
        workflows (provides conservation checking, spatial remapping, and
        unit conversion). Falls back to sequential registry-based execution
        when dCoupler is not installed or graph execution fails.

        See Also:
            ModelRegistry.get_runner(): Retrieve runner class
            ModelRegistry.get_runner_method(): Get method name
            preprocess_models(): Prepare inputs
            postprocess_results(): Extract and standardize outputs
        """
        self.logger.info("Starting model runs")

        workflow = self._resolve_model_workflow()
        self.logger.info(f"Execution workflow order: {workflow}")

        # Determine coupling mode. Default is sequential (process-based models
        # communicate via files). Use COUPLING_MODE=dcoupler to opt in to
        # graph-based execution for JAX-native or tightly-coupled workflows.
        coupling_mode = self._get_config_value(
            lambda: self.config.model.coupling_mode,
            default='sequential',
        )
        if coupling_mode == 'dcoupler' and len(workflow) > 1:
            if self._try_dcoupler_execution(workflow):
                return

        self._run_sequential(workflow)

    def _try_dcoupler_execution(self, workflow: List[str]) -> bool:
        """Try graph-based execution via dCoupler. Returns True if successful."""
        from symfluence.coupling import INSTALL_SUGGESTION, is_dcoupler_available

        if not is_dcoupler_available():
            self.logger.info(INSTALL_SUGGESTION)
            self.logger.info("Using sequential model execution.")
            return False

        try:
            from symfluence.coupling import CouplingGraphBuilder

            builder = CouplingGraphBuilder()
            graph = builder.build(self.config_dict)
            self.logger.info(
                f"Executing {len(workflow)} models via dCoupler graph "
                f"({len(graph.components)} components, "
                f"{len(graph.connections)} connections)"
            )

            # Execute graph — ProcessComponents handle their own I/O
            graph.forward(
                external_inputs={},
                n_timesteps=1,
                dt=86400.0,
            )

            self.logger.info("dCoupler graph execution completed successfully")

            # Generate conservation diagnostics if enabled
            if self.reporting_manager and getattr(graph, '_conservation', None):
                with symfluence_error_handler(
                    "generating conservation diagnostics",
                    self.logger,
                    reraise=False,
                    error_type=ModelExecutionError
                ):
                    self.reporting_manager.diagnostic_coupling_conservation(
                        graph, self.project_dir / "diagnostics"
                    )

            return True

        except (ImportError, ModuleNotFoundError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            self.logger.warning(
                f"dCoupler graph execution failed: {e}. "
                "Falling back to sequential model execution."
            )
            return False
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.exception(
                f"Unexpected dCoupler graph execution failure: {e}. "
                "Falling back to sequential model execution."
            )
            return False

    def _run_sequential(self, workflow: List[str]):
        """Execute models sequentially using registered runners (legacy path)."""
        self.resolved_executables: List[tuple] = []
        for model in workflow:
            with symfluence_error_handler(
                f"running model {model}",
                self.logger,
                error_type=ModelExecutionError
            ):
                self.logger.info(f"Running model: {model}")
                runner_class = R.runners.get(model)
                if runner_class is None:
                    self.logger.error(f"Unknown hydrological model or no runner registered: {model}")
                    continue

                runner = runner_class(self.config, self.logger, reporting_manager=self.reporting_manager)
                # Collect resolved executables for provenance
                if hasattr(runner, '_resolved_executables'):
                    self.resolved_executables.extend(runner._resolved_executables)
                method_name = R.runners.meta(model).get("runner_method", "run")
                if isinstance(runner, ModelRunner) and hasattr(runner, method_name):
                    getattr(runner, method_name)()
                else:
                    self.logger.error(f"Runner method '{method_name}' not found for model: {model}")
                    continue

            # Generate model output diagnostics
            if self.reporting_manager:
                with symfluence_error_handler(
                    f"generating output diagnostics for {model}",
                    self.logger,
                    reraise=False,
                    error_type=ModelExecutionError
                ):
                    output_dir = self.project_dir / f"{model}_output"
                    if output_dir.exists():
                        output_files = list(output_dir.glob("*.nc"))
                        if output_files:
                            self.reporting_manager.diagnostic_model_output(
                                output_nc=output_files[0],
                                model_name=model
                            )

    def postprocess_results(self):
        """
        Post-process model results using the registry.

        Extracts streamflow and other relevant outputs from model-specific result
        files and converts them to a standardized format for evaluation and comparison.
        After postprocessing, calculates and logs baseline performance metrics.

        The standardized interface expects postprocessors to implement extract_streamflow()
        method, which saves results to: project_dir/results/{experiment_id}_results.csv

        Note:
            Automatically triggers visualization of timeseries results after extraction.
            Falls back to legacy extract_results() method for backward compatibility.
        """
        self.logger.info("Starting model post-processing")

        workflow = self._resolve_model_workflow()

        for model in workflow:
            with symfluence_error_handler(
                f"post-processing model {model}",
                self.logger,
                reraise=False,
                error_type=ModelExecutionError
            ):
                # Get postprocessor class from registry
                postprocessor_class = R.postprocessors.get(model)

                if postprocessor_class is None:
                    continue

                self.logger.info(f"Post-processing {model}")
                # Create postprocessor instance
                postprocessor = postprocessor_class(self.config, self.logger, reporting_manager=self.reporting_manager)

                # Run postprocessing
                # Standardized interface: extract_streamflow is the main entry point
                if isinstance(postprocessor, ModelPostProcessor):
                    postprocessor.extract_streamflow()
                elif hasattr(postprocessor, 'extract_results'):
                    # Legacy fallback — will be removed in a future release
                    self.logger.warning(
                        f"{model} postprocessor uses deprecated extract_results(); "
                        "migrate to extract_streamflow() (ModelPostProcessor Protocol)"
                    )
                    postprocessor.extract_results()
                else:
                    self.logger.warning(f"No extraction method found for {model} postprocessor")
                    continue

        # Note: visualize_timeseries_results is now triggered automatically by extract_streamflow/save_streamflow_to_results

        # Log baseline performance metrics after postprocessing
        self.log_baseline_performance()

        # Generate model comparison overview (default visualization)
        if self.reporting_manager:
            self.reporting_manager.generate_model_comparison_overview(
                experiment_id=self.experiment_id,
                context='run_model'
            )

    def log_baseline_performance(self):
        """Log baseline model performance metrics before calibration.

        Calculates and logs performance metrics comparing simulated vs observed
        streamflow after initial model run. Provides diagnostic snapshot of model
        performance before calibration, enabling users to:
        1. Assess initial model setup quality
        2. Detect configuration issues
        3. Establish baseline for improvement assessment
        4. Identify models needing attention before calibration

        Metrics Calculated:

            **KGE (Kling-Gupta Efficiency)**:
            Formula: ``KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)``
            where r is correlation coefficient, α is ratio of simulated to observed
            std dev, β is ratio of simulated to observed mean.
            Range: [-∞, 1]. KGE >= 0.7 indicates reasonable performance,
            0.5 <= KGE < 0.7 requires calibration, KGE < 0.5 needs significant
            improvements, KGE < 0 is worse than using observed mean.

            **KGE' (Modified KGE)**:
            Symmetric variant for metric comparison. Useful when comparing
            multiple model configurations.

            **NSE (Nash-Sutcliffe Efficiency)**:
            Formula: ``NSE = 1 - (Σ(Qobs-Qsim)² / Σ(Qobs-Qmean)²)``
            Correlation-based metric with range [-∞, 1], similar interpretation
            as KGE. Less sensitive to bias and variability than KGE.

            **Bias (%)**:
            Formula: ``Bias = ((Mean_Sim - Mean_Obs) / Mean_Obs) × 100``
            Positive values indicate model overestimates, negative indicates
            underestimates. Can indicate systematic model errors.

        Data Sources:
            1. Simulation Results: results_dir/{experiment_id}_results.csv
               - Generated by postprocess_results()
               - Contains model discharge columns

            2. Observations: Multiple fallback strategies
               a. Results file column (if 'obs' or 'observed' in column name)
               b. Observations directory: project_dir/observations/streamflow/preprocessed/
               c. External observation file with datetime and discharge columns

        Workflow:
            1. Load simulation results CSV (index=datetime)
            2. Find observation column or load from observations directory
            3. For each simulation column (e.g., 'SUMMA_discharge_cms'):
               a. Align observations and simulation by datetime index
               b. Remove NaN pairs
               c. Calculate metrics (KGE, KGE', NSE, Bias)
               d. Log results with interpretation
            4. Log footer with metrics summary

        Output Format::

            ========================================================
            BASELINE MODEL PERFORMANCE (before calibration)
            ========================================================
              MODELNAME:
                KGE  = 0.7234
                KGE' = 0.7156
                NSE  = 0.6987
                Bias = +5.3%
                Valid data points: 1825
              Note: KGE >= 0.7 indicates reasonable baseline performance
            ========================================================

        Error Handling:
            - Results file not found: Skipped with debug message
            - No observations found: Skipped with debug message
            - Insufficient valid data (<10 points): Logged as warning
            - Metric calculation errors: Caught and logged as debug

        Side Effects:
            - Logs baseline metrics to logger.info()
            - Logs interpretation and recommendations
            - No files created or modified

        Examples:
            >>> # Called automatically by postprocess_results()
            >>> manager.postprocess_results()  # Includes baseline logging

            >>> # Or called directly
            >>> manager.log_baseline_performance()

        Notes:
            - Called automatically by postprocess_results()
            - Requires results file from postprocessing
            - Useful for QA/QC before calibration begins
            - Graceful degradation if data not available
            - KGE interpretation helps understand model biases

        See Also:
            postprocess_results(): Automatically calls log_baseline_performance()
            kge(), kge_prime(), nse(): Metric calculation functions
            evaluation.metrics: Metric library
        """
        with symfluence_error_handler(
            "calculating baseline metrics",
            self.logger,
            reraise=False,
            error_type=ModelExecutionError
        ):
            import numpy as np

            from symfluence.evaluation.metrics import kge, kge_prime, nse

            # Get results file path
            results_file = self.project_dir / "results" / f"{self.experiment_id}_results.csv"

            if not results_file.exists():
                self.logger.debug("Results file not found - skipping baseline performance logging")
                return

            # Load results
            results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)

            # Find observation column in results, or load from observations directory
            obs_col = None
            obs_series = None
            for col in results_df.columns:
                if 'obs' in col.lower() or 'observed' in col.lower():
                    obs_col = col
                    break

            if obs_col is None:
                # Try to load observations from standard location
                obs_dir = self.project_observations_dir / "streamflow" / "preprocessed"
                domain_name = self.config.domain.name
                obs_files = list(obs_dir.glob(f"{domain_name}*_streamflow*.csv")) if obs_dir.exists() else []

                if obs_files:
                    try:
                        obs_df = pd.read_csv(obs_files[0])
                        # Find datetime and discharge columns
                        datetime_col = None
                        discharge_col = None
                        for col in obs_df.columns:
                            if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                                datetime_col = col
                            if 'discharge' in col.lower() or 'flow' in col.lower() or col.lower() == 'q':
                                discharge_col = col

                        if datetime_col and discharge_col:
                            obs_df[datetime_col] = pd.to_datetime(obs_df[datetime_col])
                            obs_series = obs_df.set_index(datetime_col)[discharge_col]
                            obs_series = obs_series.resample('D').mean()  # Resample to daily
                            self.logger.debug(f"Loaded observations from {obs_files[0].name}")
                    except (OSError, KeyError, ValueError, TypeError) as e:
                        self.logger.debug(f"Could not load observations: {e}")

                if obs_series is None:
                    self.logger.debug("No observation data found - skipping baseline metrics")
                    return

            # Find simulation columns (model outputs)
            sim_cols = [c for c in results_df.columns if 'discharge' in c.lower()]

            if not sim_cols:
                self.logger.debug("No simulation columns found in results")
                return

            # Log header
            self.logger.info("=" * 60)
            self.logger.info("BASELINE MODEL PERFORMANCE (before calibration)")
            self.logger.info("=" * 60)

            for sim_col in sim_cols:
                sim_series = results_df[sim_col]

                # Get observations - either from results file column or externally loaded
                if obs_col is not None:
                    obs_aligned = results_df[obs_col]
                    sim_aligned = sim_series
                elif obs_series is not None:
                    # Align observations with simulation by index (datetime)
                    common_idx = sim_series.index.intersection(obs_series.index)
                    if len(common_idx) == 0:
                        self.logger.warning(f"  {sim_col}: No overlapping dates with observations")
                        continue
                    obs_aligned = obs_series.loc[common_idx]
                    sim_aligned = sim_series.loc[common_idx]
                else:
                    continue

                obs = obs_aligned.values
                sim = sim_aligned.values

                # Remove NaN pairs
                valid_mask = ~(np.isnan(obs) | np.isnan(sim))
                obs_clean = obs[valid_mask]
                sim_clean = sim[valid_mask]

                if len(obs_clean) < 10:
                    self.logger.warning(f"  {sim_col}: Insufficient valid data ({len(obs_clean)} points)")
                    continue

                # Calculate metrics
                kge_val = kge(obs_clean, sim_clean, transfo=1)
                kgep_val = kge_prime(obs_clean, sim_clean, transfo=1)
                nse_val = nse(obs_clean, sim_clean, transfo=1)

                # Calculate bias
                mean_obs = np.mean(obs_clean)
                mean_sim = np.mean(sim_clean)
                bias_pct = ((mean_sim - mean_obs) / mean_obs) * 100 if mean_obs != 0 else np.nan

                # Determine model name from column
                model_name = sim_col.replace('_discharge_cms', '').replace('_discharge', '')

                # Log metrics
                self.logger.info(f"  {model_name}:")
                self.logger.info(f"    KGE  = {kge_val:.4f}")
                self.logger.info(f"    KGE' = {kgep_val:.4f}")
                self.logger.info(f"    NSE  = {nse_val:.4f}")
                self.logger.info(f"    Bias = {bias_pct:+.1f}%")
                self.logger.info(f"    Valid data points: {len(obs_clean)}")

                # Provide interpretation
                if kge_val < 0:
                    self.logger.warning("    Note: KGE < 0 indicates model performs worse than mean observed flow")
                elif kge_val < 0.5:
                    self.logger.info("    Note: KGE < 0.5 suggests calibration may significantly improve results")
                elif kge_val >= 0.7:
                    self.logger.info("    Note: KGE >= 0.7 indicates reasonable baseline performance")

            self.logger.info("=" * 60)





    def visualize_outputs(self):
        """
        Visualize model outputs using registered model visualizers.

        Invokes visualization functions for each primary model in the configuration.
        Visualizers are registered per-model and handle model-specific output formats.

        Note:
            Requires reporting_manager to be configured. Skips visualization if not available.
            Each model can register its own visualization function with the ModelRegistry.
        """
        self.logger.info('Starting model output visualisation')

        if not self.reporting_manager:
            self.logger.info("Visualization disabled or reporting manager not available.")
            return

        workflow = self._resolve_model_workflow()
        # Primary models from configuration
        models_str = self.config.model.hydrological_model or ''
        models = [m.strip() for m in str(models_str).split(',') if m.strip()]

        for model in models:
            visualizer = R.visualizers.get(model)
            if visualizer:
                with symfluence_error_handler(
                    f"{model} visualization",
                    self.logger,
                    reraise=False,
                    error_type=ModelExecutionError
                ):
                    self.logger.info(f"Using registered visualizer for {model}")
                    visualizer(
                        self.reporting_manager,
                        self.config_dict,  # Visualizer expects flat dict
                        self.project_dir,
                        self.experiment_id,
                        workflow
                    )
            else:
                self.logger.info(f"Visualization for {model} not yet implemented or registered")

        # Generate Camille's model comparison overview (auto-detects outputs)
        with symfluence_error_handler(
            "generating model comparison overview",
            self.logger,
            reraise=False,
            error_type=ModelExecutionError
        ):
            self.reporting_manager.generate_model_comparison_overview(
                experiment_id=self.experiment_id,
                context='run_model'
            )
