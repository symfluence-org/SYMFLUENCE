# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model Manager

Lightweight facade that resolves model workflows, runs preprocess/execute/
postprocess/visualize steps, and delegates component lookups to
``ModelRegistry``. Detailed behaviour now lives in the docs (see
``docs/source/architecture`` and ``docs/source/models/*``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from symfluence.core.base_manager import BaseManager
from symfluence.core.exceptions import (
    ModelExecutionError,
    SYMFLUENCEError,
    symfluence_error_handler,
)
from symfluence.core.modeling.utilities.routing_decider import RoutingDecider
from symfluence.core.registries import R


class ModelManager(BaseManager):
    """Facade that turns a hydrological model list into an ordered workflow.

    Resolves routing dependencies, then runs preprocess → execute →
    postprocess → visualize phases using components pulled from
    ``ModelRegistry``. Full behaviour is covered in the docs; keeping this
    concise speeds imports.
    """

    # Shared routing decision logic (class-level for efficiency)
    _routing_decider = RoutingDecider()

    # Models whose runners call the router themselves (spatial orchestrator).
    # The router must still be PREPROCESSED for them (network topology,
    # parameter namelist), but must not run again as its own step.
    INTERNALLY_ROUTED_MODELS = frozenset({'FUSE', 'GR', 'LSTM'})

    def _resolve_model_workflow(self, for_preprocessing: bool = False) -> List[str]:
        """Resolve execution order and add implicit routing dependencies.

        With *for_preprocessing*, a router needed by an internally routed model
        (``INTERNALLY_ROUTED_MODELS``) is included too, so its inputs exist
        when that model's runner calls it.
        """

        models_str = self.config.model.hydrological_model or ''
        configured_models = [m.strip() for m in str(models_str).split(',') if m.strip()]
        execution_list = []

        # Models that need an EXTERNAL routing step (standalone MIZUROUTE/dRoute).
        # FUSE and GR handle routing internally in their runners (like MESH, HYPE).
        # NGEN: CFE has built-in GIUH routing, but SAC-SMA and TOPMODEL do not —
        # so NGEN needs external routing when using those modules.
        routable_models = {'SUMMA'}
        if 'NGEN' in configured_models and self._ngen_needs_external_routing():
            routable_models.add('NGEN')
        if for_preprocessing:
            routable_models |= self.INTERNALLY_ROUTED_MODELS

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

    def _ngen_needs_external_routing(self) -> bool:
        """Check if the active NGEN runoff module lacks built-in routing.

        CFE has a built-in GIUH (Geomorphologic Instantaneous Unit Hydrograph),
        so it doesn't need external routing. SAC-SMA and TOPMODEL produce raw
        runoff that requires an external routing step (mizuRoute or t-route).
        """
        modules_str = self._get_config_value(
            lambda: self.config.model.ngen.modules_selected,
            dict_key='NGEN_MODULES_SELECTED',
        )
        alias_map = {'SAC-SMA': 'SACSMA', 'SNOW-17': 'SNOW17', 'NOAH-OWP': 'NOAH'}
        selected = set()
        for m in str(modules_str).split(','):
            name = m.strip().upper()
            if name:
                selected.add(alias_map.get(name, name))

        has_cfe = 'CFE' in selected
        has_sacsma = 'SACSMA' in selected
        has_topmodel = 'TOPMODEL' in selected

        if has_sacsma or has_topmodel:
            self.logger.info(
                f"NGEN using {'SAC-SMA' if has_sacsma else 'TOPMODEL'} — "
                "no built-in routing; external routing eligible"
            )
            return True
        if has_cfe:
            self.logger.debug("NGEN using CFE with built-in GIUH — external routing not required")
        return False

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

        workflow = self._resolve_model_workflow(for_preprocessing=True)
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
                        # A configured model with no registered preprocessor is almost
                        # always a name-resolution miss (e.g. 'WRF-Hydro' not matching the
                        # 'WRFHYDRO' registry key). Silently skipping leaves the model with
                        # no inputs and a confusing downstream failure — surface it.
                        self.logger.warning(
                            f"No preprocessor registered for model '{model}'; its inputs "
                            f"will not be generated. Check that the model name resolves to a "
                            f"registered preprocessor."
                        )
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

                # Call the preprocessing entry point. We probe for the method directly
                # rather than isinstance(preprocessor, ModelPreProcessor): that Protocol is
                # runtime_checkable and also requires a `model_name` attribute, so a
                # preprocessor that defines run_preprocessing() but skips super().__init__
                # (e.g. PIHM's defensive ctor) would otherwise be silently skipped.
                run_preprocessing = getattr(preprocessor, "run_preprocessing", None)
                if callable(run_preprocessing):
                    # Many preprocessors catch their own exceptions, log them and
                    # return False; treat that as a failure rather than success.
                    if run_preprocessing() is False:
                        raise ModelExecutionError(
                            f"{model} preprocessing failed; see the errors logged above")
                else:
                    # Genuinely no preprocessing entry point (e.g. data-driven models).
                    self.logger.debug(f"{model} preprocessor exposes no run_preprocessing(); skipping")

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
        coupling_mode = self._get_config_value(lambda: self.config.model.coupling_mode, dict_key='COUPLING_MODE')
        if coupling_mode == 'dcoupler' and len(workflow) > 1:
            if self._try_dcoupler_execution(workflow):
                return

        self._run_sequential(workflow)

    def _try_dcoupler_execution(self, workflow: List[str]) -> bool:
        """Try graph-based execution via dCoupler. Returns True if successful."""
        from symfluence.core.modeling.coupling import (
            INSTALL_SUGGESTION,
            build_coupling_graph,
            is_dcoupler_available,
        )

        if not is_dcoupler_available():
            self.logger.info(INSTALL_SUGGESTION)
            self.logger.info("Using sequential model execution.")
            return False

        try:
            graph = build_coupling_graph(self.config_dict)
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
                    if model.upper() in {'LSTM', 'GNN'}:
                        from symfluence.core.exceptions import OptionalDependencyError
                        raise OptionalDependencyError(
                            f"The {model.upper()} model", "ml", dependency="torch"
                        )
                    raise ModelExecutionError(
                        f"Unknown hydrological model or no runner registered: {model}. "
                        f"Registered runners: {', '.join(sorted(R.runners.keys()))}"
                    )

                runner = runner_class(self.config, self.logger, reporting_manager=self.reporting_manager)
                # Collect resolved executables for provenance
                if hasattr(runner, '_resolved_executables'):
                    self.resolved_executables.extend(runner._resolved_executables)
                method_name = R.runners.meta(model).get("runner_method", "run")
                # Probe for the run method directly rather than isinstance(ModelRunner):
                # the Protocol also requires a `model_name` attribute, so a runner that
                # skips super().__init__ would be wrongly reported as missing its method.
                runner_method = getattr(runner, method_name, None)
                if callable(runner_method):
                    runner_method()
                else:
                    raise ModelExecutionError(
                        f"Runner method '{method_name}' not found for model: {model}"
                    )

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
        After postprocessing, scores the run and writes
        ``results/{experiment_id}/metrics.json`` (+ ``metrics.csv`` and, unless
        ``SUMMARY_REPORT: false``, ``summary.png``) -- see
        :meth:`log_baseline_performance`.

        The standardized interface expects postprocessors to implement extract_streamflow()
        method, which saves results to: project_dir/results/{experiment_id}_results.csv

        Note:
            Automatically triggers visualization of timeseries results after extraction.
            Falls back to legacy extract_results() method for backward compatibility.
        """
        self.logger.info("Starting model post-processing")

        workflow = self._resolve_model_workflow()

        # Post-process every model even when one fails, then fail the step:
        # a swallowed failure used to let the workflow write the stage marker
        # and exit 0 with no results.
        failures: List[str] = []
        for model in workflow:
            try:
                self._postprocess_model(model)
            except SYMFLUENCEError as e:
                failures.append(f"{model}: {e}")

        if failures:
            raise ModelExecutionError(
                "Post-processing failed for " + "; ".join(failures)
            )

        # Note: visualize_timeseries_results is now triggered automatically by extract_streamflow/save_streamflow_to_results

        # Score the run: results/<exp>/metrics.json (+ csv, summary figure)
        self.log_baseline_performance()

        # Generate model comparison overview (default visualization)
        if self.reporting_manager:
            self.reporting_manager.generate_model_comparison_overview(
                experiment_id=self.experiment_id,
                context='run_model'
            )

    def _postprocess_model(self, model: str) -> None:
        """Run one model's registered postprocessor (errors raise ModelExecutionError)."""
        with symfluence_error_handler(
            f"post-processing model {model}",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Get postprocessor class from registry
            postprocessor_class = R.postprocessors.get(model)

            if postprocessor_class is None:
                return

            self.logger.info(f"Post-processing {model}")
            postprocessor = postprocessor_class(self.config, self.logger, reporting_manager=self.reporting_manager)

            # Probe for the method directly rather than isinstance(ModelPostProcessor):
            # the runtime_checkable Protocol also requires a `model_name` attribute,
            # so a postprocessor that skips super().__init__ would be wrongly
            # treated as having no entry point.
            extract_streamflow = getattr(postprocessor, 'extract_streamflow', None)
            if callable(extract_streamflow):
                extract_streamflow()
            elif hasattr(postprocessor, 'extract_results'):
                # Legacy fallback — will be removed in a future release
                self.logger.warning(
                    f"{model} postprocessor uses deprecated extract_results(); "
                    "migrate to extract_streamflow() (ModelPostProcessor Protocol)"
                )
                postprocessor.extract_results()
            else:
                self.logger.warning(f"No extraction method found for {model} postprocessor")

    def log_baseline_performance(self) -> Optional[Path]:
        """Score the postprocessed run and write ``results/<exp>/metrics.json``.

        Called by :meth:`postprocess_results`. Every ``*discharge*`` column of
        ``results/<exp>_results.csv`` is scored against the observed
        streamflow on the configured calibration / evaluation / test periods
        and on the full overlapping record after spin-up (KGE with r, alpha,
        beta; NSE; logNSE; PBIAS; RMSE; n_obs; benchmark skill when available).
        See :mod:`symfluence.core.metrics.run_metrics` for the schema.

        Files written (under ``results/<experiment_id>/``):
            - ``metrics.json`` and ``metrics.csv`` -- always, even when the
              observations are missing (then with an explanatory warning);
            - ``summary.png`` -- one-page hydrograph / FDC / metrics figure,
              unless ``SUMMARY_REPORT: false``.

        Missing, unreadable or non-overlapping observations are logged at
        WARNING level naming the expected observation path, and recorded in
        the ``warnings`` list of metrics.json. Never raises: a scoring
        problem must not fail the postprocessing step.

        Returns:
            Path to metrics.json, or None if it could not be written.
        """
        from symfluence.core.metrics import run_metrics as rm

        results_file = self.project_dir / "results" / f"{self.experiment_id}_results.csv"
        out_dir = rm.run_results_dir(self.project_dir, self.experiment_id)
        try:
            doc = rm.build_postprocess_metrics(self.config, self.project_dir, results_file)
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as e:
            self.logger.warning(f"Could not compute run metrics from {results_file}: {e}")
            return None

        for message in doc['warnings']:
            self.logger.warning(f"Run metrics: {message}")

        try:
            metrics_path = rm.write_metrics(doc, out_dir)
        except (OSError, ValueError, TypeError) as e:
            self.logger.warning(f"Could not write run metrics to {out_dir}: {e}")
            return None

        rows = [r for r in rm.summary_rows(doc) if r.get('KGE') is not None]
        if rows:
            lines = ["=" * 60, "RUN PERFORMANCE (postprocessed simulation)", "=" * 60]
            for row in rows:
                lines.append(
                    f"  {row['model']:<12} {row['period']:<12} "
                    f"KGE={row['KGE']:.3f}  NSE={_fmt(row['NSE'])}  "
                    f"PBIAS={_fmt(row['PBIAS'], '+.1f')}%  n={row['n_obs']}"
                )
            lines.append("=" * 60)
            self.logger.info("\n".join(lines))
        self.logger.info(f"Run metrics written to {metrics_path}")

        summary_enabled = self._get_config_value(
            lambda: self.config.evaluation.summary_report, True, dict_key='SUMMARY_REPORT'
        )
        if summary_enabled and doc['models']:
            self._write_summary_figure(doc, results_file, out_dir / rm.SUMMARY_FIGURE_FILENAME)
        return metrics_path

    def _write_summary_figure(self, doc: Dict[str, Any], results_file: Path, output: Path) -> None:
        """Best-effort one-page summary figure (never fails the step)."""
        from symfluence.core.metrics import run_metrics as rm

        try:
            from symfluence.reporting.plotters.run_summary_plotter import plot_run_summary

            results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)
            obs = None
            obs_path = doc['target'].get('observations_path')
            if doc['target'].get('observations_found') and obs_path and Path(obs_path).suffix.lower() == '.csv':
                obs = rm.load_observed_streamflow(Path(obs_path))
            plot_run_summary(results_df, obs, doc, output, logger=self.logger)
            self.logger.info(f"Run summary figure written to {output}")
        except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
            self.logger.warning(f"Could not draw run summary figure {output}: {e}")


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


def _fmt(value: Any, spec: str = '.3f') -> str:
    """Format an optional metric for the log banner."""
    return 'n/a' if value is None else format(value, spec)
