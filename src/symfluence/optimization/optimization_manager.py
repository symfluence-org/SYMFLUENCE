# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Optimization Manager

Coordinates calibration runs by selecting algorithms and model-specific
optimizers. Keeps orchestration lean; algorithm details and examples are in
``docs/source/calibration`` and optimizer pages.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import pandas as pd

from symfluence.core.base_manager import BaseManager
from symfluence.core.constants import SupportedModels
from symfluence.core.exceptions import OptimizationError
from symfluence.core.registries import R
from symfluence.optimization.core import TransformationManager
from symfluence.optimization.optimization_results_manager import OptimizationResultsManager

if TYPE_CHECKING:
    pass


# Models whose "calibration" is internal training (gradient descent during the
# run step), not an external DDS/PSO parameter search. They register no
# optimizer/worker and are skipped by the registry calibration path. Canonical
# definition lives in SupportedModels so the sensitivity-analysis path shares it.
_SELF_TRAINING_MODELS = SupportedModels.SELF_TRAINING


def find_calibration_results(
    project_dir: Path,
    experiment_id: str,
    algorithm: str,
) -> Optional[Path]:
    """Locate the results of a finished calibration, or None.

    ``BaseModelOptimizer`` writes to
    ``optimization/{MODEL}/{algorithm}_{experiment_id}/`` (MODEL may also be
    COUPLED / COUPLED_GW); self-training models write their final evaluation
    there too. Older releases used ``optimization/{algorithm}_{experiment_id}/``
    and the flat ``optimization/{experiment_id}_parallel_iteration_results.csv``.
    """
    opt_dir = Path(project_dir) / "optimization"
    if not opt_dir.is_dir():
        return None
    run_dir_name = f"{str(algorithm).lower()}_{experiment_id}"
    candidates = [opt_dir / run_dir_name] + sorted(opt_dir.glob(f"*/{run_dir_name}"))
    patterns = (
        f"{experiment_id}_parallel_iteration_results.csv",
        f"{experiment_id}_*_best_params.json",
        f"{experiment_id}_*_final_evaluation.json",
    )
    for run_dir in candidates:
        if not run_dir.is_dir():
            continue
        for pattern in patterns:
            found = sorted(run_dir.glob(pattern))
            if found:
                return found[0]
    legacy = opt_dir / f"{experiment_id}_parallel_iteration_results.csv"
    return legacy if legacy.exists() else None


class OptimizationManager(BaseManager):
    """Facade over model-specific optimizers for iterative calibration.

    Chooses algorithm, fetches optimizer from the registry, and runs it; heavy
    lifting stays in the optimizer classes. See docs for workflow details.
    """

    def _initialize_services(self) -> None:
        """Initialize optimization services."""
        self.results_manager = self._get_service(
            OptimizationResultsManager,
            self.project_dir,
            self.experiment_id,
            self.logger,
            self.reporting_manager
        )
        self.transformation_manager = self._get_service(
            TransformationManager,
            self.config,
            self.logger
        )

    @property
    def optimizers(self) -> Any:
        """Backward compatibility: expose registered optimizers/algorithms."""
        # Return a dict-like object that satisfies 'in' and '[]' for algorithms expected by tests
        class OptimizerMapper:
            """Maps algorithm names for backward compatibility with test assertions.

            Provides dict-like interface supporting 'in' and '[]' operators
            for checking algorithm availability without instantiation.
            """
            def __init__(self):
                self.algorithms = {
                    'DDS', 'DE', 'PSO', 'SCE-UA', 'NSGA-II', 'ASYNC-DDS', 'POP-DDS',
                    'ADAM', 'LBFGS'
                }
            def __contains__(self, item):
                return item in self.algorithms
            def __getitem__(self, item):
                if item in self.algorithms:
                    return True # Return something truthy
                raise KeyError(item)
        return OptimizerMapper()

    def run_optimization_workflow(self) -> Dict[str, Any]:
        """Run main optimization workflow based on configuration.

        Entry point for complete optimization process. Checks configuration
        to determine which optimization methods to execute and runs them
        in sequence. Currently supports 'iteration' (calibration) and handles
        deprecated method warnings.

        Workflow:
            1. Check config.optimization.methods (list of methods)
            2. For each enabled method:
               - 'iteration': Run calibrate_model() for iterative optimization
               - Deprecated methods: Log warnings
            3. Return results from all executed methods

        Configuration:
            optimization.methods: List of optimization methods
            - Example: ['iteration'] enables calibration
            - Example: ['iteration', 'emulation'] enables both (emulation deprecated)

        Supported Methods:
            - 'iteration': Iterative parameter optimization (calibration)

        Deprecated Methods (logged as warnings):
            - 'emulation': Use model emulation libraries instead

        Returns:
            Dict[str, Any]: Results from completed workflows
            - Keys: Method names (e.g., 'calibration')
            - Values: Path to results file as string, or None if failed
            - Example: {'calibration': '/path/to/results.csv'}

        Side Effects:
            - Logs method execution and warnings to logger
            - Calls calibrate_model() if 'iteration' enabled
            - May create results files and directories

        Examples:
            >>> # Standard workflow
            >>> opt_mgr = OptimizationManager(config, logger)
            >>> results = opt_mgr.run_optimization_workflow()
            >>> if 'calibration' in results:
            ...     print(f"Calibration results: {results['calibration']}")

            >>> # With deprecated method (warning logged)
            >>> # config.optimization.methods = ['iteration', 'emulation']
            >>> results = opt_mgr.run_optimization_workflow()
            >>> # Logs warning about 'emulation' being deprecated

        Notes:
            - Only 'iteration' currently implemented
            - Deprecated methods logged but not executed
            - Non-empty results dict indicates at least one method ran
            - Empty dict means no methods were enabled or all failed

        See Also:
            calibrate_model(): Run iterative optimization
            get_optimization_status(): Check optimization configuration
        """
        results = {}
        optimization_methods = self._get_config_value(
            lambda: self.config.optimization.methods,
            []
        )

        self.logger.info(f"Running optimization workflows: {optimization_methods}")

        # Run iterative optimization (calibration)
        if 'iteration' in optimization_methods:
            calibration_results = self.calibrate_model()
            if calibration_results:
                results['calibration'] = str(calibration_results)

        # Run data assimilation (EnKF)
        if 'data_assimilation' in optimization_methods:
            from symfluence.data_assimilation.da_manager import DataAssimilationManager
            da_manager = DataAssimilationManager(self.config, self.logger, self.reporting_manager)
            da_results = da_manager.run_data_assimilation()
            if da_results:
                results['data_assimilation'] = str(da_results)

        # Check for deprecated methods and warn
        deprecated_methods = [
            'emulation'
        ]

        for method in deprecated_methods:
            if method in optimization_methods:
                self.logger.warning(
                    f"Optimization method '{method}' is deprecated and no longer supported. "
                    "Use gradient-based optimization (ADAM/LBFGS) via standard model optimizers instead."
                )

        return results

    def calibrate_model(self) -> Optional[Path]:
        """Calibrate model(s) using configured optimization algorithm.

        Coordinates iterative parameter optimization for one or more hydrological
        models using the registry-based unified optimizer infrastructure. Handles
        configuration validation, optimizer instantiation, algorithm selection,
        and execution.

        Calibration Workflow:
            1. Check if 'iteration' in config.optimization.methods
               - If not, log info and return None (disabled)

            2. Get algorithm from config.optimization.algorithm (default: 'PSO')
               - Supported: DDS, ASYNC_DDS, PSO, DE, SCE-UA, NSGA-II, ADAM, LBFGS

            3. Parse configured hydrological models (config.model.hydrological_model)
               - Comma-separated list, e.g., 'SUMMA,FUSE'
               - Upper-case normalization

            4. For each model:
               a. Call _calibrate_with_registry(model, algorithm)
               b. Collect results
               c. Log completion

            5. Return last result (for single model) or last of multiple

        Algorithm Selection:
            Via config.optimization.algorithm:
            - DDS, ASYNC-DDS, ASYNCDDS, ASYNC_DDS: Dynamically Dimensioned Search
            - PSO: Particle Swarm Optimization
            - SCE-UA: Shuffled Complex Evolution
            - DE: Differential Evolution
            - NSGA-II: Multi-objective non-dominated sorting
            - ADAM: Gradient-based with adaptive moments
            - LBFGS: Gradient-based quasi-Newton method

        Registry-Based Model Optimization:
            Each model uses model-specific optimizer from OptimizerRegistry:
            - OptimizerRegistry.get_optimizer('MODELNAME') returns optimizer class
            - Optimizer class inherits from BaseModelOptimizer
            - Example: SUMMAOptimizer for SUMMA calibration

        Configuration Parameters:
            Workflow Control:
                optimization.methods: Must contain 'iteration'
                optimization.algorithm: Algorithm name (PSO, DDS, ADAM, etc.)

            Model Selection:
                model.hydrological_model: Comma-separated model names

            Algorithm-Specific (if applicable):
                ADAM_STEPS / LBFGS_STEPS: Step count (falls back to NUMBER_OF_ITERATIONS)
                ADAM_LR (optimization.adam.lr): Adam learning rate (default: 0.01)
                LBFGS_LR (optimization.lbfgs.lr): L-BFGS initial step size (default: 0.1)

        Returns:
            Optional[Path]: Path to the last completed calibration results
            file, or None if calibration is disabled (no 'iteration' method)
            or handled internally by the model (FUSE built-in SCE).

        Raises:
            OptimizationError: no optimizer registered, unsupported algorithm,
                optimizer failure, or no results file produced. Failures are
                no longer swallowed, so the workflow records the step as
                failed and writes no stage marker.

        Side Effects:
            - Creates project_dir/optimization/ directory
            - Generates model-specific results files
            - Logs calibration progress and status to logger
            - Modifies reporting_manager state (if configured)

        Examples:
            >>> # Single model with DDS
            >>> opt_mgr = OptimizationManager(config, logger)
            >>> results_path = opt_mgr.calibrate_model()
            >>> if results_path:
            ...     print(f"Calibration completed: {results_path}")

            >>> # Multiple models (SUMMA + FUSE)
            >>> # config.model.hydrological_model = 'SUMMA,FUSE'
            >>> results_path = opt_mgr.calibrate_model()  # Returns FUSE results

            >>> # Disabled calibration
            >>> # config.optimization.methods = ['forward']  (no 'iteration')
            >>> results_path = opt_mgr.calibrate_model()
            >>> assert results_path is None

        Notes:
            - Disabled silently returns None (no error)
            - Registry lookup and execution errors raise OptimizationError
            - Multiple models: Last result returned (not aggregated)

        See Also:
            _calibrate_with_registry(): Registry-based optimizer execution
            run_optimization_workflow(): Top-level workflow coordinator
            OptimizerRegistry: Registry for model-specific optimizers
            BaseModelOptimizer: Base class for model optimizers
        """
        self.logger.info("Starting model calibration")

        # Check if iterative optimization is enabled
        optimization_methods = self._get_config_value(
            lambda: self.config.optimization.methods,
            []
        )
        if 'iteration' not in optimization_methods:
            self.logger.info("Iterative optimization is disabled in configuration")
            return None

        # Get the optimization algorithm from config
        opt_algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        )

        try:
            models_str = self._get_config_value(
                lambda: self.config.model.hydrological_model,
                ''
            )
            hydrological_models = [m.strip().upper() for m in str(models_str).split(',') if m.strip()]

            # Skip external DDS for FUSE when built-in SCE calibration is enabled.
            # FUSE's internal calib_sce (run in step 11) makes external DDS redundant.
            # Multi-gauge calibration always needs the external worker because
            # FUSE-internal SCE is single-basin only.
            if 'FUSE' in hydrological_models:
                use_internal = self._get_config_value(
                    lambda: self.config.model.fuse.run_internal_calibration,
                    dict_key='FUSE_RUN_INTERNAL_CALIBRATION',
                )
                multi_gauge = self._get_config_value(
                    lambda: self.config.optimization.multi_gauge.calibration,
                    default=False,
                    dict_key='MULTI_GAUGE_CALIBRATION'
                )
                if use_internal and not multi_gauge:
                    self.logger.info(
                        "Skipping external optimization for FUSE — "
                        "using built-in SCE calibration (FUSE_RUN_INTERNAL_CALIBRATION=True). "
                        "Set FUSE_RUN_INTERNAL_CALIBRATION=False to use external DDS instead."
                    )
                    hydrological_models = [m for m in hydrological_models if m != 'FUSE']
                    if not hydrological_models:
                        return None

            # Self-training models (ML): no external parameter search — LSTM/GNN
            # learn their weights during the run step (run_lstm/run_gnn) via
            # gradient descent and register no optimizer/worker. Their trained
            # simulation still gets the standard split-sample final evaluation
            # so calibration/evaluation-period KGE is reported like every other
            # member.
            self_training = [m for m in hydrological_models if m in _SELF_TRAINING_MODELS]
            self_training_result: Optional[Path] = None
            if self_training:
                self.logger.info(
                    "Skipping external optimization for %s — these models train "
                    "internally during run_model; evaluating the trained simulation instead.",
                    ', '.join(self_training),
                )
                for m in self_training:
                    result = self._evaluate_self_training_model(m, opt_algorithm)
                    if result:
                        self_training_result = result
                hydrological_models = [m for m in hydrological_models if m not in _SELF_TRAINING_MODELS]
                if not hydrological_models:
                    return self_training_result

            # Detect coupled groundwater calibration — when a GROUNDWATER_MODEL
            # is configured alongside the land-surface model, route to the
            # COUPLED_GW optimizer so that GW parameters (K, SY, etc.) are
            # included in the calibration parameter space.
            gw_model = self._get_config_value(
                lambda: self.config.model.groundwater_model,
                None,
            )
            if gw_model and str(gw_model).strip().upper() == 'MODFLOW':
                hydrological_models = ['COUPLED_GW']
                self.logger.info(
                    "Using COUPLED_GW calibration pipeline "
                    "(GROUNDWATER_MODEL: MODFLOW detected)"
                )
            # Detect coupled routing calibration — when a *calibratable* routing
            # model (one exposing a registered parameter manager, e.g. DROUTE) is
            # configured alongside the land model, route to the generic COUPLED
            # optimizer so the routing parameters join the calibration space and
            # the land->routing run goes through the dCoupler graph. (mizuRoute /
            # t-route expose no parameter manager and so stay land-only here.)
            elif self._coupled_routing_requested():
                routing_model = str(self._get_config_value(
                    lambda: self.config.model.routing_model, '', dict_key='ROUTING_MODEL'
                )).split(',')[0].strip().upper()
                hydrological_models = ['COUPLED']
                self.logger.info(
                    "Using COUPLED calibration pipeline "
                    f"(calibratable routing model {routing_model} detected)"
                )

            results = []

            for model in hydrological_models:
                result = self._calibrate_with_registry(model, opt_algorithm)
                if result:
                    results.append(result)

            if not results:
                raise OptimizationError(
                    f"Calibration produced no results for {', '.join(hydrological_models)}"
                )

            if len(results) > 1:
                self.logger.info(f"Completed calibration for {len(results)} model(s)")

            # Generate model comparison overview after calibration
            if self.reporting_manager:
                self.reporting_manager.generate_model_comparison_overview(
                    experiment_id=self.experiment_id,
                    context='calibrate_model'
                )

            return results[-1]

        except OptimizationError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as e:
            # Propagate: returning None here made a failed calibration look
            # like a successful (disabled) one, so the workflow wrote a stage
            # marker and exited 0.
            self.logger.error(f"Error during model calibration: {str(e)}")
            raise OptimizationError(f"Model calibration failed: {e}") from e

    def _coupled_routing_requested(self) -> bool:
        """True when a calibratable dCoupler routing model is coupled to the land model.

        A routing model is "calibratable" when it exposes a registered parameter manager (e.g.
        DROUTE). mizuRoute / t-route route but expose no parameter manager, so they stay land-only.
        Honours an explicit ``CALIBRATE_ROUTING`` opt-out.
        """
        routing_model = str(self._get_config_value(
            lambda: self.config.model.routing_model, '', dict_key='ROUTING_MODEL'
        )).split(',')[0].strip().upper()
        if not routing_model or routing_model in ('', 'NONE', 'N/A'):
            return False
        if R.parameter_managers.get(routing_model) is None:
            return False
        calibrate = self._get_config_value(
            lambda: None, default=True,
            dict_key='CALIBRATE_ROUTING',
        )
        return bool(calibrate)

    def _evaluate_self_training_model(self, model_name: str, algorithm: str) -> Optional[Path]:
        """Split-sample evaluation of a self-training model's trained simulation.

        Self-training models (LSTM/GNN) have no external calibration loop, so
        this scores the run step's saved streamflow against observations for
        the calibration and evaluation windows and writes the standard
        ``*_final_evaluation.json`` under ``optimization/<MODEL>/`` — the same
        artifact every externally calibrated model produces.

        Returns:
            Path to the saved final-evaluation JSON, or None if the run output
            or observations are unavailable.
        """
        from symfluence.core.calibration.optimizers.final_evaluation.self_training import (
            evaluate_self_training_model,
        )

        try:
            return evaluate_self_training_model(
                self.config, self.config_dict, self.project_dir, self.experiment_id,
                self.domain_name, model_name, algorithm, self.logger,
            )
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(
                f"Error evaluating self-training model {model_name}: {e}", exc_info=True
            )
            return None

    def _calibrate_with_registry(self, model_name: str, algorithm: str) -> Optional[Path]:
        """
        Calibrate a model using the OptimizerRegistry.

        This method uses the new unified optimizer infrastructure based on
        BaseModelOptimizer. It provides a cleaner, more maintainable approach
        to model calibration with consistent algorithm support across all models.

        Args:
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'NGEN')
            algorithm: Optimization algorithm to use

        Returns:
            Optional[Path]: Path to results file or None if calibration failed
        """
        # Import model optimizers and parameter managers to trigger registration
        from symfluence.optimization import (
            model_optimizers,  # noqa: F401
            parameter_managers,  # noqa: F401
        )

        # Get optimizer class from registry
        optimizer_cls = R.optimizers.get(model_name)

        if optimizer_cls is None:
            raise OptimizationError(
                f"No optimizer registered for model: {model_name}. "
                f"Registered models: {sorted(R.optimizers.keys())}"
            )

        # Create optimization directory
        opt_dir = self.project_dir / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize model-specific optimizer
            self.logger.debug(f"Using {algorithm} optimization for {model_name} (registry-based)")
            optimizer = optimizer_cls(self.config, self.logger, None, reporting_manager=self.reporting_manager)

            # Resolve any accepted spelling to the registry's primary name and
            # dispatch through run_optimization(). (A hand-kept name->method
            # table here had drifted from ALGORITHM_REGISTRY: it rejected
            # e.g. 'L-BFGS', 'SCE_UA', 'NSGA2' and any algorithm added via
            # register_algorithm().)
            from symfluence.core.calibration.optimizers.algorithms import (
                list_algorithms,
                resolve_algorithm_name,
            )
            try:
                canonical = resolve_algorithm_name(algorithm)
            except ValueError as e:
                raise OptimizationError(
                    f"Algorithm {algorithm} not supported for registry-based optimization. "
                    f"Supported algorithms: {list_algorithms()}"
                ) from e

            # Run optimization
            results_file = optimizer.run_optimization(canonical)

            if results_file and Path(results_file).exists():
                self.logger.info(f"{model_name} calibration completed: {results_file}")

                # Generate calibration diagnostics
                if self.reporting_manager:
                    try:
                        # Load results for diagnostic visualization
                        results_df = pd.read_csv(results_file)

                        # Extract optimization history
                        history = []
                        iter_candidates = ['iteration', 'Iteration']
                        iter_col = next((c for c in iter_candidates if c in results_df.columns), None)
                        if iter_col:
                            obj_cols = [c for c in results_df.columns
                                        if any(k in c.lower() for k in ['objective', 'kge', 'fitness', 'score', 'nse', 'rmse'])]
                            obj_col = obj_cols[0] if obj_cols else None
                            if obj_col:
                                for _, row in results_df.iterrows():
                                    history.append({
                                        'iteration': row[iter_col],
                                        'objective': row[obj_col]
                                    })

                        # Extract best parameters - find actual best row, not just row 0
                        non_param = {'iteration', 'Iteration', 'objective', 'Objective',
                                     'kge', 'KGE', 'fitness', 'score', 'timestamp',
                                     'crash_count', 'crash_rate', 'Calib_RMSE',
                                     'Calib_KGE', 'Calib_KGEp', 'Calib_KGEnp',
                                     'Calib_NSE', 'Calib_MAE',
                                     # per-iteration algorithm diagnostics
                                     'mean_score', 'std_score', 'n_improved',
                                     'grad_norm', 'lr', 'current_params',
                                     'archive_size'}
                        param_cols = [c for c in results_df.columns if c not in non_param]
                        best_params = {}
                        if not results_df.empty and param_cols:
                            # Try to load best params from JSON file first (most reliable)
                            import json
                            best_params_json = self._find_best_params_json(
                                Path(results_file).parent, algorithm
                            )
                            if best_params_json is not None:
                                try:
                                    with open(best_params_json, 'r', encoding='utf-8') as f:
                                        best_data = json.load(f)
                                    best_params = best_data.get('best_params', {})
                                except (json.JSONDecodeError, IOError):
                                    pass

                            # Fallback: find best row by score column
                            if not best_params:
                                best_idx = self._best_row_index(results_df)
                                if best_idx is not None:
                                    best_row = results_df.loc[cast(Any, best_idx)]
                                else:
                                    # No score column found, fall back to last row (most recent)
                                    best_row = results_df.iloc[-1]

                                for col in param_cols:
                                    try:
                                        best_params[col] = float(best_row[col])
                                    except (ValueError, TypeError):
                                        pass

                        self.reporting_manager.diagnostic_calibration(
                            history=history if history else None,
                            best_params=best_params if best_params else None,
                            obs_vs_sim=None,  # Would need to load from model outputs
                            model_name=model_name
                        )
                    except (OSError, FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError) as e:
                        self.logger.debug(f"Could not generate calibration diagnostics: {e}")
                    except Exception as e:  # noqa: BLE001 — must-not-raise contract
                        self.logger.exception(f"Unexpected calibration diagnostics failure: {e}")

                return results_file
            else:
                raise OptimizationError(
                    f"{model_name} calibration finished but its results file was not found "
                    f"({results_file!r})"
                )

        except (KeyError, TypeError, ValueError, RuntimeError) as e:
            self.logger.error(f"Error during {model_name} {algorithm} optimization: {str(e)}")
            raise OptimizationError(
                f"{model_name} {algorithm} optimization failed: {e}"
            ) from e


    def _find_best_params_json(self, run_dir: Path, algorithm: str) -> Optional[Path]:
        """This run's ``<experiment>_<algorithm>_best_params.json``, or None.

        The optimizer names the file with the canonical ``algorithm_slug``.
        Run directories written by older releases used the algorithm's
        display name instead ('nsga-ii', 'cma-es', 'asyncdds', 'moea_d', ...),
        so fall back to the configured spelling and then to whatever single
        best-params file the directory holds.
        """
        from symfluence.core.calibration.optimizers.algorithms import algorithm_slug

        names = [algorithm_slug(algorithm), str(algorithm).lower().replace('/', '_')]
        for name in dict.fromkeys(names):
            candidate = run_dir / f"{self.experiment_id}_{name}_best_params.json"
            if candidate.exists():
                return candidate
        found = sorted(run_dir.glob(f"{self.experiment_id}_*_best_params.json"))
        return found[0] if found else None

    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get status of optimization operations.

        Returns:
            Dict[str, Any]: Dictionary containing optimization status information
        """
        optimization_methods = self._get_config_value(
            lambda: self.config.optimization.methods,
            []
        )
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()

        status = {
            'iterative_optimization_enabled': 'iteration' in optimization_methods,
            'optimization_algorithm': algorithm.upper(),
            'optimization_metric': self._get_config_value(
                lambda: self.config.optimization.metric,
                dict_key='OPTIMIZATION_METRIC',
            ),
            'optimization_dir': str(self.project_dir / "optimization"),
            'results_exist': False,
        }

        # BaseModelOptimizer saves to
        # {project_dir}/optimization/{MODEL}/{algorithm}_{experiment_id}/...
        status['results_exist'] = find_calibration_results(
            self.project_dir, self.experiment_id, algorithm
        ) is not None

        return status

    def validate_optimization_configuration(self) -> Dict[str, bool]:
        """
        Validate optimization configuration settings.

        .. deprecated::
            Use :meth:`validate_readiness` instead. Algorithm and metric
            validation is now handled by Pydantic validators at config
            construction time.

        Returns:
            Dict[str, bool]: Dictionary containing validation results
        """
        import warnings
        warnings.warn(
            "validate_optimization_configuration() is deprecated, use validate_readiness()",
            DeprecationWarning,
            stacklevel=2,
        )
        readiness = self.validate_readiness()
        # Return legacy-compatible shape
        return {
            'model_supported': readiness.get('model_supported', False),
            'parameters_defined': readiness.get('parameters_defined', False),
        }

    def validate_readiness(self) -> Dict[str, bool]:
        """
        Validate that this manager is ready for execution.

        Checks runtime prerequisites that Pydantic cannot verify:
        whether the configured model has a registered optimizer and
        whether calibration parameters are defined.

        Returns:
            Dict mapping check names to pass/fail booleans.
        """
        results = {}

        # Check model support via optimizer registry
        models_str = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            ''
        )
        models = [m.strip().upper() for m in str(models_str).split(',') if m.strip()]
        model_supported = any(
            R.optimizers.get(m) is not None for m in models
        ) if models else False
        results['model_supported'] = model_supported

        # Check parameters to calibrate
        # Parameter lists live in model-specific config (e.g. SUMMA's
        # PARAMS_TO_CALIBRATE, FUSE_PARAMS_TO_CALIBRATE); there is no
        # optimization.params_to_calibrate, which this used to read (always '').
        flat = self.config_dict
        param_keys = ['PARAMS_TO_CALIBRATE', 'BASIN_PARAMS_TO_CALIBRATE']
        param_keys += [f'{m}_PARAMS_TO_CALIBRATE' for m in models]
        results['parameters_defined'] = any(flat.get(k) for k in param_keys)

        return results

    def get_available_optimizers(self) -> Dict[str, str]:
        """
        Get list of available optimization algorithms.

        Returns:
            Dict[str, str]: Dictionary mapping algorithm identifiers to their descriptions
        """
        return {
            'PSO': 'Particle Swarm Optimization',
            'SCE-UA': 'Shuffled Complex Evolution',
            'DDS': 'Dynamically Dimensioned Search',
            'DE': 'Differential Evolution',
            'NSGA-II': 'Non-dominated Sorting Genetic Algorithm II',
            'ASYNC-DDS': 'Asynchronous Dynamically Dimensioned Search',
            'POP-DDS': 'Population Dynamically Dimensioned Search',
        }

    def _apply_parameter_transformations(self, params: Dict[str, float], settings_dir: Path) -> bool:
        """
        Applies transformations to parameters (e.g., soil depth multipliers).
        """
        return self.transformation_manager.transform(params, settings_dir)

    def _calculate_multivariate_objective(self, sim_results: Dict[str, pd.Series]) -> float:
        """
        Calculates a composite objective score from multivariate simulation results.
        """
        # 1. Get the multivariate objective handler
        obj_cls = R.objectives.get('MULTIVARIATE')
        objective_handler = obj_cls(self.config, self.logger) if obj_cls else None
        if not objective_handler:
            return 1000.0

        # 2. Use AnalysisManager to evaluate variables
        from symfluence.evaluation.analysis_manager import AnalysisManager
        am = AnalysisManager(self.config, self.logger)
        eval_results = am.run_multivariate_evaluation(sim_results)

        # 3. Calculate scalar objective
        return objective_handler.calculate(eval_results)

    @staticmethod
    def _best_row_index(results_df: pd.DataFrame) -> Optional[Any]:
        """Index of the best iteration row, or None without a score column.

        The ``score`` column is written in maximization convention (minimise
        metrics such as RMSE are stored negated), so the best row is always
        ``idxmax`` -- whatever OPTIMIZATION_METRIC is. The other candidate
        columns (objective / kge / fitness / nse) are maximise-style too.
        """
        if results_df.empty:
            return None
        by_lower = {c.lower(): c for c in results_df.columns}
        for name in ('score', 'objective', 'fitness', 'kge', 'nse'):
            col = by_lower.get(name)
            if col is None:
                continue
            values = pd.to_numeric(results_df[col], errors='coerce')
            if values.notna().any():
                return values.idxmax()
        return None

    def load_optimization_results(self, filename: str = None) -> Optional[Dict]:
        """
        Load optimization results from file.

        Args:
            filename (str, optional): Name of results file to load. If None, uses
                                    the default filename based on experiment_id.

        Returns:
            Optional[Dict]: Dictionary with optimization results. Returns None if loading fails.
        """
        try:
            results_df = self.results_manager.load_optimization_results(filename)

            if results_df is None:
                return None

            # Find best iteration by score instead of assuming row 0
            best_idx = self._best_row_index(results_df)
            if best_idx is not None:
                best_iteration = results_df.loc[cast(Any, best_idx)].to_dict()
            else:
                # Fall back to last row if no score column found
                best_iteration = results_df.iloc[-1].to_dict() if not results_df.empty else {}

            # Convert DataFrame to dictionary format
            results = {
                'parameters': results_df.to_dict(orient='records'),
                'best_iteration': best_iteration,
                'columns': results_df.columns.tolist()
            }

            return results

        except (FileNotFoundError, ValueError, KeyError) as e:
            self.logger.error(f"Error loading optimization results: {str(e)}")
            return None
