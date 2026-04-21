# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Orchestrator for calibration and model-comparison visualizations.

Extracted from ``ReportingManager`` — handles the complex calibration-target
dispatch logic in ``visualize_calibration_results()`` and the model comparison
overview generation.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from symfluence.core.constants import ConfigKeys
from symfluence.core.exceptions import ReportingError, symfluence_error_handler
from symfluence.core.mixins import ConfigMixin
from symfluence.reporting.core.decorators import skip_if_not_visualizing

if TYPE_CHECKING:
    from symfluence.reporting.orchestrators.model_output_orchestrator import ModelOutputOrchestrator
    from symfluence.reporting.plotters.analysis_plotter import AnalysisPlotter
    from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
    from symfluence.reporting.plotters.optimization_plotter import OptimizationPlotter


class CalibrationOrchestrator(ConfigMixin):
    """Orchestrates post-calibration and model-comparison visualizations."""

    def __init__(
        self,
        config: Any,
        logger: Any,
        visualize: bool,
        project_dir: Path,
        optimization_plotter: 'OptimizationPlotter',
        model_comparison_plotter: 'ModelComparisonPlotter',
        analysis_plotter: 'AnalysisPlotter',
        *,
        model_output_orchestrator: 'ModelOutputOrchestrator',
    ) -> None:
        self._config = config
        self.logger = logger
        self.visualize = visualize
        self.project_dir = project_dir
        self.optimization_plotter = optimization_plotter
        self.model_comparison_plotter = model_comparison_plotter
        self.analysis_plotter = analysis_plotter
        self.model_output_orchestrator = model_output_orchestrator

    def _find_summa_final_evaluation_file(self, experiment_id: str) -> Optional[Path]:
        """Find calibrated SUMMA daily output from the final evaluation run."""
        algorithm = str(self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        )).lower()
        final_eval_file = (
            self.project_dir / "optimization" / "SUMMA" /
            f"{algorithm}_{experiment_id}" / "final_evaluation" /
            f"{experiment_id}_day.nc"
        )
        if final_eval_file.exists():
            return final_eval_file

        optimization_dir = self.project_dir / "optimization" / "SUMMA"
        candidates = list(optimization_dir.glob(f"*/final_evaluation/{experiment_id}_day.nc"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    @skip_if_not_visualizing()
    def generate_model_comparison_overview(
        self,
        experiment_id: Optional[str] = None,
        context: str = 'run_model',
    ) -> Optional[str]:
        """Generate model comparison overview for all models with valid output.

        Creates a comprehensive multi-panel visualization comparing observed and
        simulated streamflow across all models.

        Args:
            experiment_id: Experiment ID for loading results. If None, uses
                          config.domain.experiment_id.
            context: Context for the comparison ('run_model' or 'calibrate_model').

        Returns:
            Path to the saved overview plot, or None if failed.
        """
        # Get experiment ID from config if not provided
        if experiment_id is None:
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='default',
                dict_key=ConfigKeys.EXPERIMENT_ID,
            )

        self.logger.info(f"Generating model comparison overview for {experiment_id}...")

        with symfluence_error_handler(
            "generating model comparison overview",
            self.logger,
            reraise=False,
            error_type=ReportingError,
        ):
            return self.model_comparison_plotter.plot_model_comparison_overview(
                experiment_id=experiment_id,
                context=context,
            )

        return None

    @skip_if_not_visualizing(default={})
    def visualize_calibration_results(
        self,
        experiment_id: Optional[str] = None,
        calibration_target: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate comprehensive post-calibration visualizations.

        Creates visualizations appropriate for the calibration target:
        - Optimization progress/convergence plot
        - Model performance comparison (obs vs sim with metrics)

        Args:
            experiment_id: Experiment ID. If None, uses config value.
            calibration_target: Target variable being calibrated. If None,
                               auto-detected from config.optimization.target.

        Returns:
            Dictionary mapping plot names to file paths.
        """
        import json
        plot_paths: Dict[str, str] = {}

        # Get experiment ID from config if not provided
        if experiment_id is None:
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='default',
                dict_key=ConfigKeys.EXPERIMENT_ID,
            )

        # Get calibration target from config if not provided
        if calibration_target is None:
            calibration_target = self._get_config_value(
                lambda: self.config.optimization.target,
                default='streamflow',
                dict_key=ConfigKeys.OPTIMIZATION_TARGET,
            )
        calibration_target = str(calibration_target).lower()

        self.logger.info(f"Generating post-calibration visualizations for {experiment_id} (target: {calibration_target})")

        # 1. Generate optimization progress plot (if history exists)
        with symfluence_error_handler(
            "generating optimization progress plot",
            self.logger,
            reraise=False,
            error_type=ReportingError,
        ):
            opt_dir = self.project_dir / "optimization"
            history_files = list(opt_dir.glob("*history*.json")) + list(opt_dir.glob("*history*.csv"))

            if history_files:
                # Try to load history from JSON first
                history = []
                for hf in history_files:
                    if hf.suffix == '.json':
                        try:
                            with open(hf, encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    history = data
                                    break
                                elif isinstance(data, dict) and 'history' in data:
                                    history = data['history']
                                    break
                        except (json.JSONDecodeError, OSError, KeyError):
                            continue

                if history:
                    metric = self._get_config_value(
                        lambda: self.config.optimization.metric,
                        default='KGE',
                        dict_key=ConfigKeys.OPTIMIZATION_METRIC,
                    )
                    progress_plot = self.optimization_plotter.plot_optimization_progress(
                        history, opt_dir, calibration_target, metric,
                    )
                    if progress_plot:
                        plot_paths['optimization_progress'] = progress_plot

        # 2. Generate appropriate model comparison based on calibration target
        with symfluence_error_handler(
            "generating calibration comparison plots",
            self.logger,
            reraise=False,
            error_type=ReportingError,
        ):
            if calibration_target in ('streamflow', 'q', 'discharge', 'runoff'):
                # Streamflow calibration -> model comparison overview
                comparison_plot = self.generate_model_comparison_overview(
                    experiment_id=experiment_id,
                    context='calibrate_model',
                )
                if comparison_plot:
                    plot_paths['model_comparison'] = comparison_plot

                # Also generate default vs calibrated comparison
                with symfluence_error_handler(
                    "generating default vs calibrated comparison",
                    self.logger,
                    reraise=False,
                    error_type=ReportingError,
                ):
                    default_vs_calibrated_plot = self.model_comparison_plotter.plot_default_vs_calibrated_comparison(
                        experiment_id=experiment_id,
                    )
                    if default_vs_calibrated_plot:
                        plot_paths['default_vs_calibrated'] = default_vs_calibrated_plot

            elif calibration_target in ('swe', 'snow', 'snow_water_equivalent'):
                # SWE calibration -> SUMMA outputs with SWE observations
                final_eval_file = self._find_summa_final_evaluation_file(experiment_id)
                if final_eval_file is not None:
                    self.logger.info(f"Loading calibrated SUMMA output from: {final_eval_file}")
                    summa_plots = self.analysis_plotter.plot_summa_outputs(
                        experiment_id,
                        summa_file=final_eval_file,
                        output_suffix="calibrated",
                    )
                else:
                    self.logger.info("No calibrated SUMMA final-evaluation output found; using standard simulation output")
                    summa_plots = self.model_output_orchestrator.visualize_summa_outputs(experiment_id)
                if 'scalarSWE' in summa_plots:
                    plot_paths['scalarSWE'] = summa_plots['scalarSWE']
                    plot_paths['model_comparison'] = summa_plots['scalarSWE']
                # Include other snow-related variables if present
                for var in ['scalarSnowDepth', 'scalarSnowfall']:
                    if var in summa_plots:
                        plot_paths[var] = summa_plots[var]

            elif calibration_target in ('et', 'evapotranspiration', 'latent_heat', 'le'):
                # ET/energy flux calibration -> SUMMA outputs with energy observations
                summa_plots = self.model_output_orchestrator.visualize_summa_outputs(experiment_id)
                if 'scalarLatHeatTotal' in summa_plots:
                    plot_paths['scalarLatHeatTotal'] = summa_plots['scalarLatHeatTotal']
                    plot_paths['model_comparison'] = summa_plots['scalarLatHeatTotal']
                if 'scalarSenHeatTotal' in summa_plots:
                    plot_paths['scalarSenHeatTotal'] = summa_plots['scalarSenHeatTotal']
                # Include ET-related variables if present
                for var in ['scalarCanopyEvaporation', 'scalarGroundEvaporation', 'scalarTotalET']:
                    if var in summa_plots:
                        plot_paths[var] = summa_plots[var]

            else:
                # Unknown target - try both streamflow and SUMMA outputs
                self.logger.info(f"Unknown calibration target '{calibration_target}', generating all available plots")
                comparison_plot = self.generate_model_comparison_overview(
                    experiment_id=experiment_id,
                    context='calibrate_model',
                )
                if comparison_plot:
                    plot_paths['model_comparison'] = comparison_plot

                summa_plots = self.model_output_orchestrator.visualize_summa_outputs(experiment_id)
                plot_paths.update(summa_plots)

        self.logger.info(f"Generated {len(plot_paths)} calibration visualization(s)")
        return plot_paths
