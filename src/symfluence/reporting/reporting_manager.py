# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Central reporting facade for coordinating all SYMFLUENCE visualizations.

Provides a unified interface for generating publication-ready visualizations
across all modeling stages: domain setup, calibration, evaluation, and
multi-model comparison. Implements the Facade pattern to orchestrate
specialized plotters while hiding complexity from client code.

Heavy lifting is delegated to three orchestrators:
- ``ModelOutputOrchestrator``: registry-based model output dispatch
- ``CalibrationOrchestrator``: post-calibration target dispatch and comparison plots
- ``DiagnosticsOrchestrator``: per-workflow-step diagnostic validation plots
"""

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from symfluence.core.constants import ConfigKeys
from symfluence.core.exceptions import ReportingError, symfluence_error_handler
from symfluence.core.mixins import ConfigMixin

# Config
from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG, PlotConfig
from symfluence.reporting.core.decorators import skip_if_not_diagnostic, skip_if_not_visualizing

# Type hints only - actual imports are lazy
if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.reporting.orchestrators.calibration_orchestrator import CalibrationOrchestrator
    from symfluence.reporting.orchestrators.diagnostics_orchestrator import DiagnosticsOrchestrator
    from symfluence.reporting.orchestrators.model_output_orchestrator import ModelOutputOrchestrator
    from symfluence.reporting.plotters.analysis_plotter import AnalysisPlotter
    from symfluence.reporting.plotters.benchmark_plotter import BenchmarkPlotter
    from symfluence.reporting.plotters.diagnostic_plotter import DiagnosticPlotter
    from symfluence.reporting.plotters.domain_plotter import DomainPlotter
    from symfluence.reporting.plotters.forcing_comparison_plotter import ForcingComparisonPlotter
    from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
    from symfluence.reporting.plotters.optimization_plotter import OptimizationPlotter
    from symfluence.reporting.plotters.snow_plotter import SnowPlotter
    from symfluence.reporting.plotters.workflow_diagnostic_plotter import WorkflowDiagnosticPlotter
    from symfluence.reporting.processors.data_processor import DataProcessor
    from symfluence.reporting.processors.spatial_processor import SpatialProcessor


class ReportingManager(ConfigMixin):
    """Central facade coordinating all visualization and reporting in SYMFLUENCE.

    Orchestrates diverse visualization workflows by delegating to specialized
    plotters for domain maps, calibration analysis, performance benchmarking,
    and diagnostics. Uses Facade and Lazy Initialization patterns.

    Example:
        >>> rm = ReportingManager(config, logger, visualize=True)
        >>> rm.plot_domain()          # Generate domain overview map
        >>> rm.plot_calibration()     # Plot calibration convergence
    """

    def __init__(self, config: 'SymfluenceConfig', logger: Any, visualize: bool = False, diagnostic: bool = False):
        """Initialize the ReportingManager.

        Args:
            config: SymfluenceConfig instance.
            logger: Logger instance.
            visualize: Boolean flag indicating if visualization is enabled.
            diagnostic: Boolean flag indicating if diagnostic mode is enabled.
        """
        from symfluence.core.config.models import SymfluenceConfig
        if not isinstance(config, SymfluenceConfig):
            raise TypeError(
                f"config must be SymfluenceConfig, got {type(config).__name__}. "
                "Use SymfluenceConfig.from_file() to load configuration."
            )

        self._config = config
        self.logger = logger
        self.visualize = visualize
        self.diagnostic = diagnostic
        self.project_dir = Path(config.system.data_dir) / f"domain_{config.domain.name}"

    # =========================================================================
    # Component Properties (Lazy Initialization via cached_property)
    # =========================================================================

    @cached_property
    def plot_config(self) -> PlotConfig:
        """Lazy initialization of plot configuration."""
        return DEFAULT_PLOT_CONFIG

    @cached_property
    def data_processor(self) -> 'DataProcessor':
        """Lazy initialization of data processor."""
        from symfluence.reporting.processors.data_processor import DataProcessor
        return DataProcessor(self.config, self.logger)

    @cached_property
    def spatial_processor(self) -> 'SpatialProcessor':
        """Lazy initialization of spatial processor."""
        from symfluence.reporting.processors.spatial_processor import SpatialProcessor
        return SpatialProcessor(self.config, self.logger)

    @cached_property
    def domain_plotter(self) -> 'DomainPlotter':
        """Lazy initialization of domain plotter."""
        from symfluence.reporting.plotters.domain_plotter import DomainPlotter
        return DomainPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def optimization_plotter(self) -> 'OptimizationPlotter':
        """Lazy initialization of optimization plotter."""
        from symfluence.reporting.plotters.optimization_plotter import OptimizationPlotter
        return OptimizationPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def analysis_plotter(self) -> 'AnalysisPlotter':
        """Lazy initialization of analysis plotter."""
        from symfluence.reporting.plotters.analysis_plotter import AnalysisPlotter
        return AnalysisPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def benchmark_plotter(self) -> 'BenchmarkPlotter':
        """Lazy initialization of benchmark plotter."""
        from symfluence.reporting.plotters.benchmark_plotter import BenchmarkPlotter
        return BenchmarkPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def snow_plotter(self) -> 'SnowPlotter':
        """Lazy initialization of snow plotter."""
        from symfluence.reporting.plotters.snow_plotter import SnowPlotter
        return SnowPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def diagnostic_plotter(self) -> 'DiagnosticPlotter':
        """Lazy initialization of diagnostic plotter."""
        from symfluence.reporting.plotters.diagnostic_plotter import DiagnosticPlotter
        return DiagnosticPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def model_comparison_plotter(self) -> 'ModelComparisonPlotter':
        """Lazy initialization of model comparison plotter."""
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        return ModelComparisonPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def forcing_comparison_plotter(self) -> 'ForcingComparisonPlotter':
        """Lazy initialization of forcing comparison plotter."""
        from symfluence.reporting.plotters.forcing_comparison_plotter import ForcingComparisonPlotter
        return ForcingComparisonPlotter(self.config, self.logger, self.plot_config)

    @cached_property
    def workflow_diagnostic_plotter(self) -> 'WorkflowDiagnosticPlotter':
        """Lazy initialization of workflow diagnostic plotter."""
        from symfluence.reporting.plotters.workflow_diagnostic_plotter import WorkflowDiagnosticPlotter
        return WorkflowDiagnosticPlotter(self.config, self.logger, self.plot_config)

    # =========================================================================
    # Orchestrator Properties (Lazy Initialization)
    # =========================================================================

    @cached_property
    def _model_output_orchestrator(self) -> 'ModelOutputOrchestrator':
        from symfluence.reporting.orchestrators.model_output_orchestrator import ModelOutputOrchestrator
        return ModelOutputOrchestrator(
            config=self.config,
            logger=self.logger,
            visualize=self.visualize,
            plot_config=self.plot_config,
            analysis_plotter=self.analysis_plotter,
        )

    @cached_property
    def _calibration_orchestrator(self) -> 'CalibrationOrchestrator':
        from symfluence.reporting.orchestrators.calibration_orchestrator import CalibrationOrchestrator
        return CalibrationOrchestrator(
            config=self.config,
            logger=self.logger,
            visualize=self.visualize,
            project_dir=self.project_dir,
            optimization_plotter=self.optimization_plotter,
            model_comparison_plotter=self.model_comparison_plotter,
            analysis_plotter=self.analysis_plotter,
            model_output_orchestrator=self._model_output_orchestrator,
        )

    @cached_property
    def _diagnostics_orchestrator(self) -> 'DiagnosticsOrchestrator':
        from symfluence.reporting.orchestrators.diagnostics_orchestrator import DiagnosticsOrchestrator
        return DiagnosticsOrchestrator(
            config=self.config,
            logger=self.logger,
            diagnostic=self.diagnostic,
            project_dir=self.project_dir,
            workflow_diagnostic_plotter=self.workflow_diagnostic_plotter,
        )

    # =========================================================================
    # Public Methods — Data Processing & Utility
    # =========================================================================

    @skip_if_not_visualizing()
    def visualize_data_distribution(self, data: Any, variable_name: str, stage: str) -> None:
        """Visualize data distribution (histogram/boxplot)."""
        self.diagnostic_plotter.plot_data_distribution(data, variable_name, stage)

    @skip_if_not_visualizing()
    def visualize_spatial_coverage(self, raster_path: Path, variable_name: str, stage: str) -> None:
        """Visualize spatial coverage of raster data."""
        self.diagnostic_plotter.plot_spatial_coverage(raster_path, variable_name, stage)

    @skip_if_not_visualizing()
    def visualize_forcing_comparison(
        self,
        raw_forcing_file: Path,
        remapped_forcing_file: Path,
        forcing_grid_shp: Path,
        hru_shp: Path,
        variable: str = 'precipitation_flux',
        time_index: int = 0,
    ) -> Optional[str]:
        """Visualize raw vs. remapped forcing data comparison."""
        self.logger.info("Creating raw vs. remapped forcing comparison visualization...")
        return self.forcing_comparison_plotter.plot_raw_vs_remapped(
            raw_forcing_file=raw_forcing_file,
            remapped_forcing_file=remapped_forcing_file,
            forcing_grid_shp=forcing_grid_shp,
            hru_shp=hru_shp,
            variable=variable,
            time_index=time_index,
        )

    def is_visualization_enabled(self) -> bool:
        """Check if visualization is enabled."""
        return self.visualize

    def update_sim_reach_id(self, config_path: Optional[str] = None) -> Optional[int]:
        """Update the SIM_REACH_ID in both the config object and YAML file."""
        return self.spatial_processor.update_sim_reach_id(config_path)

    # --- Domain Visualization ---

    @skip_if_not_visualizing()
    def visualize_domain(self) -> Optional[str]:
        """Visualize the domain boundaries and features."""
        self.logger.info("Creating domain visualization...")
        return self.domain_plotter.plot_domain()

    @skip_if_not_visualizing()
    def visualize_discretized_domain(self, discretization_method: str) -> Optional[str]:
        """Visualize the discretized domain (HRUs/GRUs)."""
        self.logger.info(f"Creating discretization visualization for {discretization_method}...")
        return self.domain_plotter.plot_discretized_domain(discretization_method)

    # --- Model Output Visualization (delegates to ModelOutputOrchestrator) ---

    @skip_if_not_visualizing()
    def visualize_model_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """Visualize model outputs (streamflow comparison)."""
        return self._model_output_orchestrator.visualize_model_outputs(model_outputs, obs_files)

    @skip_if_not_visualizing()
    def visualize_lumped_model_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """Visualize lumped model outputs."""
        return self._model_output_orchestrator.visualize_lumped_model_outputs(model_outputs, obs_files)

    @skip_if_not_visualizing()
    def visualize_fuse_outputs(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[str]:
        """Visualize FUSE model outputs."""
        return self._model_output_orchestrator.visualize_fuse_outputs(model_outputs, obs_files)

    @skip_if_not_visualizing(default={})
    def visualize_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
        """Visualize SUMMA model outputs (all variables)."""
        return self._model_output_orchestrator.visualize_summa_outputs(experiment_id)

    @skip_if_not_visualizing()
    def visualize_ngen_results(self, sim_df: Any, obs_df: Optional[Any], experiment_id: str, results_dir: Path) -> None:
        """Visualize NGen streamflow plots."""
        self._model_output_orchestrator.visualize_ngen_results(sim_df, obs_df, experiment_id, results_dir)

    @skip_if_not_visualizing()
    def visualize_lstm_results(self, results_df: Any, obs_streamflow: Any, obs_snow: Any, use_snow: bool, output_dir: Path, experiment_id: str) -> None:
        """Visualize LSTM simulation results."""
        self._model_output_orchestrator.visualize_lstm_results(
            results_df, obs_streamflow, obs_snow, use_snow, output_dir, experiment_id,
        )

    @skip_if_not_visualizing()
    def visualize_hype_results(self, sim_flow: Any, obs_flow: Any, outlet_id: str, domain_name: str, experiment_id: str, project_dir: Path) -> None:
        """Visualize HYPE streamflow comparison."""
        self._model_output_orchestrator.visualize_hype_results(
            sim_flow, obs_flow, outlet_id, domain_name, experiment_id, project_dir,
        )

    @skip_if_not_visualizing()
    def visualize_model_results(self, model_name: str, **kwargs) -> Optional[Any]:
        """Visualize model results using registry-based dispatch."""
        return self._model_output_orchestrator.visualize_model_results(model_name, **kwargs)

    # --- Analysis Visualization ---

    @skip_if_not_visualizing()
    def visualize_timeseries_results(self) -> None:
        """Visualize timeseries results from the standard results file."""
        self.logger.info("Creating timeseries visualizations from results file...")

        with symfluence_error_handler(
            "creating timeseries visualizations",
            self.logger,
            reraise=False,
            error_type=ReportingError,
        ):
            df = self.data_processor.read_results_file()
            exp_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='default', dict_key=ConfigKeys.EXPERIMENT_ID)
            domain_name = self._get_config_value(lambda: self.config.domain.name, default='unknown', dict_key=ConfigKeys.DOMAIN_NAME)
            self.analysis_plotter.plot_timeseries_results(df, exp_id, domain_name)
            self.analysis_plotter.plot_diagnostics(df, exp_id, domain_name)

    @skip_if_not_visualizing(default=[])
    def visualize_benchmarks(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Visualize benchmark results."""
        self.logger.info("Creating benchmark visualizations...")
        return self.benchmark_plotter.plot_benchmarks(benchmark_results)

    @skip_if_not_visualizing(default={})
    def visualize_snow_comparison(self, model_outputs: List[List[str]]) -> Dict[str, Any]:
        """Visualize snow comparison."""
        self.logger.info("Creating snow comparison visualization...")
        formatted_outputs = [tuple(item) for item in model_outputs]
        return self.snow_plotter.plot_snow_comparison(formatted_outputs)

    @skip_if_not_visualizing()
    def visualize_optimization_progress(self, history: List[Dict], output_dir: Path, calibration_variable: str, metric: str) -> None:
        """Visualize optimization progress."""
        self.logger.info("Creating optimization progress visualization...")
        self.optimization_plotter.plot_optimization_progress(history, output_dir, calibration_variable, metric)

    @skip_if_not_visualizing()
    def visualize_optimization_depth_parameters(self, history: List[Dict], output_dir: Path) -> None:
        """Visualize depth parameter evolution."""
        self.logger.info("Creating depth parameter visualization...")
        self.optimization_plotter.plot_depth_parameters(history, output_dir)

    @skip_if_not_visualizing()
    def visualize_sensitivity_analysis(self, sensitivity_data: Any, output_file: Path, plot_type: str = 'single') -> None:
        """Visualize sensitivity analysis results."""
        self.logger.info(f"Creating sensitivity analysis visualization ({plot_type})...")
        self.analysis_plotter.plot_sensitivity_analysis(sensitivity_data, output_file, plot_type)

    @skip_if_not_visualizing()
    def visualize_decision_impacts(self, results_file: Path, output_folder: Path) -> None:
        """Visualize decision analysis impacts."""
        self.logger.info("Creating decision impact visualizations...")
        self.analysis_plotter.plot_decision_impacts(results_file, output_folder)

    @skip_if_not_visualizing()
    def visualize_hydrographs_with_highlight(self, results_file: Path, simulation_results: Dict, observed_streamflow: Any, decision_options: Dict, output_folder: Path, metric: str = 'kge') -> None:
        """Visualize hydrographs with top performers highlighted."""
        self.logger.info(f"Creating hydrograph visualization with {metric} highlight...")
        self.analysis_plotter.plot_hydrographs_with_highlight(
            results_file, simulation_results, observed_streamflow,
            decision_options, output_folder, metric,
        )

    @skip_if_not_visualizing()
    def visualize_drop_analysis(self, drop_data: List[Dict], optimal_threshold: float, project_dir: Path) -> None:
        """Visualize drop analysis for stream threshold selection."""
        self.logger.info("Creating drop analysis visualization...")
        self.analysis_plotter.plot_drop_analysis(drop_data, optimal_threshold, project_dir)

    # --- Calibration Visualization (delegates to CalibrationOrchestrator) ---

    @skip_if_not_visualizing()
    def generate_model_comparison_overview(
        self,
        experiment_id: Optional[str] = None,
        context: str = 'run_model',
    ) -> Optional[str]:
        """Generate model comparison overview for all models with valid output."""
        return self._calibration_orchestrator.generate_model_comparison_overview(
            experiment_id=experiment_id,
            context=context,
        )

    @skip_if_not_visualizing(default={})
    def visualize_calibration_results(
        self,
        experiment_id: Optional[str] = None,
        calibration_target: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate comprehensive post-calibration visualizations."""
        return self._calibration_orchestrator.visualize_calibration_results(
            experiment_id=experiment_id,
            calibration_target=calibration_target,
        )

    # =========================================================================
    # Workflow Diagnostic Methods (delegates to DiagnosticsOrchestrator)
    # =========================================================================

    @skip_if_not_diagnostic()
    def diagnostic_domain_definition(self, basin_gdf: Any, dem_path: Optional[Path] = None) -> Optional[str]:
        """Generate diagnostic plots for domain definition step."""
        return self._diagnostics_orchestrator.diagnostic_domain_definition(basin_gdf, dem_path)

    @skip_if_not_diagnostic()
    def diagnostic_discretization(self, hru_gdf: Any, method: str) -> Optional[str]:
        """Generate diagnostic plots for discretization step."""
        return self._diagnostics_orchestrator.diagnostic_discretization(hru_gdf, method)

    @skip_if_not_diagnostic()
    def diagnostic_observations(self, obs_df: Any, obs_type: str) -> Optional[str]:
        """Generate diagnostic plots for observation processing step."""
        return self._diagnostics_orchestrator.diagnostic_observations(obs_df, obs_type)

    @skip_if_not_diagnostic()
    def diagnostic_forcing_raw(self, forcing_nc: Path, domain_shp: Optional[Path] = None) -> Optional[str]:
        """Generate diagnostic plots for raw forcing acquisition step."""
        return self._diagnostics_orchestrator.diagnostic_forcing_raw(forcing_nc, domain_shp)

    @skip_if_not_diagnostic()
    def diagnostic_forcing_remapped(
        self,
        raw_nc: Path,
        remapped_nc: Path,
        hru_shp: Optional[Path] = None,
    ) -> Optional[str]:
        """Generate diagnostic plots for forcing remapping step."""
        return self._diagnostics_orchestrator.diagnostic_forcing_remapped(raw_nc, remapped_nc, hru_shp)

    @skip_if_not_diagnostic()
    def diagnostic_model_preprocessing(self, input_dir: Path, model_name: str) -> Optional[str]:
        """Generate diagnostic plots for model preprocessing step."""
        return self._diagnostics_orchestrator.diagnostic_model_preprocessing(input_dir, model_name)

    @skip_if_not_diagnostic()
    def diagnostic_model_output(self, output_nc: Path, model_name: str) -> Optional[str]:
        """Generate diagnostic plots for model output step."""
        return self._diagnostics_orchestrator.diagnostic_model_output(output_nc, model_name)

    @skip_if_not_diagnostic()
    def diagnostic_attributes(
        self,
        dem_path: Optional[Path] = None,
        soil_path: Optional[Path] = None,
        land_path: Optional[Path] = None,
    ) -> Optional[str]:
        """Generate diagnostic plots for attribute acquisition step."""
        return self._diagnostics_orchestrator.diagnostic_attributes(dem_path, soil_path, land_path)

    @skip_if_not_diagnostic()
    def diagnostic_calibration(
        self,
        history: Optional[List[Dict]] = None,
        best_params: Optional[Dict[str, float]] = None,
        obs_vs_sim: Optional[Dict[str, Any]] = None,
        model_name: str = 'Unknown',
    ) -> Optional[str]:
        """Generate diagnostic plots for calibration step."""
        return self._diagnostics_orchestrator.diagnostic_calibration(
            history=history, best_params=best_params, obs_vs_sim=obs_vs_sim, model_name=model_name,
        )

    @skip_if_not_diagnostic()
    def diagnostic_coupling_conservation(
        self, graph: Any, output_dir: Optional[Path] = None,
    ) -> Optional[str]:
        """Generate conservation diagnostic for a coupled model run."""
        return self._diagnostics_orchestrator.diagnostic_coupling_conservation(graph, output_dir)
