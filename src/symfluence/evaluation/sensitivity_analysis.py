# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Sensitivity analysis of model parameters using multiple statistical methods.

Provides parameter sensitivity analysis using Sobol, RBD-FAST, and correlation
methods on calibration results, with visualization and reporting support.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import rbd_fast, sobol
from SALib.sample import sobol as sobol_sample
from scipy.stats import spearmanr
from tqdm import tqdm

from symfluence.core.mixins import ConfigMixin

# Hoist pyviscous to module scope so the VISCOUS path is patchable from
# tests (see test_sa_viscous_nan_handling). Leaving the import inside
# perform_sensitivity_analysis kept viscous un-patchable because
# ``from pyviscous import viscous`` re-binds the real attribute inside
# the function's local scope on every call.
try:
    from pyviscous import viscous as _pyviscous
except ImportError:  # pragma: no cover — pyviscous is an optional dep
    _pyviscous = None

_NON_PARAM_COLS = frozenset({
    'Iteration', 'iteration', 'score', 'timestamp', 'crash_count', 'crash_rate',
    'Calib_RMSE', 'Calib_KGE', 'Calib_KGEp', 'Calib_KGEnp', 'Calib_NSE', 'Calib_MAE',
    'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE', 'objective', 'Objective', 'fitness',
})


class SensitivityAnalyzer(ConfigMixin):
    """
    Performs parameter sensitivity analysis on calibration results.

    Supports multiple sensitivity analysis methods including VISCOUS,
    Sobol indices, RBD-FAST, and Spearman correlation to identify
    influential parameters and their interactions.

    Attributes:
        config: Configuration dictionary with domain settings.
        logger: Logger instance for status messages.
        reporting_manager: Optional manager for generating reports.
        output_folder: Directory for sensitivity analysis outputs.
    """

    def __init__(self, config, logger, reporting_manager=None):
        """
        Initialize the sensitivity analyzer.

        Args:
            config: Configuration dictionary containing SYMFLUENCE_DATA_DIR
                and DOMAIN_NAME settings.
            logger: Logger instance for status and debug messages.
            reporting_manager: Optional reporting manager for visualizations.
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.reporting_manager = reporting_manager
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_folder = self.project_dir / "reporting" / "sensitivity_analysis"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def read_calibration_results(self, file_path):
        """
        Read calibration results from CSV file.

        Args:
            file_path: Path to calibration results CSV file.

        Returns:
            pd.DataFrame: Calibration results with NaN rows removed.
        """
        df = pd.read_csv(file_path)
        return df.dropna()

    def preprocess_data(self, samples, metric='RMSE'):
        """
        Preprocess calibration samples for sensitivity analysis.

        Steps:
          1. Drop rows whose metric value is non-finite (NaN from crashed
             iterations, inf from degenerate metric maths).
          2. Drop rows whose metric value is a known SYMFLUENCE failure
             sentinel (``<= -900``). These are the "score is invalid"
             markers several optimisers write when a model run crashed —
             their raw values aren't meaningful response values and
             poison copula-based estimators (VISCOUS) with synthetic
             low-tail mass. Co-author PW reported VISCOUS producing NaN
             on a calibration where the crash-regime score had leaked
             into the response vector.
          3. De-duplicate on parameter columns — DDS produces repeated
             rows when it restarts from the best-so-far, which biases
             sensitivity estimators toward whichever parameters happened
             to be held constant during those restarts.

        Args:
            samples: DataFrame of calibration samples with parameter values.
            metric: Metric column name used for failure-sentinel filtering.

        Returns:
            pd.DataFrame: Cleaned, deduplicated samples.
        """
        n_in = len(samples)

        if metric in samples.columns:
            metric_values = pd.to_numeric(samples[metric], errors='coerce')
            finite_mask = np.isfinite(metric_values)
            sentinel_mask = metric_values > -900
            clean = samples[finite_mask & sentinel_mask].copy()
            n_dropped = n_in - len(clean)
            if n_dropped > 0:
                self.logger.info(
                    f"Sensitivity preprocessing dropped {n_dropped}/{n_in} rows "
                    f"with non-finite or failure-sentinel {metric} values "
                    f"(NaN / <=-900). Keeping {len(clean)} usable rows."
                )
        else:
            clean = samples

        samples_unique = clean.drop_duplicates(
            subset=[col for col in clean.columns if col != 'Iteration']
        )
        if len(samples_unique) < len(clean):
            self.logger.info(
                f"Sensitivity preprocessing dropped "
                f"{len(clean) - len(samples_unique)} duplicate rows."
            )
        return samples_unique

    def perform_sensitivity_analysis(self, samples, metric='Calib_KGEnp', min_samples=60):
        """
        Perform VISCOUS sensitivity analysis on calibration samples.

        Uses the pyviscous library to compute total-order sensitivity indices
        for each parameter with respect to the specified metric.

        Args:
            samples: DataFrame with parameter values and metric columns.
            metric: Name of the objective metric column (default: 'Calib_KGEnp').
            min_samples: Minimum samples required for reliable analysis.

        Returns:
            pd.Series: Sensitivity indices for each parameter, or -999 if failed.
        """
        self.logger.info(f"Performing sensitivity analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in _NON_PARAM_COLS]

        if len(samples) < min_samples:
            self.logger.warning(f"Insufficient data for reliable sensitivity analysis. Have {len(samples)} samples, recommend at least {min_samples}.")
            return pd.Series([-999] * len(parameter_columns), index=parameter_columns)

        x = samples[parameter_columns].values.astype(float, copy=False)
        y = samples[metric].values.astype(float, copy=False).reshape(-1, 1)

        sensitivities = []

        if _pyviscous is None:
            self.logger.warning("pyviscous not installed, skipping.")
            return pd.Series([-999] * len(parameter_columns), index=parameter_columns)

        for i, param in tqdm(enumerate(parameter_columns), total=len(parameter_columns), desc="Calculating sensitivities"):
            try:
                try:
                    sensitivity_result = _pyviscous(x, y, i, sensType='total')
                except ValueError:
                    sensitivity_result = _pyviscous(x, y, i, sensType='single')

                if isinstance(sensitivity_result, tuple):
                    sensitivity = sensitivity_result[0]
                else:
                    sensitivity = sensitivity_result

                # pyviscous returns a numeric scalar on success. If the
                # GMCM fit didn't converge for any component count the
                # sensitivity can come back as NaN — previously this
                # flowed straight through to the CSV and the user saw
                # an unlabelled NaN cell with no explanation. Convert
                # NaN to the -999 failure sentinel and log specifically
                # so a co-author reading the log knows it was VISCOUS
                # non-convergence, not a missing-data error.
                try:
                    s_float = float(sensitivity)
                except (TypeError, ValueError):
                    s_float = float('nan')
                if not np.isfinite(s_float):
                    self.logger.warning(
                        f"VISCOUS returned a non-finite index for {param} "
                        f"(result={sensitivity!r}) — the GMCM copula fit "
                        "likely did not converge for any component count. "
                        "This is common when calibration samples are "
                        "highly clustered (e.g. DDS near the optimum); "
                        "cross-check with Sobol / RBD-FAST results or run "
                        "VISCOUS on a dedicated LHS/Sobol sampling pass. "
                        "Recording -999."
                    )
                    sensitivities.append(-999)
                else:
                    sensitivities.append(s_float)
                    self.logger.info(f"Successfully calculated sensitivity for {param}")
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.error(f"Error in sensitivity analysis for parameter {param}: {str(e)}")
                sensitivities.append(-999)

        self.logger.info("Sensitivity analysis completed")
        return pd.Series(sensitivities, index=parameter_columns)

    def perform_sobol_analysis(self, samples, metric='RMSE'):
        """
        Perform Sobol sensitivity analysis using SALib.

        Computes total-order Sobol indices (ST) to quantify parameter
        influence including interactions with other parameters.

        Args:
            samples: DataFrame with parameter values and metric columns.
            metric: Name of the objective metric column.

        Returns:
            pd.Series: Total-order Sobol indices for each parameter.
        """
        self.logger.info(f"Performing Sobol analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in _NON_PARAM_COLS]

        problem = {
            'num_vars': len(parameter_columns),
            'names': parameter_columns,
            'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
        }

        param_values = sobol_sample.sample(problem, 1024)

        Y = np.zeros(param_values.shape[0])
        for i in range(param_values.shape[0]):
            interpolated_values = []
            for j, col in enumerate(parameter_columns):
                interpolated_values.append(np.interp(param_values[i, j],
                                                     samples[col].sort_values().values,
                                                     samples[metric].values[samples[col].argsort()]))
            Y[i] = np.mean(interpolated_values)

        Si = sobol.analyze(problem, Y)

        self.logger.info("Sobol analysis completed")
        return pd.Series(Si['ST'], index=parameter_columns)

    def perform_rbd_fast_analysis(self, samples, metric='RMSE'):
        """
        Perform RBD-FAST sensitivity analysis using SALib.

        Random Balance Designs - Fourier Amplitude Sensitivity Test provides
        first-order sensitivity indices with lower computational cost than Sobol.

        Args:
            samples: DataFrame with parameter values and metric columns.
            metric: Name of the objective metric column.

        Returns:
            pd.Series: First-order sensitivity indices (S1) for each parameter.
        """
        self.logger.info(f"Performing RBD-FAST analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]

        problem = {
            'num_vars': len(parameter_columns),
            'names': parameter_columns,
            'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
        }

        X = samples[parameter_columns].values
        Y = samples[metric].values

        rbd_results = rbd_fast.analyze(problem, X, Y)
        self.logger.info("RBD-FAST analysis completed")
        return pd.Series(rbd_results['S1'], index=parameter_columns)

    def perform_correlation_analysis(self, samples, metric='RMSE'):
        """
        Perform Spearman correlation analysis between parameters and metric.

        Computes rank correlation coefficients to identify monotonic
        relationships between each parameter and the objective metric.

        Args:
            samples: DataFrame with parameter values and metric columns.
            metric: Name of the objective metric column.

        Returns:
            pd.Series: Spearman correlation coefficients for each parameter.
        """
        self.logger.info(f"Performing correlation analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in _NON_PARAM_COLS]
        correlations = []
        for param in parameter_columns:
            corr, _ = spearmanr(samples[param], samples[metric])
            correlations.append(abs(corr))  # Use absolute value for sensitivity
        self.logger.info("Correlation analysis completed")
        return pd.Series(correlations, index=parameter_columns)

    def run_sensitivity_analysis(self, results_file):
        """
        Run complete sensitivity analysis workflow with all methods.

        Executes VISCOUS, Sobol, RBD-FAST, and correlation analyses on the
        calibration results, saves individual and comparison outputs, and
        generates visualizations if a reporting manager is configured.

        Args:
            results_file: Path to calibration results CSV file.

        Returns:
            None. Results are saved to the output_folder as CSV files and plots.
        """
        self.logger.info("Starting sensitivity analysis")

        results = self.read_calibration_results(results_file)
        self.logger.info(f"Read {len(results)} calibration results")

        if len(results) < 10:
            self.logger.error("Error: Not enough data points for sensitivity analysis.")
            return

        # Auto-detect metric column: prefer Calib_RMSE (MESH), fall back to score (generic DDS)
        metric_col = 'Calib_RMSE'
        if metric_col not in results.columns:
            for candidate in ['score', 'Calib_KGE', 'Calib_KGEnp', 'Calib_NSE']:
                if candidate in results.columns:
                    metric_col = candidate
                    break
        self.logger.info(f"Using metric column: {metric_col}")

        results_preprocessed = self.preprocess_data(results, metric=metric_col)
        self.logger.info("Data preprocessing completed")

        methods = {
            'pyViscous': self.perform_sensitivity_analysis,
            'Sobol': self.perform_sobol_analysis,
            'RBD-FAST': self.perform_rbd_fast_analysis,
            'Correlation': self.perform_correlation_analysis
        }

        all_results = {}
        for name, method in methods.items():
            sensitivity = method(results_preprocessed, metric=metric_col)
            all_results[name] = sensitivity
            sensitivity.to_csv(self.output_folder / f'{name.lower()}_sensitivity.csv')

            if self.reporting_manager:
                self.reporting_manager.visualize_sensitivity_analysis(
                    sensitivity,
                    self.output_folder / f'{name.lower()}_sensitivity.png',
                    plot_type='single'
                )

            self.logger.info(f"Saved {name} sensitivity results and plot")

        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(self.output_folder / 'all_sensitivity_results.csv')

        if self.reporting_manager:
            self.reporting_manager.visualize_sensitivity_analysis(
                comparison_df,
                self.output_folder / 'sensitivity_comparison.png',
                plot_type='comparison'
            )

        self.logger.info("Saved comparison of all sensitivity results")

        self.logger.info("Sensitivity analysis completed successfully")
        return comparison_df
