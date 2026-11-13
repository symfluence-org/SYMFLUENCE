# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Drop analysis for objective stream threshold optimization.

Implements Tarboton et al. (1991) drop analysis method to objectively
determine optimal stream thresholds based on geomorphological principles.

Extracted from geofabric_utils.py (2026-01-01)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.mixins import ConfigMixin


class DropAnalysisMethod(ConfigMixin):
    """
    Drop analysis for objective threshold selection.

    Examines the relationship between drainage area and stream drop to identify
    where streams objectively begin based on geomorphological principles.
    """

    def __init__(self, taudem_executor: Any, config: Dict, logger: Any, interim_dir: Path, reporting_manager: Optional[Any] = None):
        """
        Initialize drop analysis.

        Args:
            taudem_executor: TauDEMExecutor instance for running commands
            config: Configuration dictionary
            logger: Logger instance
            interim_dir: Directory for interim TauDEM files
            reporting_manager: ReportingManager instance
        """
        self.taudem = taudem_executor
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.interim_dir = interim_dir
        self.reporting_manager = reporting_manager
        self.taudem_dir = taudem_executor.taudem_dir
        self.project_dir = interim_dir.parent.parent

    def run(self, mpi_prefix: str) -> Optional[float]:
        """
        Perform drop analysis to determine optimal stream threshold.

        Returns:
            Optimal threshold value if analysis succeeds, None otherwise
        """
        try:
            self.logger.info("Running drop analysis to optimize stream threshold")

            # Get parameters
            min_threshold = self._get_config_value(
                lambda: self.config.domain.delineation.drop_analysis_min_threshold,
                dict_key='DROP_ANALYSIS_MIN_THRESHOLD',
            )
            max_threshold = self._get_config_value(
                lambda: self.config.domain.delineation.drop_analysis_max_threshold,
                dict_key='DROP_ANALYSIS_MAX_THRESHOLD',
            )
            num_thresholds = self._get_config_value(
                lambda: self.config.domain.delineation.drop_analysis_num_thresholds,
                dict_key='DROP_ANALYSIS_NUM_THRESHOLDS',
            )
            use_log_spacing = self._get_config_value(
                lambda: self.config.domain.delineation.drop_analysis_log_spacing,
                dict_key='DROP_ANALYSIS_LOG_SPACING',
            )

            # Generate threshold values to test
            if use_log_spacing:
                thresholds = np.logspace(
                    np.log10(min_threshold),
                    np.log10(max_threshold),
                    num_thresholds
                )
            else:
                thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)

            drop_data = []

            # Test each threshold
            for threshold in thresholds:
                self.logger.info(f"Testing threshold: {threshold:.0f}")

                # Create stream source at this threshold
                src_file = self.interim_dir / f"elv-src-drop{int(threshold)}.tif"
                cmd = f"{mpi_prefix}{self.taudem_dir}/threshold -ssa {self.interim_dir}/elv-ad8.tif -src {src_file} -thresh {threshold}"
                self.taudem.run_command(cmd)

                # Calculate drop statistics
                drop_file = self.interim_dir / f"drop-stats-{int(threshold)}.txt"
                cmd = f"{mpi_prefix}{self.taudem_dir}/dropanalysis -p {self.interim_dir}/elv-fdir.tif -fel {self.interim_dir}/elv-fel.tif -ad8 {self.interim_dir}/elv-ad8.tif -src {src_file} -par {threshold} 0 -o {drop_file}"

                try:
                    self.taudem.run_command(cmd)

                    # Read drop analysis results
                    if drop_file.exists():
                        with open(drop_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > 1:  # Skip header
                                parts = lines[-1].strip().split()
                                if len(parts) >= 3:
                                    drop_data.append({
                                        'threshold': threshold,
                                        'num_sources': float(parts[1]),
                                        'mean_drop': float(parts[2])
                                    })
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Drop analysis failed for threshold {threshold}: {str(e)}", exc_info=True)
                    continue

            if len(drop_data) < 3:
                self.logger.warning("Insufficient data for drop analysis. Using default threshold.")
                return None

            # Find optimal threshold
            optimal_threshold = self._find_optimal_threshold(drop_data)

            # Save plot
            if self.reporting_manager:
                self.reporting_manager.visualize_drop_analysis(drop_data, optimal_threshold, self.project_dir)

            return optimal_threshold

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error in drop analysis: {str(e)}", exc_info=True)
            return None

    def _find_optimal_threshold(self, drop_data: List[Dict]) -> float:
        """
        Find the optimal threshold using the elbow method.

        Args:
            drop_data: List of dictionaries with threshold and drop statistics

        Returns:
            Optimal threshold value
        """
        thresholds = np.array([d['threshold'] for d in drop_data])
        mean_drops = np.array([d['mean_drop'] for d in drop_data])

        # Log transform for better analysis
        log_thresh = np.log10(thresholds)
        log_drops = np.log10(mean_drops + 1)  # Add 1 to avoid log(0)

        # Find point of maximum curvature (elbow)
        if len(log_thresh) >= 3:
            # Calculate finite differences for second derivative
            first_deriv = np.diff(log_drops) / np.diff(log_thresh)
            second_deriv = np.diff(first_deriv) / np.diff(log_thresh[:-1])

            # Find maximum absolute second derivative
            max_curvature_idx = np.argmax(np.abs(second_deriv))
            optimal_threshold = thresholds[max_curvature_idx + 1]
        else:
            # Fallback: use median
            optimal_threshold = np.median(thresholds)

        self.logger.info(f"Optimal threshold from drop analysis: {optimal_threshold:.0f}")

        return optimal_threshold
