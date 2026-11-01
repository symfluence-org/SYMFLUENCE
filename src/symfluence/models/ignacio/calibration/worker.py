# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO Calibration Worker.

Worker implementation for IGNACIO fire model FBP parameter calibration.
Uses spatial metrics (IoU/Dice) between simulated and observed fire
perimeters as objective functions.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add('IGNACIO')
class IGNACIOWorker(BaseWorker):
    """
    Worker for IGNACIO fire model parameter calibration.

    Calibrates FBP (Fire Behavior Prediction) parameters by comparing
    simulated fire perimeters against observed perimeters using spatial
    overlap metrics.

    Calibration Parameters:
        - ffmc: Fine Fuel Moisture Code (0-101)
        - dmc: Duff Moisture Code (0-200)
        - dc: Drought Code (0-800)
        - fmc: Foliar Moisture Content (50-150%)
        - curing: Grass curing percentage (0-100%)
        - initial_radius: Initial fire radius (1-100 m)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, logger)

        self._observed_perimeter = None
        self._ignacio_config = None

    def supports_native_gradients(self) -> bool:
        """IGNACIO does not support native gradients."""
        return False

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply FBP parameters to IGNACIO configuration.

        Updates the ignacio_config.yaml FBP defaults section with
        calibrated parameter values.

        Args:
            params: FBP parameter values
            settings_dir: Path to settings directory
            **kwargs: Additional arguments

        Returns:
            True if parameters applied successfully.
        """
        try:
            config_path = self._find_config_path(settings_dir)
            if config_path is None:
                self.logger.warning("No IGNACIO config found, storing in-memory only")
                self._current_params = params
                return True

            import yaml
            with open(config_path, encoding='utf-8') as f:
                ignacio_config = yaml.safe_load(f) or {}

            # Update FBP defaults
            if 'fbp' not in ignacio_config:
                ignacio_config['fbp'] = {}
            fbp = ignacio_config['fbp']

            param_to_yaml = {
                'ffmc': 'ffmc_default',
                'dmc': 'dmc_default',
                'dc': 'dc_default',
                'fmc': 'fmc',
                'curing': 'curing',
                'initial_radius': 'initial_radius',
            }

            for param_name, value in params.items():
                yaml_key = param_to_yaml.get(param_name, param_name)
                if yaml_key == 'initial_radius':
                    if 'simulation' not in ignacio_config:
                        ignacio_config['simulation'] = {}
                    ignacio_config['simulation']['initial_radius'] = float(value)
                else:
                    fbp[yaml_key] = float(value)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(ignacio_config, f, default_flow_style=False)

            self._current_params = params
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying IGNACIO params: {e}", exc_info=True)
            return False

    def _find_config_path(self, settings_dir: Path) -> Optional[Path]:
        """Find IGNACIO config file in settings or input directory."""
        candidates = [
            settings_dir / 'ignacio_config.yaml',
            settings_dir.parent / 'IGNACIO_input' / 'ignacio_config.yaml',
        ]
        if self.config:
            project_dir = Path(resolve_config_value(self.config, lambda: self.config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR')) / f"domain_{resolve_config_value(self.config, lambda: self.config.domain.name, '', 'DOMAIN_NAME')}"
            candidates.append(project_dir / 'IGNACIO_input' / 'ignacio_config.yaml')

        for path in candidates:
            if path.exists():
                return path
        return None

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run IGNACIO fire simulation.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if simulation succeeded.
        """
        try:
            config_path = self._find_config_path(settings_dir)
            if config_path is None:
                self.logger.error("IGNACIO config not found")
                return False

            # Try Python API first
            try:
                from ignacio.config import IgnacioConfig
                from ignacio.simulation import run_simulation

                ignacio_cfg = IgnacioConfig.from_yaml(str(config_path))
                ignacio_cfg.output_dir = str(output_dir)
                run_simulation(ignacio_cfg)
                return True

            except ImportError:
                # Fall back to CLI
                result = run_subprocess(
                    ['ignacio', 'run', str(config_path), '--output-dir', str(output_dir)],
                    capture_output=True, text=True, timeout=3600
                )
                return result.returncode == 0

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error running IGNACIO: {e}", exc_info=True)
            self.logger.debug(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate spatial metrics between simulated and observed perimeters.

        Uses IoU (Intersection over Union) and Dice coefficient.

        Args:
            output_dir: Directory containing IGNACIO outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary with metric values.
        """
        try:
            # Find simulated perimeters
            sim_perimeters = self._find_perimeters(Path(output_dir))
            if not sim_perimeters:
                return {'iou': self.penalty_score, 'error': 'No simulated perimeters'}

            # Find observed perimeter
            obs_path = self._get_observed_path(config)
            if obs_path is None:
                return {'iou': self.penalty_score, 'error': 'No observed perimeters'}

            import geopandas as gpd

            sim_gdf = gpd.read_file(sim_perimeters[0])
            obs_gdf = gpd.read_file(obs_path)

            metrics = self._compute_spatial_metrics(sim_gdf, obs_gdf)

            # Use IoU as primary score (maximized by optimizer)
            return {
                'iou': float(metrics.get('iou', 0.0)),
                'dice': float(metrics.get('dice', 0.0)),
                'area_ratio': float(metrics.get('area_ratio', 0.0)),
            }

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating IGNACIO metrics: {e}", exc_info=True)
            return {'iou': self.penalty_score, 'error': str(e)}

    def _find_perimeters(self, output_dir: Path) -> list:
        """Find fire perimeter shapefiles in output directory."""
        perimeters: list[Path] = []
        for pattern in ['*.shp', '**/perimeters/*.shp', '**/fire_*.shp']:
            perimeters.extend(output_dir.glob(pattern))
        return list(dict.fromkeys(perimeters))

    def _get_observed_path(self, config: Dict[str, Any]) -> Optional[Path]:
        """Get path to observed fire perimeter."""
        obs_path = resolve_config_value(config, lambda: config.model.ignacio.observed_perimeter_path, None, 'IGNACIO_OBSERVED_PERIMETER')
        if obs_path and Path(obs_path).exists():
            return Path(obs_path)

        wmfire_dir = resolve_config_value(config, lambda: config.model.rhessys.wmfire.perimeter_dir, None, 'WMFIRE_PERIMETER_DIR')
        if wmfire_dir and Path(wmfire_dir).exists():
            shapefiles = list(Path(wmfire_dir).glob('*.shp'))
            if shapefiles:
                return shapefiles[0]

        return None

    def _compute_spatial_metrics(self, sim_gdf, obs_gdf) -> Dict[str, float]:
        """
        Compute IoU and Dice between simulated and observed perimeters.

        Reuses logic from FirePerimeterValidator when available,
        otherwise computes directly.
        """
        try:
            from symfluence.models.wmfire import FirePerimeterValidator
            validator = FirePerimeterValidator(logger_instance=self.logger)
            return validator.compare_perimeters(sim_gdf, obs_gdf)
        except ImportError:
            pass

        # Direct computation fallback
        try:
            # Ensure same CRS
            if sim_gdf.crs != obs_gdf.crs:
                obs_gdf = obs_gdf.to_crs(sim_gdf.crs)

            sim_union = sim_gdf.geometry.unary_union
            obs_union = obs_gdf.geometry.unary_union

            intersection = sim_union.intersection(obs_union)
            union = sim_union.union(obs_union)

            intersection_area = intersection.area
            union_area = union.area
            sim_area = sim_union.area
            obs_area = obs_union.area

            iou = intersection_area / union_area if union_area > 0 else 0.0
            dice = (2 * intersection_area / (sim_area + obs_area)
                    if (sim_area + obs_area) > 0 else 0.0)

            return {
                'iou': iou,
                'dice': dice,
                'simulated_area_ha': sim_area / 10000.0,
                'observed_area_ha': obs_area / 10000.0,
                'area_ratio': sim_area / obs_area if obs_area > 0 else 0.0,
            }

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error computing spatial metrics: {e}", exc_info=True)
            return {'iou': 0.0, 'dice': 0.0}
