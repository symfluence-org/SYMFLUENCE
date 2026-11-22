# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO PostProcessor for SYMFLUENCE

Handles extraction and processing of IGNACIO fire simulation results:
- Fire perimeter shapefile collection
- Burned area statistics
- Comparison with WMFire results
- Validation against observed fire perimeters
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from symfluence.core.modeling.base.base_postprocessor import BaseModelPostProcessor
from symfluence.core.registries import R

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@R.postprocessors.add('IGNACIO')
class IGNACIOPostProcessor(BaseModelPostProcessor):
    """
    Postprocessor for IGNACIO fire spread model results.

    Handles:
    - Collecting fire perimeter shapefiles
    - Computing burned area statistics
    - Comparing IGNACIO results with WMFire
    - Validating against observed fire perimeters
    """

    def __init__(self, config, logger_instance=None, reporting_manager=None):
        """
        Initialize the IGNACIO postprocessor.

        Args:
            config: SymfluenceConfig object with domain and model settings
            logger_instance: Optional logger for status messages
            reporting_manager: Optional reporting manager for experiment tracking
        """
        super().__init__(config, logger_instance or logger, reporting_manager)

        # IGNACIO-specific paths
        self.ignacio_output_dir = self.get_simulation_dir('IGNACIO', self.config.domain.experiment_id)

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "IGNACIO"

    def extract_streamflow(self) -> Optional[Path]:
        """IGNACIO is a wildfire-perimeter model and produces no streamflow.

        The streamflow-oriented hook is required by ``BaseModelPostProcessor`` but
        is not applicable here; fire perimeters are handled by
        :meth:`run_postprocessing`. Returning ``None`` keeps the class concrete and
        instantiable without implying a streamflow output.
        """
        return None

    def run_postprocessing(self, **kwargs) -> bool:
        """
        Execute IGNACIO postprocessing.

        Collects results, computes statistics, and optionally compares
        with WMFire results.

        Returns:
            True if postprocessing completed successfully
        """
        self.logger.info("Running IGNACIO postprocessing...")

        try:
            # Verify output directory exists
            if not self.ignacio_output_dir.exists():
                self.logger.warning(f"IGNACIO output directory not found: {self.ignacio_output_dir}")
                return False

            # Collect fire perimeters
            perimeters = self._collect_perimeters()

            # Compute statistics
            stats = self._compute_statistics(perimeters)

            # Compare with WMFire if enabled
            comparison = None
            if self._should_compare_with_wmfire():
                comparison = self._compare_with_wmfire(perimeters)

            # Validate against observations if available
            validation = None
            observed_path = self._get_observed_perimeter_path()
            if observed_path:
                validation = self._validate_against_observed(perimeters, observed_path)

            # Write summary
            self._write_summary(stats, comparison, validation)

            self.logger.info("IGNACIO postprocessing complete")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"IGNACIO postprocessing failed: {e}", exc_info=True)
            return False

    def _collect_perimeters(self) -> List[Path]:
        """
        Collect fire perimeter shapefiles from output directory.

        Returns:
            List of paths to perimeter shapefiles
        """
        perimeters: List[Path] = []

        # Look for shapefiles in output directory
        for pattern in ['*.shp', '**/perimeters/*.shp', '**/fire_*.shp']:
            perimeters.extend(self.ignacio_output_dir.glob(pattern))

        # Remove duplicates while preserving order
        perimeters = list(dict.fromkeys(perimeters))

        self.logger.info(f"Found {len(perimeters)} fire perimeter file(s)")
        return perimeters

    def _compute_statistics(self, perimeters: List[Path]) -> Dict[str, Any]:
        """
        Compute burned area statistics from perimeter shapefiles.

        Args:
            perimeters: List of perimeter shapefile paths

        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_perimeters': len(perimeters),
            'total_area_ha': 0.0,
            'perimeter_files': [str(p) for p in perimeters],
        }

        try:
            import geopandas as gpd

            total_area = 0.0
            areas = []

            for perimeter_path in perimeters:
                try:
                    gdf = gpd.read_file(perimeter_path)

                    # Ensure we have a projected CRS for accurate area calculation
                    if gdf.crs and gdf.crs.is_geographic:
                        # Estimate UTM zone from centroid
                        centroid = gdf.geometry.unary_union.centroid
                        utm_zone = int((centroid.x + 180) / 6) + 1
                        utm_crs = f"EPSG:{32600 + utm_zone}" if centroid.y >= 0 else f"EPSG:{32700 + utm_zone}"
                        gdf = gdf.to_crs(utm_crs)

                    # Calculate area in hectares
                    area_ha = gdf.geometry.area.sum() / 10000
                    areas.append(area_ha)
                    total_area += area_ha

                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Could not read {perimeter_path}: {e}", exc_info=True)

            stats['total_area_ha'] = total_area
            stats['perimeter_areas_ha'] = areas

            if areas:
                stats['mean_area_ha'] = sum(areas) / len(areas)
                stats['max_area_ha'] = max(areas)
                stats['min_area_ha'] = min(areas)

        except ImportError:
            self.logger.warning("geopandas not available for statistics calculation")

        return stats

    def _should_compare_with_wmfire(self) -> bool:
        """Check if WMFire comparison is enabled."""
        # Check IGNACIO config
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'ignacio'):
            ignacio = self.config.model.ignacio
            if ignacio and hasattr(ignacio, 'compare_with_wmfire'):
                return ignacio.compare_with_wmfire

        # Check config dict
        return self._get_config_value(
            lambda: self.config.model.ignacio.compare_with_wmfire,
            dict_key='IGNACIO_COMPARE_WMFIRE',
        )

    def _compare_with_wmfire(self, ignacio_perimeters: List[Path]) -> Optional[Dict[str, Any]]:
        """
        Compare IGNACIO results with WMFire simulation.

        Args:
            ignacio_perimeters: List of IGNACIO perimeter paths

        Returns:
            Comparison metrics dictionary or None
        """
        try:
            import geopandas as gpd

            from symfluence.models.wmfire import FirePerimeterValidator

            # Find WMFire output
            wmfire_output_dir = self.get_simulation_dir('RHESSys', self.config.domain.experiment_id)

            # Look for WMFire perimeter files
            wmfire_perimeters = list(wmfire_output_dir.glob('**/fire_perimeter*.shp'))
            if not wmfire_perimeters:
                wmfire_perimeters = list(wmfire_output_dir.glob('**/*perimeter*.shp'))

            if not wmfire_perimeters:
                self.logger.warning("No WMFire perimeters found for comparison")
                return None

            if not ignacio_perimeters:
                self.logger.warning("No IGNACIO perimeters available for comparison")
                return None

            # Load perimeters
            ignacio_gdf = gpd.read_file(ignacio_perimeters[0])
            wmfire_gdf = gpd.read_file(wmfire_perimeters[0])

            # Use FirePerimeterValidator for comparison
            validator = FirePerimeterValidator(logger_instance=self.logger)
            metrics = validator.compare_perimeters(ignacio_gdf, wmfire_gdf)

            # Create comparison map
            comparison_dir = self.ignacio_output_dir / "comparison"
            comparison_dir.mkdir(exist_ok=True)

            map_path = comparison_dir / "ignacio_vs_wmfire.png"
            validator.create_comparison_map(
                ignacio_gdf,
                wmfire_gdf,
                map_path,
                title="IGNACIO vs WMFire Comparison"
            )

            # Save metrics
            metrics_path = comparison_dir / "comparison_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            self.logger.info(f"WMFire comparison: IoU={metrics.get('iou', 0):.3f}, "
                           f"Dice={metrics.get('dice', 0):.3f}")

            return metrics

        except ImportError as e:
            self.logger.warning(f"Cannot compare with WMFire: {e}")
            return None

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"WMFire comparison failed: {e}", exc_info=True)
            return None

    def _get_observed_perimeter_path(self) -> Optional[Path]:
        """Get path to observed fire perimeter for validation."""
        # Check IGNACIO config
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'ignacio'):
            ignacio = self.config.model.ignacio
            if ignacio and hasattr(ignacio, 'observed_perimeter_path'):
                path = ignacio.observed_perimeter_path
                if path and Path(path).exists():
                    return Path(path)

        # Check config dict
        obs_path = self._get_config_value(lambda: None, default=None, dict_key='IGNACIO_OBSERVED_PERIMETER')
        if obs_path and Path(obs_path).exists():
            return Path(obs_path)

        # Check WMFire perimeter dir
        wmfire_perim_dir = self._get_config_value(lambda: self.config.model.wmfire.perimeter_dir, default=None)
        if wmfire_perim_dir and Path(wmfire_perim_dir).exists():
            shapefiles = list(Path(wmfire_perim_dir).glob('*.shp'))
            if shapefiles:
                return shapefiles[0]

        return None

    def _validate_against_observed(
        self,
        simulated_perimeters: List[Path],
        observed_path: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Validate IGNACIO results against observed fire perimeter.

        Args:
            simulated_perimeters: List of simulated perimeter paths
            observed_path: Path to observed perimeter

        Returns:
            Validation metrics dictionary or None
        """
        try:
            import geopandas as gpd

            from symfluence.models.wmfire import FirePerimeterValidator

            if not simulated_perimeters:
                return None

            # Load perimeters
            sim_gdf = gpd.read_file(simulated_perimeters[0])
            obs_gdf = gpd.read_file(observed_path)

            # Use FirePerimeterValidator
            validator = FirePerimeterValidator(logger_instance=self.logger)
            metrics = validator.compare_perimeters(sim_gdf, obs_gdf)

            # Create validation map
            validation_dir = self.ignacio_output_dir / "validation"
            validation_dir.mkdir(exist_ok=True)

            map_path = validation_dir / "ignacio_vs_observed.png"
            validator.create_comparison_map(
                sim_gdf,
                obs_gdf,
                map_path,
                title="IGNACIO vs Observed Perimeter"
            )

            # Save metrics
            metrics_path = validation_dir / "validation_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            self.logger.info(f"Validation against observed: IoU={metrics.get('iou', 0):.3f}, "
                           f"Dice={metrics.get('dice', 0):.3f}")

            return metrics

        except ImportError as e:
            self.logger.warning(f"Cannot validate against observed: {e}")
            return None

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Validation failed: {e}", exc_info=True)
            return None

    def _write_summary(
        self,
        stats: Dict[str, Any],
        comparison: Optional[Dict[str, Any]],
        validation: Optional[Dict[str, Any]]
    ) -> None:
        """
        Write summary JSON file with all results.

        Args:
            stats: Burned area statistics
            comparison: WMFire comparison metrics (optional)
            validation: Observed perimeter validation metrics (optional)
        """
        summary = {
            'model': 'IGNACIO',
            'domain': self.config.domain.name,
            'experiment_id': self.config.domain.experiment_id,
            'statistics': stats,
        }

        if comparison:
            summary['wmfire_comparison'] = comparison

        if validation:
            summary['observed_validation'] = validation

        # Write summary
        summary_path = self.ignacio_output_dir / "ignacio_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary written: {summary_path}")
