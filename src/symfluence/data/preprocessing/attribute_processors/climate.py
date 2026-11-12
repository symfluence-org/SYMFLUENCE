# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Climate attribute processor.

Handles climate data processing including:
- WorldClim raw variables (precipitation, temperature, radiation, wind, vapor pressure)
- Monthly time series processing
- Derived climate indices (PET, moisture index, snow)
- Seasonality metrics (Walsh & Lawler, Markham indices)
- Aridity classification
"""
from __future__ import annotations

import os

# Security rationale: used for trusted local cache files
import pickle  # nosec B403
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats

from .base import BaseAttributeProcessor


class ClimateProcessor(BaseAttributeProcessor):
    """Processor for climate attributes."""

    def process(self) -> Dict[str, Any]:
        """
        Process climate attributes.

        Returns:
            Dictionary of climate attributes
        """
        results: Dict[str, Any] = {}

        # Find WorldClim data path
        worldclim_path = self._get_data_path('ATTRIBUTES_WORLDCLIM_PATH', 'data/attributes/climate/worldclim')

        if not worldclim_path.exists():
            self.logger.warning(f"WorldClim path not found: {worldclim_path}")
            return results

        # Process raw climate variables
        raw_climate = self._process_raw_climate_variables(worldclim_path)
        results.update(raw_climate)

        # Process derived climate indices
        derived_climate = self._process_derived_climate_indices(worldclim_path)
        results.update(derived_climate)

        return results

    def _process_raw_climate_variables(self, worldclim_path: Path) -> Dict[str, Any]:
        """
        Process raw climate variables from WorldClim.

        Args:
            worldclim_path: Path to WorldClim data

        Returns:
            Dictionary of climate attributes
        """
        results: Dict[str, Any] = {}

        # Create cache directory
        cache_dir = self.project_dir / 'cache' / 'climate'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_raw_climate_results.pickle"

        # Check cache
        if cache_file.exists():
            self.logger.info("Loading cached raw climate results")
            try:
                with open(cache_file, 'rb') as f:
                    # Security rationale: loading trusted local cache files
                    return pickle.load(f)  # nosec B301
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error loading cache: {e}")

        self.logger.info("Processing raw climate variables")

        # Define the raw climate variables to process
        raw_variables = {
            'prec': {'unit': 'mm/month', 'description': 'Precipitation'},
            'tavg': {'unit': '°C', 'description': 'Average Temperature'},
            'tmax': {'unit': '°C', 'description': 'Maximum Temperature'},
            'tmin': {'unit': '°C', 'description': 'Minimum Temperature'},
            'srad': {'unit': 'kJ/m²/day', 'description': 'Solar Radiation'},
            'wind': {'unit': 'm/s', 'description': 'Wind Speed'},
            'vapr': {'unit': 'kPa', 'description': 'Vapor Pressure'}
        }

        is_lumped = self._is_lumped()
        catchment = None if is_lumped else gpd.read_file(self.catchment_path)
        hru_id_field = self._get_config_value(
            lambda: self.config.paths.catchment_hruid,
            dict_key='CATCHMENT_SHP_HRUID',
        ) if not is_lumped else None

        # Process each climate variable
        for var, var_info in raw_variables.items():
            self.logger.info(f"Processing WorldClim {var_info['description']} ({var})")

            # Search for monthly files: flat layout (acquisition) or nested (legacy)
            monthly_files = sorted(worldclim_path.glob(f"wc2.1_30s_{var}_*.tif"))
            if not monthly_files:
                var_path = worldclim_path / 'raw' / var
                monthly_files = sorted(var_path.glob(f"wc2.1_30s_{var}_*.tif"))

            if not monthly_files:
                self.logger.warning(f"No {var} files found in {var_path}")
                continue

            # Initialize monthly values container
            monthly_values: List[List[float]] = [] if is_lumped else [[] for _ in range(len(catchment))]
            monthly_attributes = {}

            for month_file in monthly_files:
                # Extract month number from filename
                month_str = os.path.basename(month_file).split('_')[-1].split('.')[0]
                try:
                    month = int(month_str)
                except ValueError:
                    self.logger.warning(f"Could not extract month from filename: {month_file}")
                    continue

                # Calculate zonal statistics for this month
                try:
                    stats = ['mean', 'min', 'max', 'std']
                    zonal_out = zonal_stats(
                        str(self.catchment_path),
                        str(month_file),
                        stats=stats,
                        all_touched=True
                    )

                    if not zonal_out:
                        continue

                    if is_lumped:
                        # For lumped catchment
                        if len(zonal_out) > 0:
                            monthly_values.append(zonal_out[0].get('mean', np.nan))

                            for stat in stats:
                                if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                    monthly_attributes[f"climate.{var}_m{month:02d}_{stat}"] = zonal_out[0][stat]
                    else:
                        # For distributed catchment
                        for i, zonal_result in enumerate(zonal_out):
                            if i < len(catchment):
                                hru_id = catchment.iloc[i][hru_id_field]
                                prefix = f"HRU_{hru_id}_"

                                for stat in stats:
                                    if stat in zonal_result and zonal_result[stat] is not None:
                                        monthly_attributes[f"{prefix}climate.{var}_m{month:02d}_{stat}"] = zonal_result[stat]

                                # Track monthly mean values for annual calculations
                                if zonal_result.get('mean') is not None:
                                    # Ensure list is long enough
                                    while len(monthly_values[i]) < month:
                                        monthly_values[i].append(np.nan)
                                    if len(monthly_values[i]) == month - 1:
                                        monthly_values[i].append(zonal_result['mean'])
                                    else:
                                        monthly_values[i][month - 1] = zonal_result['mean']

                except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                    self.logger.error(f"Error processing {var} for month {month}: {e}")

            # Add monthly attributes to results
            results.update(monthly_attributes)

            # Calculate annual statistics
            if is_lumped:
                monthly_values_clean = [v for v in monthly_values if not np.isnan(v)]
                if monthly_values_clean:
                    self._calculate_annual_stats_lumped(results, var, monthly_values_clean)
            else:
                for i in range(len(catchment)):
                    hru_id = catchment.iloc[i][hru_id_field]
                    prefix = f"HRU_{hru_id}_"

                    hru_values_clean = [v for v in monthly_values[i] if not np.isnan(v)]
                    if hru_values_clean:
                        self._calculate_annual_stats_distributed(results, var, prefix, hru_values_clean)

            # Define aridity zones based on annual precipitation (lumped only)
            if var == 'prec' and is_lumped and 'climate.prec_annual_mean' in results:
                self._classify_aridity_zone_precip(results, results['climate.prec_annual_mean'])

        # Cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.warning(f"Error caching results: {e}")

        return results

    def _calculate_annual_stats_lumped(self, results: Dict[str, Any], var: str, monthly_values: list):
        """Calculate annual statistics for lumped catchment."""
        # Annual mean
        annual_mean = np.mean(monthly_values)
        results[f"climate.{var}_annual_mean"] = annual_mean

        # Annual range (max - min)
        annual_range = np.max(monthly_values) - np.min(monthly_values)
        results[f"climate.{var}_annual_range"] = annual_range

        # Seasonality metrics
        if len(monthly_values) >= 12:
            # Calculate seasonality index (standard deviation / mean)
            if annual_mean > 0:
                seasonality_index = np.std(monthly_values) / annual_mean
                results[f"climate.{var}_seasonality_index"] = seasonality_index

            # For precipitation: calculate precipitation seasonality
            if var == 'prec':
                # Walsh & Lawler (1981) seasonality index
                annual_total = np.sum(monthly_values)
                if annual_total > 0:
                    monthly_diff = [abs(p - annual_total / 12) for p in monthly_values]
                    walsh_index = sum(monthly_diff) / annual_total
                    results["climate.prec_walsh_seasonality_index"] = walsh_index

                # Markham Seasonality Index - vector-based measure
                if len(monthly_values) == 12:
                    month_angles = np.arange(0, 360, 30) * np.pi / 180  # in radians
                    x_sum = sum(p * np.sin(a) for p, a in zip(monthly_values, month_angles))
                    y_sum = sum(p * np.cos(a) for p, a in zip(monthly_values, month_angles))

                    # Markham concentration index (0-1, higher = more seasonal)
                    vector_magnitude = np.sqrt(x_sum**2 + y_sum**2)
                    markham_index = vector_magnitude / annual_total
                    results["climate.prec_markham_seasonality_index"] = markham_index

                    # Markham seasonality angle (direction of concentration)
                    seasonality_angle = np.arctan2(x_sum, y_sum) * 180 / np.pi
                    if seasonality_angle < 0:
                        seasonality_angle += 360

                    # Convert to month (1-12)
                    seasonality_month = round(seasonality_angle / 30) % 12
                    if seasonality_month == 0:
                        seasonality_month = 12

                    results["climate.prec_seasonality_month"] = seasonality_month

    def _calculate_annual_stats_distributed(self, results: Dict[str, Any], var: str, prefix: str, hru_values: list):
        """Calculate annual statistics for distributed catchment HRU."""
        # Annual mean
        annual_mean = np.mean(hru_values)
        results[f"{prefix}climate.{var}_annual_mean"] = annual_mean

        # Annual range
        annual_range = np.max(hru_values) - np.min(hru_values)
        results[f"{prefix}climate.{var}_annual_range"] = annual_range

        # Seasonality metrics
        if len(hru_values) >= 12:
            # Calculate seasonality index
            if annual_mean > 0:
                seasonality_index = np.std(hru_values) / annual_mean
                results[f"{prefix}climate.{var}_seasonality_index"] = seasonality_index

            # For precipitation: calculate Walsh & Lawler seasonality index
            if var == 'prec':
                annual_total = np.sum(hru_values)
                if annual_total > 0:
                    monthly_diff = [abs(p - annual_total / 12) for p in hru_values]
                    walsh_index = sum(monthly_diff) / annual_total
                    results[f"{prefix}climate.prec_walsh_seasonality_index"] = walsh_index

    def _classify_aridity_zone_precip(self, results: Dict[str, Any], annual_precip: float):
        """Classify aridity zone based on annual precipitation."""
        aridity_zones = {
            'hyperarid': (0, 100),
            'arid': (100, 400),
            'semiarid': (400, 600),
            'subhumid': (600, 1000),
            'humid': (1000, 2000),
            'superhumid': (2000, float('inf'))
        }

        for zone, (lower, upper) in aridity_zones.items():
            if lower <= annual_precip < upper:
                results['climate.aridity_zone_precip'] = zone
                break

    def _process_derived_climate_indices(self, worldclim_path: Path) -> Dict[str, Any]:
        """
        Process derived climate indices.

        Args:
            worldclim_path: Path to WorldClim data

        Returns:
            Dictionary of climate attributes
        """
        results: Dict[str, Any] = {}

        # Create cache directory
        cache_dir = self.project_dir / 'cache' / 'climate'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_derived_climate_results.pickle"

        # Check cache
        if cache_file.exists():
            self.logger.info("Loading cached derived climate results")
            try:
                with open(cache_file, 'rb') as f:
                    # Security rationale: loading trusted local cache files
                    return pickle.load(f)  # nosec B301
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error loading cache: {e}")

        self.logger.info("Processing derived climate indices")

        # Check for derived data directories
        derived_path = worldclim_path / 'derived'
        if not derived_path.exists():
            self.logger.warning(f"WorldClim derived data directory not found: {derived_path}")
            return results

        # Define the derived products to process
        derived_products = [
            {
                'name': 'pet',
                'path': derived_path / 'pet',
                'pattern': 'wc2.1_30s_pet_*.tif',
                'description': 'Potential Evapotranspiration',
                'unit': 'mm/day'
            },
            {
                'name': 'moisture_index',
                'path': derived_path / 'moisture_index',
                'pattern': 'wc2.1_30s_moisture_index_*.tif',
                'description': 'Moisture Index',
                'unit': '-'
            },
            {
                'name': 'snow',
                'path': derived_path / 'snow',
                'pattern': 'wc2.1_30s_snow_*.tif',
                'description': 'Snow',
                'unit': 'mm/month'
            },
            {
                'name': 'climate_index_im',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_im.tif',
                'description': 'Humidity Index',
                'unit': '-'
            },
            {
                'name': 'climate_index_imr',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_imr.tif',
                'description': 'Relative Humidity Index',
                'unit': '-'
            },
            {
                'name': 'climate_index_fs',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_fs.tif',
                'description': 'Fraction of Precipitation as Snow',
                'unit': '-'
            }
        ]

        is_lumped = self._is_lumped()
        catchment = None if is_lumped else gpd.read_file(self.catchment_path)
        hru_id_field = self._get_config_value(
            lambda: self.config.paths.catchment_hruid,
            dict_key='CATCHMENT_SHP_HRUID',
        ) if not is_lumped else None

        # Process each derived product
        for product in derived_products:
            product_path = product['path']

            if not product_path.exists():
                self.logger.warning(f"WorldClim {product['description']} directory not found: {product_path}")
                continue

            self.logger.info(f"Processing WorldClim {product['description']} ({product['name']})")

            # Find files for this product
            product_files = sorted(product_path.glob(product['pattern']))

            if not product_files:
                self.logger.warning(f"No {product['name']} files found in {product_path}")
                continue

            # Check if this is a monthly product
            has_monthly_files = len(product_files) > 1 and any('_01.' in file.name or '_1.' in file.name for file in product_files)

            if has_monthly_files:
                self._process_monthly_derived_product(product, product_files, results, is_lumped, catchment, hru_id_field)
            else:
                self._process_annual_derived_product(product, product_files, results, is_lumped, catchment, hru_id_field)

        # Calculate composite climate indicators
        self._calculate_composite_climate_indicators(results, is_lumped, catchment, hru_id_field)

        # Cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.warning(f"Error caching results: {e}")

        return results

    def _process_monthly_derived_product(self, product: Dict[str, Any], product_files: list,
                                         results: Dict[str, Any], is_lumped: bool,
                                         catchment, hru_id_field: str):
        """Process monthly derived climate product."""
        # container for monthly values
        monthly_values: List[List[float]] = [] if is_lumped else [[] for _ in range(len(catchment))]

        for month_file in product_files:
            # Extract month number
            month_part = month_file.name.split('_')[-1].split('.')[0]
            try:
                month = int(month_part)
            except ValueError:
                self.logger.warning(f"Could not extract month from filename: {month_file}")
                continue

            # Calculate zonal statistics
            try:
                stats = ['mean', 'min', 'max', 'std']
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(month_file),
                    stats=stats,
                    all_touched=True
                )

                if not zonal_out:
                    continue

                if is_lumped:
                    if len(zonal_out) > 0:
                        monthly_values.append(zonal_out[0].get('mean', np.nan))

                        for stat in stats:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                results[f"climate.{product['name']}_m{month:02d}_{stat}"] = zonal_out[0][stat]
                else:
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"

                            for stat in stats:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    results[f"{prefix}climate.{product['name']}_m{month:02d}_{stat}"] = zonal_result[stat]

                            # Track monthly values
                            if zonal_result.get('mean') is not None:
                                while len(monthly_values[i]) < month:
                                    monthly_values[i].append(np.nan)
                                if len(monthly_values[i]) == month - 1:
                                    monthly_values[i].append(zonal_result['mean'])
                                else:
                                    monthly_values[i][month - 1] = zonal_result['mean']

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error processing {product['name']} for month {month}: {e}")

        # Calculate annual statistics for monthly products
        if is_lumped:
            monthly_values_clean = [v for v in monthly_values if not np.isnan(v)]
            if monthly_values_clean:
                results[f"climate.{product['name']}_annual_mean"] = np.mean(monthly_values_clean)
                results[f"climate.{product['name']}_annual_min"] = np.min(monthly_values_clean)
                results[f"climate.{product['name']}_annual_max"] = np.max(monthly_values_clean)
                results[f"climate.{product['name']}_annual_range"] = np.max(monthly_values_clean) - np.min(monthly_values_clean)

                # Calculate seasonality index for applicable variables
                if product['name'] in ['pet', 'snow']:
                    annual_mean = results[f"climate.{product['name']}_annual_mean"]
                    if annual_mean > 0:
                        seasonality_index = np.std(monthly_values_clean) / annual_mean
                        results[f"climate.{product['name']}_seasonality_index"] = seasonality_index

    def _process_annual_derived_product(self, product: Dict[str, Any], product_files: list,
                                        results: Dict[str, Any], is_lumped: bool,
                                        catchment, hru_id_field: str):
        """Process annual (non-monthly) derived climate product."""
        for product_file in product_files:
            try:
                stats = ['mean', 'min', 'max', 'std']
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(product_file),
                    stats=stats,
                    all_touched=True
                )

                if not zonal_out:
                    continue

                if is_lumped:
                    if len(zonal_out) > 0:
                        for stat in stats:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                results[f"climate.{product['name']}_{stat}"] = zonal_out[0][stat]
                else:
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"

                            for stat in stats:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    results[f"{prefix}climate.{product['name']}_{stat}"] = zonal_result[stat]

            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.error(f"Error processing {product['name']}: {e}")

    def _calculate_composite_climate_indicators(self, results: Dict[str, Any], is_lumped: bool,
                                                 catchment, hru_id_field: str):
        """Calculate composite climate indicators from raw and derived data."""
        if is_lumped:
            # Aridity index (PET/P)
            if "climate.prec_annual_mean" in results and "climate.pet_annual_mean" in results:
                precip = results["climate.prec_annual_mean"]
                pet = results["climate.pet_annual_mean"]

                if precip > 0:
                    aridity_index = pet / precip
                    results["climate.aridity_index"] = aridity_index

                    # Classify aridity zone based on UNEP criteria
                    if aridity_index < 0.03:
                        results["climate.aridity_zone"] = "hyperhumid"
                    elif aridity_index < 0.2:
                        results["climate.aridity_zone"] = "humid"
                    elif aridity_index < 0.5:
                        results["climate.aridity_zone"] = "subhumid"
                    elif aridity_index < 0.65:
                        results["climate.aridity_zone"] = "dry_subhumid"
                    elif aridity_index < 1.0:
                        results["climate.aridity_zone"] = "semiarid"
                    elif aridity_index < 3.0:
                        results["climate.aridity_zone"] = "arid"
                    else:
                        results["climate.aridity_zone"] = "hyperarid"

            # Snow fraction
            if "climate.snow_annual_mean" in results and "climate.prec_annual_mean" in results:
                snow = results["climate.snow_annual_mean"] * 12  # Convert mean to annual total
                precip = results["climate.prec_annual_mean"] * 12

                if precip > 0:
                    snow_fraction = snow / precip
                    results["climate.snow_fraction"] = snow_fraction
        else:
            # For distributed catchment
            if catchment is not None:
                for i in range(len(catchment)):
                    hru_id = catchment.iloc[i][hru_id_field]
                    prefix = f"HRU_{hru_id}_"

                    # Aridity index
                    prec_key = f"{prefix}climate.prec_annual_mean"
                    pet_key = f"{prefix}climate.pet_annual_mean"

                    if prec_key in results and pet_key in results:
                        precip = results[prec_key]
                        pet = results[pet_key]

                        if precip > 0:
                            aridity_index = pet / precip
                            results[f"{prefix}climate.aridity_index"] = aridity_index

                            # Classify aridity zone
                            if aridity_index < 0.03:
                                results[f"{prefix}climate.aridity_zone"] = "hyperhumid"
                            elif aridity_index < 0.2:
                                results[f"{prefix}climate.aridity_zone"] = "humid"
                            elif aridity_index < 0.5:
                                results[f"{prefix}climate.aridity_zone"] = "subhumid"
                            elif aridity_index < 0.65:
                                results[f"{prefix}climate.aridity_zone"] = "dry_subhumid"
                            elif aridity_index < 1.0:
                                results[f"{prefix}climate.aridity_zone"] = "semiarid"
                            elif aridity_index < 3.0:
                                results[f"{prefix}climate.aridity_zone"] = "arid"
                            else:
                                results[f"{prefix}climate.aridity_zone"] = "hyperarid"

                    # Snow fraction
                    snow_key = f"{prefix}climate.snow_annual_mean"
                    if snow_key in results and prec_key in results:
                        snow = results[snow_key] * 12
                        precip = results[prec_key] * 12

                        if precip > 0:
                            snow_fraction = snow / precip
                            results[f"{prefix}climate.snow_fraction"] = snow_fraction
