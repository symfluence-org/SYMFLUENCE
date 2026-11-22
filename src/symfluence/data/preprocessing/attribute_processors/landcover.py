# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Land cover attribute processor.

Handles land cover and vegetation attributes including:
- GLCLU2019 classification
- Leaf Area Index (LAI) monthly processing
- Forest height metrics
- Irrigation attributes
- Composite ecological groupings and diversity metrics
"""
from __future__ import annotations

import pickle  # nosec B403
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats

from .base import BaseAttributeProcessor


class LandCoverProcessor(BaseAttributeProcessor):
    """Processor for land cover and vegetation attributes."""

    def process(self) -> Dict[str, Any]:
        """
        Process land cover attributes.

        Returns:
            Dictionary of land cover attributes
        """
        results = {}

        # Process GLCLU2019 land cover classification
        glclu_results = self._process_glclu2019_landcover()
        results.update(glclu_results)

        # Process LAI (Leaf Area Index) data
        lai_results = self._process_lai_data()
        results.update(lai_results)

        # Process forest height data
        forest_results = self._process_forest_height()
        results.update(forest_results)

        # Calculate composite landcover metrics
        composite_results = self._calculate_composite_metrics(results)
        results.update(composite_results)

        return results

    def _process_glclu2019_landcover(self) -> Dict[str, Any]:
        """
        Process land cover data from the GLCLU2019 dataset.

        Returns:
            Dict[str, Any]: Dictionary of land cover attributes
        """
        results: Dict[str, Any] = {}

        # Define path to GLCLU data (configurable via GLCLU2019_DIR)
        glclu_path = Path(self._get_config_value(
            lambda: None,
            default=str(self.project_dir / 'data' / 'attributes' / 'landclass' / 'glclu'),
            dict_key='GLCLU2019_DIR',
        ))

        # Find GLCLU raster: acquisition naming or legacy
        main_tif = None
        for pattern in ("*glclu*.tif", "glclu2019_map.tif"):
            matches = list(glclu_path.glob(pattern)) if glclu_path.exists() else []
            if matches:
                main_tif = matches[0]
                break

        if main_tif is None:
            self.logger.warning(f"GLCLU file not found in {glclu_path}")
            return results

        # Create cache directory
        cache_dir = self.project_dir / 'cache' / 'landcover'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_glclu2019_results.pickle"

        # Check cache
        if cache_file.exists():
            self.logger.info("Loading cached GLCLU2019 results")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)  # nosec B301
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error loading cache: {e}")

        self.logger.info("Processing GLCLU2019 land cover data")

        try:
            # Define land cover classes
            landcover_classes = {
                1: 'true_desert', 2: 'semi_arid', 3: 'dense_short_vegetation',
                4: 'open_tree_cover', 5: 'dense_tree_cover', 6: 'tree_cover_gain',
                7: 'tree_cover_loss', 8: 'salt_pan', 9: 'wetland_sparse_vegetation',
                10: 'wetland_dense_short_vegetation', 11: 'wetland_open_tree_cover',
                12: 'wetland_dense_tree_cover', 13: 'wetland_tree_cover_gain',
                14: 'wetland_tree_cover_loss', 15: 'ice', 16: 'water',
                17: 'cropland', 18: 'built_up', 19: 'ocean', 20: 'no_data'
            }

            # Broader categories
            category_mapping = {
                'forest': [4, 5, 6, 11, 12, 13],
                'wetland': [9, 10, 11, 12, 13, 14],
                'barren': [1, 2, 8],
                'water': [15, 16, 19],
                'agricultural': [17],
                'urban': [18],
                'other_vegetation': [3]
            }

            # Calculate zonal statistics
            zonal_out = zonal_stats(
                str(self.catchment_path),
                str(main_tif),
                categorical=True,
                nodata=20,
                all_touched=True
            )

            if not zonal_out:
                self.logger.warning("No valid zonal statistics for GLCLU2019")
                return results

            is_lumped = self._is_lumped()

            if is_lumped:
                if zonal_out and len(zonal_out) > 0:
                    valid_pixels = sum(count for class_id, count in zonal_out[0].items()
                                     if class_id is not None and class_id != 20)

                    if valid_pixels > 0:
                        # Class fractions
                        for class_id, class_name in landcover_classes.items():
                            if class_id != 20:
                                pixel_count = zonal_out[0].get(class_id, 0)
                                fraction = pixel_count / valid_pixels if valid_pixels > 0 else 0
                                results[f"landcover.{class_name}_fraction"] = fraction

                        # Category fractions
                        for category, class_ids in category_mapping.items():
                            category_count = sum(zonal_out[0].get(class_id, 0) for class_id in class_ids)
                            fraction = category_count / valid_pixels if valid_pixels > 0 else 0
                            results[f"landcover.{category}_fraction"] = fraction

                        # Dominant class
                        dominant_class_id = max(
                            ((class_id, count) for class_id, count in zonal_out[0].items()
                             if class_id is not None and class_id != 20),
                            key=lambda x: x[1],
                            default=(None, 0)
                        )[0]

                        if dominant_class_id is not None:
                            results["landcover.dominant_class"] = landcover_classes[dominant_class_id]
                            results["landcover.dominant_fraction"] = zonal_out[0][dominant_class_id] / valid_pixels

                        # Diversity (Shannon entropy)
                        shannon_entropy = 0
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None and class_id != 20 and count > 0:
                                p = count / valid_pixels
                                shannon_entropy -= p * np.log(p)
                        results["landcover.diversity_index"] = shannon_entropy
            else:
                # Distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self._get_config_value(
                    lambda: self.config.paths.catchment_hruid,
                    dict_key='CATCHMENT_SHP_HRUID',
                )

                for i, zonal_result in enumerate(zonal_out):
                    if i < len(catchment):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"

                        valid_pixels = sum(count for class_id, count in zonal_result.items()
                                        if class_id is not None and class_id != 20)

                        if valid_pixels > 0:
                            for class_id, class_name in landcover_classes.items():
                                if class_id != 20:
                                    pixel_count = zonal_result.get(class_id, 0)
                                    fraction = pixel_count / valid_pixels
                                    results[f"{prefix}landcover.{class_name}_fraction"] = fraction

            # Cache results
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error caching results: {e}")

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing GLCLU2019: {e}")

        return results

    def _process_lai_data(self) -> Dict[str, Any]:
        """
        Process Leaf Area Index (LAI) data from MODIS.

        Returns:
            Dict[str, Any]: Dictionary of LAI attributes
        """
        results: Dict[str, Any] = {}

        # Configurable via LAI_DIR — check acquisition output first
        lai_path = Path(self._get_config_value(
            lambda: None,
            default=str(self.project_dir / 'data' / 'attributes' / 'vegetation' / 'modis_lai'),
            dict_key='LAI_DIR',
        ))
        use_water_mask = self._get_config_value(lambda: None, default=True, dict_key='USE_WATER_MASKED_LAI')

        # Try acquisition NetCDF first, then legacy monthly TIF folders
        lai_nc = list(lai_path.glob("*MODIS_LAI*.nc")) if lai_path.exists() else []
        if lai_nc:
            return self._process_lai_from_netcdf(lai_nc[0])

        lai_folder = lai_path / ('monthly_lai_with_water_mask' if use_water_mask else 'monthly_lai_no_water_mask')
        if not lai_folder.exists():
            lai_folder = lai_path / ('monthly_lai_no_water_mask' if use_water_mask else 'monthly_lai_with_water_mask')
            if not lai_folder.exists():
                # Also try legacy path
                legacy = self.data_dir / 'geospatial' / 'lai' / 'monthly_average_2013_2023'
                lai_folder = legacy / ('monthly_lai_with_water_mask' if use_water_mask else 'monthly_lai_no_water_mask')
                if not lai_folder.exists():
                    self.logger.warning("LAI folder not found")
                    return results

        # Cache
        cache_dir = self.project_dir / 'cache' / 'lai'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_lai_results.pickle"

        if cache_file.exists():
            self.logger.info("Loading cached LAI results")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)  # nosec B301
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error loading cache: {e}")

        self.logger.info("Processing LAI data")

        try:
            lai_files = sorted(lai_folder.glob("*.tif"))
            if not lai_files:
                return results

            monthly_lai_values = []
            is_lumped = self._is_lumped()

            for lai_file in lai_files:
                # Extract month from filename
                try:
                    month = int(lai_file.name.split('_')[2])
                except (IndexError, ValueError):
                    continue

                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(lai_file),
                    stats=['mean', 'min', 'max', 'std'],
                    nodata=255,
                    all_touched=True
                )

                if not zonal_out:
                    continue

                scale = 0.1  # MODIS LAI scale factor

                if is_lumped:
                    if zonal_out and len(zonal_out) > 0 and 'mean' in zonal_out[0]:
                        for stat in ['mean', 'min', 'max', 'std']:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                value = zonal_out[0][stat] * scale
                                results[f"vegetation.lai_month{month:02d}_{stat}"] = value

                        if 'mean' in zonal_out[0] and zonal_out[0]['mean'] is not None:
                            monthly_lai_values.append((month, zonal_out[0]['mean'] * scale))
                else:
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self._get_config_value(
                        lambda: self.config.paths.catchment_hruid,
                        dict_key='CATCHMENT_SHP_HRUID',
                    )

                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"

                            for stat in ['mean', 'min', 'max', 'std']:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat] * scale
                                    results[f"{prefix}vegetation.lai_month{month:02d}_{stat}"] = value

            # Seasonal metrics
            if is_lumped and monthly_lai_values:
                monthly_lai_values.sort(key=lambda x: x[0])
                lai_values = [v[1] for v in monthly_lai_values if not np.isnan(v[1])]

                if lai_values:
                    results["vegetation.lai_annual_min"] = min(lai_values)
                    results["vegetation.lai_annual_mean"] = sum(lai_values) / len(lai_values)
                    results["vegetation.lai_annual_max"] = max(lai_values)
                    results["vegetation.lai_seasonal_amplitude"] = max(lai_values) - min(lai_values)

            # Cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error caching: {e}")

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing LAI: {e}")

        return results

    def _process_lai_from_netcdf(self, nc_path) -> Dict[str, Any]:
        """Process LAI from acquisition NetCDF (MODIS MCD15A2H)."""
        results: Dict[str, Any] = {}
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_path)
            if 'lai_mean' in ds:
                lai = ds['lai_mean'].values
                results['vegetation.lai_annual_mean'] = float(np.nanmean(lai))
                results['vegetation.lai_annual_min'] = float(np.nanmin(lai))
                results['vegetation.lai_annual_max'] = float(np.nanmax(lai))
                results['vegetation.lai_seasonal_amplitude'] = float(
                    np.nanmax(lai) - np.nanmin(lai)
                )
            elif 'Lai_500m' in ds:
                lai = ds['Lai_500m'].values
                results['vegetation.lai_annual_mean'] = float(np.nanmean(lai))
                results['vegetation.lai_annual_min'] = float(np.nanmin(lai))
                results['vegetation.lai_annual_max'] = float(np.nanmax(lai))
                results['vegetation.lai_seasonal_amplitude'] = float(
                    np.nanmax(lai) - np.nanmin(lai)
                )
            ds.close()
            self.logger.info(f"Processed LAI from NetCDF: {nc_path.name}")
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Error processing LAI NetCDF: {e}")
        return results

    def _process_forest_height(self) -> Dict[str, Any]:
        """
        Process forest height data.

        Returns:
            Dict[str, Any]: Dictionary of forest height attributes
        """
        results = {}

        # Check acquisition output first, then legacy path
        acq_dir = self.project_dir / 'data' / 'attributes' / 'vegetation' / 'canopy_height' / 'glad'
        forest_dir = Path(self._get_config_value(
            lambda: None,
            default=str(acq_dir),
            dict_key='FOREST_HEIGHT_DIR',
        ))

        # Find forest height files — acquisition naming or legacy
        forest_files = {}
        if forest_dir.exists():
            for tif in forest_dir.glob("*.tif"):
                if 'tree_height' in tif.name or 'treecover' in tif.name:
                    forest_files['acquired'] = tif
                    break
                elif 'forest_height_2020' in tif.name:
                    forest_files['2020'] = tif
                elif 'forest_height_2000' in tif.name:
                    forest_files['2000'] = tif

        if not forest_files:
            legacy_dir = self.data_dir / 'geospatial' / 'forest_height' / 'raw'
            if (legacy_dir / 'forest_height_2020.tif').exists():
                forest_files['2020'] = legacy_dir / 'forest_height_2020.tif'
            if (legacy_dir / 'forest_height_2000.tif').exists():
                forest_files['2000'] = legacy_dir / 'forest_height_2000.tif'

        # Cache
        cache_dir = self.project_dir / 'cache' / 'forest_height'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_forest_height_results.pickle"

        if cache_file.exists():
            self.logger.info("Loading cached forest height results")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)  # nosec B301
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error loading cache: {e}")

        self.logger.info("Processing forest height data")

        try:
            for year, forest_file in forest_files.items():
                if not forest_file.exists():
                    continue

                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(forest_file),
                    stats=['mean', 'min', 'max', 'std'],
                    nodata=-9999,
                    all_touched=True
                )

                if not zonal_out:
                    continue

                is_lumped = self._is_lumped()

                if is_lumped:
                    if zonal_out and len(zonal_out) > 0:
                        for stat in ['mean', 'min', 'max', 'std']:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                results[f"forest.height_{year}_{stat}"] = zonal_out[0][stat]
                else:
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self._get_config_value(
                        lambda: self.config.paths.catchment_hruid,
                        dict_key='CATCHMENT_SHP_HRUID',
                    )

                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"

                            for stat in ['mean', 'min', 'max', 'std']:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    results[f"{prefix}forest.height_{year}_{stat}"] = zonal_result[stat]

            # Cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
                self.logger.warning(f"Error caching: {e}")

        except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError, ImportError, LookupError) as e:
            self.logger.error(f"Error processing forest height: {e}")

        return results

    def _calculate_composite_metrics(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate composite land cover metrics from individual datasets.

        Args:
            current_results: Current results dictionary

        Returns:
            Dict[str, Any]: Composite metrics
        """
        results = {}

        # Define ecological groupings

        # Calculate anthropogenic influence
        urban_key = "landcover.urban_fraction"
        crop_key = "landcover.agricultural_fraction"

        if urban_key in current_results and crop_key in current_results:
            results["landcover.anthropogenic_influence"] = (
                current_results[urban_key] + current_results[crop_key]
            )

        return results

    def _read_scale_and_offset(self, geotiff_path: Path):
        """Helper to read scale and offset from GeoTIFF."""
        try:
            from osgeo import gdal
            dataset = gdal.Open(str(geotiff_path))
            if dataset is None:
                return None, None

            scale = dataset.GetRasterBand(1).GetScale()
            offset = dataset.GetRasterBand(1).GetOffset()
            dataset = None

            return scale, offset
        except (ImportError, RuntimeError, AttributeError):
            return None, None
