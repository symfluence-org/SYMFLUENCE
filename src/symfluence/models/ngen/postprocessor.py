# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGen Model Postprocessor.

Processes simulation outputs from the NOAA NextGen Framework (ngen).
Migrated to use StandardModelPostProcessor with multi-file support (Phase 1.5).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('NGEN')
class NgenPostProcessor(StandardModelPostProcessor):
    """
    Postprocessor for NextGen Framework outputs.

    Handles extraction and analysis of simulation results from multiple nexus
    output files. Uses StandardModelPostProcessor with multi-file aggregation.

    NGEN outputs streamflow to multiple nex-*_output.csv files, one per nexus.
    This postprocessor aggregates them based on CALIBRATION_NEXUS_ID config
    or sums all nexus outputs.

    Special handling:
    - Multi-file glob pattern (nex-*_output.csv)
    - Headerless CSV detection
    - Multiple nexus aggregation

    Attributes:
        model_name: "NGEN"
        output_file_glob: "nex-*_output.csv"
        aggregation_method: "sum" (all nexus outputs summed)
        streamflow_unit: "cms"
    """

    # Model identification
    model_name = "NGEN"

    # Multi-file configuration
    output_file_glob = "nex-*_output.csv"
    aggregation_method = "sum"

    # Text file parsing
    text_file_separator = ","

    # Streamflow is already in cms from NGEN
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        """Return model name for NGEN."""
        return "NGEN"

    def _get_output_dir(self) -> Path:
        """
        Get NGEN output directory.

        Returns:
            Path to ngen output directory within simulations folder
        """
        experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='run_1')
        return self.get_simulation_dir(self.model_name, experiment_id)

    def extract_streamflow(self, experiment_id: str = None) -> Optional[Path]:
        """
        Extract streamflow from ngen nexus outputs.

        Handles NGEN's multi-file output format with optional nexus filtering
        based on CALIBRATION_NEXUS_ID configuration.

        Note: NGEN postprocessor accepts an optional experiment_id parameter,
        which differs from the base class signature. This is necessary to support
        NGEN's multi-experiment workflow.

        Args:
            experiment_id: Experiment identifier (default: from config)

        Returns:
            Path to extracted streamflow CSV file, or None if extraction fails
        """
        self.logger.info("Extracting streamflow from ngen outputs")

        if experiment_id is None:
            experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='run_1')

        # Get output directory
        output_dir = self.get_simulation_dir(self.model_name, experiment_id)

        # Find nexus output files
        nexus_files = list(output_dir.glob(self.output_file_glob))

        if not nexus_files:
            self.logger.error(f"No nexus output files found in {output_dir}")
            return None

        # Filter by CALIBRATION_NEXUS_ID if configured
        target_nexus = self._get_config_value(lambda: self.config.model.ngen.calibration_nexus_id, default=None)
        if target_nexus:
            # Normalize ID
            target_files = [
                f for f in nexus_files
                if f.stem == f"{target_nexus}_output" or f.stem == target_nexus
            ]

            if target_files:
                self.logger.info(f"Post-processing restricted to target nexus: {target_nexus}")
                nexus_files = target_files
            else:
                self.logger.warning(
                    f"Configured CALIBRATION_NEXUS_ID '{target_nexus}' not found in output files. "
                    "Processing all files."
                )

        self.logger.info(f"Found {len(nexus_files)} nexus output file(s)")

        # Read and process each nexus file
        all_streamflow: List[pd.DataFrame] = []
        for nexus_file in nexus_files:
            nexus_id = nexus_file.stem.replace('_output', '')

            try:
                df = self._read_ngen_nexus_file(nexus_file)
                if df is not None:
                    df['nexus_id'] = nexus_id
                    all_streamflow.append(df)

            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.error(f"Error processing {nexus_file}: {e}", exc_info=True)
                continue

        if not all_streamflow:
            self.logger.error("No streamflow data could be extracted")
            return None

        # Combine all nexus outputs
        combined_streamflow = pd.concat(all_streamflow, ignore_index=True)

        # Aggregate by time (sum for multiple nexuses)
        if self.aggregation_method == "sum":
            aggregated_flow = combined_streamflow.groupby('datetime')['streamflow_cms'].sum()
        elif self.aggregation_method == "mean":
            aggregated_flow = combined_streamflow.groupby('datetime')['streamflow_cms'].mean()
        else:
            # Default to sum
            aggregated_flow = combined_streamflow.groupby('datetime')['streamflow_cms'].sum()

        # Save using standard method
        result = self.save_streamflow_to_results(
            aggregated_flow,
            model_column_name=f"NGEN_{experiment_id}_discharge_cms"
        )

        # If mizuRoute is the routing model, convert nexus CSVs to NetCDF
        routing_model = self._get_config_value(
            lambda: self.config.model.routing_model, default=None
        )
        if routing_model and str(routing_model).upper() in ('MIZUROUTE', 'MIZU_ROUTE', 'MIZU'):
            self.logger.info("mizuRoute routing configured — converting NGEN output to NetCDF")
            self.convert_output_to_mizuroute_netcdf(experiment_id=experiment_id)

        return result

    def _read_ngen_nexus_file(self, nexus_file: Path) -> Optional[pd.DataFrame]:
        """
        Read a single NGEN nexus output file.

        Handles NGEN's potentially headerless CSV format by detecting the
        format from the first row.

        Args:
            nexus_file: Path to the nexus output CSV file

        Returns:
            DataFrame with 'datetime' and 'streamflow_cms' columns, or None if failed
        """
        try:
            # First try reading with header
            df = pd.read_csv(nexus_file, skipinitialspace=True)

            # Check for standard NGEN headerless format (index, time, flow)
            is_headerless = False
            if len(df.columns) == 3:
                # Check if first row's second column looks like a date
                # (indicating header is actually data)
                try:
                    pd.to_datetime(df.columns[1])
                    is_headerless = True
                except (ValueError, TypeError):
                    pass

            if is_headerless:
                df = pd.read_csv(
                    nexus_file,
                    header=None,
                    names=['index', 'time', 'flow'],
                    skipinitialspace=True,
                )
                flow_col = 'flow'
            else:
                # Find flow column from common names
                flow_col = None
                for col_name in ['flow', 'Flow', 'Q_OUT', 'streamflow', 'discharge']:
                    if col_name in df.columns:
                        flow_col = col_name
                        break

            if flow_col is None:
                self.logger.warning(
                    f"No flow column found in {nexus_file}. Columns: {df.columns.tolist()}"
                )
                return None

            # Find time column
            if 'time' in df.columns:
                time = pd.to_datetime(df['time'])
            elif 'Time' in df.columns:
                time = pd.to_datetime(df['Time'], unit='ns')
            else:
                self.logger.warning(f"No time column found in {nexus_file}")
                return None

            return pd.DataFrame({
                'datetime': time,
                'streamflow_cms': pd.to_numeric(df[flow_col], errors='coerce')
            })

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error reading {nexus_file}: {e}", exc_info=True)
            return None

    def _calculate_nse(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]

        if len(obs) == 0:
            return np.nan

        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)

        if denominator == 0:
            return np.nan

        return 1 - (numerator / denominator)

    def _calculate_kge(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]

        if len(obs) == 0:
            return np.nan

        # Calculate components
        r = np.corrcoef(obs, sim)[0, 1]  # Correlation
        alpha = np.std(sim) / np.std(obs)  # Variability ratio
        beta = np.mean(sim) / np.mean(obs)  # Bias ratio

        # Calculate KGE
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        return kge

    # -----------------------------------------------------------------
    # mizuRoute integration: convert NGEN CSV output → NetCDF
    # -----------------------------------------------------------------

    def convert_output_to_mizuroute_netcdf(
        self,
        experiment_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Convert NGEN nexus CSV outputs to a mizuRoute-compatible NetCDF.

        mizuRoute expects a NetCDF file with dimensions (time, hru) containing
        a runoff variable in depth/time units (m/s). NGEN outputs per-nexus
        CSV files with flow in m³/s. This method reads all nexus CSVs,
        converts flow to runoff depth using catchment areas, and writes the
        result as a single NetCDF.

        Args:
            experiment_id: Override experiment ID (default: from config)

        Returns:
            Path to the generated NetCDF file, or None on failure.
        """
        import xarray as xr

        if experiment_id is None:
            experiment_id = self._get_config_value(
                lambda: self.config.domain.experiment_id, default='run_1'
            )

        output_dir = self.get_simulation_dir(self.model_name, experiment_id)
        nexus_files = sorted(output_dir.glob(self.output_file_glob))

        if not nexus_files:
            self.logger.error(f"No nexus output files in {output_dir}")
            return None

        # Load catchment areas from the GeoJSON written by the preprocessor
        domain_name = self._get_config_value(
            lambda: self.config.domain.name, default='domain'
        )
        areas_km2 = self._load_catchment_areas(domain_name)

        # Read each nexus file into a per-HRU timeseries
        hru_data: Dict[str, pd.Series] = {}
        for nexus_file in nexus_files:
            df = self._read_ngen_nexus_file(nexus_file)
            if df is None:
                continue

            nexus_id = nexus_file.stem.replace('_output', '')
            cat_id = nexus_id.replace('nex-', 'cat-')
            hru_data[cat_id] = df.set_index('datetime')['streamflow_cms']

        if not hru_data:
            self.logger.error("No valid nexus data could be read")
            return None

        # Build a DataFrame: columns=catchment IDs, index=time, values=flow (m³/s)
        flow_df = pd.DataFrame(hru_data)
        flow_df.index = pd.to_datetime(flow_df.index)
        flow_df = flow_df.sort_index()

        # Convert m³/s → m/s (runoff depth) using catchment area
        runoff_df = flow_df.copy()
        for cat_id in runoff_df.columns:
            area_km2 = areas_km2.get(cat_id, None)
            if area_km2 is None or area_km2 <= 0:
                self.logger.warning(
                    f"No area for {cat_id}, using 1.0 km² — "
                    "runoff depth will be approximate"
                )
                area_km2 = 1.0
            area_m2 = area_km2 * 1e6
            runoff_df[cat_id] = flow_df[cat_id] / area_m2

        # Assign integer HRU IDs (1-based)
        hru_ids = list(range(1, len(runoff_df.columns) + 1))

        # Build xarray Dataset
        time_values = runoff_df.index.values
        runoff_array = runoff_df.values

        ds = xr.Dataset(
            {
                'runoff': (['time', 'hru'], runoff_array, {
                    'units': 'm/s',
                    'long_name': 'NGEN catchment runoff',
                }),
                'hruId': (['hru'], np.array(hru_ids, dtype=np.int32), {
                    'long_name': 'HRU identifier',
                }),
            },
            coords={
                'time': time_values,
            },
            attrs={
                'source': 'NGEN via SYMFLUENCE',
                'experiment_id': experiment_id,
                'history': 'Converted from NGEN nexus CSV outputs for mizuRoute routing',
            },
        )

        nc_path = output_dir / f"{experiment_id}_runoff.nc"
        ds.to_netcdf(nc_path, format='NETCDF4')
        self.logger.info(
            f"NGEN → mizuRoute NetCDF written: {nc_path} "
            f"({len(hru_ids)} HRUs, {len(time_values)} timesteps)"
        )
        return nc_path

    def _load_catchment_areas(self, domain_name: str) -> Dict[str, float]:
        """Load catchment areas (km²) from the NGEN catchments GeoJSON."""
        settings_dir = self.project_dir / "settings" / "NGEN"

        geojson_path = settings_dir / f"{domain_name}_catchments.geojson"
        if geojson_path.exists():
            try:
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                areas: Dict[str, float] = {}
                for feature in data.get('features', []):
                    props = feature.get('properties', {})
                    cat_id = props.get('id', props.get('cat_id', ''))
                    for area_key in ['areasqkm', 'area_km2', 'area_sqkm', 'AREA']:
                        if area_key in props and props[area_key] is not None:
                            area_val = float(props[area_key])
                            if area_val > 1e6:
                                area_val /= 1e6
                            areas[cat_id] = area_val
                            break
                return areas
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Could not read catchment areas from {geojson_path}: {e}")

        gpkg_path = settings_dir / f"{domain_name}_catchments.gpkg"
        if gpkg_path.exists():
            try:
                import geopandas as gpd
                gdf = gpd.read_file(gpkg_path)
                areas = {}
                id_col = next((c for c in ['id', 'cat_id', 'HRU_ID'] if c in gdf.columns), None)
                area_col = next(
                    (c for c in ['areasqkm', 'area_km2', 'area_sqkm', 'AREA', 'HRU_area']
                     if c in gdf.columns), None
                )
                if id_col and area_col:
                    for _, row in gdf.iterrows():
                        area_val = float(row[area_col])
                        if area_val > 1e6:
                            area_val /= 1e6
                        areas[str(row[id_col])] = area_val
                return areas
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Could not read catchment areas from {gpkg_path}: {e}")

        self.logger.warning("No catchment area file found — using default 1.0 km² for all HRUs")
        return {}
