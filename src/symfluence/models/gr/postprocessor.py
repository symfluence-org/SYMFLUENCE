# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR model postprocessor.

Handles extraction and processing of GR (GR4J/CemaNeige) simulation results.
Supports both lumped and distributed modes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import BaseModelPostProcessor
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.registries import R

from .r_environment import (
    configure_r_dll_search,
    describe_rpy2_import_failure,
    r_path,
    rpy2_installed,
    run_r_script,
)

# Optional R/rpy2 support - only needed for GR models
# Broad exception handling is intentional here: rpy2 can raise RuntimeError, RRuntimeError,
# ImportError, or other exceptions when R is installed but broken (missing core packages,
# incompatible versions, etc.). We must catch all to provide graceful fallback.
# rpy2 prints noisy messages to stderr during R initialization — redirect to suppress.
# configure_r_dll_search() must precede the rpy2 import: see r_environment.
# _RPY2_IMPORT_ERROR keeps the real cause (see the runner for the full rationale).
_RPY2_IMPORT_ERROR: Optional[BaseException] = None
configure_r_dll_search()
try:
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except Exception:  # noqa: BLE001 - Broad exception required for rpy2 import failures
    import sys as _sys
    _RPY2_IMPORT_ERROR = _sys.exc_info()[1]
    HAS_RPY2 = False
    robjects = None
    pandas2ri = None
    localconverter = None


@R.postprocessors.add('GR')
class GRPostProcessor(BaseModelPostProcessor):
    """
    Postprocessor for GR (GR4J/CemaNeige) model outputs.
    Handles extraction and processing of simulation results.
    Supports both lumped and distributed modes.
    """

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "GR"

    def _setup_model_specific_paths(self) -> None:
        """Set up GR-specific paths and check dependencies."""
        # Check for R/rpy2 dependency
        if not HAS_RPY2:
            # Installed-but-unimportable rpy2 => surface the real embedded-R
            # failure the module-level import swallowed, not "not installed".
            if _RPY2_IMPORT_ERROR is not None and rpy2_installed():
                raise ModelExecutionError(
                    describe_rpy2_import_failure(_RPY2_IMPORT_ERROR)
                ) from _RPY2_IMPORT_ERROR
            raise ImportError(
                "GR models require R and rpy2. "
                "Please install R and rpy2, or use a different model. "
                "See https://rpy2.github.io/doc/latest/html/overview.html#installation"
            )

        # GR-specific configuration
        self.spatial_mode = self._get_config_value(
            lambda: self.config.model.gr.spatial_mode if self.config.model and self.config.model.gr else None,
            default='lumped'
        )
        self._output_path = self.sim_dir  # Alias for consistency with existing code

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from GR output and append to results CSV.
        Handles both lumped and distributed modes.
        """
        try:
            self.logger.info(f"Extracting GR streamflow results ({self.spatial_mode} mode)")

            if self.spatial_mode == SpatialMode.LUMPED:
                return self._extract_lumped_streamflow()
            else:  # distributed
                return self._extract_distributed_streamflow()

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting GR streamflow: {str(e)}")
            raise

    def _extract_lumped_streamflow(self) -> Optional[Path]:
        """Extract streamflow from lumped GR4J run"""

        # Check for R data file
        r_results_path = self._output_path / 'GR_results.Rdata'
        if not r_results_path.exists():
            self.logger.error(f"GR results file not found at: {r_results_path}")
            return None

        # Load R data
        run_r_script(robjects, f'load({r_path(r_results_path)})', "GR results loader")

        # Extract simulated streamflow
        r_script = """
        data.frame(
            date = format(OutputsModel$DatesR, "%Y-%m-%d"),
            flow = OutputsModel$Qsim
        )
        """

        sim_df = run_r_script(robjects, r_script, "GR streamflow extractor")

        # Convert to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            sim_df = robjects.conversion.rpy2py(sim_df)

        sim_df['date'] = pd.to_datetime(sim_df['date'])
        sim_df.set_index('date', inplace=True)

        # Convert units from mm/day to m3/s (cms) using base method
        # Note: GR4J lumped outputs are in mm/day
        q_sim_cms = self.convert_mm_per_day_to_cms(sim_df['flow'])

        # Save using standard method
        return self.save_streamflow_to_results(
            q_sim_cms,
            model_column_name='GR_discharge_cms'
        )

    def _extract_distributed_streamflow(self) -> Optional[Path]:
        """Extract streamflow from a distributed GR4J run.

        Both branches produce m³/s directly. They used to share a single
        ``convert_mm_per_day_to_cms`` call at the end, which was wrong for each
        of them in a different way -- see the two helpers for the units each
        source actually carries.
        """
        needs_routing = self._get_config_value(
            lambda: self.config.model.gr.routing_integration if self.config.model and self.config.model.gr else None,
            default=None
        ) == 'mizuRoute'

        q_cms = (
            self._routed_streamflow_cms() if needs_routing
            else self._unrouted_streamflow_cms()
        )
        if q_cms is None:
            return None

        return self.save_streamflow_to_results(
            q_cms,
            model_column_name='GR_discharge_cms'
        )

    def _routed_streamflow_cms(self) -> Optional[pd.Series]:
        """Outlet discharge (m³/s) from mizuRoute output.

        mizuRoute routes runoff into discharge: ``IRFroutedRunoff`` and its
        siblings are already m³/s, so no unit conversion is applied. This
        matches ``StandardModelPostProcessor``, which declares
        ``streamflow_unit = "cms"  # Routing output is already in cms`` for the
        same variables. The previous code pushed this through
        ``convert_mm_per_day_to_cms``, inflating routed GR discharge by
        ``area_km2 / 86.4`` -- a factor of ~11.6 for a 1000 km² basin.
        """
        exp_id = self._get_config_value(lambda: self.config.domain.experiment_id)
        mizuroute_output_dir = self.get_simulation_dir('mizuRoute', exp_id)

        output_files = list(mizuroute_output_dir.glob(f"{exp_id}*.nc"))
        if not output_files:
            self.logger.error(f"No mizuRoute output files found in {mizuroute_output_dir}")
            return None

        mizuroute_file = output_files[0]
        self.logger.info(f"Reading routed streamflow from: {mizuroute_file}")

        with xr.open_dataset(mizuroute_file) as ds:
            # NOTE: outlet is taken as the last segment, as before. Picking the
            # outlet by position is a separate question from units and is left
            # unchanged here.
            streamflow_var = next(
                (v for v in ('IRFroutedRunoff', 'dlayRunoff', 'KWTroutedRunoff')
                 if v in ds.variables),
                None
            )
            if streamflow_var is None:
                self.logger.error(
                    "Could not find streamflow variable in mizuRoute output. "
                    f"Available: {list(ds.variables)}"
                )
                return None

            # to_pandas() is typed as DataArray | Series | DataFrame; a 1-D
            # selection is always a Series.
            series = cast(pd.Series, ds[streamflow_var].isel(seg=-1).to_pandas())

        series.index = pd.to_datetime(series.index)
        return series

    def _unrouted_streamflow_cms(self) -> Optional[pd.Series]:
        """Basin discharge (m³/s) from GR's own distributed output.

        ``GRRunner._save_distributed_results_for_routing`` writes ``q_routed``
        as a per-GRU runoff **depth rate in m/s** (it divides mm/day by
        ``1000 * 86400`` and labels the variable ``units = 'm/s'``). Basin
        discharge is therefore the area-weighted sum

            Q [m³/s] = sum_i runoff_i [m/s] * area_i [m²]

        The previous code did ``.sum(dim='gru')`` and then applied the mm/day
        conversion (``* area_km2 / 86.4``), which is wrong twice over: it
        treated an m/s depth rate as mm/day, and it summed depth rates across
        GRUs instead of weighting them by area. For a single-GRU basin the
        result was low by a factor of 8.64e7.
        """
        exp_id = self._get_config_value(lambda: self.config.domain.experiment_id)
        gr_output = self.get_simulation_dir('GR', exp_id) / \
            f"{self.domain_name}_{exp_id}_runs_def.nc"

        if not gr_output.exists():
            self.logger.error(f"GR output not found: {gr_output}")
            return None

        routing_var_config = self._get_config_value(
            lambda: self.config.model.mizuroute.routing_var if self.config.model and self.config.model.mizuroute else None,
            default='q_routed'
        )
        routing_var = (
            'q_routed' if routing_var_config in ('default', None, '')
            else routing_var_config
        )

        with xr.open_dataset(gr_output) as ds:
            if routing_var not in ds.variables:
                self.logger.error(
                    f"Variable '{routing_var}' not found in {gr_output}. "
                    f"Available: {list(ds.variables)}"
                )
                return None

            runoff_ms = ds[routing_var]  # (time, gru), m/s
            areas_m2 = self._gru_areas_m2(ds)

            if areas_m2 is not None:
                weights = xr.DataArray(areas_m2, dims=('gru',))
                q_cms = (runoff_ms * weights).sum(dim='gru')
            else:
                # Equal-area fallback: sum_i runoff_i * (A_total / N) is exactly
                # the basin mean times total area. Correct when GRUs are
                # equal-area, approximate otherwise -- hence the warning.
                total_area_m2 = self.get_catchment_area_km2() * 1e6
                self.logger.warning(
                    "Per-GRU areas unavailable; converting distributed GR runoff "
                    "with an equal-area assumption (basin mean x total area). "
                    "Discharge will be approximate if GRU areas differ."
                )
                q_cms = runoff_ms.mean(dim='gru') * total_area_m2

            # 1-D after the gru reduction, so always a Series.
            series = cast(pd.Series, q_cms.to_pandas())

        series.index = pd.to_datetime(series.index)
        return series

    def _gru_areas_m2(self, ds) -> Optional[np.ndarray]:
        """Per-GRU areas in m², aligned to the dataset's ``gruId`` order.

        Returns None (rather than raising) whenever the mapping cannot be built
        completely -- a partial area map would silently mis-weight the basin,
        so the caller falls back to an explicit equal-area assumption instead.
        """
        if 'gruId' not in ds.variables:
            return None

        try:
            import geopandas as gpd

            basin_name = self._get_config_value(
                lambda: self.config.paths.river_basins_name, default=None
            )
            if basin_name in ('default', None):
                basin_name = (
                    f"{self.domain_name}_riverBasins_{self.domain_definition_method}.shp"
                )

            basin_path = self._get_file_path(
                path_key='RIVER_BASINS_PATH',
                name_key='RIVER_BASINS_NAME',
                default_subpath='shapefiles/river_basins',
                default_name=basin_name,
            )
            if not basin_path.exists():
                return None

            gdf = gpd.read_file(basin_path)
            id_col = self._get_config_value(
                lambda: self.config.paths.river_basin_rm_gruid, default='GRU_ID'
            )
            area_col = self._get_config_value(
                lambda: self.config.paths.river_basin_area, default='GRU_area'
            )
            if id_col not in gdf.columns or area_col not in gdf.columns:
                return None

            areas = {
                str(k): float(v)
                for k, v in zip(gdf[id_col], gdf[area_col])
            }
            gru_ids = [str(g) for g in np.asarray(ds['gruId'].values).ravel()]
            if not gru_ids or not all(g in areas for g in gru_ids):
                return None

            return np.array([areas[g] for g in gru_ids], dtype=float)
        except Exception as exc:  # noqa: BLE001 — area lookup is best-effort
            self.logger.debug(f"Per-GRU areas unavailable: {exc}")
            return None


    @property
    def output_path(self):
        """Get output path for backwards compatibility"""
        return self.get_simulation_dir('GR', self._get_config_value(lambda: self.config.domain.experiment_id))
