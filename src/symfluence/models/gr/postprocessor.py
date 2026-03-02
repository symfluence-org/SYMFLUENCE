# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR model postprocessor.

Handles extraction and processing of GR (GR4J/CemaNeige) simulation results.
Supports both lumped and distributed modes.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from ..base import BaseModelPostProcessor
from ..registry import ModelRegistry
from ..spatial_modes import SpatialMode

# Optional R/rpy2 support - only needed for GR models
# Broad exception handling is intentional here: rpy2 can raise RuntimeError, RRuntimeError,
# ImportError, or other exceptions when R is installed but broken (missing core packages,
# incompatible versions, etc.). We must catch all to provide graceful fallback.
# rpy2 prints noisy messages to stderr during R initialization — redirect to suppress.
try:
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except Exception:  # noqa: BLE001 - Broad exception required for rpy2 import failures
    HAS_RPY2 = False
    robjects = None
    pandas2ri = None
    localconverter = None


@ModelRegistry.register_postprocessor('GR')
class GRPostprocessor(BaseModelPostProcessor):
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
        robjects.r(f'load("{str(r_results_path)}")')

        # Extract simulated streamflow
        r_script = """
        data.frame(
            date = format(OutputsModel$DatesR, "%Y-%m-%d"),
            flow = OutputsModel$Qsim
        )
        """

        sim_df = robjects.r(r_script)

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
        """Extract streamflow from distributed GR4J run (after routing)"""

        # Check if routing was performed
        needs_routing = self._get_config_value(
            lambda: self.config.model.gr.routing_integration if self.config.model and self.config.model.gr else None,
            default=None
        ) == 'mizuRoute'

        if needs_routing:
            # Get routed streamflow from mizuRoute output
            exp_id = self._get_config_value(lambda: self.config.domain.experiment_id)
            mizuroute_output_dir = self.project_dir / 'simulations' / exp_id / 'mizuRoute'

            # Find mizuRoute output file
            output_files = list(mizuroute_output_dir.glob(f"{exp_id}*.nc"))

            if not output_files:
                self.logger.error(f"No mizuRoute output files found in {mizuroute_output_dir}")
                return None

            # Use the first output file
            mizuroute_file = output_files[0]
            self.logger.info(f"Reading routed streamflow from: {mizuroute_file}")

            ds = xr.open_dataset(mizuroute_file)

            # Extract streamflow at outlet (typically the last reach)
            # mizuRoute typically names the variable 'IRFroutedRunoff' or similar
            streamflow_vars = ['IRFroutedRunoff', 'dlayRunoff', 'KWTroutedRunoff']
            streamflow_var = None

            for var in streamflow_vars:
                if var in ds.variables:
                    streamflow_var = var
                    break

            if streamflow_var is None:
                self.logger.error(f"Could not find streamflow variable in mizuRoute output. Available: {list(ds.variables)}")
                return None

            # Get streamflow at outlet (last segment)
            q_routed = ds[streamflow_var].isel(seg=-1)

            # Convert to DataFrame
            q_df = q_routed.to_dataframe(name='flow')
            q_df = q_df.reset_index()

            # Convert time if needed
            if 'time' in q_df.columns:
                q_df['time'] = pd.to_datetime(q_df['time'])
                q_df.set_index('time', inplace=True)

        else:
            # No routing - sum all HRU outputs
            exp_id = self._get_config_value(lambda: self.config.domain.experiment_id)
            gr_output = self.project_dir / 'simulations' / exp_id / 'GR' / \
                        f"{self.domain_name}_{exp_id}_runs_def.nc"

            if not gr_output.exists():
                self.logger.error(f"GR output not found: {gr_output}")
                return None

            ds = xr.open_dataset(gr_output)

            # Sum across all GRUs
            # Handle 'default' config value - use model-specific default
            routing_var_config = self._get_config_value(
                lambda: self.config.model.mizuroute.routing_var if self.config.model and self.config.model.mizuroute else None,
                default='q_routed'
            )
            if routing_var_config in ('default', None, ''):
                routing_var = 'q_routed'  # GR4J default for routing
            else:
                routing_var = routing_var_config
            q_total = ds[routing_var].sum(dim='gru')

            # Convert to DataFrame
            q_df = q_total.to_dataframe(name='flow')

        # Convert from mm/day to m3/s using base method
        # Assumes GR output in mm/day. If mizuRoute, it might be in m3/s already depending on config,
        # but typically routing input is mm/day and output is m3/s?
        # Looking at original code:
        # q_cms = q_df['flow'] * area_km2 / UnitConversion.MM_DAY_TO_CMS
        # This implies the input was mm/day.

        q_cms = self.convert_mm_per_day_to_cms(q_df['flow'])

        # Save using standard method
        return self.save_streamflow_to_results(
            q_cms,
            model_column_name='GR_discharge_cms'
        )


    @property
    def output_path(self):
        """Get output path for backwards compatibility"""
        return self.project_dir / 'simulations' / self._get_config_value(lambda: self.config.domain.experiment_id) / 'GR'
