# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR Result Extractor.

Handles extraction of simulation results from GR model outputs.
GR models (GR4J/GR5J/GR6J) can run in lumped (CSV) or distributed (NetCDF) modes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor


class GRResultExtractor(ModelResultExtractor):
    """GR-specific result extraction.

    Handles GR model's output characteristics:
    - Lumped mode: CSV file (GR_results.csv), streamflow column ``q_sim``
    - Distributed mode: NetCDF file (*_runs_def.nc), variable ``q_routed``
    - Routing: Can use mizuRoute for distributed runs

    Known gap: the unrouted distributed file is listed in
    :meth:`get_output_file_patterns` but cannot be served yet -- its
    ``q_routed`` is per-GRU runoff needing a sum over ``gru`` (what
    ``GRPostProcessor`` does), while :meth:`_extract_from_netcdf` reduces
    multi-dimensional variables by taking the first spatial index. It therefore
    raises rather than returning a single GRU's series. mizuRoute-routed output
    is unaffected: those variables are matched ahead of the GR-specific list and
    reduced by outlet selection.
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for GR outputs."""
        return {
            'streamflow': [
                # Lumped mode CSV
                'GR_results.csv',
                # Distributed mode NetCDF
                '*_runs_def.nc',
                '*_runs_best.nc',
                # mizuRoute routing
                'mizuRoute/*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get GR variable names for different types.

        ``q_sim`` leads because it is the column GR actually writes in the
        lumped-mode ``GR_results.csv``; ``Qsim`` is the airGR *R object* field
        name, not a column header, so leading with it matched nothing and
        reached the data only through :meth:`_extract_from_csv`'s positional
        fallback. The rest are tolerant fallbacks for hand-edited output.

        ``q_routed`` (distributed mode) is deliberately absent: this list is
        shared with :meth:`_extract_from_netcdf`, which would reduce it to a
        single GRU rather than the basin total -- see the class docstring.
        """
        variable_mapping = {
            'streamflow': ['q_sim', 'Qsim', 'Q', 'streamflow', 'discharge'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from GR output.

        Args:
            output_file: Path to GR output file (CSV or NetCDF)
            variable_type: Type of variable to extract
            **kwargs: Additional options

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable cannot be extracted
        """
        if output_file.suffix == '.csv':
            # Lumped mode CSV output
            return self._extract_from_csv(output_file)
        elif output_file.suffix == '.nc':
            # Distributed mode NetCDF or mizuRoute output
            return self._extract_from_netcdf(output_file, kwargs.get('sim_reach_id'))

        raise ValueError(f"Unsupported file format: {output_file.suffix}")

    def _extract_from_csv(self, output_file: Path) -> pd.Series:
        """Extract streamflow from GR CSV output.

        Args:
            output_file: Path to GR_results.csv

        Returns:
            Time series of streamflow
        """
        try:
            df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
            # Named lookup first, from the single declaration in
            # get_variable_names (a second hardcoded copy here is how this
            # drifted out of step with the column GR writes in the first place).
            for col in self.get_variable_names('streamflow'):
                if col in df.columns:
                    return df[col]
            # Last resort: first column, whatever it is named.
            return df.iloc[:, 0]
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(f"Failed to parse GR CSV output {output_file}: {e}") from e

    def _extract_from_netcdf(self, output_file: Path, sim_reach_id=None) -> pd.Series:
        """Extract streamflow from GR NetCDF output.

        Args:
            output_file: Path to GR NetCDF output

        Returns:
            Time series of streamflow

        Raises:
            ValueError: if no streamflow variable is present, or if the file is
                GR's per-GRU distributed runoff, which this interface cannot
                convert to discharge (see :meth:`_reject_per_gru_runoff`).
        """
        import xarray as xr

        with xr.open_dataset(output_file) as ds:
            # Try mizuRoute variables first
            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims or 'reachID' in var.dims:
                        # Shared reach rule (SIM_REACH_ID, topology outlet,
                        # else highest mean flow)
                        from symfluence.core.modeling.utilities.reach_selection import (
                            select_outlet_series,
                            topology_candidates,
                        )
                        return select_outlet_series(
                            ds, var, sim_reach_id,
                            topology_candidates(near=output_file),
                        )

            # Routed output is handled above; anything still holding per-GRU
            # runoff cannot be turned into discharge here.
            self._reject_per_gru_runoff(ds, output_file)

            # Try GR-specific variables
            for var_name in self.get_variable_names('streamflow'):
                if var_name in ds.variables:
                    var = ds[var_name]
                    # Handle spatial dimensions if present
                    if len(var.shape) > 1:
                        spatial_dims = [d for d in var.dims if d != 'time']
                        if spatial_dims:
                            var = var.isel({spatial_dims[0]: 0})
                    return cast(pd.Series, var.to_pandas())

        raise ValueError(f"No suitable streamflow variable found in {output_file}")

    @staticmethod
    def _reject_per_gru_runoff(ds, output_file: Path) -> None:
        """Refuse GR's distributed runoff file with an actionable message.

        ``{domain}_{experiment_id}_runs_def.nc`` holds ``q_routed`` as a
        per-GRU runoff *depth rate* in m/s. Turning that into basin discharge
        requires per-GRU areas -- ``Q = sum_i runoff_i * area_i`` -- which this
        interface cannot reach: :meth:`extract_variable` receives only a file
        path, with no config, project directory or basin shapefile.

        Reducing it the way :meth:`_extract_from_netcdf` reduces other
        multi-dimensional variables (``isel`` of the first spatial index) would
        return one GRU's runoff labelled as basin streamflow, so this fails
        loudly instead. ``GRPostProcessor._unrouted_streamflow_cms`` does have
        the config and performs the area-weighted conversion; routed output
        (mizuRoute, already m³/s at a reach) is handled normally above.
        """
        if 'q_routed' not in ds.variables:
            return
        if 'gru' not in ds['q_routed'].dims:
            return

        raise ModelExecutionError(
            f"{output_file.name} is GR's per-GRU distributed runoff "
            "('q_routed', m/s over the 'gru' dimension), not basin streamflow. "
            "Converting it to discharge needs per-GRU areas, which this "
            "extractor has no access to — use GRPostProcessor "
            "(extract_streamflow) for distributed GR output, or point this at "
            "the mizuRoute routed output instead."
        )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """GR outputs are typically in mm/day or m³/s depending on mode."""
        return False  # Units handled by evaluator if needed

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """GR can be lumped or distributed."""
        return 'outlet_selection'  # For distributed mode
