# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mizuRoute Result Extractor.

Handles extraction of routed streamflow from mizuRoute model outputs.
Encapsulates mizuRoute-specific logic for reach/segment identification
and routing variable names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor


class MizuRouteResultExtractor(ModelResultExtractor):
    """mizuRoute-specific result extraction.

    Handles mizuRoute's unique output characteristics:
    - Variable naming: IRFroutedRunoff, KWTroutedRunoff, averageRoutedRunoff
    - Spatial dimension: seg (river segments) or reachID (reaches)
    - Outlet identification: reach/segment with highest mean discharge
    - File patterns: mizuRoute/*.nc, *.h.*.nc
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for mizuRoute outputs."""
        return {
            'streamflow': [
                'mizuRoute/*.nc',
                '*.h.*.nc',
                'mizuRoute/**/*.nc',
                '*_routed.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get mizuRoute variable names for different types."""
        variable_mapping = {
            'streamflow': ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract routed streamflow from mizuRoute output.

        Args:
            output_file: Path to mizuRoute NetCDF output
            variable_type: Type of variable (typically 'streamflow')
            **kwargs: Additional options (currently unused)

        Returns:
            Time series of routed discharge at outlet

        Raises:
            ValueError: If no routed runoff variable found
        """
        if variable_type != 'streamflow':
            raise ModelExecutionError(
                f"mizuRoute extractor only supports 'streamflow', got '{variable_type}'"
            )

        var_names = self.get_variable_names(variable_type)

        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Find outlet: shared reach rule (SIM_REACH_ID, topology
                    # outlet, else highest mean flow)
                    if 'seg' in var.dims or 'reachID' in var.dims:
                        from symfluence.core.modeling.utilities.reach_selection import (
                            select_outlet_series,
                            topology_candidates,
                        )
                        return select_outlet_series(
                            ds, var, kwargs.get('sim_reach_id'),
                            topology_candidates(near=output_file),
                        )
                    # No spatial dimension - use as-is
                    return cast(pd.Series, var.to_pandas())

            raise ModelExecutionError(
                f"No suitable routed runoff variable found in {output_file}. "
                f"Tried: {var_names}"
            )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """mizuRoute outputs are already in m³/s, no conversion needed."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """mizuRoute automatically aggregates to reaches/segments."""
        return 'outlet_selection'  # Select outlet reach with max discharge
