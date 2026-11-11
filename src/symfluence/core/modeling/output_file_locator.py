# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Output File Locator for Model Evaluation

Provides centralized logic for finding model output files across different
hydrological models. This consolidates the duplicated file-finding patterns
that were scattered across individual evaluators.

Usage:
    from symfluence.core.modeling.output_file_locator import OutputFileLocator

    locator = OutputFileLocator(logger)

    # Find files for a specific output type
    files = locator.find_output_files(sim_dir, 'streamflow')
    files = locator.find_output_files(sim_dir, 'et')

    # Model-specific file finding
    files = locator.find_hype_output(sim_dir)
    files = locator.find_rhessys_output(sim_dir)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Union


class OutputFileLocator:
    """
    Centralized utility for locating model output files.

    Consolidates the file-finding patterns used across evaluators:
    - NetCDF patterns: *_day.nc, *timestep.nc, *_streamflow.nc
    - Model-specific: timeCOUT.txt (HYPE), *_basin.daily (RHESSys)
    - Routing outputs: mizuRoute/*.nc

    The locator uses a priority-based search, returning the first
    matching pattern to avoid returning incorrect file types.
    """

    # Standard NetCDF patterns in priority order (includes recursive variants)
    NETCDF_PATTERNS = {
        'daily': ['*_day.nc', '*_daily.nc', '**/*_day.nc', '**/*_daily.nc'],
        'timestep': ['*timestep.nc', '*_timestep.nc', '**/*timestep.nc', '**/*_timestep.nc'],
        'streamflow': ['*_streamflow.nc', '*streamflow*.nc', '**/*_streamflow.nc', '**/*streamflow*.nc'],
        'output': ['*output*.nc', '*_runs_best.nc', '*_runs_def.nc', '**/*output*.nc', '**/*_runs_best.nc', '**/*_runs_def.nc'],
        'generic': ['*.nc', '**/*.nc'],
    }

    # Model-specific file patterns (includes recursive variants)
    MODEL_PATTERNS = {
        'HYPE': {
            'streamflow': ['timeCOUT.txt', 'HYPE/timeCOUT.txt'],
            'patterns': ['**/timeCOUT.txt'],
        },
        'RHESSys': {
            'streamflow': ['rhessys_results.csv', 'rhessys_basin.daily'],
            'patterns': ['*_basin.daily', '**/*_basin.daily'],
        },
        'GR': {
            'streamflow': ['GR_results.csv'],
            'patterns': ['*_runs_def.nc', '**/*_runs_def.nc'],
        },
        'TROUTE': {
            'streamflow': ['troute_output.nc', 'nex-troute-out.nc'],
            'patterns': ['**/troute_output.nc', '**/nex-troute-out.nc', '**/troute_*.nc'],
        },
        'NGEN': {
            'streamflow': ['nexus_data.nc', 'catchment_data.nc'],
            'patterns': ['**/nexus_data.nc', '**/catchment_data.nc'],
        },
        'mizuRoute': {
            'patterns': ['mizuRoute/*.nc', '*.h.*.nc', '**/*.h.*.nc', 'mizuRoute/**/*.nc'],
        },
        'JFUSE': {
            'streamflow': [],
            'patterns': ['*_jfuse_output.nc', '*_jfuse_output.csv', '**/*_jfuse_output.nc', '**/*_jfuse_output.csv'],
        },
        'CFUSE': {
            'streamflow': [],
            'patterns': ['*_cfuse_output.nc', '*_cfuse_output.csv', '**/*_cfuse_output.nc', '**/*_cfuse_output.csv'],
        },
        'CLM': {
            'streamflow': ['*.clm2.h0.*.nc'],
            'patterns': ['*.clm2.h0.*.nc', '**/*.clm2.h0.*.nc', 'CLM/*.clm2.h0.*.nc'],
        },
    }

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize OutputFileLocator.

        Args:
            logger: Optional logger instance
            experiment_id: Optional experiment ID for filtering files.
                           If provided, files containing this ID in their name
                           are prioritized over modification time sorting.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.experiment_id = experiment_id

    def find_output_files(
        self,
        directory: Union[str, Path],
        output_type: str = 'streamflow',
        model: Optional[str] = None,
        prefer_daily: bool = True,
    ) -> List[Path]:
        """
        Find output files in a directory.

        Args:
            directory: Directory to search
            output_type: Type of output ('streamflow', 'et', 'soil_moisture', etc.)
            model: Optional model name for model-specific patterns
            prefer_daily: If True, prefer daily files over timestep files

        Returns:
            List of matching file paths, sorted by modification time (newest first)
        """
        directory = Path(directory)
        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return []

        # Check for model-specific patterns first
        if model:
            files = self._find_model_specific(directory, model, output_type)
            if files:
                return files

        # Check for routing model outputs (mizuRoute, t-route)
        if 'mizuroute' in str(directory).lower() or output_type == 'streamflow':
            files = self._find_mizuroute_output(directory)
            if files:
                return files

        if 'troute' in str(directory).lower() or output_type == 'streamflow':
            files = self._find_troute_output(directory)
            if files:
                return files

        # Standard NetCDF search with priority
        return self._find_netcdf_files(directory, output_type, prefer_daily)

    def _find_netcdf_files(
        self,
        directory: Path,
        output_type: str,
        prefer_daily: bool
    ) -> List[Path]:
        """Find NetCDF files using standard patterns."""
        # Define search order based on output type
        if output_type == 'streamflow':
            search_order = ['streamflow', 'timestep', 'output', 'generic']
        elif output_type in ['et', 'soil_moisture', 'groundwater', 'snow', 'tws']:
            # These outputs prefer daily aggregated files
            if prefer_daily:
                search_order = ['daily', 'timestep', 'output', 'generic']
            else:
                search_order = ['timestep', 'daily', 'output', 'generic']
        else:
            search_order = ['daily', 'timestep', 'output', 'generic']

        for pattern_key in search_order:
            patterns = self.NETCDF_PATTERNS.get(pattern_key, [])
            for pattern in patterns:
                files = list(directory.glob(pattern))
                if files:
                    self.logger.debug(
                        f"Found {len(files)} files matching '{pattern}' in {directory}"
                    )
                    return self._sort_by_mtime(files)

        return []

    def _find_model_specific(
        self,
        directory: Path,
        model: str,
        output_type: str
    ) -> List[Path]:
        """Find model-specific output files."""
        # Case-insensitive: the table has mixed-case keys ('RHESSys',
        # 'mizuRoute') that an upper-cased lookup could never hit.
        by_upper = {name.upper(): spec for name, spec in self.MODEL_PATTERNS.items()}
        patterns = by_upper.get(model.upper())
        if patterns is None:
            return []

        # Check specific files first
        if output_type in patterns:
            for filename in patterns[output_type]:
                filepath = directory / filename
                if filepath.exists():
                    self.logger.debug(f"Found model-specific file: {filepath}")
                    return [filepath]

        # Then check glob patterns
        if 'patterns' in patterns:
            for pattern in patterns['patterns']:
                files = list(directory.glob(pattern))
                if files:
                    self.logger.debug(
                        f"Found {len(files)} files matching '{pattern}' for {model}"
                    )
                    return self._sort_by_mtime(files)

        return []

    def _find_mizuroute_output(self, directory: Path) -> List[Path]:
        """Find mizuRoute output files."""
        patterns = self.MODEL_PATTERNS['mizuRoute']['patterns']

        for pattern in patterns:
            files = list(directory.glob(pattern))
            if files:
                self.logger.debug(f"Found {len(files)} mizuRoute files")
                return self._sort_by_mtime(files)

        return []

    def _find_troute_output(self, directory: Path) -> List[Path]:
        """Find t-route output files."""
        patterns = self.MODEL_PATTERNS['TROUTE']['patterns']

        # Check streamflow-specific files first
        for filename in self.MODEL_PATTERNS['TROUTE'].get('streamflow', []):
            filepath = directory / filename
            if filepath.exists():
                self.logger.debug(f"Found t-route output: {filepath}")
                return [filepath]

        # Then glob patterns
        for pattern in patterns:
            files = list(directory.glob(pattern))
            if files:
                self.logger.debug(f"Found {len(files)} t-route files")
                return self._sort_by_mtime(files)

        # Also check troute subdirectory
        troute_dir = directory / 'troute'
        if troute_dir.exists():
            for pattern in ['*.nc', '**/*.nc']:
                files = list(troute_dir.glob(pattern))
                if files:
                    self.logger.debug(f"Found {len(files)} t-route files in subdirectory")
                    return self._sort_by_mtime(files)

        return []

    def _sort_by_mtime(self, files: List[Path]) -> List[Path]:
        """Sort files by experiment ID match (if set) then modification time.

        If experiment_id is configured, files containing the ID in their name
        are prioritized. Within matching/non-matching groups, files are sorted
        by modification time (newest first).

        Args:
            files: List of file paths to sort

        Returns:
            Sorted list of file paths
        """
        if self.experiment_id:
            # Prioritize files whose name contains the experiment id as a
            # whole token: a bare substring test let 'run_1' claim
            # 'run_10_timestep.nc'.
            token = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(self.experiment_id)}(?![A-Za-z0-9])"
            )
            matching = [f for f in files if token.search(f.name)]
            if matching:
                self.logger.debug(
                    f"Found {len(matching)} files matching experiment_id '{self.experiment_id}'"
                )
                return sorted(matching, key=lambda f: f.stat().st_mtime, reverse=True)
        return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

    # Convenience methods for specific output types

    def find_streamflow_files(
        self,
        directory: Union[str, Path],
        model: Optional[str] = None
    ) -> List[Path]:
        """
        Find streamflow output files in a simulation directory.

        Searches for streamflow outputs using model-specific patterns first
        (e.g., timeCOUT.txt for HYPE, mizuRoute/*.nc), then falls back to
        generic NetCDF patterns (*_streamflow.nc, *timestep.nc).

        Args:
            directory: Simulation output directory to search
            model: Optional model name for model-specific file patterns

        Returns:
            List of matching file paths, sorted by modification time (newest first)
        """
        return self.find_output_files(directory, 'streamflow', model)

    def find_et_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find evapotranspiration output files.

        Searches for ET outputs, preferring daily aggregated files (*_day.nc,
        *_daily.nc) over timestep files for more efficient processing.

        Args:
            directory: Simulation output directory to search

        Returns:
            List of matching file paths, sorted by modification time (newest first)
        """
        return self.find_output_files(directory, 'et', prefer_daily=True)

    def find_soil_moisture_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find soil moisture output files.

        Searches for soil moisture outputs (volumetric water content, saturation),
        preferring daily aggregated files for efficient comparison with satellite
        observations (e.g., SMAP, SMOS).

        Args:
            directory: Simulation output directory to search

        Returns:
            List of matching file paths, sorted by modification time (newest first)
        """
        return self.find_output_files(directory, 'soil_moisture', prefer_daily=True)

    def find_snow_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find snow output files (SWE, snow depth, snow cover area).

        Searches for snow-related outputs, preferring daily aggregated files
        for comparison with SNODAS, SNOTEL, or satellite snow products.

        Args:
            directory: Simulation output directory to search

        Returns:
            List of matching file paths, sorted by modification time (newest first)
        """
        return self.find_output_files(directory, 'snow', prefer_daily=True)

    def find_tws_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find total water storage output files.

        Searches for TWS outputs (sum of all water storage components),
        preferring daily files for comparison with GRACE/GRACE-FO observations.

        Args:
            directory: Simulation output directory to search

        Returns:
            List of matching file paths, sorted by modification time (newest first)
        """
        return self.find_output_files(directory, 'tws', prefer_daily=True)

    def find_groundwater_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find groundwater output files.

        Searches for groundwater/aquifer storage outputs, preferring daily
        aggregated files for comparison with well observations or derived products.

        Args:
            directory: Simulation output directory to search

        Returns:
            List of matching file paths, sorted by modification time (newest first)
        """
        return self.find_output_files(directory, 'groundwater', prefer_daily=True)

    # Model-specific convenience methods

    def find_hype_output(
        self,
        directory: Union[str, Path],
        output_type: str = 'streamflow'
    ) -> List[Path]:
        """Find HYPE model output files."""
        directory = Path(directory)

        if output_type == 'streamflow':
            # Priority order for HYPE streamflow
            candidates = [
                directory / 'timeCOUT.txt',
                directory / 'HYPE' / 'timeCOUT.txt',
            ]
            for candidate in candidates:
                if candidate.exists():
                    return [candidate]

            # Recursive search
            files = list(directory.glob('**/timeCOUT.txt'))
            if files:
                return files

            # Fall back to mizuRoute
            return self._find_mizuroute_output(directory)

        return self.find_output_files(directory, output_type, model='HYPE')

    def find_rhessys_output(
        self,
        directory: Union[str, Path],
        output_type: str = 'streamflow'
    ) -> List[Path]:
        """Find RHESSys model output files."""
        directory = Path(directory)

        if output_type == 'streamflow':
            # Priority order for RHESSys streamflow
            candidates = [
                directory / 'rhessys_results.csv',
                directory / 'rhessys_basin.daily',
            ]
            for candidate in candidates:
                if candidate.exists():
                    return [candidate]

            # Pattern search
            files = list(directory.glob('*_basin.daily'))
            if files:
                return self._sort_by_mtime(files)

        return self.find_output_files(directory, output_type, model='RHESSys')

    def find_gr_output(
        self,
        directory: Union[str, Path],
        domain_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> List[Path]:
        """Find GR model output files."""
        directory = Path(directory)

        # Check for mizuRoute first
        files = self._find_mizuroute_output(directory)
        if files:
            return files

        # Check for CSV (lumped mode)
        csv_file = directory / 'GR_results.csv'
        if csv_file.exists():
            return [csv_file]

        # Check for NetCDF (distributed mode)
        if domain_name and experiment_id:
            nc_file = directory / f'{domain_name}_{experiment_id}_runs_def.nc'
            if nc_file.exists():
                return [nc_file]

        # Fall back to generic NetCDF search
        return self._find_netcdf_files(directory, 'streamflow', prefer_daily=False)

    def get_most_recent(
        self,
        directory: Union[str, Path],
        output_type: str = 'streamflow'
    ) -> Optional[Path]:
        """
        Get the most recently modified output file.

        Useful for TWS and other evaluators that need a single file.
        """
        files = self.find_output_files(directory, output_type)
        return files[0] if files else None


# Module-level instance for convenience
_default_locator: Optional[OutputFileLocator] = None


def get_output_file_locator(logger: Optional[logging.Logger] = None) -> OutputFileLocator:
    """Get a shared OutputFileLocator instance."""
    global _default_locator
    if _default_locator is None:
        _default_locator = OutputFileLocator(logger)
    return _default_locator
