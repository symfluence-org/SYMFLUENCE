# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Model Preprocessor

Orchestrates CLM preprocessing by delegating to focused sub-modules:
- CLMDomainGenerator: Domain file, ESMF mesh, catchment geometry
- CLMSurfaceGenerator: Surface data, parameter files
- CLMNuopcGenerator: NUOPC runtime configuration files

Forcing is handled by the existing CLMForcingProcessor.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.preprocessors.add("CLM")
class CLMPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a CLM5 model run.

    Delegates to sub-modules for each domain:
    - domain_generator: Domain file and ESMF mesh
    - surface_generator: Surface data and default parameters
    - nuopc_generator: All NUOPC runtime config files
    """

    MODEL_NAME = "CLM"

    def __init__(self, config, logger):
        super().__init__(config, logger)

        # Use standard SYMFLUENCE directory layout (inherited from base):
        #   self.setup_dir   -> {project_dir}/settings/CLM
        #   self.forcing_dir -> {project_dir}/data/forcing/CLM_input
        self.settings_dir = self.setup_dir
        self.params_dir = self.setup_dir / "parameters"

        # Lazy-init sub-modules
        self._domain_generator = None
        self._surface_generator = None
        self._nuopc_generator = None

    # ------------------------------------------------------------------ #
    #  Lazy-init sub-module properties
    # ------------------------------------------------------------------ #

    @property
    def domain_generator(self):
        if self._domain_generator is None:
            from .domain_generator import CLMDomainGenerator
            self._domain_generator = CLMDomainGenerator(self)
        return self._domain_generator

    @property
    def surface_generator(self):
        if self._surface_generator is None:
            from .surface_generator import CLMSurfaceGenerator
            self._surface_generator = CLMSurfaceGenerator(self)
        return self._surface_generator

    @property
    def nuopc_generator(self):
        if self._nuopc_generator is None:
            from .nuopc_generator import CLMNuopcGenerator
            self._nuopc_generator = CLMNuopcGenerator(self)
        return self._nuopc_generator

    # ------------------------------------------------------------------ #
    #  Install path
    # ------------------------------------------------------------------ #

    def _get_install_path(self) -> Path:
        """Resolve CLM install path."""
        install_path = self._get_config_value(lambda: self.config.model.clm.install_path, dict_key='CLM_INSTALL_PATH')
        if install_path == 'default':
            code_dir = self._get_config_value(
                lambda: self.config.system.code_dir,
                default=None, dict_key='SYMFLUENCE_CODE_DIR'
            )
            if code_dir:
                code_path = Path(code_dir)
                return code_path.parent / (code_path.name + '_data') / 'installs' / 'clm'
            return self.project_dir.parents[1] / 'installs' / 'clm'
        return Path(install_path)

    # ------------------------------------------------------------------ #
    #  Main pipeline
    # ------------------------------------------------------------------ #

    def run_preprocessing(self) -> bool:  # type: ignore[override]
        """Run CLM preprocessing pipeline."""
        logger.info("Starting CLM preprocessing")

        self._create_directory_structure()
        self.domain_generator.generate_domain_file()
        self.domain_generator.generate_esmf_mesh()
        self.surface_generator.generate_surface_data()
        self.surface_generator.copy_default_params()
        self._generate_forcing()
        self._generate_topo_forcing()
        self.nuopc_generator.generate_nuopc_runtime()

        logger.info("CLM preprocessing complete")
        return True

    def _create_directory_structure(self) -> None:
        """Create CLM input directory structure."""
        for d in [self.settings_dir, self.forcing_dir, self.params_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created CLM directory structure: settings={self.settings_dir}, forcing={self.forcing_dir}")

    # ------------------------------------------------------------------ #
    #  Forcing (thin wrappers around existing CLMForcingProcessor)
    # ------------------------------------------------------------------ #

    def _generate_forcing(self) -> None:
        """Generate CLM forcing files in DATM format."""
        from .forcing_processor import CLMForcingProcessor

        lat, lon, _ = self.domain_generator.get_catchment_centroid()
        # Store-first (model_ready/forcings), else legacy basin_averaged_data.
        forcing_data_dir = self.forcing_basin_path
        if not forcing_data_dir.exists():
            forcing_data_dir = self.project_forcing_dir

        start_date = self._get_config_value(
            lambda: self.config.domain.time_start,
            default='2000-01-01', dict_key='EXPERIMENT_TIME_START'
        )
        end_date = self._get_config_value(
            lambda: self.config.domain.time_end,
            default='2010-12-31', dict_key='EXPERIMENT_TIME_END'
        )

        processor = CLMForcingProcessor(self.config_dict, logger)
        processor.process_forcing(
            forcing_data_dir=forcing_data_dir,
            output_dir=self.forcing_dir,
            lat=lat, lon=lon,
            start_date=str(start_date), end_date=str(end_date),
            forcing_selector=self._select_run_forcing,
        )

    def _generate_topo_forcing(self) -> None:
        """Generate single-point topography forcing file for DATM."""
        lat, lon, _ = self.domain_generator.get_catchment_centroid()
        mean_elev = self.domain_generator.get_mean_elevation()

        time_val = np.array([0.0], dtype=np.float64)

        ds = xr.Dataset({
            'LONGXY': xr.DataArray(np.array([[[lon]]]), dims=['time', 'lat', 'lon'],
                                   attrs={'units': 'degrees_east'}),
            'LATIXY': xr.DataArray(np.array([[[lat]]]), dims=['time', 'lat', 'lon'],
                                   attrs={'units': 'degrees_north'}),
            'TOPO': xr.DataArray(np.array([[[mean_elev]]]), dims=['time', 'lat', 'lon'],
                                 attrs={'units': 'm', 'long_name': 'topography height'}),
        }, coords={
            'time': xr.DataArray(time_val, dims=['time'], attrs={
                'units': 'days since 0001-01-01 00:00:00',
                'calendar': 'noleap',
                'long_name': 'time',
            }),
        }, attrs={'title': f'Topography forcing for {self.domain_name}'})

        filepath = self.forcing_dir / 'topo_forcing.nc'
        ds.to_netcdf(filepath, format='NETCDF4',
                     encoding={'time': {'dtype': 'float64', '_FillValue': None}})
        logger.info(f"Generated topo forcing: {filepath}")
