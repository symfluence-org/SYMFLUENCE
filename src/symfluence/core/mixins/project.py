# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Project context mixin for SYMFLUENCE modules.

Provides standard project context attributes like paths and directories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .config import ConfigMixin


def resolve_data_subdir(project_dir: Path, subdir: str) -> Path:
    """Resolve data subdirectory with backward compatibility.

    Prefers data/{subdir}, falls back to {subdir} if it exists.
    """
    new_path = project_dir / 'data' / subdir
    old_path = project_dir / subdir
    if new_path.exists():
        return new_path
    if old_path.exists():
        return old_path
    return new_path


def resolve_forcing_basin_path(project_dir: Path) -> Path:
    """Resolve the basin-averaged forcing directory, store-first.

    Prefers the model-ready forcings store (``data/model_ready/forcings``) when
    it exists and holds NetCDF files, otherwise falls back to the legacy
    ``{data/}forcing/basin_averaged_data`` location. Mirrors the logic
    ``BaseModelPreProcessor`` uses to set ``self.forcing_basin_path`` so that
    preprocessors which do not call ``super().__init__`` (and therefore lack
    that attribute) resolve forcing identically.
    """
    store = project_dir / 'data' / 'model_ready' / 'forcings'
    if store.exists() and any(store.glob('*.nc')):
        return store
    return resolve_data_subdir(project_dir, 'forcing') / 'basin_averaged_data'


def simulation_dir(project_dir: Path, experiment_id: str, model_name: str) -> Path:
    """Conventional model output directory: ``{project_dir}/simulations/{experiment_id}/{model_name}``.

    Pure convention: this deliberately does NOT consult ``EXPERIMENT_OUTPUT_<MODEL>``
    (callers that honour that override resolve it themselves).
    """
    return Path(project_dir) / 'simulations' / experiment_id / model_name


class ProjectContextMixin(ConfigMixin):
    """
    Mixin providing standard project context attributes.

    Extracts core project parameters (data_dir, domain_name, project_dir)
    from the typed SymfluenceConfig, providing a consistent interface across modules.
    """

    @property
    def data_dir(self) -> Path:
        """Root data directory from configuration."""
        _data_dir = getattr(self, '_data_dir', None)
        if _data_dir is not None:
            return Path(_data_dir)

        return Path(self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.',
            dict_key='SYMFLUENCE_DATA_DIR'
        ))

    @data_dir.setter
    def data_dir(self, value: Union[str, Path]) -> None:
        """Set the data directory."""
        self._data_dir = Path(value)

    @data_dir.deleter
    def data_dir(self) -> None:
        """Delete the data directory override."""
        if hasattr(self, '_data_dir'):
            del self._data_dir

    @property
    def domain_name(self) -> str:
        """Domain name from configuration."""
        _domain_name = getattr(self, '_domain_name', None)
        if _domain_name is not None:
            return _domain_name

        return self._get_config_value(
            lambda: self.config.domain.name,
            default='domain',
            dict_key='DOMAIN_NAME'
        )

    @domain_name.setter
    def domain_name(self, value: str) -> None:
        """Set the domain name."""
        self._domain_name = value

    @domain_name.deleter
    def domain_name(self) -> None:
        """Delete the domain name override."""
        if hasattr(self, '_domain_name'):
            del self._domain_name

    @property
    def project_dir(self) -> Path:
        """
        Resolved project directory: {data_dir}/domain_{domain_name}.
        """
        _project_dir = getattr(self, '_project_dir', None)
        if _project_dir is not None:
            return Path(_project_dir)

        return self.data_dir / f"domain_{self.domain_name}"

    @project_dir.setter
    def project_dir(self, value: Union[str, Path]) -> None:
        """Set the project directory."""
        self._project_dir = Path(value)

    @project_dir.deleter
    def project_dir(self) -> None:
        """Delete the project directory override."""
        if hasattr(self, '_project_dir'):
            del self._project_dir

    # Standard subdirectories (based on project convention)

    @property
    def project_shapefiles_dir(self) -> Path:
        """Directory for shapefiles: {project_dir}/data/shapefiles (or legacy {project_dir}/shapefiles)"""
        return resolve_data_subdir(self.project_dir, 'shapefiles')

    @property
    def project_attributes_dir(self) -> Path:
        """Directory for catchment attributes: {project_dir}/data/attributes (or legacy {project_dir}/attributes)"""
        return resolve_data_subdir(self.project_dir, 'attributes')

    @property
    def project_forcing_dir(self) -> Path:
        """Directory for forcing data: {project_dir}/data/forcing (or legacy {project_dir}/forcing)"""
        return resolve_data_subdir(self.project_dir, 'forcing')

    @property
    def project_observations_dir(self) -> Path:
        """Directory for observation data: {project_dir}/data/observations (or legacy {project_dir}/observations)"""
        return resolve_data_subdir(self.project_dir, 'observations')

    @property
    def project_simulations_dir(self) -> Path:
        """Directory for model simulations: {project_dir}/simulations"""
        return self.project_dir / 'simulations'

    def get_simulation_dir(self, model_name: str, experiment_id: Optional[str] = None) -> Path:
        """Model output directory ``{project_dir}/simulations/{experiment_id}/{model_name}``.

        ``experiment_id`` defaults to ``self.experiment_id``. Does not consult
        ``EXPERIMENT_OUTPUT_<MODEL>``; see :func:`simulation_dir`.
        """
        exp_id = self.experiment_id if experiment_id is None else experiment_id
        return simulation_dir(self.project_dir, exp_id, model_name)

    @property
    def project_settings_dir(self) -> Path:
        """Directory for model settings: {project_dir}/settings"""
        return self.project_dir / 'settings'

    @property
    def project_cache_dir(self) -> Path:
        """Directory for cached data: {project_dir}/cache"""
        return self.project_dir / 'cache'
