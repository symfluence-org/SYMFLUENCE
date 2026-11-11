# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Path resolution utilities for SYMFLUENCE configuration management.

Provides standardized path resolution with default fallback handling to eliminate
code duplication across model preprocessors, managers, and utilities.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from symfluence.core.mixins.project import resolve_data_subdir

# Subdirectories that have been migrated under data/
_DATA_SUBDIRS = ('forcing', 'attributes', 'observations')


def resolve_path(
    config: Dict[str, Any],
    config_key: str,
    project_dir: Path,
    default_subpath: str,
    logger: Optional[logging.Logger] = None,
    must_exist: bool = False
) -> Path:
    """
    Resolve a path from config or use default relative to project directory.

    This function handles the common pattern of checking a config key for a path value,
    with special handling for 'default' keyword and None values which trigger use of
    a default path relative to the project directory.

    For default subpaths starting with forcing/, attributes/, or observations/,
    uses resolve_data_subdir for backward-compatible resolution (prefers data/{subdir},
    falls back to {subdir}).

    Args:
        config: Configuration dictionary
        config_key: Key to look up in configuration (e.g., 'FORCING_PATH')
        project_dir: Project base directory (typically data_dir/domain_{domain_name})
        default_subpath: Default path relative to project_dir (e.g., 'forcing/merged_data')
        logger: Optional logger for debug messages
        must_exist: If True, raise FileNotFoundError if resolved path doesn't exist

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist

    Examples:
        >>> config = {'FORCING_PATH': 'default'}
        >>> resolve_path(config, 'FORCING_PATH', Path('/data/domain_test'), 'forcing/merged_data')
        Path('/data/domain_test/data/forcing/merged_data')

        >>> config = {'FORCING_PATH': '/custom/forcing/path'}
        >>> resolve_path(config, 'FORCING_PATH', Path('/data/domain_test'), 'forcing/merged_data')
        Path('/custom/forcing/path')
    """
    path_value = config.get(config_key)
    if isinstance(path_value, str):
        path_value = path_value.strip()
        # '', 'default' (any case) mean "unset" -> use the default location.
        if path_value == '' or path_value.lower() == 'default':
            path_value = None
        else:
            path_value = os.path.expandvars(os.path.expanduser(path_value))

    # Handle unset values -> use default path relative to project_dir
    if path_value is None:
        # Check if default_subpath starts with a migrated data subdirectory
        parts = default_subpath.split('/', 1)
        if parts[0] in _DATA_SUBDIRS:
            base = resolve_data_subdir(project_dir, parts[0])
            resolved = base / parts[1] if len(parts) > 1 else base
        else:
            resolved = project_dir / default_subpath
        if logger:
            logger.debug(f"Using default path for {config_key}: {resolved}")
    else:
        resolved = Path(path_value)
        if logger:
            logger.debug(f"Using configured path for {config_key}: {resolved}")

    # Validate existence if required
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Required path does not exist: {resolved} (config_key: {config_key})"
        )

    return resolved


def resolve_file_path(
    config: Dict[str, Any],
    project_dir: Path,
    path_key: str,
    name_key: str,
    default_subpath: str,
    default_name: str,
    logger: Optional[logging.Logger] = None,
    must_exist: bool = False
) -> Path:
    """
    Resolve complete file path from separate directory and filename config keys.

    This function handles the common pattern where a file path is specified via two
    config keys: one for the directory and one for the filename. Both support the
    'default' keyword for fallback values.

    Args:
        config: Configuration dictionary
        project_dir: Project base directory
        path_key: Config key for directory path (e.g., 'DEM_PATH')
        name_key: Config key for file name (e.g., 'DEM_NAME')
        default_subpath: Default directory relative to project_dir
        default_name: Default file name
        logger: Optional logger for debug messages
        must_exist: If True, raise FileNotFoundError if file doesn't exist

    Returns:
        Complete file path (Path object)

    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist

    Examples:
        >>> config = {'DEM_PATH': 'default', 'DEM_NAME': 'default'}
        >>> resolve_file_path(
        ...     config, Path('/data/domain_test'), 'DEM_PATH', 'DEM_NAME',
        ...     'attributes/elevation/dem', 'domain_test_elv.tif'
        ... )
        Path('/data/domain_test/attributes/elevation/dem/domain_test_elv.tif')

        >>> config = {'DEM_PATH': '/custom/dem', 'DEM_NAME': 'my_dem.tif'}
        >>> resolve_file_path(
        ...     config, Path('/data/domain_test'), 'DEM_PATH', 'DEM_NAME',
        ...     'attributes/elevation/dem', 'domain_test_elv.tif'
        ... )
        Path('/custom/dem/my_dem.tif')
    """
    # Get directory path using resolve_path
    dir_path = resolve_path(
        config=config,
        config_key=path_key,
        project_dir=project_dir,
        default_subpath=default_subpath,
        logger=logger,
        must_exist=False  # Check file existence below, not directory
    )

    # Get filename (handle 'default' or None)
    file_name = config.get(name_key)
    if file_name == 'default' or file_name is None:
        file_name = default_name

    # Construct complete file path
    file_path = dir_path / file_name

    # Validate file existence if required
    if must_exist and not file_path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {file_path} "
            f"(path_key: {path_key}, name_key: {name_key})"
        )

    return file_path


# Values of ``<MODEL>_INSTALL_PATH`` that select the default install location.
_INSTALL_DEFAULT_VALUES = ('default', '')


def _is_default_install(install_path: Any) -> bool:
    return install_path is None or (
        isinstance(install_path, str) and install_path in _INSTALL_DEFAULT_VALUES
    )


def resolve_install_dir(
    install_path: Any,
    data_dir: Union[str, Path],
    default_subpath: str,
) -> Path:
    """Resolve a model install directory from a ``<MODEL>_INSTALL_PATH`` value.

    A standalone (no-``self``) companion to ``BaseModelRunner.get_install_path``
    for calibration workers and other code that reads the setting itself (via
    ``resolve_config_value`` / ``self._get_config_value``) and passes it in.

    * ``'default'``, ``None`` or ``''`` -> ``data_dir / default_subpath``
      (no code_dir / sibling-data fallback search; that is runner-only).
    * any other value -> ``Path(install_path)`` unchanged: an absolute path is
      used as-is and a relative path stays relative to the CWD, exactly as
      ``get_install_path`` treats it.

    Args:
        install_path: The configured ``<MODEL>_INSTALL_PATH`` value.
        data_dir: SYMFLUENCE data directory the default is relative to.
        default_subpath: Default location under ``data_dir``, e.g. ``'installs/vic/bin'``.
    """
    if _is_default_install(install_path):
        return Path(data_dir) / default_subpath
    return Path(install_path)


def resolve_install_executable(
    install_path: Any,
    data_dir: Union[str, Path],
    default_subpath: str,
    exe_name: str,
) -> Path:
    """Resolve a model executable from a ``<MODEL>_INSTALL_PATH`` value and exe name.

    * default install path -> ``data_dir / default_subpath / exe_name``
    * override naming an existing directory -> ``override / exe_name``
    * any other override (a file, or a path that does not exist) -> the
      override itself, taken as the full executable path.

    Callers whose override is always a directory should use
    ``resolve_install_dir(...) / exe_name`` instead.
    """
    if _is_default_install(install_path):
        return Path(data_dir) / default_subpath / exe_name
    path = Path(install_path)
    return path / exe_name if path.is_dir() else path


def find_basin_shapefile(
    shapefiles_dir: Path,
    domain_name: str,
    definition_method: str,
    experiment_id: str,
    *,
    explicit_path: Optional[str] = None,
    explicit_name: Optional[str] = None,
    include_river_basins: bool = True,
    required: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Locate an EXISTING catchment/basin polygon shapefile on disk.

    Discretization writes the catchment under an experiment-scoped nested layout
    (``catchment/{definition_method}/{experiment_id}/{domain}_HRUs_{disc}.shp``)
    and may upper-case the discretization suffix (GRUs -> GRUS), so a flat
    ``catchment/{domain}_catchment.shp`` lookup never matches. This finder is a
    READ-side complement to :meth:`PathResolverMixin._get_catchment_file_path`
    (which builds the canonical WRITE path). Resolution order:

    1. ``explicit_path`` (+ ``explicit_name``) when provided and non-'default'.
    2. The canonical nested layout, then the legacy flat ``catchment`` directory.
    3. A deep search under ``catchment/`` for ``{domain}_HRUs_*.shp`` (handles
       experiment-id and casing drift).
    4. The ``river_basins`` outline (any single-polygon basin works for masking
       or area), unless ``include_river_basins`` is False.

    Args:
        shapefiles_dir: The domain's ``shapefiles`` directory.
        domain_name: Domain name (filename prefix).
        definition_method: e.g. ``lumped`` / ``semidistributed`` (nesting level 1).
        experiment_id: Experiment id (nesting level 2).
        explicit_path: Optional CATCHMENT_PATH override (dir or 'default').
        explicit_name: Optional CATCHMENT_SHP_NAME override (file or 'default').
        include_river_basins: Allow falling back to the river_basins outline.
        required: Raise FileNotFoundError when nothing is found.
        logger: Optional logger for a debug breadcrumb.

    Returns:
        The resolved Path, or None when not found and not ``required``.
    """
    def _log(found: Path) -> Path:
        if logger is not None:
            logger.debug(f"Resolved basin shapefile: {found}")
        return found

    # 1. Explicit user-provided path/name.
    if explicit_path and explicit_path != 'default':
        base_dir = Path(explicit_path)
        if explicit_name and explicit_name != 'default':
            explicit = base_dir / explicit_name
            if explicit.exists():
                return _log(explicit)
        found = sorted(base_dir.rglob("*.shp")) if base_dir.exists() else []
        if found:
            return _log(found[0])

    catchment_dir = shapefiles_dir / "catchment"

    # 2. Canonical nested layout, then legacy flat directory.
    nested_dir = catchment_dir / definition_method / experiment_id
    for candidate_dir in (nested_dir, catchment_dir):
        if not candidate_dir.exists():
            continue
        matches = (
            sorted(candidate_dir.glob(f"{domain_name}_HRUs_*.shp"))
            or sorted(candidate_dir.glob(f"{domain_name}*.shp"))
        )
        if not matches and candidate_dir == nested_dir:
            matches = sorted(candidate_dir.glob("*.shp"))
        if matches:
            return _log(matches[0])

    # 3. Deep search anywhere under catchment/ (experiment/casing drift).
    if catchment_dir.exists():
        deep = sorted(catchment_dir.rglob(f"{domain_name}_HRUs_*.shp"))
        if deep:
            return _log(deep[0])

    # 4. River basins outline as a last resort.
    if include_river_basins:
        river_basins_dir = shapefiles_dir / "river_basins"
        if river_basins_dir.exists():
            rb_matches = (
                sorted(river_basins_dir.glob(f"{domain_name}_riverBasins_*.shp"))
                or sorted(river_basins_dir.glob("*.shp"))
            )
            if rb_matches:
                return _log(rb_matches[0])

    if required:
        raise FileNotFoundError(
            f"Catchment/basin shapefile not found. Searched '{catchment_dir}' "
            f"(incl. {definition_method}/{experiment_id})"
            f"{' and river_basins' if include_river_basins else ''}. Run "
            f"define_domain + discretize_domain first, or set "
            f"CATCHMENT_PATH/CATCHMENT_SHP_NAME."
        )
    return None


def find_catchment_subfile(
    shapefiles_dir: Path,
    definition_method: str,
    experiment_id: str,
    filename: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Find a specifically-named file under the nested catchment layout.

    For artifacts written with a fixed name (e.g.
    ``{domain}_catchment_delineated.shp``) that discretization places under
    ``catchment/{definition_method}/{experiment_id}/``. Checks the nested path,
    then the legacy flat ``catchment`` directory, then deep-searches. Returns
    None when not found.
    """
    catchment_dir = shapefiles_dir / "catchment"
    for candidate in (
        catchment_dir / definition_method / experiment_id / filename,
        catchment_dir / filename,
    ):
        if candidate.exists():
            if logger is not None:
                logger.debug(f"Resolved catchment file: {candidate}")
            return candidate
    if catchment_dir.exists():
        deep = sorted(catchment_dir.rglob(filename))
        if deep:
            if logger is not None:
                logger.debug(f"Resolved catchment file (deep search): {deep[0]}")
            return deep[0]
    return None


def find_river_basins_shapefile(
    shapefiles_dir: Path,
    domain_name: str,
    *,
    required: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Locate the EXISTING river-basins (domain outline) shapefile.

    Domain definition writes ``river_basins/{domain}_riverBasins_{method}.shp``
    (flat, method-suffixed — not experiment-nested). Returns None when not found
    unless ``required``.
    """
    river_basins_dir = shapefiles_dir / "river_basins"
    if river_basins_dir.exists():
        matches = (
            sorted(river_basins_dir.glob(f"{domain_name}_riverBasins_*.shp"))
            or sorted(river_basins_dir.glob("*.shp"))
        )
        if matches:
            if logger is not None:
                logger.debug(f"Resolved river_basins shapefile: {matches[0]}")
            return matches[0]
    if required:
        raise FileNotFoundError(
            f"River-basins shapefile not found under '{river_basins_dir}'. "
            f"Run define_domain first."
        )
    return None


from .mixins import ConfigurableMixin


class PathResolverMixin(ConfigurableMixin):
    """
    Mixin providing path resolution methods for classes with config and project_dir.

    This mixin eliminates duplicate `_get_default_path()` implementations across
    model preprocessors, managers, and utilities by providing a standardized interface.

    Usage:
        class MyPreprocessor(BaseModelPreProcessor, PathResolverMixin):
            def __init__(self, config, logger):
                super().__init__(config, logger)

                # Use the mixin's methods
                self.forcing_path = self._get_default_path(
                    'FORCING_PATH', 'forcing/merged_data'
                )

    Examples:
        >>> class TestClass(PathResolverMixin):
        ...     def __init__(self):
        ...         self.config = {'MY_PATH': 'default', 'DOMAIN_NAME': 'test'}
        ...
        >>> obj = TestClass()
        >>> obj._get_default_path('MY_PATH', 'some/path')
        Path('./domain_test/some/path')
    """

    def _get_default_path(
        self,
        config_key: str,
        default_subpath: str,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve path from config or use default (instance method).

        This method provides the standard interface that existing code expects,
        delegating to the standalone resolve_path() function.

        Args:
            config_key: Configuration key to lookup
            default_subpath: Default path relative to project_dir
            must_exist: If True, raise FileNotFoundError if path doesn't exist

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If must_exist=True and path doesn't exist
        """
        return resolve_path(
            config=self.config_dict,
            config_key=config_key,
            project_dir=self.project_dir,
            default_subpath=default_subpath,
            logger=self.logger,
            must_exist=must_exist
        )

    def resolve(
        self,
        config_key: str,
        default_subpath: str,
        filename: Optional[str] = None,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve a path from config, with optional filename (alias for _get_default_path).

        Matches the PathManager.resolve() API for easier migration and consistency.
        """
        path = self._get_default_path(config_key, default_subpath, must_exist=must_exist)
        if filename:
            return path / filename
        return path

    def _get_file_path(
        self,
        path_key: str,
        name_key: str,
        default_subpath: str,
        default_name: str,
        must_exist: bool = False
    ) -> Path:
        """
        Resolve complete file path from directory and name keys (instance method).

        This method provides a convenient interface for resolving file paths that
        are specified via separate directory and filename config keys.

        Args:
            path_key: Config key for directory path
            name_key: Config key for file name
            default_subpath: Default directory relative to project_dir
            default_name: Default file name
            must_exist: If True, raise FileNotFoundError if file doesn't exist

        Returns:
            Complete file path (Path object)

        Raises:
            FileNotFoundError: If must_exist=True and file doesn't exist
        """
        return resolve_file_path(
            config=self.config_dict,
            project_dir=self.project_dir,
            path_key=path_key,
            name_key=name_key,
            default_subpath=default_subpath,
            default_name=default_name,
            logger=self.logger,
            must_exist=must_exist
        )

    def _get_catchment_file_path(
        self,
        catchment_name: Optional[str] = None
    ) -> Path:
        """
        Resolve catchment shapefile path with backward compatibility.

        Checks for existing file in the legacy location first, then returns
        the new organized path if not found.

        The new path structure includes domain_definition_method and experiment_id:
            shapefiles/catchment/{domain_definition_method}/{experiment_id}/

        For backward compatibility:
            - If file exists at old path (shapefiles/catchment/), use that
            - Otherwise use new organized path

        Args:
            catchment_name: Optional catchment shapefile name. If None,
                           constructs default name from domain and discretization.

        Returns:
            Full path to the catchment shapefile
        """
        # Construct catchment name if not provided
        if catchment_name is None or catchment_name == 'default':
            catchment_name = self._get_config_value(
                lambda: self.config.paths.catchment_name,
                default='default',
                dict_key='CATCHMENT_SHP_NAME'
            )

        if catchment_name is None or catchment_name == 'default':
            discretization = self.domain_discretization
            # Handle comma-separated attributes for output filename
            if "," in discretization:
                method_suffix = discretization.replace(",", "_")
            else:
                method_suffix = discretization
            catchment_name = f"{self.domain_name}_HRUs_{method_suffix}.shp"

        # Check old path first for backward compatibility
        old_path = self.project_dir / "shapefiles" / "catchment" / catchment_name
        if old_path.exists():
            return old_path

        # Return new organized path
        new_path = (
            self.project_dir / "shapefiles" / "catchment" /
            self.domain_definition_method / self.experiment_id / catchment_name
        )
        return new_path

    def _select_run_forcing(self, forcing_files, pattern: str = '*.nc') -> List[Path]:
        """Select the remapped forcing for THIS run's HRU set (issue #339).

        *forcing_files* is a directory (globbed with *pattern*) or a file list.
        Selection uses the run's discretization and a content hash of its
        catchment shapefile; see
        :func:`symfluence.core.modeling.forcing_naming.select_forcing_files`.
        """
        from symfluence.core.modeling.forcing_naming import resolve_forcing_files

        hru_field = self._get_config_value(
            lambda: self.config.paths.catchment_hruid, default='HRU_ID', dict_key='CATCHMENT_SHP_HRUID')
        catchment = self._get_catchment_file_path()
        return resolve_forcing_files(
            forcing_files, self.domain_discretization, catchment, hru_field, pattern)

    def _find_basin_shapefile(
        self,
        required: bool = False,
        include_river_basins: bool = True,
    ) -> Optional[Path]:
        """Locate an EXISTING catchment/basin shapefile for reading.

        Read-side complement to :meth:`_get_catchment_file_path` (which builds
        the canonical write path). Honours explicit CATCHMENT_PATH /
        CATCHMENT_SHP_NAME, searches the nested discretization layout, and falls
        back to the river_basins outline. Returns None when not found unless
        ``required``. See :func:`find_basin_shapefile`.
        """
        explicit_path = self._get_config_value(
            lambda: self.config.paths.catchment_path, default='default', dict_key='CATCHMENT_PATH'
        )
        explicit_name = self._get_config_value(
            lambda: self.config.paths.catchment_name, default='default', dict_key='CATCHMENT_SHP_NAME'
        )
        return find_basin_shapefile(
            self.project_dir / "shapefiles",
            self.domain_name,
            self.domain_definition_method,
            self.experiment_id,
            explicit_path=explicit_path,
            explicit_name=explicit_name,
            include_river_basins=include_river_basins,
            required=required,
            logger=getattr(self, 'logger', None),
        )

    def _get_method_suffix(self) -> str:
        """
        Get delineation suffix encoding all relevant configuration.

        Returns unique suffixes based on definition method and associated options:

        | Method           | Subset | Grid Source | Suffix                              |
        |------------------|--------|-------------|-------------------------------------|
        | point            | —      | —           | point                               |
        | lumped           | false  | —           | lumped                              |
        | lumped           | true   | —           | lumped_subset_{geofabric}           |
        | semidistributed  | false  | —           | semidistributed                     |
        | semidistributed  | true   | —           | semidistributed_subset_{geofabric}  |
        | distributed      | false  | generate    | distributed_{cellsize}m             |
        | distributed      | true   | —           | distributed_subset_{geofabric}      |
        | distributed      | —      | native      | distributed_native_{dataset}        |
        """
        cfg = self.config
        if cfg is None:
            return 'semidistributed'

        # Handle both typed config objects and plain dicts
        if isinstance(cfg, dict):
            method = cfg.get('DOMAIN_DEFINITION_METHOD', cfg.get('DEFINITION_METHOD', 'semidistributed'))
            subset = cfg.get('SUBSET_FROM_GEOFABRIC', False)
            grid_source = cfg.get('GRID_SOURCE', 'generate')
            geofabric_type = cfg.get('GEOFABRIC_TYPE', 'na')
            grid_cell_size = int(cfg.get('GRID_CELL_SIZE', 1000))
            native_dataset = cfg.get('NATIVE_GRID_DATASET', 'era5')
        else:
            method = cfg.domain.definition_method or 'semidistributed'
            subset = getattr(cfg.domain, 'subset_from_geofabric', False)
            grid_source = getattr(cfg.domain, 'grid_source', 'generate')
            geofabric_type = cfg.domain.delineation.geofabric_type or 'na'
            grid_cell_size = int(cfg.domain.grid_cell_size)
            native_dataset = getattr(cfg.domain, 'native_grid_dataset', 'era5')

        # Point method - simple suffix
        if method == 'point':
            return 'point'

        # Lumped method
        if method == 'lumped':
            if subset:
                return f"lumped_subset_{geofabric_type}"
            return 'lumped'

        # Semi-distributed method
        if method == 'semidistributed':
            if subset:
                return f"semidistributed_subset_{geofabric_type}"
            return 'semidistributed'

        # Distributed method
        if method == 'distributed':
            if subset:
                return f"distributed_subset_{geofabric_type}"
            if grid_source == 'native':
                return f"distributed_native_{native_dataset}"
            return f"distributed_{grid_cell_size}m"

        return method
