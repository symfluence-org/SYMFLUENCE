# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SpatialOrchestrator - Centralized spatial mode handling and routing integration.

This module consolidates the spatial mode logic (lumped, semi_distributed, distributed)
and mizuRoute integration that was previously scattered across FUSE, GR, SUMMA, and
other model runners.

Key Responsibilities:
    - Validate and normalize spatial mode configurations
    - Determine if routing is required based on model and domain settings
    - Convert model outputs to mizuRoute-compatible format
    - Orchestrate routing execution when needed

Usage:
    class MyRunner(BaseModelRunner, ModelExecutor, SpatialOrchestrator):
        def run_model(self):
            # Validate spatial setup
            spatial_config = self.get_spatial_config()

            # Run model
            output = self._execute_model()

            # Handle routing if needed
            if self.requires_routing():
                output = self.route_model_output(output)
"""

import shutil
import subprocess
import tempfile
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from symfluence.core.exceptions import GeospatialError, ModelExecutionError
from symfluence.models.spatial_modes import SpatialMode


class RoutingModel(Enum):
    """Supported routing models."""
    NONE = "none"
    MIZUROUTE = "mizuRoute"
    TROUTE = "troute"
    DROUTE = "dRoute"

    @classmethod
    def from_string(cls, value: str) -> 'RoutingModel':
        """Parse routing model from string."""
        if value is None or value.lower() in ('none', 'null', ''):
            return cls.NONE
        normalized = value.lower()
        mapping = {
            'mizuroute': cls.MIZUROUTE,
            'mizu_route': cls.MIZUROUTE,
            'mizu': cls.MIZUROUTE,
            'troute': cls.TROUTE,
            't_route': cls.TROUTE,
            'droute': cls.DROUTE,
            'd_route': cls.DROUTE,
        }
        return mapping.get(normalized, cls.NONE)


@dataclass
class RoutingConfig:
    """Configuration for routing integration.

    Attributes:
        model: Which routing model to use
        topology_file: Path to river network topology
        routing_var: Variable name for runoff input
        routing_units: Units of the runoff variable
        output_var: Variable name for routed output
    """
    model: RoutingModel = RoutingModel.NONE
    topology_file: Optional[Path] = None
    routing_var: str = "averageRoutedRunoff"
    routing_units: str = "m/s"
    output_var: str = "IRFroutedRunoff"
    control_file: Optional[Path] = None


@dataclass
class SpatialConfig:
    """Complete spatial configuration for a model run.

    Combines spatial mode with routing requirements and provides
    validation of the configuration consistency.
    """
    mode: SpatialMode
    routing: RoutingConfig
    n_hrus: int = 1
    hru_id_var: str = "gruId"

    # Domain info
    domain_method: str = "lumped"  # DOMAIN_DEFINITION_METHOD
    routing_delineation: str = "lumped"  # ROUTING_DELINEATION

    def requires_routing(self) -> bool:
        """Determine if routing is needed based on configuration."""
        if self.routing.model == RoutingModel.NONE:
            return False

        # Distributed modes always need routing
        if self.mode in (SpatialMode.DISTRIBUTED, SpatialMode.SEMI_DISTRIBUTED):
            return True

        # Lumped with river network delineation needs routing
        if self.mode == SpatialMode.LUMPED and self.routing_delineation == 'river_network':
            return True

        return False

    def validate(self) -> List[str]:
        """Validate configuration consistency, return list of warnings."""
        warnings = []

        if self.requires_routing() and self.routing.topology_file:
            if not self.routing.topology_file.exists():
                warnings.append(f"Topology file not found: {self.routing.topology_file}")

        if self.mode == SpatialMode.LUMPED and self.n_hrus > 1:
            warnings.append(f"Lumped mode but n_hrus={self.n_hrus}")

        return warnings


class SpatialOrchestrator(ABC):
    """
    Mixin class providing centralized spatial mode handling.

    This consolidates the scattered spatial mode logic from FUSE, GR, SUMMA
    into a single, tested implementation that can be reused across all models.

    Key Methods:
        get_spatial_config: Build SpatialConfig from model configuration
        requires_routing: Check if routing is needed
        convert_to_routing_format: Transform output for routing input
        route_model_output: Execute routing and return results
    """

    # These should be provided by BaseModelRunner
    logger: Any
    project_dir: Path
    config_dict: Dict[str, Any]
    config: Any  # Typed config
    domain_name: str

    # =========================================================================
    # Configuration
    # =========================================================================

    def get_spatial_config(
        self,
        model_name: str,
        spatial_mode_key: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> SpatialConfig:
        """
        Build SpatialConfig from model configuration.

        Reads relevant configuration keys and builds a validated
        SpatialConfig object.

        Args:
            model_name: Name of the model (FUSE, GR, SUMMA, etc.)
            spatial_mode_key: Config key for spatial mode (auto-detected if None)
            routing_key: Config key for routing integration (auto-detected if None)

        Returns:
            SpatialConfig with validated settings
        """
        # Auto-detect spatial mode key
        if spatial_mode_key is None:
            spatial_mode_key = f"{model_name.upper()}_SPATIAL_MODE"

        # Get spatial mode with fallback to domain method
        mode_str = self._get_config_value(lambda: None, default=None, dict_key=spatial_mode_key)

        # Handle 'auto' mode - resolve from DOMAIN_DEFINITION_METHOD
        if mode_str in (None, 'auto', 'default'):
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method, default='lumped')
            # Map domain definition method to spatial mode
            if domain_method == 'delineate':
                mode_str = 'distributed'
            elif domain_method in ('lumped', 'point'):
                mode_str = domain_method
            else:
                mode_str = 'lumped'

        mode = SpatialMode.from_string(mode_str)

        # Auto-detect routing key
        if routing_key is None:
            routing_key = f"{model_name.upper()}_ROUTING_INTEGRATION"

        # Get routing configuration
        routing_str = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key=routing_key
        )
        routing_model = RoutingModel.from_string(routing_str)

        # Build routing config - handle 'default' config value
        routing_var_config = self._get_config_value(
            lambda: None, default='averageRoutedRunoff', dict_key='SETTINGS_MIZU_ROUTING_VAR')
        if routing_var_config in ('default', None, ''):
            routing_var = 'averageRoutedRunoff'  # Default for SUMMA
        else:
            routing_var = routing_var_config
        routing_config = RoutingConfig(
            model=routing_model,
            routing_var=routing_var,
            routing_units=self._get_config_value(
                lambda: None, default='m/s', dict_key='SETTINGS_MIZU_ROUTING_UNITS'),
        )

        # Add topology file if routing is configured
        if routing_model != RoutingModel.NONE:
            topology_file = self._get_config_value(
                lambda: None, default='topology.nc', dict_key='SETTINGS_MIZU_TOPOLOGY')

            # Determine settings dir based on model
            if routing_model == RoutingModel.MIZUROUTE:
                settings_subdir = 'mizuRoute'
            elif routing_model == RoutingModel.DROUTE:
                settings_subdir = 'dRoute'
            elif routing_model == RoutingModel.TROUTE:
                settings_subdir = 'troute'
            else:
                settings_subdir = routing_model.value

            routing_config.topology_file = (
                self.project_dir / 'settings' / settings_subdir / topology_file
            )

        return SpatialConfig(
            mode=mode,
            routing=routing_config,
            domain_method=self._get_config_value(
                lambda: self.config.domain.definition_method, default='lumped'),
            routing_delineation=self._get_config_value(
                lambda: None, default='lumped', dict_key='ROUTING_DELINEATION'),
        )

    def requires_routing(self, spatial_config: Optional[SpatialConfig] = None) -> bool:
        """
        Determine if routing is required for the current model run.

        This consolidates the routing checks scattered across FUSE, GR, etc.

        Args:
            spatial_config: Pre-built config (will be created if None)

        Returns:
            True if routing should be executed
        """
        if spatial_config is None:
            model_name = getattr(self, 'model_name', 'UNKNOWN')
            spatial_config = self.get_spatial_config(model_name)

        return spatial_config.requires_routing()

    # =========================================================================
    # Output Conversion
    # =========================================================================

    def convert_to_routing_format(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        routing_config: Optional[RoutingConfig] = None,
        source_var: Optional[str] = None,
    ) -> Path:
        """
        Convert model output to mizuRoute-compatible format.

        This consolidates the conversion logic from:
        - FUSE: _convert_fuse_distributed_to_mizuroute_format
        - SUMMA: _convert_lumped_to_distributed_routing
        - GR: Similar conversion patterns

        The conversion:
        1. Identifies the spatial dimension (latitude/longitude/gru)
        2. Squeezes singleton dimensions
        3. Renames to (time, gru) format
        4. Adds gruId variable
        5. Ensures correct variable naming

        Args:
            input_file: Path to model output file
            output_file: Where to write converted output (in-place if None)
            routing_config: Routing configuration
            source_var: Source variable name for runoff (auto-detected if None)

        Returns:
            Path to converted file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # If input_file is a directory, find the actual output file
        if input_file.is_dir():
            self.logger.debug(f"Input is a directory, searching for output file in: {input_file}")
            # Look for timestep or day output files (prefer timestep for higher temporal resolution)
            candidates = list(input_file.glob("*_timestep.nc"))
            if not candidates:
                candidates = list(input_file.glob("*_day.nc"))
            if not candidates:
                candidates = list(input_file.glob("*.nc"))

            if not candidates:
                raise FileNotFoundError(f"No NetCDF output files found in directory: {input_file}")

            # Use most recently modified file
            input_file = max(candidates, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"Using output file: {input_file}")

        if routing_config is None:
            routing_config = RoutingConfig()

        target_var = routing_config.routing_var

        self.logger.debug(f"Converting {input_file} to routing format")

        with xr.open_dataset(input_file) as ds:
            converted = self._create_routing_dataset(
                ds,
                target_var=target_var,
                source_var=source_var,
                units=routing_config.routing_units
            )

        # Determine output path
        if output_file is None:
            # Write to temp file then move (atomic update)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix='.nc', dir=input_file.parent
            ) as tmp:
                temp_path = Path(tmp.name)
            converted.to_netcdf(temp_path, format='NETCDF4')
            converted.close()
            shutil.move(temp_path, input_file)
            output_file = input_file
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            converted.to_netcdf(output_file, format='NETCDF4')
            converted.close()

        self.logger.info(f"Converted to routing format: {output_file}")
        return output_file

    def _create_routing_dataset(
        self,
        source_ds: xr.Dataset,
        target_var: str,
        source_var: Optional[str] = None,
        units: str = "m/s"
    ) -> xr.Dataset:
        """
        Create mizuRoute-compatible dataset from source model output.

        Args:
            source_ds: Source xarray Dataset
            target_var: Target variable name for routing
            source_var: Source variable name (auto-detected if None)
            units: Units for the output variable

        Returns:
            xr.Dataset with (time, gru) dimensions and gruId variable
        """
        # Find runoff variable if not specified
        if source_var is None:
            candidates = [
                target_var, 'q_routed', 'q_instnt', 'averageRoutedRunoff',
                'basin__TotalRunoff', 'qsim', 'runoff'
            ]
            # Also check for any variable starting with q_ or containing runoff
            candidates.extend([str(v) for v in source_ds.data_vars if str(v).lower().startswith('q_')])
            candidates.extend([str(v) for v in source_ds.data_vars if 'runoff' in str(v).lower()])

            source_var = next((v for v in candidates if v in source_ds.data_vars), None)
            if source_var is None:
                raise ValueError(
                    f"No suitable runoff variable found. Available: {list(source_ds.data_vars)}"
                )

        self.logger.debug(f"Using source variable: {source_var}")

        # Determine spatial dimension structure
        lat_len = source_ds.sizes.get('latitude', 0)
        lon_len = source_ds.sizes.get('longitude', 0)
        gru_len = source_ds.sizes.get('gru', 0)

        # Get the data and identify spatial axis
        data = source_ds[source_var]

        if gru_len > 0:
            # Already has gru dimension
            spatial_name = 'gru'
            ids = source_ds['gruId'].values if 'gruId' in source_ds else np.arange(1, gru_len + 1)
            data = data.transpose('time', 'gru')

        elif lat_len > 1 and lon_len in (0, 1):
            # Latitude is the spatial dimension
            if 'longitude' in data.dims:
                data = data.squeeze('longitude', drop=True)
            data = data.transpose('time', 'latitude')
            spatial_name = 'latitude'
            ids = source_ds['latitude'].values

        elif lon_len > 1 and lat_len in (0, 1):
            # Longitude is the spatial dimension
            if 'latitude' in data.dims:
                data = data.squeeze('latitude', drop=True)
            data = data.transpose('time', 'longitude')
            spatial_name = 'longitude'
            ids = source_ds['longitude'].values

        elif lat_len == 1 and lon_len == 1:
            # Lumped case - squeeze both
            data = data.squeeze(['latitude', 'longitude'], drop=True)
            # Add a synthetic gru dimension
            data = data.expand_dims('gru')
            data = data.transpose('time', 'gru')
            spatial_name = 'gru'
            ids = np.array([1])

        else:
            raise ValueError(
                f"Cannot determine spatial structure from dims: {dict(source_ds.dims)}"
            )

        # Rename spatial dimension to 'gru' if needed
        if spatial_name != 'gru':
            data = data.rename({spatial_name: 'gru'})

        # Build output dataset
        mizu_ds = xr.Dataset()

        # Copy time coordinate
        mizu_ds['time'] = source_ds['time'].copy()
        if 'units' in mizu_ds['time'].attrs:
            # Normalize time units format (mizuRoute expects specific format)
            mizu_ds['time'].attrs['units'] = mizu_ds['time'].attrs['units'].replace('T', ' ')

        # Add gru dimension and gruId variable
        n_gru = data.sizes['gru']
        mizu_ds['gru'] = xr.DataArray(range(n_gru), dims=('gru',))

        # Convert IDs to int if possible
        try:
            gru_ids = ids.astype('int32')
        except (ValueError, TypeError):
            gru_ids = np.arange(1, n_gru + 1, dtype='int32')

        mizu_ds['gruId'] = xr.DataArray(
            gru_ids,
            dims=('gru',),
            attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
        )

        # Add the runoff variable with correct name
        mizu_ds[target_var] = data.rename(target_var) if data.name != target_var else data
        mizu_ds[target_var].attrs.update({
            'long_name': 'Runoff for routing',
            'units': units
        })

        # Preserve useful global attributes
        for key in ['model', 'domain', 'experiment_id', 'creation_date']:
            if key in source_ds.attrs:
                mizu_ds.attrs[key] = source_ds.attrs[key]

        return mizu_ds

    # =========================================================================
    # Routing Execution
    # =========================================================================

    def route_model_output(
        self,
        model_output: Path,
        spatial_config: Optional[SpatialConfig] = None,
        routing_control_file: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Execute routing on model output.

        Orchestrates the complete routing workflow:
        1. Convert model output to routing format
        2. Create/update routing control file
        3. Run routing model
        4. Return path to routed output

        Args:
            model_output: Path to model output file
            spatial_config: Spatial configuration (auto-built if None)
            routing_control_file: Custom control file path

        Returns:
            Path to routed output, or None if routing fails
        """
        if spatial_config is None:
            model_name = getattr(self, 'model_name', 'UNKNOWN')
            spatial_config = self.get_spatial_config(model_name)

        if not spatial_config.requires_routing():
            self.logger.debug("Routing not required")
            return model_output

        routing = spatial_config.routing

        self.logger.info(f"Routing output via {routing.model.value}")

        try:
            # Convert output to routing format
            self.convert_to_routing_format(
                model_output,
                routing_config=routing
            )

            # Execute routing
            if routing.model == RoutingModel.MIZUROUTE:
                return self._run_mizuroute(spatial_config, routing_control_file)
            elif routing.model == RoutingModel.TROUTE:
                return self._run_troute(spatial_config, routing_control_file)
            elif routing.model == RoutingModel.DROUTE:
                return self._run_droute(spatial_config, routing_control_file)
            else:
                self.logger.warning(f"Unknown routing model: {routing.model}")
                return model_output

        except FileNotFoundError as e:
            self.logger.error(f"Routing failed - required file not found: {e}")
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Routing subprocess failed with exit code {e.returncode}: {e}")
            return None
        except (OSError, IOError) as e:
            self.logger.error(f"Routing failed - I/O error: {e}")
            return None
        except ValueError as e:
            self.logger.error(f"Routing failed - invalid data or configuration: {e}")
            return None
        except (ModelExecutionError, GeospatialError) as e:
            self.logger.error(f"Routing failed: {e}")
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Routing failed with unexpected error ({type(e).__name__}): {e}")
            return None

    def _run_mizuroute(
        self,
        spatial_config: SpatialConfig,
        control_file: Optional[Path] = None,
        model_name: Optional[str] = None,
        create_control_file: bool = True
    ) -> Optional[Path]:
        """
        Run mizuRoute for the model output.

        This method imports and uses the existing mizuRoute runner.
        Optionally creates model-specific control files via the preprocessor.

        Args:
            spatial_config: Spatial configuration
            control_file: Custom control file path
            model_name: Model name (e.g., 'fuse', 'gr', 'summa') for control file creation
            create_control_file: Whether to create a model-specific control file

        Returns:
            Path to routed output, or None if routing fails
        """
        try:
            from symfluence.models.mizuroute import MizuRouteRunner

            # Create model-specific control file if requested
            if create_control_file and model_name:
                self._create_mizuroute_control_file(model_name)

            runner = MizuRouteRunner(
                self.config,
                self.logger
            )
            result = runner.run_mizuroute()

            if result:
                self.logger.info("mizuRoute completed successfully")
                return result
            else:
                self.logger.error("mizuRoute failed")
                return None

        except ImportError:
            self.logger.error("mizuRoute runner not available")
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"mizuRoute execution failed: {e}")
            return None

    def _create_mizuroute_control_file(self, model_name: str) -> None:
        """
        Create model-specific mizuRoute control file.

        Uses naming convention: create_{model}_control_file() in MizuRoutePreProcessor.

        Args:
            model_name: Model name (e.g., 'fuse', 'gr', 'summa')
        """
        try:
            from symfluence.models.mizuroute import MizuRoutePreProcessor

            preprocessor = MizuRoutePreProcessor(
                self.config,
                self.logger
            )

            # Use naming convention to find the appropriate method
            method_name = f"create_{model_name.lower()}_control_file"

            if hasattr(preprocessor, method_name):
                getattr(preprocessor, method_name)()
                self.logger.debug(f"Created {model_name} control file for mizuRoute")
            else:
                # Fall back to default control file creation if available
                if hasattr(preprocessor, 'create_control_file'):
                    preprocessor.create_control_file()
                    self.logger.debug("Created default control file for mizuRoute")
                else:
                    self.logger.warning(
                        f"No control file method found for {model_name}. "
                        f"Available: {[m for m in dir(preprocessor) if m.startswith('create_') and m.endswith('_control_file')]}"
                    )

        except ImportError:
            self.logger.warning("MizuRoutePreProcessor not available - skipping control file creation")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error creating mizuRoute control file: {e}")

    def _run_troute(
        self,
        spatial_config: SpatialConfig,
        control_file: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Run t-route for the model output.
        """
        try:
            from symfluence.models.troute import TrouteRunner

            runner = TrouteRunner(
                self.config,
                self.logger
            )
            result = runner.run_troute()

            if result:
                self.logger.info("t-route completed successfully")
                return result
            else:
                self.logger.error("t-route failed")
                return None

        except ImportError:
            self.logger.error("t-route runner not available")
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"t-route execution failed: {e}")
            return None

    def _run_droute(
        self,
        spatial_config: SpatialConfig,
        control_file: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Run dRoute for the model output.
        """
        try:
            from droute.runner import DRouteRunner

            runner = DRouteRunner(
                self.config,
                self.logger
            )
            result = runner.run_droute()

            if result:
                self.logger.info("dRoute completed successfully")
                return result
            else:
                self.logger.error("dRoute failed")
                return None

        except ImportError:
            self.logger.error("dRoute runner not available")
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"dRoute execution failed: {e}")
            return None

    # =========================================================================
    # Helpers
    # =========================================================================

    def get_hru_count(self, shapefile_path: Optional[Path] = None) -> int:
        """
        Get the number of HRUs from catchment shapefile.

        Args:
            shapefile_path: Path to shapefile (auto-detected if None)

        Returns:
            Number of unique HRUs
        """
        import geopandas as gpd

        if shapefile_path is None:
            catchment_name = self._get_config_value(
                lambda: None, default='default', dict_key='CATCHMENT_SHP_NAME')
            if catchment_name == 'default':
                discretization = self._get_config_value(
                    lambda: self.config.domain.discretization, default='catchment')
                catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"
            shapefile_path = self.project_dir / 'shapefiles' / 'catchment' / catchment_name

        if not shapefile_path.exists():
            self.logger.warning(f"Shapefile not found: {shapefile_path}")
            return 1

        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.empty:
                self.logger.warning(f"Shapefile is empty: {shapefile_path}")
                return 1
            gru_col = self._get_config_value(
                lambda: None, default='GRU_ID', dict_key='CATCHMENT_SHP_GRUID')
            if gru_col in gdf.columns:
                return len(gdf[gru_col].unique())
            else:
                return len(gdf)
        except FileNotFoundError as e:
            self.logger.error(f"Shapefile not found: {e}")
            return 1
        except (OSError, IOError) as e:
            self.logger.error(f"Could not read shapefile (I/O error): {e}")
            return 1
        except ValueError as e:
            self.logger.error(f"Invalid shapefile format: {e}")
            return 1
        except KeyError as e:
            self.logger.error(f"Missing expected column in shapefile: {e}")
            return 1
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Unexpected error reading shapefile ({type(e).__name__}): {e}")
            return 1

    def normalize_spatial_output(
        self,
        input_file: Path,
        from_mode: SpatialMode,
        to_mode: SpatialMode
    ) -> Path:
        """
        Normalize model output between spatial modes.

        Handles conversions like:
        - Lumped to distributed (replication)
        - Distributed to lumped (aggregation)

        Args:
            input_file: Path to input file
            from_mode: Source spatial mode
            to_mode: Target spatial mode

        Returns:
            Path to normalized output
        """
        if from_mode == to_mode:
            return input_file

        self.logger.info(f"Normalizing output: {from_mode.value} -> {to_mode.value}")

        with xr.open_dataset(input_file) as ds:
            if from_mode == SpatialMode.LUMPED and to_mode != SpatialMode.LUMPED:
                # Replicate lumped output to HRUs
                n_hrus = self.get_hru_count()
                normalized = self._replicate_to_distributed(ds, n_hrus)
            elif from_mode != SpatialMode.LUMPED and to_mode == SpatialMode.LUMPED:
                # Aggregate distributed to lumped
                normalized = self._aggregate_to_lumped(ds)
            else:
                # Semi-distributed <-> distributed: no change needed
                return input_file

            # Save normalized output
            output_file = input_file.with_stem(f"{input_file.stem}_normalized")
            normalized.to_netcdf(output_file)
            normalized.close()

        return output_file

    def _replicate_to_distributed(self, ds: xr.Dataset, n_hrus: int) -> xr.Dataset:
        """Replicate lumped dataset to n_hrus."""
        # Implementation for lumped -> distributed replication
        result = xr.Dataset()
        result['time'] = ds['time'].copy()
        result['gru'] = xr.DataArray(range(n_hrus), dims=('gru',))
        result['gruId'] = xr.DataArray(np.arange(1, n_hrus + 1), dims=('gru',))

        for var in ds.data_vars:
            # Broadcast to (time, gru)
            data = ds[var].values
            if data.ndim == 1:  # (time,)
                data = np.tile(data[:, np.newaxis], (1, n_hrus))
            result[var] = xr.DataArray(data, dims=('time', 'gru'), attrs=ds[var].attrs)

        return result

    def _aggregate_to_lumped(self, ds: xr.Dataset) -> xr.Dataset:
        """Aggregate distributed dataset to lumped (mean over spatial dim)."""
        result = xr.Dataset()
        result['time'] = ds['time'].copy()

        for var in ds.data_vars:
            if 'gru' in ds[var].dims:
                result[var] = ds[var].mean(dim='gru')
            else:
                result[var] = ds[var].copy()

        return result
