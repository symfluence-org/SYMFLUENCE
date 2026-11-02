# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Model Optimizer

FUSE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with FUSE.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.file_utils import copy_file
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.path_resolver import resolve_install_dir
from symfluence.core.registries import R

from .parameter_manager import FUSEParameterManager  # noqa: F401 - trigger param manager registration
from .worker import FUSEWorker  # noqa: F401 - trigger worker registration


@R.optimizers.add('FUSE')
class FUSEModelOptimizer(BaseModelOptimizer):
    """
    FUSE-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization
    - run_lbfgs(): L-BFGS gradient-based optimization

    Example:
        optimizer = FUSEModelOptimizer(config, logger)
        results_path = optimizer.run_pso()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize FUSE optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        # Initialize FUSE-specific paths before super().__init__
        # because parent calls _setup_parallel_dirs()
        # Note: super().__init__ hasn't run yet, so _get_config_value() isn't
        # available; resolve_config_value works for both typed and dict configs.
        exp_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')
        data_dir_str = config.get('SYMFLUENCE_DATA_DIR') if isinstance(config, dict) else config.system.data_dir
        domain_name_str = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        self.data_dir = Path(data_dir_str)
        self.domain_name = domain_name_str
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.fuse_sim_dir = self.get_simulation_dir('FUSE', exp_id)
        self.fuse_setup_dir = self.project_dir / 'settings' / 'FUSE'
        self.fuse_exe_path = self._get_fuse_executable_path_pre_init(config)
        # Use 'or' to treat None as "not set" and fallback to exp_id; hash long IDs.
        from symfluence.models.fuse.calibration.file_manager import resolve_fuse_id
        fuse_file_id = resolve_config_value(config, lambda: config.model.fuse.file_id, None, 'FUSE_FILE_ID')
        self.fuse_id = resolve_fuse_id({'FUSE_FILE_ID': fuse_file_id, 'EXPERIMENT_ID': exp_id})

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        # Validate that calibrated params match active decisions
        self._validate_decision_param_consistency()

        self.logger.debug("FUSEModelOptimizer initialized")

    def _get_fuse_executable_path_pre_init(self, config) -> Path:
        """Helper to get FUSE executable path before full initialization."""
        fuse_install = resolve_config_value(config, lambda: config.model.fuse.install_path, 'default', 'FUSE_INSTALL_PATH')
        return resolve_install_dir(fuse_install, self.data_dir, 'installs/fuse/bin') / 'fuse.exe'

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'FUSE'

    def _create_parameter_manager(self):
        """Create FUSE parameter manager."""
        return FUSEParameterManager(
            self.config,
            self.logger,
            self.fuse_setup_dir
        )

    def _validate_decision_param_consistency(self) -> None:
        """Validate that calibrated parameters cover active FUSE decisions."""
        try:
            fuse_id = self._get_config_value(
                lambda: self.config.model.fuse.file_id,
                dict_key='FUSE_FILE_ID'
            ) or self.experiment_id
            decisions_path = self.fuse_setup_dir / f"fuse_zDecisions_{fuse_id}.txt"
            if hasattr(self, 'param_manager') and self.param_manager is not None:
                warnings = self.param_manager.validate_params_for_decisions(decisions_path)
                if warnings:
                    self.logger.warning(
                        f"FUSE decision-parameter validation found {len(warnings)} issue(s). "
                        f"Calibration will proceed, but results may be suboptimal."
                    )
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.debug(f"Decision-param validation skipped: {e}", exc_info=True)

    def _create_para_def_nc(self, param_file: Path) -> bool:
        """
        Create a fresh para_def.nc file with default parameter values.

        FUSE's run_def mode reads parameters from para_def.nc, so this file
        must exist for calibration to work. This method creates a properly
        structured NetCDF file with all calibration parameters set to their
        default (middle of bounds) values.

        Note: Elevation band parameters (N_BANDS, Z_FORCING, etc.) are NOT
        included here - FUSE reads them directly from the elevation bands file.

        Args:
            param_file: Path where para_def.nc should be created

        Returns:
            True if file was created successfully
        """
        try:
            import netCDF4 as nc

            # Ensure parent directory exists
            param_file.parent.mkdir(parents=True, exist_ok=True)

            # Get parameters to calibrate from config
            fuse_params_str = self._get_config_value(
                lambda: self.config.model.fuse.params_to_calibrate,
                default='',
                dict_key='SETTINGS_FUSE_PARAMS_TO_CALIBRATE'
            )
            if not fuse_params_str or fuse_params_str == 'default':
                fuse_params_str = 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE'
            fuse_params = [p.strip() for p in fuse_params_str.split(',') if p.strip()]

            # Get parameter bounds from config or use defaults
            from symfluence.core.calibration.parameters.parameter_bounds_registry import get_fuse_bounds
            bounds = get_fuse_bounds()

            # Override with config bounds if specified (preserve transform metadata)
            config_bounds = self._get_config_value(
                lambda: self.config.model.fuse.param_bounds,
                default={},
                dict_key='FUSE_PARAM_BOUNDS'
            ) or {}
            for param, bound_list in config_bounds.items():
                if isinstance(bound_list, (list, tuple)) and len(bound_list) == 2:
                    existing = bounds.get(param, {})
                    new_entry = {'min': float(bound_list[0]), 'max': float(bound_list[1])}
                    if 'transform' in existing:
                        new_entry['transform'] = existing['transform']
                    bounds[param] = new_entry

            # CRITICAL: Include elevation band parameters (N_BANDS, Z_MID, AF) in para_def.nc
            # FUSE's run_def mode reads ALL parameters from para_def.nc, including elevation bands.
            # If these are missing, FUSE defaults them to 0, which causes zero runoff output!
            elev_params = self._get_elevation_band_params()
            self.logger.debug(f"Adding elevation band params to para_def.nc: {elev_params}")

            # Create NetCDF file
            with nc.Dataset(param_file, 'w', format='NETCDF4') as ds:
                # Create dimensions - CRITICAL: Use size 1 for single parameter set
                ds.createDimension('par', 1)

                # Create coordinate variable
                par_var = ds.createVariable('par', 'i4', ('par',))
                par_var[:] = [0]  # 0-based indexing

                # Create parameter variables with default values (middle of bounds)
                for param_name in fuse_params:
                    param_var = ds.createVariable(param_name, 'f8', ('par',))
                    if param_name in bounds:
                        default_val = (bounds[param_name]['min'] + bounds[param_name]['max']) / 2.0
                    else:
                        default_val = 1.0  # Fallback
                    param_var[:] = [default_val]
                    self.logger.debug(f"  Created param {param_name} = {default_val:.4f}")

                # Add elevation band parameters (CRITICAL for non-zero runoff!)
                for param_name, value in elev_params.items():
                    if param_name not in ds.variables:
                        param_var = ds.createVariable(param_name, 'f8', ('par',))
                        param_var[:] = [value]
                        self.logger.debug(f"  Created elev param {param_name} = {value:.4f}")

                # Add global attributes
                ds.setncattr('title', 'FUSE parameter file for calibration')
                ds.setncattr('created_by', 'SYMFLUENCE FUSEModelOptimizer')

                ds.sync()

            self.logger.info(f"Created para_def.nc with {len(fuse_params)} calibration parameters and {len(elev_params)} elevation band parameters")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to create para_def.nc: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _get_elevation_band_params(self) -> Dict[str, float]:
        """
        Read elevation band parameters from the elevation bands file.

        These parameters are required for FUSE's run_def mode to properly
        aggregate band-level snow results to catchment totals.

        Returns:
            Dictionary with N_BANDS, Z_FORCING, Z_MIDxx, AFxx parameters
        """
        import xarray as xr

        params: Dict[str, float] = {}

        try:
            # Find the elevation bands file
            forcing_dir = self.project_forcing_dir / 'FUSE_input'
            elev_bands_file = forcing_dir / f"{self.domain_name}_elev_bands.nc"

            if not elev_bands_file.exists():
                self.logger.warning(f"Elevation bands file not found: {elev_bands_file}")
                # Use defaults for single band
                params['N_BANDS'] = 1.0
                params['Z_FORCING'] = 1000.0  # Default forcing elevation
                params['Z_MID01'] = 1000.0
                params['AF01'] = 1.0
                return params

            with xr.open_dataset(elev_bands_file) as ds:
                # Get number of bands
                if 'elevation_band' in ds.dims:
                    n_bands = ds.sizes['elevation_band']
                else:
                    n_bands = 1
                params['N_BANDS'] = float(n_bands)

                # Get mean elevations and area fractions for each band
                if 'mean_elev' in ds:
                    mean_elevs = ds['mean_elev'].values.flatten()
                    for i, elev in enumerate(mean_elevs[:n_bands]):
                        params[f'Z_MID{i+1:02d}'] = float(elev)

                if 'area_frac' in ds:
                    area_fracs = ds['area_frac'].values.flatten()
                    for i, af in enumerate(area_fracs[:n_bands]):
                        params[f'AF{i+1:02d}'] = float(af)

                # Z_FORCING: elevation of forcing data
                # This can be set from config or default to catchment elevation
                # (same as Z_MID01 = no lapse correction) or use ERA5 orography
                z_forcing = self._get_config_value(
                    lambda: self.config.model.fuse.forcing_elevation if self.config.model and self.config.model.fuse else None,
                    default=None,
                    dict_key='FUSE_FORCING_ELEVATION'
                )
                if z_forcing is not None:
                    params['Z_FORCING'] = float(z_forcing)
                elif 'Z_MID01' in params:
                    # Default: same as catchment elevation (no lapse correction)
                    # User should set FUSE_FORCING_ELEVATION for proper lapse rate
                    params['Z_FORCING'] = params['Z_MID01']
                else:
                    params['Z_FORCING'] = 1000.0

            self.logger.info(f"Read elevation band params: N_BANDS={params.get('N_BANDS')}, "
                           f"Z_FORCING={params.get('Z_FORCING'):.0f}, Z_MID01={params.get('Z_MID01', 'N/A')}")

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Error reading elevation bands: {e}. Using defaults.", exc_info=True)
            params['N_BANDS'] = 1.0
            params['Z_FORCING'] = 1000.0
            params['Z_MID01'] = 1000.0
            params['AF01'] = 1.0

        return params

    def _find_complete_fuse_template(self) -> Optional[Path]:
        """
        Find an existing complete FUSE para_def.nc template with all required variables.

        FUSE's run_pre mode requires a para_def.nc with all ~89 variables including:
        - Calibration parameters (MAXWATR_1, BASERTE, etc.)
        - Derived parameters (MAXTENS_1, MAXFREE_1, etc.)
        - Numerix settings (SOLUTION, TIMSTEP_TYP, etc.)
        - Statistics placeholders (nash_sutt, kge, etc.)

        This method searches for existing FUSE-generated templates that contain
        all required variables. If found, this template should be used instead of
        creating a minimal file via _create_para_def_nc().

        Search locations (in priority order):
        0. User-specified FUSE_TEMPLATE_PATH (highest priority)
        1. Current simulation directory (from previous run)
        2. Other simulation directories in the project
        3. Parallel process directories (from calibration runs)

        Returns:
            Path to complete template if found, None otherwise
        """
        import netCDF4 as nc

        # Minimum number of variables expected in a complete FUSE template
        # FUSE generates ~89 variables; we require at least 70 to be safe
        MIN_REQUIRED_VARS = 70

        def is_complete_template(path: Path) -> bool:
            """Check if a para_def.nc has all required variables."""
            try:
                with nc.Dataset(path, 'r') as ds:
                    n_vars = len([v for v in ds.variables if v != 'par'])
                    if n_vars >= MIN_REQUIRED_VARS:
                        self.logger.debug(f"Found complete template with {n_vars} vars: {path}")
                        return True
                    else:
                        self.logger.debug(f"Template {path} has only {n_vars} vars (need {MIN_REQUIRED_VARS}+)")
                        return False
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.debug(f"Could not check template {path}: {e}", exc_info=True)
                return False

        # 0. Check for user-specified template path (highest priority)
        user_template = self._get_config_value(
            lambda: self.config.model.fuse.template_path,
            default=None,
            dict_key='FUSE_TEMPLATE_PATH'
        )
        if user_template:
            user_template_path = Path(user_template)
            if user_template_path.exists() and is_complete_template(user_template_path):
                self.logger.info(f"Using user-specified FUSE template: {user_template_path}")
                return user_template_path
            else:
                self.logger.warning(
                    f"User-specified FUSE_TEMPLATE_PATH '{user_template}' does not exist "
                    f"or is not a complete template. Searching for alternatives..."
                )

        # Search locations
        search_paths: List[Path] = []

        # 1. Current simulation directory
        if self.fuse_sim_dir.exists():
            search_paths.extend(self.fuse_sim_dir.glob("*_para_def.nc"))
            search_paths.extend(self.fuse_sim_dir.glob("sim_*_para_def.nc"))

        # 2. Other simulation directories
        sim_base = self.project_dir / 'simulations'
        if sim_base.exists():
            for sim_dir in sim_base.iterdir():
                if sim_dir.is_dir() and sim_dir.name != self.experiment_id:
                    fuse_dir = sim_dir / 'FUSE'
                    if fuse_dir.exists():
                        search_paths.extend(fuse_dir.glob("*_para_def.nc"))
                        search_paths.extend(fuse_dir.glob("sim_*_para_def.nc"))

        # 3. Parallel process directories (may have FUSE-generated templates)
        run_dirs = list(sim_base.glob("run_*/process_*/settings/FUSE"))
        for run_dir in run_dirs:
            if run_dir.exists():
                search_paths.extend(run_dir.glob("sim_*_para_def.nc"))

        # Check each path for completeness
        for path in search_paths:
            if path.exists() and is_complete_template(path):
                self.logger.info(f"Found complete FUSE template: {path}")
                return path

        self.logger.debug("No complete FUSE template found")
        return None

    def _add_elevation_params_to_constraints(self, constraints_file: Path) -> bool:
        """
        Add or update elevation band parameters in the FUSE constraints file as FIXED parameters.

        CRITICAL: FUSE's run_def mode needs elevation band parameters in the constraints
        file to properly initialize the aggregation from band-level results (swe_z01)
        to catchment totals (swe_tot, eff_ppt). Without these, run_def mode will compute
        band-level snow but fail to aggregate it, causing massive flow underestimation.

        This method will UPDATE existing values if parameters already exist, ensuring
        correct elevations are used even if the file was previously modified with wrong values.

        Args:
            constraints_file: Path to fuse_zConstraints_snow.txt

        Returns:
            True if successful
        """
        try:
            # Get elevation band parameters from the elevation bands file
            elev_params = self._get_elevation_band_params()

            if not constraints_file.exists():
                self.logger.warning(f"Constraints file not found: {constraints_file}")
                return False

            # Read existing constraints file with encoding fallback
            try:
                with open(constraints_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                self.logger.warning(
                    f"UTF-8 decode error reading {constraints_file}, falling back to latin-1"
                )
                with open(constraints_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Define all elevation parameter names we want to manage
            # Format: (L1,1X,I1,1X,3(F9.3,1X),F3.2,1X,F5.1,1X,3(I1,1X),I2,1X,2(I1,1X),3(A9,1X))
            # F 0   value    lower    upper  frac scale t1 t2 t3 nL im nH NAME      CHILD1    CHILD2
            elev_param_names = ['N_BANDS', 'Z_FORCING']
            for i in range(1, 11):  # Support up to 10 bands
                elev_param_names.append(f'Z_MID{i:02d}')
                elev_param_names.append(f'AF{i:02d}')

            # Create new line for each elevation parameter
            def make_elev_line(param_name: str, val: float, comment: str) -> str:
                return f"F 0 {val:9.3f} {val:9.3f} {val:9.3f} .10   1.0 0 0 0  0 0 0 {param_name:9s} NO_CHILD1 NO_CHILD2 ! {comment}\n"

            # Process lines: update existing or track for adding
            new_lines = []
            found_params = set()
            insert_pos = len(lines)

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Track where description section starts
                if stripped.startswith('*****'):
                    insert_pos = min(insert_pos, i)
                    new_lines.append(line)
                    continue

                # Skip non-parameter lines
                if stripped.startswith('(') or stripped.startswith('!') or not stripped:
                    new_lines.append(line)
                    continue

                # Try to parse parameter name from line
                parts = line.split()
                if len(parts) >= 14:
                    param_name = parts[13]

                    # Check if this is an elevation parameter we manage
                    if param_name in elev_param_names:
                        found_params.add(param_name)

                        # Update with correct value if we have it
                        if param_name in elev_params:
                            val = elev_params[param_name]
                            if param_name == 'N_BANDS':
                                new_lines.append(make_elev_line(param_name, val, "number of elevation bands"))
                            elif param_name == 'Z_FORCING':
                                new_lines.append(make_elev_line(param_name, val, "elevation of forcing data (m)"))
                            elif param_name.startswith('Z_MID'):
                                band_num = int(param_name[5:])
                                new_lines.append(make_elev_line(param_name, val, f"elevation of band {band_num} (m)"))
                            elif param_name.startswith('AF'):
                                band_num = int(param_name[2:])
                                new_lines.append(make_elev_line(param_name, val, f"area fraction of band {band_num}"))
                            else:
                                new_lines.append(line)  # Keep original if unknown
                        else:
                            new_lines.append(line)  # Keep original if no value
                        continue

                # Keep other lines unchanged
                new_lines.append(line)

            # Add any elevation parameters that weren't in the file
            elev_lines_to_add = []
            for param_name in ['N_BANDS', 'Z_FORCING']:
                if param_name not in found_params and param_name in elev_params:
                    val = elev_params[param_name]
                    if param_name == 'N_BANDS':
                        elev_lines_to_add.append(make_elev_line(param_name, val, "number of elevation bands"))
                    else:
                        elev_lines_to_add.append(make_elev_line(param_name, val, "elevation of forcing data (m)"))

            # Add Z_MID and AF parameters for each band
            n_bands = int(elev_params.get('N_BANDS', 1))
            for i in range(1, n_bands + 1):
                z_mid_name = f'Z_MID{i:02d}'
                af_name = f'AF{i:02d}'

                if z_mid_name not in found_params and z_mid_name in elev_params:
                    elev_lines_to_add.append(make_elev_line(z_mid_name, elev_params[z_mid_name], f"elevation of band {i} (m)"))

                if af_name not in found_params and af_name in elev_params:
                    elev_lines_to_add.append(make_elev_line(af_name, elev_params[af_name], f"area fraction of band {i}"))

            # Insert new elevation parameters before the description section
            if elev_lines_to_add:
                # Find insert position in new_lines (before ***** section)
                new_insert_pos = len(new_lines)
                for i, line in enumerate(new_lines):
                    if line.strip().startswith('*****'):
                        new_insert_pos = i
                        break
                new_lines = new_lines[:new_insert_pos] + elev_lines_to_add + new_lines[new_insert_pos:]

            # Write updated constraints file
            with open(constraints_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            updated_count = len(found_params)
            added_count = len(elev_lines_to_add)
            self.logger.info(f"Elevation params in constraints: updated {updated_count}, added {added_count}")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to add elevation params to constraints file: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _check_routing_needed(self) -> bool:
        """
        Determine if routing is needed for FUSE calibration.

        Returns:
            True if mizuRoute routing should be used
        """
        # Check FUSE routing integration setting
        routing_integration = self._get_config_value(lambda: self.config.model.fuse.routing_integration, default='none', dict_key='FUSE_ROUTING_INTEGRATION')

        # If 'default', inherit from ROUTING_MODEL
        if routing_integration == 'default':
            routing_model = self._get_config_value(lambda: self.config.model.routing_model, default='none', dict_key='ROUTING_MODEL')
            routing_integration = 'mizuRoute' if routing_model == 'mizuRoute' else routing_integration

        if routing_integration != 'mizuRoute':
            return False

        # Check calibration variable (only streamflow calibration uses routing)
        calibration_var = self._get_config_value(lambda: self.config.optimization.calibration_variable, default='streamflow', dict_key='CALIBRATION_VARIABLE')
        if calibration_var != 'streamflow':
            return False

        # Check spatial mode and routing delineation
        spatial_mode = self._get_config_value(lambda: self.config.model.fuse.spatial_mode, default='lumped', dict_key='FUSE_SPATIAL_MODE')
        routing_delineation = self._get_config_value(lambda: self.config.domain.delineation.routing, default='lumped', dict_key='ROUTING_DELINEATION')

        # Distributed or semi-distributed modes need routing
        if spatial_mode in ['semi_distributed', 'distributed']:
            return True

        # Lumped with river network routing needs routing
        if spatial_mode == 'lumped' and routing_delineation == 'river_network':
            return True

        return False

    def _copy_default_initial_params_to_sce(self):
        """Helper to ensure para_sce.nc exists by copying para_def.nc."""
        if self.fuse_sim_dir.exists():
            default_params = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc"
            sce_params = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_sce.nc"
            if default_params.exists():
                copy_file(default_params, sce_params)
                self.logger.info("Copied DDS best parameters to para_sce.nc")

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """
        Apply best parameters for final evaluation.

        Ensures a complete para_def.nc exists in the settings directory
        (required by run_pre mode), populates any zero-valued non-calibrated
        parameters from the constraints file defaults, then applies calibrated
        best parameters to both the constraints file and para_def.nc.
        """
        try:
            # Ensure para_def.nc exists in settings/FUSE/ for the final run.
            # During calibration, para_def.nc only exists in parallel worker dirs
            # and simulations dir, but run_pre mode executes from settings/FUSE/.
            settings_para_def = self.fuse_setup_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc"
            if not settings_para_def.exists():
                source = self._find_para_def_source()
                if source:
                    copy_file(source, settings_para_def)
                    self.logger.info(f"Copied para_def.nc to settings dir from: {source}")
                else:
                    self.logger.error("No para_def.nc source found for final evaluation")
                    return False

            # Populate zero-valued non-calibrated parameters from constraints.
            # The para_def.nc may have been overwritten with zeros for parameters
            # that FUSE doesn't calibrate (e.g. LOGLAMB, TISHAPE) but which are
            # required by certain model decisions (e.g. TOPMODEL surface runoff).
            self._populate_non_calibrated_params(settings_para_def)

            # Update constraints file (for record-keeping)
            self.param_manager.update_model_files(best_params)

            # Apply best parameters to para_def.nc via worker (updates both
            # constraints and para_def.nc in the settings directory)
            if not self.worker.apply_parameters(
                best_params, self.fuse_setup_dir, config=self.config
            ):
                self.logger.error("Failed to apply best parameters to FUSE files")
                return False

            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying FUSE parameters for final evaluation: {e}", exc_info=True)
            return False

    def _parse_constraints_defaults(self) -> Dict[str, float]:
        """
        Parse parameter default values from the FUSE constraints file.

        Returns:
            Dictionary mapping parameter names to their default values.
        """
        defaults: Dict[str, float] = {}
        constraints_file = self.fuse_setup_dir / 'fuse_zConstraints_snow.txt'
        if not constraints_file.exists():
            self.logger.warning(f"Constraints file not found: {constraints_file}")
            return defaults

        try:
            try:
                with open(constraints_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(constraints_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            for line in lines:
                stripped = line.strip()
                # Skip format line, comments, empty lines, description section
                if (not stripped or stripped.startswith('(') or
                        stripped.startswith('!') or stripped.startswith('*')):
                    continue
                parts = stripped.split()
                # Format: F/T flag default lower upper ... param_name ...
                # Indices: 0=fit_flag, 1=stoch, 2=default, ..., 13=param_name
                if len(parts) >= 14:
                    try:
                        default_val = float(parts[2])
                        param_name = parts[13]
                        defaults[param_name] = default_val
                    except (ValueError, IndexError):
                        continue
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Error parsing constraints file: {e}", exc_info=True)

        return defaults

    def _populate_non_calibrated_params(self, para_def_path: Path) -> None:
        """
        Populate zero-valued non-calibrated parameters in para_def.nc
        from the constraints file defaults.

        FUSE's run_pre mode reads ALL parameters from para_def.nc. If a
        para_def.nc was regenerated or corrupted, non-calibrated parameters
        (like LOGLAMB, TISHAPE) may be zero, causing Fortran crashes
        (e.g. gammp_s args from the incomplete gamma function).

        This method reads the constraints file for default values and fills
        in any zero-valued parameters in para_def.nc that have non-zero
        defaults in the constraints.
        """
        try:
            import netCDF4 as nc

            if not para_def_path.exists():
                return

            defaults = self._parse_constraints_defaults()
            if not defaults:
                self.logger.debug("No constraint defaults parsed; skipping non-calibrated param fill")
                return

            updated = []
            with nc.Dataset(para_def_path, 'r+') as ds:
                for var_name in ds.variables:
                    if var_name in defaults and defaults[var_name] != 0.0:
                        var = ds.variables[var_name]
                        current_val = float(var[:].flat[0])
                        if current_val == 0.0:
                            var[:] = defaults[var_name]
                            updated.append((var_name, defaults[var_name]))

            if updated:
                self.logger.info(
                    f"Populated {len(updated)} zero-valued params in para_def.nc from constraints: "
                    + ", ".join(f"{name}={val}" for name, val in updated)
                )
            else:
                self.logger.debug("All non-calibrated params in para_def.nc already have non-zero values")

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Error populating non-calibrated params: {e}", exc_info=True)

    def _find_para_def_source(self) -> Optional[Path]:
        """Find an existing complete para_def.nc to use as source for final evaluation."""
        candidates = [
            # Simulations directory (from initial FUSE run)
            self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc",
            # Parallel worker directory (from calibration)
            self.project_dir / 'simulations' / 'run_dds' / 'process_0' / 'settings' / 'FUSE' / f"{self.domain_name}_{self.fuse_id}_para_def.nc",
            self.project_dir / 'simulations' / 'run_pso' / 'process_0' / 'settings' / 'FUSE' / f"{self.domain_name}_{self.fuse_id}_para_def.nc",
            self.project_dir / 'simulations' / 'run_sce' / 'process_0' / 'settings' / 'FUSE' / f"{self.domain_name}_{self.fuse_id}_para_def.nc",
        ]
        # Also try the general template search
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Fall back to the comprehensive template search
        return self._find_complete_fuse_template()

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run FUSE for final evaluation.

        Uses run_pre mode which reads all parameters from para_def.nc.
        The para_def.nc must have been prepared with correct non-calibrated
        parameters (including derived params like POWLAMB, MAXPOW) and
        numerics settings before calling this method.

        Note: run_def mode would be preferable (recomputes derived params
        from constraints) but is broken in many FUSE builds due to
        NC_UNLIMITED conflicts in NETCDF3_CLASSIC format.
        """
        self._copy_default_initial_params_to_sce()
        return self.worker.run_model(
            self.config,
            self.fuse_setup_dir,
            output_dir,
            mode='run_pre'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to FUSE file manager."""
        fuse_fm = self._get_config_value(lambda: self.config.model.fuse.filemanager, default='fm_catch.txt', dict_key='SETTINGS_FUSE_FILEMANAGER')
        if fuse_fm == 'default':
            fuse_fm = 'fm_catch.txt'
        return self.fuse_setup_dir / fuse_fm

    def _transfer_sce_to_para_def(self) -> None:
        """
        Transfer SCE-optimized non-calibrated parameters from para_sce.nc to para_def.nc.

        When FUSE runs calib_sce (internal SCE calibration), it optimizes ALL parameters
        and writes results to para_sce.nc. However, external DDS calibration only updates
        the user-specified calibration parameters. Non-calibrated parameters (LOGLAMB,
        TISHAPE, IFLWRTE, etc.) remain at their initial defaults in para_def.nc.

        This method reads para_sce.nc and copies non-calibrated parameter values to
        para_def.nc, so DDS starts with SCE-seeded values for all parameters.
        """
        try:
            import netCDF4 as nc

            fuse_id = self._get_config_value(
                lambda: self.config.model.fuse.file_id,
                dict_key='FUSE_FILE_ID'
            ) or self.experiment_id

            para_sce_path = self.fuse_sim_dir / f"{self.domain_name}_{fuse_id}_para_sce.nc"
            para_def_path = self.fuse_sim_dir / f"{self.domain_name}_{fuse_id}_para_def.nc"

            if not para_sce_path.exists() or not para_def_path.exists():
                self.logger.debug(
                    "SCE→DDS transfer skipped: para_sce.nc or para_def.nc not found"
                )
                return

            # Get the set of calibrated parameter names
            calibrated_params = set()
            if hasattr(self, 'param_manager') and self.param_manager is not None:
                calibrated_params = set(self.param_manager.all_param_names)

            transferred = []
            with nc.Dataset(para_sce_path, 'r') as ds_sce:
                with nc.Dataset(para_def_path, 'r+') as ds_def:
                    if 'par' not in ds_sce.dimensions or ds_sce.dimensions['par'].size == 0:
                        return
                    if 'par' not in ds_def.dimensions or ds_def.dimensions['par'].size == 0:
                        return

                    for var_name in ds_sce.variables:
                        if var_name == 'par':
                            continue
                        # Only transfer non-calibrated params (DDS handles calibrated ones)
                        if var_name in calibrated_params:
                            continue
                        if var_name not in ds_def.variables:
                            continue

                        sce_val = float(ds_sce.variables[var_name][0])
                        def_val = float(ds_def.variables[var_name][0])

                        # Only transfer if SCE has a meaningful value and def is different
                        if sce_val != 0.0 and abs(sce_val - def_val) > 1e-6:
                            ds_def.variables[var_name][0] = sce_val
                            transferred.append((var_name, def_val, sce_val))

                    ds_def.sync()

            if transferred:
                self.logger.info(
                    f"SCE→DDS transfer: copied {len(transferred)} non-calibrated params "
                    f"from para_sce.nc to para_def.nc"
                )
                for name, old_val, new_val in transferred[:10]:
                    self.logger.debug(f"  {name}: {old_val:.4f} → {new_val:.4f}")
            else:
                self.logger.debug("SCE→DDS transfer: no parameters needed updating")

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"SCE→DDS parameter transfer failed (non-fatal): {e}", exc_info=True)

    def _setup_parallel_dirs(self) -> None:
        """Setup FUSE-specific parallel directories."""
        # Transfer SCE-optimized non-calibrated params to para_def.nc before copying
        self._transfer_sce_to_para_def()

        algorithm = self._get_config_value(lambda: self.config.optimization.algorithm, default='optimization', dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM').lower()
        base_dir = self._resolve_sim_base_dir(algorithm)
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'FUSE',
            self.experiment_id
        )

        # Copy FUSE settings to each parallel directory
        if self.fuse_setup_dir.exists():
            self.copy_base_settings(self.fuse_setup_dir, self.parallel_dirs, 'FUSE')

        # Copy parameter file to each parallel directory
        # This is critical for parallel workers to modify parameters in isolation
        param_file = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc"

        # If para_def.nc doesn't exist, try to find a complete FUSE template
        # CRITICAL: For run_pre mode, FUSE requires all ~89 variables including:
        # - Derived parameters (MAXTENS_*, MAXFREE_*, etc.)
        # - Numerix settings (SOLUTION, TIMSTEP_TYP, etc.)
        # - Statistics placeholders (nash_sutt, kge, etc.)
        # A minimal file created by _create_para_def_nc() will NOT work for run_pre!
        if not param_file.exists():
            # First, try to find an existing complete FUSE template
            template_path = self._find_complete_fuse_template()

            if template_path is not None:
                # Copy the complete template to the expected location
                self.logger.info(f"Using complete FUSE template from: {template_path}")
                param_file.parent.mkdir(parents=True, exist_ok=True)
                copy_file(template_path, param_file)
            else:
                # No complete template found - create minimal file
                # This will work for run_def mode but NOT for run_pre mode
                self.logger.warning(
                    "No complete FUSE template found. Creating minimal para_def.nc. "
                    "This will work for run_def mode, but run_pre mode requires a complete "
                    "template with all ~89 variables. To generate a complete template, first "
                    "run FUSE in run_def or calib_sce mode, or provide FUSE_TEMPLATE_PATH in config."
                )
                self._create_para_def_nc(param_file)

                # run_pre (the calibration mode) needs a complete template
                # with all ~89 FUSE variables. A minimal file may not work.
                self.logger.error(
                    "FUSE calibration (run_pre) requires a complete para_def.nc "
                    "template with all ~89 variables, but none was found. "
                    "Calibration may fail. To fix:\n"
                    "  1. Run the FUSE model step once — its initial default run "
                    "(run_def) generates a complete para_def.nc\n"
                    "  2. Or set FUSE_TEMPLATE_PATH to an existing complete para_def.nc"
                )

        if param_file.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                # Copy directly to settings_dir (which already includes model name, e.g., .../settings/FUSE)
                dest_file = dirs['settings_dir'] / param_file.name
                try:
                    copy_file(param_file, dest_file)
                    self.logger.debug(f"Copied parameter file to {dest_file}")
                except Exception as e:  # noqa: BLE001 — calibration resilience
                    self.logger.error(f"Failed to copy parameter file to {dest_file}: {e}", exc_info=True)
        else:
            self.logger.error(f"Failed to create parameter file: {param_file} - FUSE calibration will fail")

        # NOTE: Do NOT add elevation band parameters to constraints file!
        # FUSE reads N_BANDS, Z_MID, AF etc. from the elevation bands file (_elev_bands.nc)
        # directly, not from the constraints. Adding them to constraints causes FUSE to
        # error with "parameter name (N_BANDS) does not exist" because it then expects
        # them in para_def.nc in a specific FUSE-internal format.
        # The ellioaar_iceland working example confirms this: constraints has NO elevation
        # params, but FUSE calib_sce creates para_def.nc with them from _elev_bands.nc.

        # If routing needed, also copy and configure mizuRoute settings
        if self._check_routing_needed():
            mizu_settings = self.project_dir / 'settings' / 'mizuRoute'
            if mizu_settings.exists():
                for proc_id, dirs in self.parallel_dirs.items():
                    mizu_dest = dirs['root'] / 'settings' / 'mizuRoute'
                    mizu_dest.mkdir(parents=True, exist_ok=True)
                    for item in mizu_settings.iterdir():
                        if item.is_file():
                            copy_file(item, mizu_dest / item.name)

                # Update mizuRoute control files with process-specific paths
                self.update_mizuroute_controls(
                    self.parallel_dirs,
                    'FUSE',
                    self.experiment_id
                )
                self.logger.debug("Copied and configured mizuRoute settings for parallel processes")


# Backward compatibility alias
FUSEOptimizer = FUSEModelOptimizer
