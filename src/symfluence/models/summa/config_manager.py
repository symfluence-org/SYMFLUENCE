# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Configuration Manager.

This module contains the SummaConfigManager class which handles configuration
file creation and management for SUMMA model runs.
"""

# Standard library imports
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

# Third-party imports
import netCDF4 as nc4
import numpy as np
import xarray as xr

from symfluence.core.exceptions import ConfigValidationError

# SYMFLUENCE imports
from symfluence.core.path_resolver import PathResolverMixin


class SummaConfigManager(PathResolverMixin):
    """
    Manager for SUMMA configuration files.

    This class handles the creation and management of SUMMA configuration files,
    including file managers, initial conditions, trial parameters, and attributes.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], 'SymfluenceConfig'],
        logger: Any,
        project_dir: Path,
        setup_dir: Path,
        forcing_summa_path: Path,
        catchment_path: Path,
        catchment_name: str,
        dem_path: Path,
        hruId: str,
        gruId: str,
        data_step: int,
        coldstate_name: str,
        parameter_name: str,
        attribute_name: str,
        forcing_measurement_height: float,
        filter_forcing_hru_ids_callback: Optional[Callable] = None,
        get_base_settings_source_dir_callback: Optional[Callable] = None,
        get_default_path_callback: Optional[Callable] = None,
        get_simulation_times_callback: Optional[Callable] = None
    ):
        """
        Initialize the SUMMA Configuration Manager.

        Args:
            config: Configuration dict or typed SymfluenceConfig
            logger: Logger object for recording processing information
            project_dir: Path to the project directory
            setup_dir: Path to the setup directory
            forcing_summa_path: Path to SUMMA forcing files
            catchment_path: Path to catchment shapefiles
            catchment_name: Name of the catchment shapefile
            dem_path: Path to the DEM file
            hruId: Column name for HRU IDs in shapefiles
            gruId: Column name for GRU IDs in shapefiles
            data_step: Time step size for forcing data
            coldstate_name: Name of the cold state file
            parameter_name: Name of the trial parameters file
            attribute_name: Name of the attributes file
            forcing_measurement_height: Height of forcing measurements
            filter_forcing_hru_ids_callback: Callback function to filter forcing HRU IDs
            get_base_settings_source_dir_callback: Callback function to get base settings source directory
            get_default_path_callback: Callback function to get default paths
            get_simulation_times_callback: Callback function to get simulation times
        """
        # Store config for ConfigMixin compatibility (inherited via PathResolverMixin)
        if isinstance(config, dict):
            self._config = None
            self._config_dict_override = config
        else:
            self._config = config
        self.logger = logger
        self.project_dir = project_dir
        self.setup_dir = setup_dir
        self.forcing_summa_path = forcing_summa_path
        self.catchment_path = catchment_path
        self.catchment_name = catchment_name
        self.dem_path = dem_path
        self.hruId = hruId
        self.gruId = gruId
        self.data_step = data_step
        self.coldstate_name = coldstate_name
        self.parameter_name = parameter_name
        self.attribute_name = attribute_name
        self.forcing_measurement_height = forcing_measurement_height

        # Callbacks for methods that need to delegate back to parent
        self._filter_forcing_hru_ids_callback = filter_forcing_hru_ids_callback
        self._get_base_settings_source_dir_callback = get_base_settings_source_dir_callback
        self._get_default_path_callback = get_default_path_callback
        self._get_simulation_times_callback = get_simulation_times_callback

    def _filter_forcing_hru_ids(self, forcing_hru_ids):
        """Filter forcing HRU IDs using the callback if provided."""
        if self._filter_forcing_hru_ids_callback:
            return self._filter_forcing_hru_ids_callback(forcing_hru_ids)
        return forcing_hru_ids

    def _get_default_path(self, path_key: str, default_subpath: str, must_exist: bool = False) -> Path:
        """Get a path from config or use a default based on the project directory."""
        # Use callback if provided (delegates to parent preprocessor's mixin method)
        if self._get_default_path_callback:
            return self._get_default_path_callback(path_key, default_subpath)

        # Otherwise use the inherited PathResolverMixin method
        return super()._get_default_path(path_key, default_subpath, must_exist)

    def _get_simulation_times(self) -> Tuple[str, str]:
        """Get the simulation start and end times from config or calculate defaults."""
        if self._get_simulation_times_callback:
            return self._get_simulation_times_callback()

        # Fallback implementation
        sim_start = str(self._get_config_value(lambda: self.config.domain.time_start, default='', dict_key='EXPERIMENT_TIME_START'))
        sim_end = str(self._get_config_value(lambda: self.config.domain.time_end, default='', dict_key='EXPERIMENT_TIME_END'))
        return sim_start, sim_end

    def copy_base_settings(self):
        """
        Copy SUMMA base settings from the source directory to the project's settings directory.

        This method performs the following steps:
        1. Determines the source directory for base settings
        2. Determines the destination directory for settings
        3. Creates the destination directory if it doesn't exist
        4. Copies all files from the source directory to the destination directory

        Raises:
            FileNotFoundError: If the source directory or any source file is not found.
            PermissionError: If there are permission issues when creating directories or copying files.
        """
        self.logger.info("Copying SUMMA base settings")

        # Get base settings source directory
        if self._get_base_settings_source_dir_callback:
            base_settings_path = self._get_base_settings_source_dir_callback()
        else:
            # Fallback to direct resource loading if no callback provided
            from symfluence.core.modeling.base_settings import get_base_settings_dir
            base_settings_path = get_base_settings_dir('SUMMA')

        settings_path = self._get_default_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA')

        try:
            settings_path.mkdir(parents=True, exist_ok=True)

            for file in os.listdir(base_settings_path):
                source_file = base_settings_path / file
                dest_file = settings_path / file
                copyfile(source_file, dest_file)
                self.logger.debug(f"Copied {source_file} to {dest_file}")

            additional_outputs = self._get_config_value(
                lambda: self.config.model.summa.additional_outputs, default={},
                dict_key='SUMMA_ADDITIONAL_OUTPUTS')
            if additional_outputs:
                output_path = settings_path / self._get_config_value(
                    lambda: self.config.model.summa.output, default='outputControl.txt')
                lines = output_path.read_text().splitlines()
                # Replace configured variables, retaining the rest of the template.
                lines = [line for line in lines if line.split('|')[0].strip() not in additional_outputs]
                lines.extend(f'{name} | {frequency}' for name, frequency in additional_outputs.items())
                output_path.write_text('\n'.join(lines) + '\n')

            # Ensure TWS variables are in outputControl if doing TWS optimization
            # Check both primary and secondary targets for multi-objective calibration
            target = (self._get_config_value(lambda: self.config.optimization.target, default='', dict_key='OPTIMIZATION_TARGET') or '').lower()
            target2 = (self._get_config_value(lambda: self.config.optimization.nsga2.secondary_target, default='', dict_key='OPTIMIZATION_TARGET2') or '').lower()
            tws_targets = ['tws', 'grace', 'grace_tws', 'total_storage', 'stor_grace']
            if target in tws_targets or target2 in tws_targets:
                output_control_path = settings_path / 'outputControl.txt'
                if output_control_path.exists():
                    with open(output_control_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    required_vars = ['scalarSWE', 'scalarCanopyWat', 'scalarTotalSoilWat', 'scalarAquiferStorage']
                    # Get from config if specified
                    storage_str = self._get_config_value(lambda: self.config.evaluation.tws.storage_components, default='', dict_key='TWS_STORAGE_COMPONENTS')
                    if storage_str:
                        required_vars = [v.strip() for v in storage_str.split(',') if v.strip()]

                    # Filter out existing entries for these variables to avoid duplicates
                    new_lines = []
                    for line in lines:
                        is_required = False
                        for var in required_vars:
                            if line.strip().startswith(var):
                                is_required = True
                                break
                        if not is_required:
                            new_lines.append(line)

                    # Append them cleanly at the end with frequency 1
                    new_lines.append("\n! TWS Optimization required variables (every timestep)\n")
                    for v in required_vars:
                        new_lines.append(f"{v} | 1\n")

                    with open(output_control_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                    self.logger.info(f"Updated outputControl.txt with TWS variables: {required_vars}")

            # Ensure coupling variables are in outputControl if a groundwater model is configured
            gw_model = self._get_config_value(lambda: self.config.model.groundwater_model, default='', dict_key='GROUNDWATER_MODEL')
            if gw_model and str(gw_model).upper() not in ('', 'NONE'):
                output_control_path = settings_path / 'outputControl.txt'
                if output_control_path.exists():
                    with open(output_control_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    coupling_vars = {
                        'scalarSoilDrainage': '1',
                        'scalarSurfaceRunoff': '1',
                    }
                    added = []
                    for var, freq in coupling_vars.items():
                        if var not in content:
                            content += f"{var} | {freq}\n"
                            added.append(var)

                    if added:
                        with open(output_control_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        self.logger.info(
                            f"Added groundwater coupling variables to outputControl.txt: {added}"
                        )

                # Disable SUMMA's internal aquifer when coupling with an external
                # groundwater model.  bigBuckt routes soil drainage through an
                # internal aquifer, which double-counts recharge that also feeds
                # MODFLOW.  noXplict passes drainage straight through so only
                # MODFLOW handles the groundwater component.
                decisions_path = settings_path / 'modelDecisions.txt'
                if decisions_path.exists():
                    with open(decisions_path, 'r', encoding='utf-8') as f:
                        dec_content = f.read()

                    if 'bigBuckt' in dec_content:
                        import re
                        dec_content_new = re.sub(
                            r'(groundwatr\s+)\S+',
                            r'\1noXplict',
                            dec_content,
                        )
                        if dec_content_new != dec_content:
                            with open(decisions_path, 'w', encoding='utf-8') as f:
                                f.write(dec_content_new)
                            self.logger.info(
                                "Set groundwatr=noXplict for external GW coupling "
                                "(disabled SUMMA internal aquifer to avoid double-counting)"
                            )

            # Apply user-specified SUMMA_DECISION_OPTIONS overrides
            decisions_path = settings_path / 'modelDecisions.txt'
            decision_options = self._get_config_value(lambda: self.config.model.summa.decision_options, default={}, dict_key='SUMMA_DECISION_OPTIONS')
            if decision_options and decisions_path.exists():
                import re
                with open(decisions_path, 'r', encoding='utf-8') as f:
                    dec_content = f.read()
                for key, values in decision_options.items():
                    value = values[0] if isinstance(values, list) else values
                    dec_content, n = re.subn(
                        rf'({key}\s+)\S+',
                        rf'\g<1>{value}',
                        dec_content,
                    )
                    if n > 0:
                        self.logger.info(f"Set {key}={value} from SUMMA_DECISION_OPTIONS")
                with open(decisions_path, 'w', encoding='utf-8') as f:
                    f.write(dec_content)

            self.logger.info(f"SUMMA base settings copied to {settings_path}")
        except FileNotFoundError as e:
            self.logger.error(f"Source file or directory not found: {e}")
            raise
        except PermissionError as e:
            self.logger.error(f"Permission error when copying files: {e}")
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error copying base settings: {e}")
            raise

    def create_file_manager(self):
        """
        Create the SUMMA file manager configuration file.

        This method generates a file manager configuration for SUMMA, including:
        - Control version
        - Simulation start and end times
        - Output file prefix
        - Paths for various settings and data files

        The method uses configuration values and default paths where appropriate.

        Raises:
            ValueError: If required configuration values are missing or invalid.
            IOError: If there's an error writing the file manager configuration.
        """
        self.logger.info("Creating SUMMA file manager")

        try:
            experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, dict_key='EXPERIMENT_ID')
            if not experiment_id:
                raise ConfigValidationError("EXPERIMENT_ID is missing from configuration")

            sim_start, sim_end = self._get_simulation_times()

            filemanager_name = self._get_config_value(lambda: self.config.model.summa.filemanager, dict_key='SETTINGS_SUMMA_FILEMANAGER')
            if not filemanager_name:
                raise ConfigValidationError("SETTINGS_SUMMA_FILEMANAGER is missing from configuration")

            filemanager_path = self.setup_dir / filemanager_name

            with open(filemanager_path, 'w', encoding='utf-8') as fm:
                fm.write("controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n")
                fm.write(f"simStartTime         '{sim_start}'\n")
                fm.write(f"simEndTime           '{sim_end}'\n")
                fm.write("tmZoneInfo           'utcTime'\n")
                fm.write(f"outFilePrefix        '{experiment_id}'\n")
                fm.write(f"settingsPath         '{self._get_default_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA')}/'\n")
                fm.write(f"forcingPath          '{self._get_default_path('FORCING_SUMMA_PATH', 'forcing/SUMMA_input')}/'\n")
                fm.write(f"outputPath           '{self.project_dir / 'simulations' / experiment_id / 'SUMMA'}/'\n")

                fm.write(f"initConditionFile    '{self._get_config_value(lambda: self.config.model.summa.coldstate, dict_key='SETTINGS_SUMMA_COLDSTATE')}'\n")
                fm.write(f"attributeFile        '{self._get_config_value(lambda: self.config.model.summa.attributes, dict_key='SETTINGS_SUMMA_ATTRIBUTES')}'\n")
                fm.write(f"trialParamFile       '{self._get_config_value(lambda: self.config.model.summa.trialparams, dict_key='SETTINGS_SUMMA_TRIALPARAMS')}'\n")
                fm.write(f"forcingListFile      '{self._get_config_value(lambda: self.config.model.summa.forcing_list, dict_key='SETTINGS_SUMMA_FORCING_LIST')}'\n")
                fm.write("decisionsFile        'modelDecisions.txt'\n")
                fm.write("outputControlFile    'outputControl.txt'\n")
                fm.write("globalHruParamFile   'localParamInfo.txt'\n")
                fm.write("globalGruParamFile   'basinParamInfo.txt'\n")
                fm.write("vegTableFile         'TBL_VEGPARM.TBL'\n")
                fm.write("soilTableFile        'TBL_SOILPARM.TBL'\n")
                fm.write("generalTableFile     'TBL_GENPARM.TBL'\n")
                fm.write("noahmpTableFile      'TBL_MPTABLE.TBL'\n")

            self.logger.info(f"SUMMA file manager created at {filemanager_path}")

        except ValueError as ve:
            self.logger.error(f"Configuration error: {str(ve)}")
            raise
        except IOError as io_err:
            self.logger.error(f"Error writing file manager configuration: {str(io_err)}")
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Unexpected error in create_file_manager: {str(e)}")
            raise

    def create_initial_conditions(self):
        """
        Create the initial conditions (cold state) file for SUMMA.

        This method performs the following steps:
        1. Define the dimensions and variables for the cold state file
        2. Set default values for all state variables
        3. Create the netCDF file with the defined structure and values
        4. Ensure consistency with the forcing data (e.g., number of HRUs)

        The resulting file provides SUMMA with a starting point for model simulations.

        Raises:
            FileNotFoundError: If required input files (e.g., forcing file template) are not found.
            IOError: If there are issues creating or writing to the cold state file.
            ValueError: If there are inconsistencies between the cold state and forcing data.
        """
        self.logger.info("Creating initial conditions (cold state) file")

        # Find a forcing file to use as a template for hruId order
        forcing_files = list(self.forcing_summa_path.glob('*.nc'))
        if not forcing_files:
            self.logger.error("No forcing files found in the SUMMA input directory")
            return
        forcing_file = forcing_files[0]

        # Get the sorting order from the forcing file
        with xr.open_dataset(forcing_file) as forc:
            forcing_hruIds = forc['hruId'].values.astype(int)
        forcing_hruIds = list(forcing_hruIds)
        forcing_hruIds = self._filter_forcing_hru_ids(forcing_hruIds)

        num_hru = len(forcing_hruIds)

        # Define the dimensions and fill values
        nSnow = 0
        scalarv = 1

        soil_setups = {
            "FA": {
                "mLayerDepth":  np.asarray([0.2, 0.3, 0.5]),
                "iLayerHeight": np.asarray([0.0, 0.2, 0.5, 1.0]),
            },
            "CWARHM": {
                "mLayerDepth":  np.asarray([0.025, 0.075, 0.15, 0.25, 0.5, 0.5, 1.0, 1.5]),
                "iLayerHeight": np.asarray([0, 0.025, 0.1, 0.25, 0.5, 1, 1.5, 2.5, 4]),
            },
        }

        choice = self._get_config_value(lambda: self.config.model.summa.soilprofile, default='FA', dict_key='SETTINGS_SUMMA_SOILPROFILE')
        mLayerDepth  = soil_setups[choice]["mLayerDepth"]
        iLayerHeight = soil_setups[choice]["iLayerHeight"]

        midToto = len(mLayerDepth)
        ifcToto = len(iLayerHeight)
        midSoil = midToto
        nSoil   = midToto
        MatricHead = self._get_config_value(lambda: self.config.model.summa.init_matric_head, default=-1.0, dict_key='SUMMA_INIT_MATRIC_HEAD')

        # States
        states = {
            'scalarCanopyIce': 0,
            'scalarCanopyLiq': 0,
            'scalarSnowDepth': 0,
            'scalarSWE': 0,
            'scalarSfcMeltPond': 0,
            'scalarAquiferStorage': 2.5,
            'scalarSnowAlbedo': 0,
            'scalarCanairTemp': 283.16,
            'scalarCanopyTemp': 283.16,
            'mLayerTemp': 283.16,
            'mLayerVolFracIce': 0,
            'mLayerVolFracLiq': 0.2,
            'mLayerMatricHead': MatricHead
        }

        coldstate_path = self.setup_dir / self.coldstate_name

        def create_and_fill_nc_var(nc, newVarName, newVarVal, fillDim1, fillDim2, newVarDim, newVarType, fillVal):
            if newVarName in ['iLayerHeight', 'mLayerDepth']:
                fillWithThis = np.full((fillDim1, fillDim2), newVarVal).transpose()
            else:
                fillWithThis = np.full((fillDim1, fillDim2), newVarVal)

            ncvar = nc.createVariable(newVarName, newVarType, (newVarDim, 'hru'), fill_value=fillVal)
            ncvar[:] = fillWithThis

        with nc4.Dataset(coldstate_path, "w", format="NETCDF4") as cs:
            # Set attributes
            cs.setncattr('Author', "Created by SUMMA workflow scripts")
            cs.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            cs.setncattr('Purpose', 'Create a cold state .nc file for initial SUMMA runs')

            # Define dimensions
            cs.createDimension('hru', num_hru)
            cs.createDimension('midSoil', midSoil)
            cs.createDimension('midToto', midToto)
            cs.createDimension('ifcToto', ifcToto)
            cs.createDimension('scalarv', scalarv)

            # Create variables
            var = cs.createVariable('hruId', 'i4', 'hru', fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of hydrological response unit (HRU)')
            var[:] = forcing_hruIds

            create_and_fill_nc_var(cs, 'dt_init', self.data_step, 1, num_hru, 'scalarv', 'f8', False)
            create_and_fill_nc_var(cs, 'nSoil', nSoil, 1, num_hru, 'scalarv', 'i4', False)
            create_and_fill_nc_var(cs, 'nSnow', nSnow, 1, num_hru, 'scalarv', 'i4', False)

            for var_name, var_value in states.items():
                if var_name.startswith('mLayer'):
                    create_and_fill_nc_var(cs, var_name, var_value, midToto, num_hru, 'midToto', 'f8', False)
                else:
                    create_and_fill_nc_var(cs, var_name, var_value, 1, num_hru, 'scalarv', 'f8', False)

            create_and_fill_nc_var(cs, 'iLayerHeight', iLayerHeight, num_hru, ifcToto, 'ifcToto', 'f8', False)
            create_and_fill_nc_var(cs, 'mLayerDepth', mLayerDepth, num_hru, midToto, 'midToto', 'f8', False)

        self.logger.info(f"Initial conditions file created at: {coldstate_path}")

    def create_trial_parameters(self):
        """
        Create the trial parameters file for SUMMA.

        This method performs the following steps:
        1. Read trial parameter configurations from the main configuration
        2. Find a forcing file to use as a template for HRU order
        3. Create a netCDF file with the trial parameters
        4. Set the parameters for each HRU based on the configuration

        The resulting file provides SUMMA with parameter values to use in simulations.

        Raises:
            FileNotFoundError: If required input files (e.g., forcing file template) are not found.
            IOError: If there are issues creating or writing to the trial parameters file.
            ValueError: If there are inconsistencies in the parameter configurations.
        """
        self.logger.info("Creating trial parameters file")

        # Find a forcing file to use as a template for hruId order
        forcing_files = list(self.forcing_summa_path.glob('*.nc'))
        if not forcing_files:
            self.logger.error("No forcing files found in the SUMMA input directory")
            return
        forcing_file = forcing_files[0]

        # Get the sorting order from the forcing file
        with xr.open_dataset(forcing_file) as forc:
            forcing_hruIds = forc['hruId'].values.astype(int)
        forcing_hruIds = self._filter_forcing_hru_ids(forcing_hruIds)

        num_hru = len(forcing_hruIds)

        # Setup example trial parameter file initialisation
        num_tp = 1
        all_tp = {}
        for i in range(num_tp):
            par_and_val = 'maxstep,900'
            if par_and_val:
                arr = par_and_val.split(',')
                if len(arr) > 2:
                    val = np.array(arr[1:], dtype=np.float32)
                else:
                    val = float(arr[1])
                all_tp[arr[0]] = val

        parameter_path = self.setup_dir / self.parameter_name

        # Read GRU IDs from attributes file for the gru dimension
        attr_path = self.setup_dir / self.attribute_name if hasattr(self, 'attribute_name') else self.setup_dir / 'attributes.nc'
        try:
            with xr.open_dataset(attr_path) as attr_ds:
                gru_ids = attr_ds['gruId'].values.astype(int)
                num_gru = len(gru_ids)
                self.logger.debug(f"Read {num_gru} GRU IDs from attributes file")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            # Fallback: use hruIds as gruIds (one GRU per HRU)
            self.logger.warning(f"Could not read gruId from {attr_path}: {e}. Using HRU IDs as GRU IDs.", exc_info=True)
            gru_ids = forcing_hruIds
            num_gru = len(gru_ids)

        with nc4.Dataset(parameter_path, "w", format="NETCDF4") as tp:
            # Set attributes
            tp.setncattr('Author', "Created by SUMMA workflow scripts")
            tp.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            tp.setncattr('Purpose', 'Create a trial parameter .nc file for initial SUMMA runs')

            # Define dimensions
            tp.createDimension('hru', num_hru)
            tp.createDimension('gru', num_gru)

            # Create hruId variable
            var = tp.createVariable('hruId', 'i4', 'hru', fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of hydrological response unit (HRU)')
            var[:] = forcing_hruIds

            # Create gruId variable
            gru_var = tp.createVariable('gruId', 'i4', 'gru', fill_value=False)
            gru_var.setncattr('units', '-')
            gru_var.setncattr('long_name', 'Index of grouped response unit (GRU)')
            gru_var[:] = gru_ids

            # Create variables for specified trial parameters
            if self._get_config_value(lambda: self.config.model.summa.trialparam_n, default=0, dict_key='SETTINGS_SUMMA_TRIALPARAM_N') != 0:
                for var_name, val in all_tp.items():
                    tp_var = tp.createVariable(var_name, 'f8', 'hru', fill_value=False)
                    tp_var[:] = val

        self.logger.info(f"Trial parameters file created at: {parameter_path}")
