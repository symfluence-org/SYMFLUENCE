# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Configuration management utilities for HYPE model.

This module provides the HYPEConfigManager class for generating HYPE configuration
files including info.txt, filedir.txt, and par.txt. It supports parameter substitution
for calibration workflows and dynamic land-use parameter generation based on the
domain's land cover classes.

Example usage:
    >>> from symfluence.models.hype import HYPEConfigManager
    >>> manager = HYPEConfigManager(config, logger, output_path)
    >>> manager.write_info_filedir(spinup_days=365, results_dir='/path/to/results/')
    >>> manager.write_par_file(params={'cevp': 0.5}, land_uses=np.array([1, 5, 10]))
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Match

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError
from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HYPEConfigManager(ConfigMixin):
    """
    Manager for HYPE configuration and control files.

    This class handles the generation of HYPE's three main configuration files:

    - **info.txt**: Simulation control settings including dates, model options,
      and output specifications
    - **filedir.txt**: File path references for HYPE I/O
    - **par.txt**: Model parameters with support for calibration substitution

    The manager also provides dynamic land-use parameter generation based on
    IGBP land cover classifications (types 1-17).

    Attributes:
        config: Configuration dictionary containing HYPE settings.
        logger: Logger instance for status messages and warnings.
        output_path: Path to the HYPE settings directory where files are written.

    Example:
        >>> manager = HYPEConfigManager(
        ...     config={'EXPERIMENT_TIME_START': '2000-01-01'},
        ...     logger=logging.getLogger(__name__),
        ...     output_path=Path('/project/settings/HYPE')
        ... )
        >>> land_uses = geodata_manager.create_geofiles(...)
        >>> manager.write_par_file(params={'cevp': 0.3}, land_uses=land_uses)
    """

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger | Any | None,
        output_path: Path | str
    ) -> None:
        """
        Initialize the HYPE configuration manager.

        Args:
            config: Configuration dictionary containing experiment settings.
                Expected keys include:
                - EXPERIMENT_TIME_START: Simulation start date (YYYY-MM-DD)
                - EXPERIMENT_TIME_END: Simulation end date (YYYY-MM-DD)
            logger: Logger instance for status messages. If None, creates a
                module-level logger.
            output_path: Path to the HYPE settings directory where configuration
                files will be written. Will be created if it doesn't exist.
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.output_path = Path(output_path)

    def write_info_filedir(
        self,
        spinup_days: int,
        results_dir: str,
        experiment_start: str | None = None,
        experiment_end: str | None = None,
        forcing_data_dir: str | Path | None = None
    ) -> None:
        """
        Write info.txt and filedir.txt configuration files.

        This method creates the two primary HYPE control files:

        - **filedir.txt**: Contains the path for HYPE I/O (absolute path to forcing data)
        - **info.txt**: Contains simulation settings including:
          - Simulation period (bdate, cdate, edate)
          - Model options (snow, evaporation, routing, etc.)
          - Output specifications (timeoutput variables)
          - Input file toggles (readtminobs, readtmaxobs, etc.)

        The simulation dates are determined from (in priority order):
        1. Explicit experiment_start/experiment_end parameters
        2. Forcing file dates (from Pobs.txt if it exists)
        3. Config defaults (EXPERIMENT_TIME_START/END)

        Args:
            spinup_days: Number of days for model spinup. The 'cdate' (criteria
                start date) will be set to bdate + spinup_days. If spinup_days
                exceeds the simulation period, it's set to 0 with a warning.
            results_dir: Output directory path for HYPE results. Must include
                trailing slash (e.g., '/path/to/results/').
            experiment_start: Optional start date override (YYYY-MM-DD format).
                If not provided, uses forcing file start date or config default.
            experiment_end: Optional end date override (YYYY-MM-DD format).
                If not provided, uses forcing file end date or config default.
            forcing_data_dir: Optional path to forcing data directory. If not
                provided, uses the output_path (where config files are written).
                For calibration workers, this should point to the original
                HYPE settings directory containing Pobs.txt, Tobs.txt, etc.

        Raises:
            OSError: If the output directory cannot be written to.

        Note:
            HYPE requires the results_dir to have a trailing slash. This method
            ensures one is present.
        """
        # 1. Write filedir.txt - use absolute path to forcing data
        # For calibration workers, forcing_data_dir points to original HYPE settings
        # where Pobs.txt, Tobs.txt, etc. are located
        if forcing_data_dir is not None:
            forcing_path = str(Path(forcing_data_dir).resolve()).rstrip('/') + '/'
        else:
            forcing_path = str(self.output_path.resolve()).rstrip('/') + '/'

        filedir_path = self.output_path / 'filedir.txt'
        with open(filedir_path, 'w', encoding='utf-8') as f:
            f.write(f'{forcing_path}\n')

        # 2. Determine simulation period from Pobs.txt or config
        # Pobs.txt may be in forcing_data_dir (separate from config output_path)
        forcing_dir = Path(forcing_data_dir) if forcing_data_dir else self.output_path
        pobs_path = forcing_dir / 'Pobs.txt'
        if not pobs_path.exists():
            pobs_path = self.output_path / 'Pobs.txt'
        forcing_start: pd.Timestamp | None = None
        forcing_end: pd.Timestamp | None = None

        if pobs_path.exists():
            try:
                pobs_meta = pd.read_csv(pobs_path, sep='\t', parse_dates=['time'], nrows=2)
                forcing_start = pd.to_datetime(pobs_meta['time'].iloc[0])
                forcing_end = pd.to_datetime(
                    pd.read_csv(pobs_path, sep='\t', usecols=['time']).iloc[-1, 0]
                )
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Could not read Pobs.txt for period: {e}", exc_info=True)

        # Use experiment dates if provided, otherwise use forcing dates
        start_date: pd.Timestamp
        end_date: pd.Timestamp

        if experiment_start:
            start_date = pd.to_datetime(experiment_start)
        elif forcing_start is not None:
            start_date = forcing_start
        else:
            start_date = pd.to_datetime(
                self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01', dict_key='EXPERIMENT_TIME_START')
            )

        if experiment_end:
            end_date = pd.to_datetime(experiment_end)
        elif forcing_end is not None:
            end_date = forcing_end
        else:
            end_date = pd.to_datetime(
                self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31', dict_key='EXPERIMENT_TIME_END')
            )

        # Ensure start_date is not before forcing starts
        if forcing_start is not None and start_date < forcing_start:
            start_date = forcing_start

        # Daily observation dates are inclusive. Dropping a day here silently
        # shortens every run, including a complete calendar-month smoke test.
        if forcing_end is not None:
            end_date = min(end_date.normalize(), forcing_end.normalize())

        spinup_date = start_date + pd.Timedelta(days=spinup_days)

        # Handle edge cases
        if spinup_date >= end_date:
            spinup_date = start_date
            self.logger.debug(
                f"Spinup days ({spinup_days}) exceeds simulation period. Setting spinup to 0."
            )

        if start_date == end_date:
            end_date = end_date + pd.Timedelta(days=1)

        # Ensure results directory has trailing slash
        results_dir = str(results_dir).rstrip('/') + '/'

        # Format dates for info.txt
        start_str = start_date.strftime('%Y-%m-%d')
        spinup_str = spinup_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Disable HYPE internal obs loading when using external multi-gauge metrics
        multi_gauge = self._get_config_value(
            lambda: self.config.calibration.multi_gauge_calibration,
            default=False, dict_key='MULTI_GAUGE_CALIBRATION',
        )
        readobsid_flag = 'n' if multi_gauge else 'y'

        # Resolve configurable model options
        infilt_opt = self._get_config_value(
            lambda: self.config.model.hype.infiltration_model,
            default=3, dict_key='HYPE_INFILTRATION_MODEL',
        )
        petmodel_opt = self._get_config_value(
            lambda: self.config.model.hype.pet_model,
            default=1, dict_key='HYPE_PET_MODEL',
        )
        frozensoil_opt = self._get_config_value(
            lambda: self.config.model.hype.frozen_soil_model,
            default=2, dict_key='HYPE_FROZEN_SOIL_MODEL',
        )
        snowevap_opt = self._get_config_value(
            lambda: self.config.model.hype.snow_evaporation,
            default=0, dict_key='HYPE_SNOW_EVAPORATION',
        )
        deepground_opt = self._get_config_value(
            lambda: self.config.model.hype.deep_ground,
            default=0, dict_key='HYPE_DEEP_GROUND',
        )
        surfrunoff_opt = self._get_config_value(
            lambda: self.config.model.hype.surface_runoff,
            default=0, dict_key='HYPE_SURFACE_RUNOFF',
        )
        soiliniwet_opt = 'y' if self._get_config_value(
            lambda: self.config.model.hype.soil_init_wet,
            default=False, dict_key='HYPE_SOIL_INIT_WET',
        ) else 'n'

        # Build info.txt content
        glacier_outputs = '\tGLCV\tGLCA\tGMLT' if self.config.get('HYPE_GLACIER_FRACTION') is not None else ''
        if self.config.get('HYPE_CRYOSPHERE_CONSTRAINTS', False):
            glacier_outputs = '\tGLCV\tGLCA\tGMLT\tCFSC'
        if self.config.get('HYPE_WATER_BALANCE_DIAGNOSTICS', False):
            glacier_outputs += '\tSOIM\tSML1\tSML2\tSML3\tCLRV\tCMRV\tCPRC'
        snow_melt_model = self.config.get('HYPE_SNOW_MELT_MODEL', 0)
        read_sw = 'y' if snow_melt_model == 2 else 'n'
        info_content = f"""!! ----------------------------------------------------------------------------
!!
!! HYPE - Model Agnostic Framework
!!
!! -----------------------------------------------------------------------------
!! Check Indata during first runs (deactivate after first runs)
indatacheckonoff\t0
indatachecklevel\t0
!! -----------------------------------------------------------------------------
!!
!! Simulation settings:
!!
!! -----------------
bdate\t{start_str}
cdate\t{spinup_str}
edate\t{end_str}
resultdir\t{results_dir}
instate\tn
warning\ty
readdaily\ty
submodel\tn
calibration\tn
readobsid\t{readobsid_flag}
soilstretch\tn
!! Soilstretch enable the use of soilcorr parameters (strech soildepths in layer 2 and 3)
steplength\t1d
!! -----------------------------------------------------------------------------
!!
!! Enable/disable optional input files
!!
!! -----------------
readsfobs\tn
readswobs\t{read_sw}
readuobs\tn
readrhobs\tn
readtminobs\ty
readtmaxobs\ty
soiliniwet\t{soiliniwet_opt}
usestop84\tn
!! -----------------------------------------------------------------------------
!!
!! Define model options (optional)
!!
!! -----------------
modeloption snowfallmodel\t0
modeloption snowdensity\t0
modeloption snowfalldist\t2
modeloption snowheat\t0
modeloption snowmeltmodel\t{snow_melt_model}
modeloption snowevaporation\t{snowevap_opt}
modeloption lakeriverice\t0
modeloption deepground\t{deepground_opt}
modeloption glacierini\t1
modeloption floodmodel\t0
modeloption frozensoil\t{frozensoil_opt}
modeloption infiltration\t{infilt_opt}
modeloption surfacerunoff\t{surfrunoff_opt}
modeloption petmodel\t{petmodel_opt}
modeloption wetlandmodel\t2
modeloption connectivity\t0
!! ------------------------------------------------------------------------------------
!!
!! Define outputs
!!
!! -----------------
timeoutput variable COUT\tEVAP\tSNOW{glacier_outputs}
timeoutput meanperiod\t1
timeoutput decimals\t3
!! ------------------------------------------------------------------------------------
!!
!! Select criteria for model evaluation and automatic calibration
!!
!! -----------------
!! crit 1 criterion\tMKG
!! crit 1 cvariable\tcout
!! crit 1 rvariable\trout
!! crit 1 weight\t1
"""
        info_path = self.output_path / 'info.txt'
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(info_content)

        self.logger.debug(f"Created info.txt and filedir.txt in {self.output_path}")

    def write_par_file(
        self,
        params: dict[str, Any] | None = None,
        template_file: Path | str | None = None,
        land_uses: NDArray[np.integer] | None = None
    ) -> None:
        """
        Write par.txt parameter file with optional calibration substitution.

        This method generates the HYPE parameter file containing all model
        parameters. It supports three modes of operation:

        1. **Default generation**: Creates par.txt with sensible defaults based
           on IGBP land cover types and domain characteristics
        2. **Template-based**: Uses an existing par.txt as a template
        3. **Calibration substitution**: Replaces parameter values for optimization

        Land-use dependent parameters (ttmp, cmlt, cevp, etc.) are automatically
        generated based on IGBP land cover classifications when land_uses is provided.

        Args:
            params: Dictionary of parameter names to values for substitution.
                Supports both scalar values (replicated across all classes) and
                list/array values (applied directly). Example:
                ``{'cevp': 0.5, 'cmlt': [1.0, 2.0, 3.0]}``
            template_file: Path to an existing par.txt to use as template.
                If None, generates default content.
            land_uses: Array of land use type IDs (IGBP classes 1-17) present
                in the domain. Used to generate appropriate land-use dependent
                parameter columns. If None, reads from GeoClass.txt.

        Note:
            If an existing par.txt exists at output_path, it will be deleted
            and replaced.

        Example:
            >>> # For calibration with DDS optimizer
            >>> manager.write_par_file(
            ...     params={'cevp': trial_value, 'cmlt': trial_value},
            ...     land_uses=np.array([1, 5, 7, 10, 12])
            ... )
        """
        output_file = self.output_path / 'par.txt'
        if output_file.exists():
            output_file.unlink()

        # Read class counts from GeoClass.txt if it exists
        geoclass_file = self.output_path / 'GeoClass.txt'
        num_lu = 5  # Default
        num_soil = 2  # Default

        if geoclass_file.exists():
            try:
                # Skip the first row (header starting with !)
                geoclass_df = pd.read_csv(geoclass_file, sep='\t', skiprows=1, header=None)
                num_lu = int(geoclass_df.iloc[:, 1].max())  # Max LULC ID
                num_soil = int(geoclass_df.iloc[:, 2].max())  # Max Soil ID

                # Extract land use IDs if not provided
                if land_uses is None:
                    land_uses = geoclass_df.iloc[:, 1].unique()
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Could not read GeoClass.txt for class counts: {e}", exc_info=True)

        # Generate dynamic land use parameters
        lu_params: dict[str, str] | None
        if land_uses is not None and len(land_uses) > 0:
            lu_params, max_lu_id = self._generate_landuse_params(land_uses)
            max_lu = max(max_lu_id, num_lu)
        else:
            lu_params = None
            max_lu = num_lu

        # Build header strings
        lu_header = '\t'.join([f'LU{i}' for i in range(1, max_lu + 1)])
        soil_header = '\t'.join([f'S{i}' for i in range(1, num_soil + 1)])

        if template_file and Path(template_file).exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                par_content = f.read()
        else:
            par_content = self._build_default_par_content(
                lu_params, lu_header, soil_header, max_lu, num_soil
            )

        # Glacier melt uses general parameters, not the land-use snow cmlt.
        # Keep these explicit so an active glacier class cannot silently have zero melt.
        if land_uses is not None and 15 in land_uses and not template_file:
            melt = self.config.get('HYPE_GLACIER_MELT_FACTOR', 5.0)
            threshold = self.config.get('HYPE_GLACIER_MELT_THRESHOLD', 0.0)
            par_content += f'\n!! Glacier degree-day melt; initial volume uses HYPE area-volume defaults\nglaccmlt\t{melt}\nglacttmp\t{threshold}\nglacdens\t0.85\n'

        # Apply parameter substitutions
        percolation = self.config.get('HYPE_SOIL_PERCOLATION')
        if percolation is not None:
            if template_file or len(percolation) != 2 or any(not math.isfinite(v) or v < 0 for v in percolation):
                raise ValidationError('Soil percolation requires two finite nonnegative values and no custom template')
            for name, value in zip(('mperc1', 'mperc2'), percolation):
                par_content += '\n' + name + '\t' + '\t'.join([str(value)] * num_soil) + '\n'
        if self.config.get('HYPE_SNOW_MELT_MODEL', 0) == 2:
            if template_file:
                raise ValidationError('Set radiation melt parameters directly in a custom HYPE template')
            par_content += '\n!! Temperature and radiation melt (SWobs in MJ/m2/day)\n'
            # Keep the snow radiation term zero initially so this candidate
            # tests glacier radiation first. Both radiation coefficients can
            # subsequently be calibrated explicitly.
            for name, value in {'cmrad': 0.0, 'snalbmin': 0.5, 'snalbmax': 0.85,
                                'snalbkexp': 0.05}.items():
                par_content += name + '\t' + '\t'.join([str(value)] * max_lu) + '\n'
            par_content += f"glaccmrad\t{self.config.get('HYPE_GLACIER_RADIATION_MELT_FACTOR', 0.5)}\n"
            par_content += f"glacalb\t{self.config.get('HYPE_GLACIER_ALBEDO', 0.35)}\n"
            par_content += 'glaccmrefr\t0.2\n'
        if self.config.get('HYPE_FRACTIONAL_SNOW_COVER', False):
            if template_file:
                raise ValidationError('Set fractional snow-cover parameters directly in a custom HYPE template')
            # HYPE model_hype.f90 reference values. Without fscmax the model
            # treats any nonzero snow pack as 100% covered within each class.
            par_content += '\n!! Fractional snow cover: accumulation and depletion\n'
            for name, value in {'fscmax': 0.95, 'fscmin': 0.001, 'fsclim': 0.001,
                                'fsck1': 0.2, 'fsckexp': 1e-6}.items():
                par_content += f'{name}\t{value}\n'
            for name, value in {'fscdistmax': 0.8, 'fscdist0': 0.6, 'fscdist1': 0.001}.items():
                par_content += name + '\t' + '\t'.join([str(value)] * max_lu) + '\n'
        if self.config.get('HYPE_PRESERVE_ICECAP_VOLUME', False):
            if template_file or any(k in (params or {}) for k in ('glacvcoef1', 'glacvexp1')):
                raise ValidationError('Ice-cap volume preservation requires the default area-volume relation')
            par_content += '\n!! Relation used to preserve parent ice-cap volume across bands\nglacvcoef1\t1.701\nglacvexp1\t1.25\n'
        if params:
            par_content = self._apply_param_substitutions(par_content, params)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(par_content)

        self.logger.debug(f"Created par.txt in {self.output_path}")

    def _generate_landuse_params(
        self,
        land_uses: NDArray[np.integer]
    ) -> tuple[dict[str, str], int]:
        """
        Generate land-use-dependent parameter values for HYPE.

        This method creates parameter value strings for land-use dependent
        parameters based on IGBP land cover classifications. Each parameter
        has pre-defined values for the 17 IGBP classes, derived from S-HYPE
        (Swedish HYPE) baseline parameters.

        IGBP Land Cover Classes:
            1: Evergreen Needleleaf Forest
            2: Evergreen Broadleaf Forest
            3: Deciduous Needleleaf Forest
            4: Deciduous Broadleaf Forest
            5: Mixed Forest
            6: Closed Shrublands
            7: Open Shrublands
            8: Woody Savannas
            9: Savannas
            10: Grasslands
            11: Permanent Wetlands
            12: Croplands
            13: Urban and Built-Up
            14: Cropland/Natural Vegetation Mosaic
            15: Snow and Ice
            16: Barren or Sparsely Vegetated
            17: Water Bodies

        Args:
            land_uses: Array of land use type IDs present in the domain.
                Should contain values from 1-17 (IGBP classification).

        Returns:
            A tuple containing:
                - dict[str, str]: Mapping of parameter names to tab-separated
                  value strings. Keys include: ttmp, cmlt, cevp, ttrig, treda,
                  tredb, fepotsnow, srrcs, surfmem, depthrel, frost.
                - int: Maximum land use ID found (for sizing parameter arrays).

        Example:
            >>> lu_params, max_lu = manager._generate_landuse_params(np.array([1, 5, 10]))
            >>> print(lu_params['cevp'])
            '0.4689\\t0.7925\\t0.6317\\t0.6317\\t0.6317\\t0.4689\\t...'
        """
        # Base parameter values for common IGBP land use types (1-17)
        # Values derived from S-HYPE baseline parameters
        base_values: dict[str, dict[int, float | int]] = {
            'ttmp': {  # Snowmelt threshold temperature (deg C)
                1: -0.9253, 2: -1.5960, 3: -0.9620, 4: -0.9620, 5: -0.9620,
                6: -2.7121, 7: -2.7121, 8: -0.9620, 9: -0.9620, 10: -0.9253,
                11: 2.6945, 12: -0.9253, 13: -1.5960, 14: -2.7121, 15: 2.6945,
                16: -2.7121, 17: 0.0,
            },
            'cmlt': {  # Snowmelt degree day coefficient (mm/deg/day)
                1: 9.6497, 2: 9.2928, 3: 9.8897, 4: 9.8897, 5: 9.8897,
                6: 5.5393, 7: 5.5393, 8: 9.8897, 9: 9.8897, 10: 9.6497,
                11: 2.5333, 12: 9.6497, 13: 9.2928, 14: 5.5393, 15: 2.5333,
                16: 5.5393, 17: 0.0,
            },
            'cevp': {  # Evapotranspiration coefficient (-)
                1: 0.4689, 2: 0.7925, 3: 0.6317, 4: 0.6317, 5: 0.6317,
                6: 0.1699, 7: 0.1699, 8: 0.6317, 9: 0.6317, 10: 0.4689,
                11: 0.4506, 12: 0.4689, 13: 0.7925, 14: 0.1699, 15: 0.4506,
                16: 0.1699, 17: 0.0,
            },
            'ttrig': {  # Soil temperature threshold for transpiration (deg C)
                1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
                10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0,
            },
            'treda': {  # Root water uptake coefficient A (-)
                1: 0.84, 2: 0.84, 3: 0.84, 4: 0.84, 5: 0.84, 6: 0.84, 7: 0.84,
                8: 0.84, 9: 0.84, 10: 0.84, 11: 0.95, 12: 0.84, 13: 0.84,
                14: 0.84, 15: 0.95, 16: 0.84, 17: 0.0,
            },
            'tredb': {  # Root water uptake coefficient B (-)
                1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 5: 0.4, 6: 0.4, 7: 0.4,
                8: 0.4, 9: 0.4, 10: 0.4, 11: 0.4, 12: 0.4, 13: 0.4,
                14: 0.4, 15: 0.4, 16: 0.4, 17: 0.0,
            },
            'fepotsnow': {  # Fraction of PET for snow sublimation (-)
                1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8, 6: 0.8, 7: 0.8,
                8: 0.8, 9: 0.8, 10: 0.8, 11: 0.8, 12: 0.8, 13: 0.8,
                14: 0.8, 15: 0.8, 16: 0.8, 17: 0.0,
            },
            'srrcs': {  # Surface runoff coefficient (-)
                1: 0.0673, 2: 0.1012, 3: 0.1984, 4: 0.1984, 5: 0.1984,
                6: 0.0202, 7: 0.0202, 8: 0.1984, 9: 0.1984, 10: 0.0673,
                11: 0.0202, 12: 0.0673, 13: 0.1012, 14: 0.0202, 15: 0.0202,
                16: 0.0202, 17: 0.0,
            },
            'surfmem': {  # Upper soil temperature memory (days)
                1: 17.8, 2: 17.8, 3: 17.8, 4: 17.8, 5: 17.8, 6: 17.8, 7: 17.8,
                8: 17.8, 9: 17.8, 10: 17.8, 11: 5.15, 12: 17.8, 13: 17.8,
                14: 17.8, 15: 5.15, 16: 17.8, 17: 5.15,
            },
            'depthrel': {  # Depth relation for soil temperature (-)
                1: 1.1152, 2: 1.1152, 3: 1.1152, 4: 1.1152, 5: 1.1152,
                6: 1.1152, 7: 1.1152, 8: 1.1152, 9: 1.1152, 10: 1.1152,
                11: 2.47, 12: 1.1152, 13: 1.1152, 14: 1.1152, 15: 2.47,
                16: 1.1152, 17: 2.47,
            },
            'frost': {  # Frost depth parameter (cm/deg C)
                1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
                10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2,
            },
        }

        # Get maximum land use ID
        max_lu = int(max(land_uses))

        # Generate parameter strings
        param_strings: dict[str, str] = {}
        for param_name, value_dict in base_values.items():
            values: list[float | int] = []
            for lu_id in range(1, max_lu + 1):
                if lu_id in value_dict:
                    values.append(value_dict[lu_id])
                else:
                    # Use a sensible default for missing land uses
                    values.append(value_dict.get(1, 0.0))

            # Format as tab-separated string
            param_strings[param_name] = '\t'.join(
                f'{v:.4f}' if isinstance(v, float) else str(v) for v in values
            )

        return param_strings, max_lu

    def _build_default_par_content(
        self,
        lu_params: dict[str, str] | None,
        lu_header: str,
        soil_header: str,
        max_lu: int,
        num_soil: int
    ) -> str:
        """
        Build default par.txt content with dynamic land-use parameters.

        This method generates a complete HYPE parameter file with all required
        sections including snow, evapotranspiration, soil hydraulics, and
        river routing parameters.

        Args:
            lu_params: Dictionary of land-use parameter strings from
                _generate_landuse_params(). If None, uses default values.
            lu_header: Tab-separated header string for land use columns
                (e.g., 'LU1\\tLU2\\tLU3').
            soil_header: Tab-separated header string for soil columns
                (e.g., 'S1\\tS2').
            max_lu: Maximum land use ID (determines number of columns).
            num_soil: Number of soil types (determines number of columns).

        Returns:
            Complete par.txt file content as a string.
        """
        # Land-use dependent values
        ttmp_val = lu_params['ttmp'] if lu_params else '\t'.join(['-0.9253'] * max_lu)
        cmlt_val = lu_params['cmlt'] if lu_params else '\t'.join(['9.6497'] * max_lu)
        cevp_val = lu_params['cevp'] if lu_params else '\t'.join(['0.4689'] * max_lu)
        ttrig_val = lu_params['ttrig'] if lu_params else '\t'.join(['0'] * max_lu)
        treda_val = lu_params['treda'] if lu_params else '\t'.join(['0.84'] * max_lu)
        tredb_val = lu_params['tredb'] if lu_params else '\t'.join(['0.4'] * max_lu)
        fepotsnow_val = lu_params['fepotsnow'] if lu_params else '\t'.join(['0.8'] * max_lu)
        srrcs_val = lu_params['srrcs'] if lu_params else '\t'.join(['0.0673'] * max_lu)
        surfmem_val = lu_params['surfmem'] if lu_params else '\t'.join(['17.8'] * max_lu)
        depthrel_val = lu_params['depthrel'] if lu_params else '\t'.join(['1.1152'] * max_lu)
        frost_val = lu_params['frost'] if lu_params else '\t'.join(['2'] * max_lu)

        # Soil-dependent defaults
        bfrozn_val = '\t'.join(['3.7518'] * num_soil)
        logsatmp_val = '\t'.join(['1.15'] * num_soil)
        bcosby_val = '\t'.join(['11.2208'] * num_soil)
        rrcs1_val = '\t'.join(['0.4345'] * num_soil)
        rrcs2_val = '\t'.join(['0.1201'] * num_soil)
        sfrost_val = '\t'.join(['1'] * num_soil)
        wcwp_val = '\t'.join(['0.1171'] * num_soil)
        wcfc_val = '\t'.join(['0.3771'] * num_soil)
        wcep_val = '\t'.join(['0.4047'] * num_soil)

        return f"""!!	=======================================================================================================
!! Parameter file for:
!! HYPE -- Generated by the Model Agnostic Framework (hypeflow)
!!	=======================================================================================================
!!
!!	------------------------
!!
!!	=======================================================================================================
!!	SNOW - MELT, ACCUMULATION, AND DISTRIBUTION
!!	-----
!!	General snow accumulation and melt related parameters (baseline values from SHYPE)
ttpi	1.7083	!! width of the temperature interval with mixed precipitation
sdnsnew	0.13	!! density of fresh snow (kg/dm3)
snowdensdt	0.0016	!! snow densification parameter
fsceff	1	!! efficiency of fractional snow cover to reduce melt and evap
cmrefr	0.2	!! snow refreeze capacity (fraction of degreeday melt factor)
!!	-----
!!	Landuse dependent snow melt parameters
!!LUSE:	{lu_header}
ttmp	{ttmp_val}	!! Snowmelt threshold temperature (deg)
cmlt	{cmlt_val}	!! Snowmelt degree day coef (mm/deg/timestep)
!!	-----
!!	=======================================================================================================
!!	EVAPOTRANSPIRATION PARAMETERS
!!	-----
!!	General evapotranspiration parameters
lp	    0.6613	!! Threshold for water content reduction of transpiration
epotdist	   4.7088	!! Coefficient in exponential function for PET depth dependency
!!	-----
!!
!!LUSE:	{lu_header}
cevp	{cevp_val}	!! Evapotranspiration coefficient
ttrig	{ttrig_val}	!! Soil temperature threshold to allow transpiration
treda	{treda_val}	!! Coefficient in soil temperature response function
tredb	{tredb_val}	!! Coefficient in soil temperature response fuction
fepotsnow	{fepotsnow_val}	!! Fraction of PET used for snow sublimation
!!
!! Frozen soil infiltration parameters
!! SOIL:	{soil_header}
bfroznsoil  {bfrozn_val}  !! frozen soil infiltration parameter
logsatmp	{logsatmp_val}	!! saturated matric potential
bcosby	    {bcosby_val}	!! Cosby B parameter
!!	=======================================================================================================
!!	SOIL/LAND HYDRAULIC RESPONSE PARAMETERS
!!	-----
!!	Soil-class parameters
!!	SOIL:	{soil_header}
rrcs1   {rrcs1_val}	!! recession coefficients uppermost layer
rrcs2   {rrcs2_val}	!! recession coefficients bottom layer
rrcs3	    0.0939	!! Recession coefficient slope dependance
sfrost  {sfrost_val}	!! frost depth parameter (cm/degree Celsius)
wcwp    {wcwp_val}	!! Soil water content at wilting point
wcfc    {wcfc_val}	!! Field capacity
wcep    {wcep_val}	!! Effective porosity
!!	-----
!!	Landuse-class parameters
!!LUSE:	{lu_header}
srrcs	{srrcs_val}	!! Runoff coefficient for surface runoff
!!	-----
!!	Regional groundwater outflow
rcgrw	0.1	!! recession coefficient for regional groundwater outflow
!!	=======================================================================================================
!!	SOIL TEMPERATURE AND SOIL FROST DEPT
!!	-----
!!	General
deepmem	1000	!! temperature memory of deep soil (days)
!!-----
!!LUSE:	{lu_header}
surfmem	{surfmem_val}	!! upper soil layer soil temperature memory (days)
depthrel	{depthrel_val}	!! depth relation for soil temperature memory
frost	{frost_val}	!! frost depth parameter
!!	-----
!!	=======================================================================================================
!!	LAKE DISCHARGE
!!	-----
!!	ILAKE and OLAKE REGIONAL PARAMETERS
!!	ILAKE parameters
!! ilRegion	PPR 1
ilratk  149.9593
ilratp  4.9537
illdepth    0.33
ilicatch    1.0
!!
!!	=======================================================================================================
!!	RIVER ROUTING
!!	-----
damp	   0.2719	!! fraction of delay in the watercourse which also causes damping
rivvel	     9.7605	!! celerity of flood in watercourse
qmean 	200	!! initial value for calculation of mean flow (mm/yr)"""

    def _apply_param_substitutions(
        self,
        par_content: str,
        params: dict[str, Any]
    ) -> str:
        """
        Apply parameter value substitutions to par.txt content.

        This method replaces parameter values in the par.txt content with
        values from the params dictionary. It handles two scenarios:

        1. **Scalar values**: Replicated across all land-use/soil classes
        2. **List/array values**: Applied directly without replication

        For scalar values, the method preserves the original number of values
        (columns) in the parameter line, ensuring multi-class parameters
        remain properly formatted.

        Args:
            par_content: Original par.txt file content as a string.
            params: Dictionary mapping parameter names to new values.
                Values can be:
                - Scalar (int, float): Replicated for all classes
                - List/array: Used directly as space-separated values

        Returns:
            Modified par.txt content with substituted values.

        Note:
            A safety check is applied for 'cevp' values: if the value is
            less than 0.1, it's clamped to 0.1 with a warning, as very low
            cevp values can cause numerical instability.

        Example:
            >>> content = manager._apply_param_substitutions(
            ...     par_content,
            ...     {'cevp': 0.5, 'cmlt': [1.0, 2.0, 3.0, 4.0, 5.0]}
            ... )
        """
        self.logger.debug(f"Applying parameter substitutions: {list(params.keys())}")

        for key, value in params.items():
            # Handle list/array values
            if isinstance(value, (list, np.ndarray)):
                val_str = "  ".join(map(str, value))
                # Simple substitution for list/array values
                par_content = re.sub(
                    fr'^({key}\s+)[^\!\n]*',
                    fr'\g<1>{val_str}  ',
                    par_content,
                    flags=re.MULTILINE
                )
            else:
                val_str = str(value)

                def replicate_value(match: Match[str], _val_str: str = val_str) -> str:
                    """Replace parameter values while preserving column count."""
                    prefix = match.group(1)
                    content = match.group(0)[len(prefix):]
                    comment_parts = content.split('!!', 1)
                    existing_vals = comment_parts[0].strip()
                    comment = "!!" + comment_parts[1] if len(comment_parts) > 1 else ""

                    val_count = len(existing_vals.split())
                    if val_count > 1:
                        return f"{prefix}{' '.join([_val_str] * val_count)}  {comment}"
                    return f"{prefix}{_val_str}  {comment}"

                # Match parameter at start of line with values
                par_content = re.sub(
                    fr'^({key}\s+)([^\n]*)',
                    replicate_value,
                    par_content,
                    flags=re.MULTILINE
                )

        # Safety check for critical parameters
        if 'cevp' in params:
            cevp_val = params['cevp']
            if isinstance(cevp_val, (int, float)) and cevp_val < 0.1:
                self.logger.debug(
                    f"cevp value {cevp_val} is too low. Clamping to 0.1."
                )
                clamped_val = 0.1

                def replicate_clamped(match: Match[str]) -> str:
                    """Replace cevp with clamped minimum value."""
                    prefix = match.group(1)
                    content = match.group(0)[len(prefix):]
                    comment_parts = content.split('!!', 1)
                    existing_vals = comment_parts[0].strip()
                    comment = "!!" + comment_parts[1] if len(comment_parts) > 1 else ""
                    val_count = len(existing_vals.split())
                    return f"{prefix}{' '.join([str(clamped_val)] * val_count)}  {comment}"

                par_content = re.sub(
                    r'^(cevp\s+)([^\n]*)',
                    replicate_clamped,
                    par_content,
                    flags=re.MULTILINE
                )

        return par_content
