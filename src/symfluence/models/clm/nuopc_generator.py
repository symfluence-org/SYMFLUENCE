# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM NUOPC Runtime Configuration Generator

Generates all NUOPC runtime configuration files required for a CLM5
single-point run, including nuopc.runconfig, nuopc.runseq, lnd_in,
datm_in, datm.streams.xml, drv_in, drv_flds_in, fd.yaml, CASEROOT,
and user_nl_clm.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Default CESM inputdata root (downloaded on demand). Configurable via
# CLM_CESM_INPUTDATA_PATH — see _resolve_cesm_inputdata().
DEFAULT_CESM_INPUTDATA = Path.home() / 'projects' / 'cesm-inputdata'

# Base URL for CESM inputdata SVN server
_CESM_INPUTDATA_URL = (
    'https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata'
)

# Required CESM inputdata files (relative to CESM_INPUTDATA root)
_REQUIRED_INPUTDATA = [
    'lnd/clm2/snicardata/snicar_optics_5bnd_c013122.nc',
    'lnd/clm2/snicardata/snicar_drdt_bst_fit_60_c070416.nc',
    'lnd/clm2/urbandata/CLM50_tbuildmax_Oleson_2016_0.9x1.25_simyr1849-2106_c160923.nc',
    'lnd/clm2/urbandata/CLM50_tbuildmax_Oleson_2016_0.9x1_ESMFmesh_cdf5_100621.nc',
    'atm/cam/chem/trop_mozart/emis/megan21_emis_factors_78pft_c20161108.nc',
]


def _resolve_cesm_inputdata(preprocessor) -> Path:
    """Resolve the CESM inputdata root, defaulting to a SYMFLUENCE-managed dir.

    Precedence:

    1. ``CLM_CESM_INPUTDATA_PATH`` (``config.model.clm.cesm_inputdata_path``)
       when set to anything other than the ``'default'`` sentinel.
    2. Otherwise a writable, SYMFLUENCE-managed location alongside the model
       installs (``<data_dir>/installs/cesm-inputdata``), resolved the same way
       as ``CLM_INSTALL_PATH: default``. This shares the (large, static)
       inputdata across domains and avoids the home directory, which is often
       not writable from HPC compute nodes — so no manual override is needed.
    3. As a last resort (no project/data dir available), the home-directory
       default. This keeps the function total even outside a real project.
    """
    configured = preprocessor._get_config_value(
        lambda: preprocessor.config.model.clm.cesm_inputdata_path,
        default='default',
        dict_key='CLM_CESM_INPUTDATA_PATH',
    )
    if configured and str(configured).strip().lower() != 'default':
        return Path(configured).expanduser()

    # Mirror CLMPreProcessor._get_install_path() so inputdata lands next to the
    # model installs under the SYMFLUENCE data dir (writable scratch on HPC).
    code_dir = preprocessor._get_config_value(
        lambda: preprocessor.config.system.code_dir,
        default=None,
        dict_key='SYMFLUENCE_CODE_DIR',
    )
    if code_dir:
        code_path = Path(code_dir)
        return code_path.parent / (code_path.name + '_data') / 'installs' / 'cesm-inputdata'
    try:
        return preprocessor.project_dir.parents[1] / 'installs' / 'cesm-inputdata'
    except (AttributeError, IndexError):
        return DEFAULT_CESM_INPUTDATA


def _ensure_cesm_inputdata(root: Path) -> None:
    """Download required CESM inputdata files if missing."""
    for rel_path in _REQUIRED_INPUTDATA:
        local = root / rel_path
        if local.exists() and local.stat().st_size > 0:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f'{_CESM_INPUTDATA_URL}/{rel_path}'
        logger.info(f"Downloading CESM inputdata: {rel_path}")
        try:
            # Security rationale: trusted CESM inputdata URL
            urllib.request.urlretrieve(url, local)  # nosec B310
            logger.info(f"  Saved: {local} ({local.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 — model execution resilience
            logger.warning(f"  Failed to download {rel_path}: {exc}", exc_info=True)


class CLMNuopcGenerator:
    """Generates NUOPC runtime configuration files for CLM5.

    Parameters
    ----------
    preprocessor : CLMPreProcessor
        Parent preprocessor instance providing config, paths, and
        geometry helpers.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor
        self.cesm_inputdata = _resolve_cesm_inputdata(preprocessor)

    def generate_nuopc_runtime(self) -> None:
        """Generate all NUOPC runtime configuration files."""
        _ensure_cesm_inputdata(self.cesm_inputdata)
        logger.info("Generating NUOPC runtime configuration files")

        lat, lon, area_km2 = self.pp.domain_generator.get_catchment_centroid()
        mean_elev = self.pp.domain_generator.get_mean_elevation()

        start_date = self.pp._get_config_value(
            lambda: self.pp.config.domain.time_start,
            default='2000-01-01', dict_key='EXPERIMENT_TIME_START'
        )
        end_date = self.pp._get_config_value(
            lambda: self.pp.config.domain.time_end,
            default='2010-12-31', dict_key='EXPERIMENT_TIME_END'
        )
        start_ymd = str(start_date)[:10].replace('-', '')
        stop_ymd = str(end_date)[:10].replace('-', '')
        start_year = int(str(start_date)[:4])
        end_year = int(str(end_date)[:4])

        hist_nhtfrq = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.hist_nhtfrq,
            default=-24, dict_key='CLM_HIST_NHTFRQ'
        )
        hist_mfilt = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.hist_mfilt,
            default=365, dict_key='CLM_HIST_MFILT'
        )

        params_file = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.params_file,
            default='clm5_params.nc', dict_key='CLM_PARAMS_FILE'
        )
        surfdata_file = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.surfdata_file,
            default='surfdata_clm.nc', dict_key='CLM_SURFDATA_FILE'
        )

        mesh_path = str(self.pp.params_dir / 'esmf_mesh.nc')
        forcing_files = sorted(self.pp.forcing_dir.glob('clmforc.*.nc'))

        ctx = {
            'lat': lat, 'lon': lon, 'area_km2': area_km2,
            'mean_elev': mean_elev,
            'start_ymd': start_ymd, 'stop_ymd': stop_ymd,
            'start_year': start_year, 'end_year': end_year,
            'hist_nhtfrq': hist_nhtfrq, 'hist_mfilt': hist_mfilt,
            'params_file': str(self.pp.params_dir / params_file),
            'surfdata_file': str(self.pp.params_dir / surfdata_file),
            'mesh_path': mesh_path,
            'forcing_dir': str(self.pp.forcing_dir),
            'forcing_files': forcing_files,
            'topo_file': str(self.pp.forcing_dir / 'topo_forcing.nc'),
            'domain_name': self.pp.domain_name,
        }

        self._write_nuopc_runconfig(ctx)
        self._write_nuopc_runseq()
        self._write_datm_in(ctx)
        self._write_datm_streams_xml(ctx)
        self._write_lnd_in(ctx)
        self._write_drv_in()
        self._write_drv_flds_in()
        self._copy_fd_yaml()
        self._write_caseroot()
        self._write_user_nl_clm(ctx)

        logger.info(f"Generated NUOPC runtime files in {self.pp.settings_dir}")

    def _write_nuopc_runconfig(self, ctx: dict) -> None:
        """Generate nuopc.runconfig for single-point CLM."""
        sections = [
            self._nuopc_driver_section(),
            self._nuopc_pelayout_section(),
            self._nuopc_allcomp_section(ctx),
            self._nuopc_med_section(),
            self._nuopc_clock_section(ctx),
            self._nuopc_component_attrs_section(),
            self._nuopc_modelio_sections(),
        ]
        content = '\n'.join(sections) + '\n'
        (self.pp.settings_dir / 'nuopc.runconfig').write_text(content)
        logger.debug("Generated nuopc.runconfig")

    @staticmethod
    def _nuopc_driver_section() -> str:
        return """DRIVER_attributes::
     Verbosity = off
     cime_model = cesm
     drv_restart_pointer = rpointer.cpl
     logFilePostFix = .log
     outPathRoot = ./
     pio_blocksize = -1
     pio_buffer_size_limit = -1
     pio_debug_level = 0
     pio_rearr_comm_enable_hs_comp2io = .true.
     pio_rearr_comm_enable_hs_io2comp = .false.
     pio_rearr_comm_enable_isend_comp2io = .false.
     pio_rearr_comm_enable_isend_io2comp = .true.
     pio_rearr_comm_fcd = 2denable
     pio_rearr_comm_max_pend_req_comp2io = -2
     pio_rearr_comm_max_pend_req_io2comp = 64
     pio_rearr_comm_type = p2p
     reprosum_diffmax = -1.0e-8
     reprosum_recompute = .false.
     reprosum_use_ddpdd = .false.
     tchkpt_dir = ./timing/checkpoints
     timing_dir = ./timing
     wv_sat_scheme = GoffGratch
     wv_sat_table_spacing = 1.0D0
     wv_sat_transition_start = 20.0D0
     wv_sat_use_tables = .false.
::"""

    @staticmethod
    def _nuopc_pelayout_section() -> str:
        return """PELAYOUT_attributes::
     atm_ntasks = 1
     atm_nthreads = 1
     atm_pestride = 1
     atm_rootpe = 0
     cpl_ntasks = 1
     cpl_nthreads = 1
     cpl_pestride = 1
     cpl_rootpe = 0
     esp_ntasks = 1
     esp_nthreads = 1
     esp_pestride = 1
     esp_rootpe = 0
     glc_ntasks = 1
     glc_nthreads = 1
     glc_pestride = 1
     glc_rootpe = 0
     ice_ntasks = 1
     ice_nthreads = 1
     ice_pestride = 1
     ice_rootpe = 0
     lnd_ntasks = 1
     lnd_nthreads = 1
     lnd_pestride = 1
     lnd_rootpe = 0
     ninst = 1
     ocn_ntasks = 1
     ocn_nthreads = 1
     ocn_pestride = 1
     ocn_rootpe = 0
     pio_asyncio_ntasks = 0
     pio_asyncio_rootpe = 1
     pio_asyncio_stride = 0
     rof_ntasks = 1
     rof_nthreads = 1
     rof_pestride = 1
     rof_rootpe = 0
     wav_ntasks = 1
     wav_nthreads = 1
     wav_pestride = 1
     wav_rootpe = 0
::"""

    @staticmethod
    def _nuopc_allcomp_section(ctx: dict) -> str:
        mesh = ctx['mesh_path']
        return f"""component_list: MED ATM LND
ALLCOMP_attributes::
     ATM_model = datm
     GLC_model = sglc
     ICE_model = sice
     LND_model = clm
     MED_model = cesm
     OCN_model = socn
     Profiling = 0
     ROF_model = srof
     ScalarFieldCount = 4
     ScalarFieldIdxGridNX = 1
     ScalarFieldIdxGridNY = 2
     ScalarFieldIdxNextSwCday = 3
     ScalarFieldIdxPrecipFactor = 0
     ScalarFieldName = cpl_scalars
     WAV_model = swav
     brnch_retain_casename = .false.
     case_desc = SYMFLUENCE CLM single-point
     case_name = {ctx['domain_name']}
     cism_evolve = .false.
     coldair_outbreak_mod = .true.
     data_assimilation_atm = .false.
     data_assimilation_cpl = .false.
     data_assimilation_glc = .false.
     data_assimilation_ice = .false.
     data_assimilation_lnd = .false.
     data_assimilation_ocn = .false.
     data_assimilation_rof = .false.
     data_assimilation_wav = .false.
     flds_bgc_oi = .false.
     flds_co2a = .false.
     flds_co2b = .false.
     flds_co2c = .false.
     flds_i2o_per_cat = .false.
     flds_r2l_stream_channel_depths = .false.
     flds_wiso = .false.
     flux_convergence = 0.0
     flux_max_iteration = 5
     glc_nec = 10
     histaux_l2x1yrg = .false.
     histaux_wav2med_file1_enabled = .false.
     history_n = -999
     history_option = never
     hostname = symfluence
     ice_ncat = 1
     mediator_present = true
     mesh_atm = {mesh}
     mesh_glc = UNSET
     mesh_ice = UNSET
     mesh_lnd = {mesh}
     mesh_mask = {mesh}
     mesh_ocn = UNSET
     model_version = ctsm5.3
     ocn2glc_coupling = .false.
     orb_eccen = 1.e36
     orb_iyear = 2000
     orb_iyear_align = 2000
     orb_mode = fixed_year
     orb_mvelp = 1.e36
     orb_obliq = 1.e36
     scol_lat = -999.99
     scol_lon = -999.99
     single_column_lnd_domainfile = UNSET
     start_type = startup
     tfreeze_option = mushy
     username = symfluence
     wav_coupling_to_cice = .false.
     write_restart_at_endofrun = .false.
::"""

    @staticmethod
    def _nuopc_med_section() -> str:
        return """MED_attributes::
     Verbosity = off
     add_gusts = .false.
     aoflux_grid = ogrid
     atm2ice_map = unset
     atm2lnd_map = idmap
     atm2ocn_map = unset
     atm2wav_map = unset
     atm_nx = 1
     atm_ny = 1
     budget_ann = 1
     budget_daily = 0
     budget_inst = 0
     budget_ltann = 1
     budget_ltend = 0
     budget_month = 1
     budget_table_version = v0
     check_for_nans = .true.
     coupling_mode = cesm
     do_budgets = .false.
     flux_albav = .false.
     glc_renormalize_smb = on_if_glc_coupled_fluxes
     gust_fac = 0.0D0
     histaux_atm2med_file1_enabled = .false.
     histaux_atm2med_file2_enabled = .false.
     histaux_atm2med_file3_enabled = .false.
     histaux_atm2med_file4_enabled = .false.
     histaux_atm2med_file5_enabled = .false.
     histaux_lnd2med_file1_enabled = .false.
     histaux_ocn2med_file1_enabled = .false.
     histaux_rof2med_file1_enabled = .false.
     histaux_wav2med_file1_enabled = .false.
     history_n_atm_avg = -999
     history_n_atm_inst = -999
     history_n_glc_avg = -999
     history_n_glc_inst = -999
     history_n_ice_avg = -999
     history_n_ice_inst = -999
     history_n_lnd_avg = -999
     history_n_lnd_inst = -999
     history_n_med_inst = -999
     history_n_ocn_avg = -999
     history_n_ocn_inst = -999
     history_n_rof_avg = -999
     history_n_rof_inst = -999
     history_n_wav_avg = -999
     history_n_wav_inst = -999
     history_option_atm_avg = never
     history_option_atm_inst = never
     history_option_glc_avg = never
     history_option_glc_inst = never
     history_option_ice_avg = never
     history_option_ice_inst = never
     history_option_lnd_avg = never
     history_option_lnd_inst = never
     history_option_med_inst = never
     history_option_ocn_avg = never
     history_option_ocn_inst = never
     history_option_rof_avg = never
     history_option_rof_inst = never
     history_option_wav_avg = never
     history_option_wav_inst = never
     ice2atm_map = unset
     ice_nx = 0
     ice_ny = 0
     info_debug = 1
     lnd2atm_map = idmap
     lnd2rof_map = unset
     lnd_nx = 1
     lnd_ny = 1
     mapuv_with_cart3d = .true.
     ocn2atm_map = unset
     ocn_nx = 0
     ocn_ny = 0
     ocn_surface_flux_scheme = 0
     remove_negative_runoff = .true.
     rof2lnd_map = unset
     rof2ocn_ice_rmapname = unset
     rof2ocn_liq_rmapname = unset
     rof_nx = 0
     rof_ny = 0
     wav_nx = 0
     wav_ny = 0
::"""

    @staticmethod
    def _nuopc_clock_section(ctx: dict) -> str:
        start_ymd = ctx['start_ymd']
        n_years = ctx['end_year'] - ctx['start_year'] + 1
        return f"""CLOCK_attributes::
     atm_cpl_dt = 3600
     calendar = GREGORIAN
     end_restart = .false.
     glc_avg_period = yearly
     glc_cpl_dt = 3600
     history_ymd = -999
     ice_cpl_dt = 3600
     lnd_cpl_dt = 3600
     ocn_cpl_dt = 3600
     restart_n = 1
     restart_option = nyears
     restart_ymd = -999
     rof_cpl_dt = 10800
     start_tod = 0
     start_ymd = {start_ymd}
     stop_n = {n_years}
     stop_option = nyears
     stop_tod = 0
     stop_ymd = -999
     tprof_n = -999
     tprof_option = never
     tprof_ymd = -999
     wav_cpl_dt = 3600
::"""

    @staticmethod
    def _nuopc_component_attrs_section() -> str:
        return """ATM_attributes::
     Verbosity = off
     aqua_planet = .false.
     perpetual = .false.
     perpetual_ymd = -999
::

ICE_attributes::
     Verbosity = off
::

GLC_attributes::
     Verbosity = off
::

LND_attributes::
     Verbosity = off
::

OCN_attributes::
     Verbosity = off
::

ROF_attributes::
     Verbosity = off
     mesh_rof = UNSET
::

WAV_attributes::
     Verbosity = off
     mesh_wav = UNSET
::"""

    @staticmethod
    def _nuopc_modelio_sections() -> str:
        """Generate all modelio blocks for NUOPC components."""
        def _modelio_block(comp: str, logname: str, pio_rearranger: int = 1,
                           pio_root: int = 0, pio_stride: int = 1,
                           pio_typename: str = 'netcdf') -> str:
            return f"""{comp}_modelio::
     diro = ./
     logfile = {logname}.log
     pio_async_interface = .false.
     pio_netcdf_format = 64bit_offset
     pio_numiotasks = -99
     pio_rearranger = {pio_rearranger}
     pio_root = {pio_root}
     pio_stride = {pio_stride}
     pio_typename = {pio_typename}
::"""

        blocks = []
        for comp, logname in [
            ('MED', 'med'), ('ATM', 'atm'), ('LND', 'lnd'), ('ICE', 'ice'),
            ('OCN', 'ocn'), ('ROF', 'rof'), ('GLC', 'glc'), ('WAV', 'wav'),
        ]:
            blocks.append(_modelio_block(comp, logname))

        # ESP has non-standard PIO values
        blocks.append(_modelio_block('ESP', 'esp', pio_rearranger=-99,
                                     pio_root=-99, pio_stride=-99,
                                     pio_typename='nothing'))

        # DRV is minimal (no PIO settings)
        blocks.append("DRV_modelio::\n     diro = ./\n     logfile = drv.log\n::")

        return '\n\n'.join(blocks)

    def _write_nuopc_runseq(self) -> None:
        """Generate NUOPC run sequence."""
        content = """runSeq::
@3600
  MED med_phases_prep_lnd
  MED -> LND :remapMethod=redist
  LND
  LND -> MED :remapMethod=redist
  MED med_phases_post_lnd
  ATM
  ATM -> MED :remapMethod=redist
  MED med_phases_post_atm
  MED med_phases_history_write
  MED med_phases_restart_write
  MED med_phases_profile
@
::
"""
        (self.pp.settings_dir / 'nuopc.runseq').write_text(content)
        logger.debug("Generated nuopc.runseq")

    def _write_datm_in(self, ctx: dict) -> None:
        """Generate NUOPC-format datm_in."""
        mesh = ctx['mesh_path']
        content = f"""&const_forcing_nml
  dn10 = 1.204
  peak_lwdn = 450.0
  peak_swdn = 330.0
  q = 0.0
  slp = 101325.0
  t = 273.15
  u = 0.0
  v = 0.0
/
&datm_nml
  datamode = "CLMNCEP"
  factorfn_data = "null"
  factorfn_mesh = "null"
  flds_co2 = .false.
  flds_presaero = .false.
  flds_presndep = .false.
  flds_preso3 = .false.
  flds_wiso = .false.
  iradsw = 1
  model_maskfile = "{mesh}"
  model_meshfile = "{mesh}"
  nx_global = 1
  ny_global = 1
  restfilm = "null"
  skip_restart_read = .false.
/
"""
        (self.pp.settings_dir / 'datm_in').write_text(content)
        logger.debug("Generated datm_in")

    def _write_datm_streams_xml(self, ctx: dict) -> None:
        """Generate datm.streams.xml with custom forcing streams."""
        mesh = ctx['mesh_path']
        forcing_files = ctx['forcing_files']
        start_year = ctx['start_year']
        end_year = ctx['end_year']
        topo_file = ctx['topo_file']

        # Build file list XML
        file_entries = '\n'.join(
            f'      <file>{f}</file>' for f in forcing_files
        )

        content = f"""<?xml version="1.0"?>
<file id="stream" version="2.0">

  <stream_info name="CLMNCEP.Solar">
   <taxmode>cycle</taxmode>
   <tintalgo>coszen</tintalgo>
   <readmode>single</readmode>
   <mapalgo>nn</mapalgo>
   <dtlimit>3.0</dtlimit>
   <year_first>{start_year}</year_first>
   <year_last>{end_year}</year_last>
   <year_align>{start_year}</year_align>
   <vectors>null</vectors>
   <meshfile>{mesh}</meshfile>
   <lev_dimname>null</lev_dimname>
   <datafiles>
{file_entries}
   </datafiles>
   <datavars>
      <var>FSDS Faxa_swdn</var>
   </datavars>
   <offset>0</offset>
  </stream_info>

  <stream_info name="CLMNCEP.Precip">
   <taxmode>cycle</taxmode>
   <tintalgo>nearest</tintalgo>
   <readmode>single</readmode>
   <mapalgo>nn</mapalgo>
   <dtlimit>3.0</dtlimit>
   <year_first>{start_year}</year_first>
   <year_last>{end_year}</year_last>
   <year_align>{start_year}</year_align>
   <vectors>null</vectors>
   <meshfile>{mesh}</meshfile>
   <lev_dimname>null</lev_dimname>
   <datafiles>
{file_entries}
   </datafiles>
   <datavars>
      <var>PRECTmms Faxa_precn</var>
   </datavars>
   <offset>0</offset>
  </stream_info>

  <stream_info name="CLMNCEP.TPQW">
   <taxmode>cycle</taxmode>
   <tintalgo>linear</tintalgo>
   <readmode>single</readmode>
   <mapalgo>nn</mapalgo>
   <dtlimit>3.0</dtlimit>
   <year_first>{start_year}</year_first>
   <year_last>{end_year}</year_last>
   <year_align>{start_year}</year_align>
   <vectors>null</vectors>
   <meshfile>{mesh}</meshfile>
   <lev_dimname>null</lev_dimname>
   <datafiles>
{file_entries}
   </datafiles>
   <datavars>
      <var>TBOT     Sa_tbot</var>
      <var>WIND     Sa_wind</var>
      <var>QBOT     Sa_shum</var>
      <var>PSRF     Sa_pbot</var>
      <var>FLDS     Faxa_lwdn</var>
   </datavars>
   <offset>0</offset>
  </stream_info>

  <stream_info name="topo.observed">
   <taxmode>cycle</taxmode>
   <tintalgo>lower</tintalgo>
   <readmode>single</readmode>
   <mapalgo>nn</mapalgo>
   <dtlimit>1.5</dtlimit>
   <year_first>1</year_first>
   <year_last>1</year_last>
   <year_align>1</year_align>
   <vectors>null</vectors>
   <meshfile>{mesh}</meshfile>
   <lev_dimname>null</lev_dimname>
   <datafiles>
      <file>{topo_file}</file>
   </datafiles>
   <datavars>
      <var>TOPO Sa_topo</var>
   </datavars>
   <offset>0</offset>
  </stream_info>

</file>
"""
        (self.pp.settings_dir / 'datm.streams.xml').write_text(content)
        logger.debug("Generated datm.streams.xml")

    def _write_lnd_in(self, ctx: dict) -> None:
        """Generate CLM5 lnd_in namelist for single-point SP run."""
        surfdata = ctx['surfdata_file']
        paramfile = ctx['params_file']
        hist_nhtfrq = ctx['hist_nhtfrq']
        hist_mfilt = ctx['hist_mfilt']
        # SNICAR data paths (from CESM inputdata)
        snicar_dir = self.cesm_inputdata / 'lnd' / 'clm2' / 'snicardata'
        fsnowaging = snicar_dir / 'snicar_drdt_bst_fit_60_c070416.nc'
        fsnowoptics = snicar_dir / 'snicar_optics_5bnd_c013122.nc'

        # MEGAN emissions (from CESM inputdata)
        # Urban stream data (from CESM inputdata)
        urban_dir = self.cesm_inputdata / 'lnd' / 'clm2' / 'urbandata'
        urban_tv = urban_dir / 'CLM50_tbuildmax_Oleson_2016_0.9x1.25_simyr1849-2106_c160923.nc'
        urban_mesh = urban_dir / 'CLM50_tbuildmax_Oleson_2016_0.9x1_ESMFmesh_cdf5_100621.nc'

        content = f"""&clm_inparm
 albice = 0.50,0.30
 co2_ppmv = 367.0
 co2_type = 'constant'
 collapse_urban = .false.
 compname = 'clm2'
 convert_ocean_to_land = .true.
 create_crop_landunit = .true.
 crop_fsat_equals_zero = .false.
 do_sno_oc = .false.
 downscale_hillslope_meteorology = .false.
 finidat = ''
 flush_gdd20 = .false.
 for_testing_no_crop_seed_replenishment = .false.
 for_testing_run_ncdiopio_tests = .false.
 for_testing_use_repr_structure_pool = .false.
 for_testing_use_second_grain_pool = .false.
 fsnowaging = '{fsnowaging}'
 fsnowoptics = '{fsnowoptics}'
 fsurdat = '{surfdata}'
 glc_do_dynglacier = .false.
 glc_snow_persistence_max_days = 0
 h2osno_max = 10000.0
 hillslope_fsat_equals_zero = .false.
 hist_fields_list_file = .false.
 hist_fincl1 = 'QRUNOFF','QOVER','QDRAI','QFLX_EVAP_TOT','EFLX_LH_TOT','H2OSNO','SNOWDP','FSNO','SOILWATER_10CM','TWS','RAIN','SNOW'
 hist_mfilt = {hist_mfilt}
 hist_nhtfrq = {hist_nhtfrq}
 hist_wrt_matrixcn_diag = .false.
 irrigate = .false.
 maxpatch_glc = 10
 n_dom_landunits = 0
 n_dom_pfts = 0
 nlevsno = 12
 nsegspc = 35
 o3_veg_stress_method = 'unset'
 paramfile = '{paramfile}'
 run_zero_weight_urban = .false.
 snicar_dust_optics = 'sahara'
 snicar_numrad_snw = 5
 snicar_snobc_intmix = .false.
 snicar_snodst_intmix = .false.
 snicar_snw_shape = 'sphere'
 snicar_solarspec = 'mid_latitude_winter'
 snicar_use_aerosol = .true.
 snow_cover_fraction_method = 'SwensonLawrence2012'
 snow_thermal_cond_method = 'Jordan1991'
 soil_layerstruct_predefined = '20SL_8.5m'
 toosmall_crop = 0.d00
 toosmall_glacier = 0.d00
 toosmall_lake = 0.d00
 toosmall_soil = 0.d00
 toosmall_urban = 0.d00
 toosmall_wetland = 0.d00
 use_bedrock = .true.
 use_cn = .false.
 use_crop = .false.
 use_excess_ice = .false.
 use_fates = .false.
 use_fertilizer = .false.
 use_fun = .false.
 use_grainproduct = .false.
 use_hillslope = .false.
 use_hillslope_routing = .false.
 use_hydrstress = .true.
 use_init_interp = .false.
 use_lai_streams = .false.
 use_lch4 = .false.
 use_luna = .true.
 use_matrixcn = .false.
 use_nitrif_denitrif = .false.
 use_snicar_frc = .false.
 use_soil_matrixcn = .false.
 use_soil_moisture_streams = .false.
 use_subgrid_fluxes = .true.
 use_z0m_snowmelt = .false.
 z0param_method = 'ZengWang2007'
/
&ndepdyn_nml
/
&popd_streams
/
&urbantv_streams
 stream_fldfilename_urbantv = '{urban_tv}'
 stream_meshfile_urbantv = '{urban_mesh}'
 stream_year_first_urbantv = 2000
 stream_year_last_urbantv = 2000
 urbantvmapalgo = 'nn'
/
&light_streams
/
&soil_moisture_streams
/
&lai_streams
/
&atm2lnd_inparm
 glcmec_downscale_longwave = .true.
 lapse_rate = 0.006
 lapse_rate_longwave = 0.032
 longwave_downscaling_limit = 0.5
 precip_repartition_glc_all_rain_t = 0.
 precip_repartition_glc_all_snow_t = -2.
 precip_repartition_nonglc_all_rain_t = 2.
 precip_repartition_nonglc_all_snow_t = 0.
 repartition_rain_snow = .true.
/
&lnd2atm_inparm
 melt_non_icesheet_ice_runoff = .true.
/
&clm_canopyhydrology_inparm
 use_clm5_fpi = .true.
/
&cnphenology
 generate_crop_gdds = .false.
 use_mxmat = .true.
/
&cropcal_streams
 cropcals_rx = .false.
 cropcals_rx_adapt = .false.
 stream_gdd20_seasons = .false.
/
&clm_soilhydrology_inparm
/
&dynamic_subgrid
 reset_dynbal_baselines = .false.
/
&cnvegcarbonstate
/
&finidat_consistency_checks
/
&dynpft_consistency_checks
/
&clm_initinterp_inparm
 init_interp_method = 'general'
/
&century_soilbgcdecompcascade
/
&soilhydrology_inparm
 baseflow_scalar = 0.001d00
/
&luna
/
&friction_velocity
 zetamaxstable = 0.5d00
/
&mineral_nitrogen_dynamics
/
&soilwater_movement_inparm
 dtmin = 60.
 expensive = 42
 flux_calculation = 1
 inexpensive = 1
 lower_boundary_condition = 2
 soilwater_movement_method = 1
 upper_boundary_condition = 1
 verysmall = 1.e-8
 xtolerlower = 1.e-2
 xtolerupper = 1.e-1
/
&rooting_profile_inparm
 rooting_profile_method_carbon = 1
 rooting_profile_method_water = 1
/
&soil_resis_inparm
 soil_resis_method = 1
/
&bgc_shared
/
&canopyfluxes_inparm
 itmax_canopy_fluxes = 40
 use_biomass_heat_storage = .false.
 use_undercanopy_stability = .false.
/
&aerosol
/
&clmu_inparm
 building_temp_method = 1
 urban_explicit_ac = .false.
 urban_hac = 'ON_WASTEHEAT'
 urban_traffic = .false.
/
&clm_soilstate_inparm
 organic_frac_squared = .false.
/
&clm_nitrogen
 lnc_opt = .false.
/
&clm_snowhydrology_inparm
 lotmp_snowdensity_method = 'Slater2017'
 reset_snow = .false.
 reset_snow_glc = .false.
 reset_snow_glc_ela = 1.e9
 snow_dzmax_l_1 = 0.03d00
 snow_dzmax_l_2 = 0.07d00
 snow_dzmax_u_1 = 0.02d00
 snow_dzmax_u_2 = 0.05d00
 snow_dzmin_1 = 0.010d00
 snow_dzmin_2 = 0.015d00
 snow_overburden_compaction_method = 'Vionnet2012'
 wind_dependent_snow_density = .true.
/
&hillslope_hydrology_inparm
 hillslope_head_gradient_method = 'Darcy'
 hillslope_transmissivity_method = 'LayerSum'
/
&hillslope_properties_inparm
 hillslope_pft_distribution_method = 'Standard'
 hillslope_soil_profile_method = 'Uniform'
/
&cnprecision_inparm
/
&clm_glacier_behavior
 glacier_region_behavior = 'single_at_atm_topo','UNSET','virtual','multiple'
 glacier_region_ice_runoff_behavior = 'melted','UNSET','remains_ice','remains_ice'
 glacier_region_melt_behavior = 'remains_in_place','UNSET','replaced_by_ice','replaced_by_ice'
/
&crop_inparm
/
&irrigation_inparm
 irrig_depth = 0.6
 irrig_length = 14400
 irrig_method_default = 'drip'
 irrig_min_lai = 0.0
 irrig_start_time = 21600
 irrig_target_smp = -3400.
 irrig_threshold_fraction = 1.0
 limit_irrigation_if_rof_enabled = .false.
 use_groundwater_irrigation = .false.
/
&surfacealbedo_inparm
 snowveg_affects_radiation = .true.
/
&water_tracers_inparm
 enable_water_isotopes = .false.
 enable_water_tracer_consistency_checks = .false.
/
&tillage_inparm
 tillage_mode = 'off'
/
&ctsm_nuopc_cap
/
&clm_humanindex_inparm
 calc_human_stress_indices = 'FAST'
/
&cnmresp_inparm
/
&photosyns_inparm
 leafresp_method = 0
 light_inhibit = .true.
 modifyphoto_and_lmr_forcrop = .true.
 rootstem_acc = .false.
 stomatalcond_method = 'Medlyn2011'
/
&cnfire_inparm
/
&cn_general
/
&nitrif_inparm
/
&lifire_inparm
/
&ch4finundated
/
&exice_streams
/
&clm_temperature_inparm
 excess_ice_coldstart_depth = 0.5
 excess_ice_coldstart_temp = -1.0
/
&soilbgc_decomp
 soil_decomp_method = 'None'
/
&clm_canopy_inparm
 leaf_mr_vcm = 0.015d00
/
&prigentroughness
 use_prigent_roughness = .false.
/
&zendersoilerod
/
&scf_swenson_lawrence_2012_inparm
 int_snow_max = 2000.
 n_melt_glcmec = 10.0d00
/
"""
        (self.pp.settings_dir / 'lnd_in').write_text(content)
        logger.debug("Generated lnd_in")

    def _write_drv_in(self) -> None:
        """Generate NUOPC-format drv_in."""
        content = """&debug_inparm
  create_esmf_pet_files = .false.
/
&papi_inparm
  papi_ctr1_str = "PAPI_FP_OPS"
  papi_ctr2_str = "PAPI_NO_CTR"
  papi_ctr3_str = "PAPI_NO_CTR"
  papi_ctr4_str = "PAPI_NO_CTR"
/
&prof_inparm
  profile_add_detail = .false.
  profile_barrier = .false.
  profile_depth_limit = 4
  profile_detail_limit = 2
  profile_disable = .false.
  profile_global_stats = .true.
  profile_outpe_num = 1
  profile_outpe_stride = 0
  profile_ovhd_measurement = .false.
  profile_papi_enable = .false.
  profile_single_file = .false.
  profile_timer = 4
/
"""
        (self.pp.settings_dir / 'drv_in').write_text(content)
        logger.debug("Generated drv_in")

    def _write_drv_flds_in(self) -> None:
        """Generate drv_flds_in."""
        megan_file = (self.cesm_inputdata / 'atm' / 'cam' / 'chem'
                      / 'trop_mozart' / 'emis'
                      / 'megan21_emis_factors_78pft_c20161108.nc')

        content = f"""&dust_emis_inparm
  dust_emis_method = 'Zender_2003'
  zender_soil_erod_source = 'atm'
/
&megan_emis_nl
  megan_factors_file = '{megan_file}'
  megan_specifier = 'ISOP = isoprene',
      'C10H16 = pinene_a + carene_3 + thujene_a', 'CH3OH = methanol',
      'C2H5OH = ethanol', 'CH2O = formaldehyde', 'CH3CHO = acetaldehyde',
      'CH3COOH = acetic_acid', 'CH3COCH3 = acetone'
/
&ozone_coupling_nl
  atm_ozone_frequency = 'multiday_average'
/
"""
        (self.pp.settings_dir / 'drv_flds_in').write_text(content)
        logger.debug("Generated drv_flds_in")

    def _build_case_path(self) -> Path:
        """Resolve the case belonging to the installed binary, including symlink installs."""
        install = self.pp._get_install_path()
        for exe in (install / 'bin' / 'cesm.exe', install / 'cesm.exe'):
            if exe.exists():
                candidate = exe.resolve().parent.parent
                if (candidate / 'run' / 'fd.yaml').is_file():
                    return candidate
        return install / 'cases' / 'symfluence_build'

    def _copy_fd_yaml(self) -> None:
        """Copy fd.yaml field dictionary from CLM install."""
        install_path = self.pp._get_install_path()
        fd_src = self._build_case_path() / 'run' / 'fd.yaml'

        if fd_src.exists():
            shutil.copy2(fd_src, self.pp.settings_dir / 'fd.yaml')
            logger.debug("Copied fd.yaml from build")
        else:
            logger.warning(f"fd.yaml not found at {fd_src}")

    def _write_caseroot(self) -> None:
        """Write CASEROOT pointer file."""
        install_path = self.pp._get_install_path()
        case_dir = self._build_case_path()
        (self.pp.settings_dir / 'CASEROOT').write_text(str(case_dir) + '\n')
        logger.debug("Generated CASEROOT")

    def _write_user_nl_clm(self, ctx: dict) -> None:
        """Write user_nl_clm (for calibration worker to modify)."""
        surfdata = ctx['surfdata_file']
        paramfile = ctx['params_file']
        hist_nhtfrq = ctx['hist_nhtfrq']
        hist_mfilt = ctx['hist_mfilt']

        content = f"""! CLM5 namelist for {self.pp.domain_name}
! Generated by SYMFLUENCE CLM preprocessor

fsurdat = '{surfdata}'
paramfile = '{paramfile}'
finidat = ''

hist_nhtfrq = {hist_nhtfrq}
hist_mfilt = {hist_mfilt}
hist_fincl1 = 'QRUNOFF','QOVER','QDRAI','QFLX_EVAP_TOT','EFLX_LH_TOT','H2OSNO','SNOWDP','FSNO','SOILWATER_10CM','TWS','RAIN','SNOW'

use_init_interp = .false.
"""
        (self.pp.settings_dir / 'user_nl_clm').write_text(content)
        logger.debug("Generated user_nl_clm")
