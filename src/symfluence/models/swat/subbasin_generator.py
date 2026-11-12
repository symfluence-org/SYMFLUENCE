# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SWAT Sub-basin File Generator

Generates sub-basin (.sub), HRU (.hru), groundwater (.gw), management (.mgt),
and soil (.sol) files required by SWAT.
"""
from __future__ import annotations

import logging

from .parameters import DEFAULT_PARAMS

logger = logging.getLogger(__name__)


class SWATSubbasinGenerator:
    """Generates sub-basin-level SWAT input files.

    Args:
        preprocessor: Parent SWATPreProcessor instance providing access
            to config, logger, paths, and helper methods.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    def generate_subbasin_files(self) -> None:
        """Generate sub-basin, HRU, groundwater, management, and soil files.

        The .sub file must conform to the layout expected by both
        ``getallo.f`` and ``readsub.f``.

        ``getallo.f`` does::

            do j = 1, 52
              read(25,6000) titldum      ! skip 52 lines
            end do
            read(25,*) numhru            ! line 53 = number of HRUs
            do j = 1, 8
              read(25,6000) titldum      ! skip 8 lines (54-61)
            end do
            call hruallo                 ! reads HRU file refs at line 62+

        ``readsub.f`` reads every line sequentially (title, SUB_KM,
        lat, elev, gage codes, WGN filename, elevation bands, lapse
        rates, channel params, pond/wus files, climate-change
        adjustments, then hrutot on line 53, 8 skip lines, and HRU
        file references on line 62+).

        ``hruallo.f`` reads each HRU line with format ``(4a13,52x,i6)``
        (4 filenames of 13 chars, 52-char gap, then an integer).
        ``readsub.f`` reads with format ``(8a13,i6)`` (8 filenames of
        13 chars, then an integer).  Both consume the same physical
        line, so filenames must sit in exact 13-char columns.
        """
        props = self.pp._get_catchment_properties()

        # Helper: format a 10-value elevation-band line (format 10f8.1)
        def fmt_10f81(vals):
            return ''.join(f"{v:8.1f}" for v in vals)

        # Helper: format a 6-value monthly-adjustment line (format 10f8.1)
        def fmt_6f81(vals):
            return ''.join(f"{v:8.1f}" for v in vals)

        # ------------------------------------------------------------------
        # Generate auxiliary files referenced by .sub
        # ------------------------------------------------------------------
        self.pp.basin_generator.generate_wgn_file(props)
        self.pp.basin_generator.generate_pnd_stub()
        self.pp.basin_generator.generate_wus_stub()
        self.pp.basin_generator.generate_chm_stub()

        # ------------------------------------------------------------------
        # Build .sub file -- exactly 62+ lines
        # ------------------------------------------------------------------
        sub_path = self.pp.txtinout_dir / '000010001.sub'

        zeros10 = [0.0] * 10
        zeros6  = [0.0] * 6

        sub_lines = []

        # Line  1: Title
        sub_lines.append(" Subbasin: 1 -- SYMFLUENCE generated")

        # Line  2: SUB_KM (read with read *)
        sub_lines.append(
            f"  {props['area_km2']:14.4f}    | SUB_KM : Subbasin area [km2]")

        # Line  3: skip line (titldum) -- readsub reads this as titldum
        #          (in isproj==3 branch it reads harg_petco etc; else just titldum)
        sub_lines.append(
            f"  {0.0023:14.4f}    | HARG_PETCO / or skip line")

        # Line  4: skip line (titldum)
        sub_lines.append(" Subbasin climate and channel data")

        # Line  5: SUB_LAT
        sub_lines.append(
            f"  {props['lat']:14.4f}    | SUB_LAT : Latitude [deg]")

        # Line  6: SUB_ELEV
        sub_lines.append(
            f"  {props['elev']:14.4f}    | SUB_ELEV : Mean elevation [m]")

        # Line  7: IRGAGE (rain gage code)
        sub_lines.append(
            f"  {1:14d}    | IRGAGE : Precipitation gage data code")

        # Line  8: ITGAGE (temp gage code)
        sub_lines.append(
            f"  {1:14d}    | ITGAGE : Temperature gage data code")

        # Line  9: ISGAGE (solar gage code)
        sub_lines.append(
            f"  {0:14d}    | ISGAGE : Solar radiation gage data code")

        # Line 10: IHGAGE (humidity gage code)
        sub_lines.append(
            f"  {0:14d}    | IHGAGE : Relative humidity gage data code")

        # Line 11: IWGAGE (wind gage code)
        sub_lines.append(
            f"  {0:14d}    | IWGAGE : Wind speed gage data code")

        # Line 12: WGNFILE -- read with format (8a13,i6) => first 13 chars
        sub_lines.append(
            f"{'000010001.wgn':13s}"
            f"             | WGNFILE : Weather generator file")

        # Line 13: FCST_REG (forecast region)
        sub_lines.append(
            f"  {0:14d}    | FCST_REG : Forecast region")

        # Line 14: skip (elevation band header)
        sub_lines.append(" Elevation Band Data")

        # Line 15: skip (titldum before elevb values)
        sub_lines.append(" Elevation of band center [m]:")

        # Line 16: ELEVB(1:10) -- format 10f8.1
        sub_lines.append(fmt_10f81(zeros10))

        # Line 17: skip (titldum)
        sub_lines.append(" Fraction of subbasin area in band:")

        # Line 18: ELEVB_FR(1:10) -- format 10f8.1
        sub_lines.append(fmt_10f81(zeros10))

        # Line 19: skip (titldum)
        sub_lines.append(" Snow water equivalent in band [mm]:")

        # Line 20: SNOEB(1:10) -- format 10f8.1
        sub_lines.append(fmt_10f81(zeros10))

        # Line 21: PLAPS (precip lapse rate)
        plaps = self.pp._get_config_value(lambda: self.pp.config.model.swat.plaps, dict_key='SWAT_PLAPS')
        sub_lines.append(
            f"  {plaps:14.4f}    | PLAPS : Precip lapse rate [mm H2O/km]")

        # Line 22: TLAPS (temp lapse rate)
        tlaps = self.pp._get_config_value(lambda: self.pp.config.model.swat.tlaps, dict_key='SWAT_TLAPS')
        sub_lines.append(
            f"  {tlaps:14.4f}    | TLAPS : Temp lapse rate [deg C/km]")

        # Line 23: SNO_SUB (initial snow)
        sub_lines.append(
            f"  {0.0:14.4f}    | SNO_SUB : Initial snow water [mm H2O]")

        # Line 24: skip (titldum -- channel header)
        sub_lines.append(" Channel data:")

        # Line 25: CH_L1 (longest tributary channel length)
        sub_lines.append(
            f"  {1.0:14.4f}    | CH_L1 : Longest trib channel length [km]")

        # Line 26: CH_S1 (average slope of tributary channel)
        sub_lines.append(
            f"  {0.05:14.4f}    | CH_S1 : Avg slope of trib channel [m/m]")

        # Line 27: CH_W1 (average width of tributary channel)
        sub_lines.append(
            f"  {10.0:14.4f}    | CH_W1 : Avg width of trib channel [m]")

        # Line 28: CH_K1 (effective hydraulic conductivity)
        sub_lines.append(
            f"  {0.0:14.4f}    | CH_K1 : Eff hydraulic conductivity [mm/hr]")

        # Line 29: CH_N1 (Manning's n for tributary channel)
        sub_lines.append(
            f"  {0.014:14.4f}    | CH_N1 : Manning's n for trib channel")

        # Line 30: skip (titldum -- pond header)
        sub_lines.append(" Pond/Wetland data:")

        # Line 31: PNDFILE -- read with format (8a13,i6) => first 13 chars
        sub_lines.append(
            f"{'000010001.pnd':13s}"
            f"             | PNDFILE : Pond input file")

        # Line 32: skip (titldum -- water use header)
        sub_lines.append(" Water Use data:")

        # Line 33: WUSFILE -- read with format (8a13,i6) => first 13 chars
        sub_lines.append(
            f"{'000010001.wus':13s}"
            f"             | WUSFILE : Water use file")

        # Line 34: SNOFILE / Climate Change -- read as titldum/snofile
        sub_lines.append("              ")  # 13+ blank chars = no snow file

        # Line 35: CO2
        sub_lines.append(
            f"  {330.0:14.4f}    | CO2 : CO2 concentration [ppmv]")

        # Line 36: skip (titldum -- rainfall adj header)
        sub_lines.append(" Rainfall adjustment (months 1-6):")

        # Line 37: RFINC(1:6) -- format 10f8.1
        sub_lines.append(fmt_6f81(zeros6))

        # Line 38: skip (titldum)
        sub_lines.append(" Rainfall adjustment (months 7-12):")

        # Line 39: RFINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 40: skip (titldum -- temp adj header)
        sub_lines.append(" Temperature adjustment (months 1-6):")

        # Line 41: TMPINC(1:6)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 42: skip (titldum)
        sub_lines.append(" Temperature adjustment (months 7-12):")

        # Line 43: TMPINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 44: skip (titldum -- radiation adj header)
        sub_lines.append(" Solar radiation adjustment (months 1-6):")

        # Line 45: RADINC(1:6)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 46: skip (titldum)
        sub_lines.append(" Solar radiation adjustment (months 7-12):")

        # Line 47: RADINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 48: skip (titldum -- humidity adj header)
        sub_lines.append(" Humidity adjustment (months 1-6):")

        # Line 49: HUMINC(1:6)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 50: skip (titldum)
        sub_lines.append(" Humidity adjustment (months 7-12):")

        # Line 51: HUMINC(7:12)
        sub_lines.append(fmt_6f81(zeros6))

        # Line 52: skip (titldum -- HRU section header)
        sub_lines.append(" HRU data:")

        # ---- Line 53: NUMBER OF HRUs (critical for getallo.f) ----
        sub_lines.append(
            f"  {1:14d}    | HRUTOT : Number of HRUs in subbasin")

        # Lines 54-61: 8 skip/header lines before HRU file references
        sub_lines.append(
            " HRU: Num  Luse   Soil         Slope        Fraction")
        sub_lines.append(
            "  1  FRST  SoilA  0-9999       1.000")
        sub_lines.append(
            f"  {'FRST':14s}    | LUSE : Land use")
        sub_lines.append(
            f"  {'SoilA':14s}    | SOIL : Soil type")
        sub_lines.append(
            f"  {'0-9999':14s}    | SLOPE_CD : Slope class")
        sub_lines.append(
            f"  {1.0:14.4f}    | HRU_FR : Fraction of subbasin in HRU")
        sub_lines.append(
            f"  {props['slope']:14.4f}    | HRU_SLP : Average slope steepness [m/m]")
        sub_lines.append(
            f"  {50.0:14.4f}    | OV_N : Manning's n for overland flow")

        # ---- Line 62: HRU file references ----
        # Format (8a13,i6) for readsub: hrufile, mgtfile, solfile,
        #   chmfile, gwfile, opsfile, septfile, sdrfile, ils2
        # Format (4a13,52x,i6) for hruallo: hrufile, mgtfile, solfile,
        #   chmfile, (skip 52 covering gw+ops+sep+sdr), ils2
        # Both read the same 110-char line.
        hru_ref_line = (
            f"{'000010001.hru':13s}"
            f"{'000010001.mgt':13s}"
            f"{'000010001.sol':13s}"
            f"{'000010001.chm':13s}"
            f"{'000010001.gw':13s}"
            f"{'':13s}"  # opsfile (blank)
            f"{'':13s}"  # septfile (blank)
            f"{'':13s}"  # sdrfile (blank)
            f"{0:6d}"    # ils2 (landscape routing flag)
        )
        sub_lines.append(hru_ref_line)

        # Verify line count (should be at least 62)
        assert len(sub_lines) >= 62, (
            f".sub file has {len(sub_lines)} lines, need >= 62"
        )

        sub_path.write_text('\n'.join(sub_lines) + '\n', encoding='utf-8')
        logger.info(
            f"Sub-basin file written: {sub_path} ({len(sub_lines)} lines)")

        # Generate HRU, GW, MGT, SOL files
        self._generate_hru_file()
        self._generate_gw_file()
        self._generate_mgt_file()
        self._generate_sol_file()

        logger.info(f"Generated sub-basin, HRU, GW, MGT, SOL files in {self.pp.txtinout_dir}")

    def _generate_hru_file(self) -> None:
        """Generate the HRU file (.hru) -- readhru.f reads ~53 values.

        Line 1 is title, then all remaining are free-format (* reads).
        """
        hru_path = self.pp.txtinout_dir / '000010001.hru'
        props = self.pp._get_catchment_properties()
        hru_lines = [
            " HRU:1 Subbasin:1 -- SYMFLUENCE generated",
            # Line  2: HRU_FR
            f"  {1.0:14.4f}    | HRU_FR : Fraction of subbasin area",
            # Line  3: SLSUBBSN
            f"  {91.46:14.4f}    | SLSUBBSN : Avg slope length [m]",
            # Line  4: HRU_SLP (mean catchment slope from the DEM)
            f"  {props['slope']:14.4f}    | HRU_SLP : Avg slope steepness [m/m]",
            # Line  5: OV_N
            f"  {0.14:14.4f}    | OV_N : Manning's n for overland flow",
            # Line  6: LAT_TTIME
            f"  {0.0:14.4f}    | LAT_TTIME : Lateral flow travel time [days]",
            # Line  7: LAT_SED
            f"  {0.0:14.4f}    | LAT_SED : Sediment conc in lateral flow [mg/l]",
            # Line  8: SLSOIL
            f"  {0.0:14.4f}    | SLSOIL : Slope length for lateral subsurface flow [m]",
            # Line  9: CANMX
            f"  {0.0:14.4f}    | CANMX : Maximum canopy storage [mm H2O]",
            # Line 10: ESCO
            f"  {DEFAULT_PARAMS['ESCO']:14.4f}    | ESCO : Soil evaporation compensation factor",
            # Line 11: EPCO
            f"  {0.95:14.4f}    | EPCO : Plant uptake compensation factor",
            # Line 12: RSDIN
            f"  {0.0:14.4f}    | RSDIN : Initial residue cover [kg/ha]",
            # Line 13: ERORGN
            f"  {0.0:14.4f}    | ERORGN : Organic N enrichment ratio",
            # Line 14: ERORGP
            f"  {0.0:14.4f}    | ERORGP : Organic P enrichment ratio",
            # Line 15: POT_FR
            f"  {0.0:14.4f}    | POT_FR : Fraction draining to pothole",
            # Line 16: FLD_FR
            f"  {0.0:14.4f}    | FLD_FR : Fraction of HRU in floodplain",
            # Line 17: RIP_FR
            f"  {0.0:14.4f}    | RIP_FR : Fraction of HRU in riparian zone",
            # Line 18: title
            " Pothole data:",
            # Line 19: POT_TILEMM
            f"  {0.0:14.4f}    | POT_TILEMM",
            # Line 20: POT_VOLXMM
            f"  {0.0:14.4f}    | POT_VOLXMM",
            # Line 21: POT_VOLMM
            f"  {0.0:14.4f}    | POT_VOLMM",
            # Line 22: POT_NSED
            f"  {0.0:14.4f}    | POT_NSED",
            # Line 23: POT_NO3L
            f"  {0.0:14.4f}    | POT_NO3L",
            # Line 24: DEP_IMP
            f"  {6000.0:14.4f}    | DEP_IMP : Depth to impervious layer [mm]",
            # Line 25: title
            " Urban data:",
            # Line 26: title
            " Consumptive water use:",
            # Line 27: title
            " Tile drain:",
            # Line 28: EVPOT
            f"  {0.0:14.4f}    | EVPOT",
            # Line 29: DIS_STREAM
            f"  {0.0:14.4f}    | DIS_STREAM : Distance to stream [m]",
            # Line 30: CF (concentration factor)
            f"  {0.0:14.4f}    | CF",
            # Line 31: CFH
            f"  {0.0:14.4f}    | CFH",
            # Line 32: CFDEC
            f"  {0.0:14.4f}    | CFDEC",
            # Line 33: SED_CON
            f"  {0.0:14.4f}    | SED_CON : Sediment concentration [mg/l]",
            # Line 34: ORGN_CON
            f"  {0.0:14.4f}    | ORGN_CON : Organic N concentration [mg/l]",
            # Line 35: ORGP_CON
            f"  {0.0:14.4f}    | ORGP_CON : Organic P concentration [mg/l]",
            # Line 36: SOLN_CON
            f"  {0.0:14.4f}    | SOLN_CON : Soluble N concentration [mg/l]",
            # Line 37: SOLP_CON
            f"  {0.0:14.4f}    | SOLP_CON : Soluble P concentration [mg/l]",
            # Line 38: POT_SOLPL
            f"  {0.0:14.4f}    | POT_SOLPL",
            # Line 39: POT_K
            f"  {0.0:14.4f}    | POT_K",
            # Line 40: N_REDUC
            f"  {0.0:14.4f}    | N_REDUC",
            # Line 41: N_LAG
            f"  {0.0:14.4f}    | N_LAG",
            # Line 42: N_LN
            f"  {0.0:14.4f}    | N_LN",
            # Line 43: N_LNCO
            f"  {0.0:14.4f}    | N_LNCO",
            # Line 44: SURLAG (HRU-level override)
            f"  {DEFAULT_PARAMS['SURLAG']:14.4f}    | SURLAG : Surface runoff lag coefficient",
            # Line 45: R2ADJ
            f"  {0.0:14.4f}    | R2ADJ",
            # Line 46: CMN
            f"  {0.0003:14.4f}    | CMN",
            # Line 47: CDN
            f"  {1.4:14.4f}    | CDN",
            # Line 48: NPERCO
            f"  {0.20:14.4f}    | NPERCO",
            # Line 49: PHOSKD
            f"  {175.0:14.4f}    | PHOSKD",
            # Line 50: PSP
            f"  {0.4:14.4f}    | PSP",
            # Line 51: SDNCO
            f"  {1.1:14.4f}    | SDNCO",
            # Line 52: IWETILE (integer)
            f"  {0:14d}    | IWETILE",
            # Line 53: IWETGW (integer)
            f"  {0:14d}    | IWETGW",
        ]
        hru_path.write_text('\n'.join(hru_lines) + '\n', encoding='utf-8')

    def _generate_gw_file(self) -> None:
        """Generate the groundwater file (.gw) -- readgw.f reads title + 17 data values.

        Line 1: title; Lines 2-18: free-format read * values.
        """
        gw_path = self.pp.txtinout_dir / '000010001.gw'
        gw_lines = [
            " Groundwater parameters -- SYMFLUENCE generated",
            # Line  2: SHALLST (must exceed GWQMN for immediate baseflow)
            f"  {1000.0:14.4f}    | SHALLST : Initial shallow aquifer storage [mm]",
            # Line  3: DEEPST
            f"  {2000.0:14.4f}    | DEEPST : Initial deep aquifer storage [mm]",
            # Line  4: GW_DELAY (= delay)
            f"  {DEFAULT_PARAMS['GW_DELAY']:14.4f}    | GW_DELAY : Groundwater delay time [days]",
            # Line  5: ALPHA_BF
            f"  {DEFAULT_PARAMS['ALPHA_BF']:14.4f}    | ALPHA_BF : Baseflow alpha factor [1/days]",
            # Line  6: GWQMN
            f"  {DEFAULT_PARAMS['GWQMN']:14.4f}    | GWQMN : Threshold depth for return flow [mm]",
            # Line  7: GW_REVAP
            f"  {DEFAULT_PARAMS['GW_REVAP']:14.4f}    | GW_REVAP : Groundwater revap coefficient",
            # Line  8: REVAPMN
            f"  {500.0:14.4f}    | REVAPMN : Threshold depth for revap [mm]",
            # Line  9: RCHRG_DP
            f"  {0.05:14.4f}    | RCHRG_DP : Deep aquifer percolation fraction",
            # Line 10: GWHT
            f"  {10.0:14.4f}    | GWHT : Initial groundwater height [m]",
            # Line 11: GW_SPYLD
            f"  {0.003:14.4f}    | GW_SPYLD : Specific yield of shallow aquifer [m3/m3]",
            # Line 12: SHALLST_N
            f"  {0.0:14.4f}    | SHALLST_N : Initial NO3 in shallow aquifer [mg/l]",
            # Line 13: GWMINP (=gwsolp in some versions)
            f"  {0.0:14.4f}    | GWMINP : Minimum P in groundwater [mg/l]",
            # Line 14: HLIFE_NGW
            f"  {0.0:14.4f}    | HLIFE_NGW : Half-life of NO3 in shallow aquifer [days]",
            # Line 15: LAT_ORGN (NEW)
            f"  {0.0:14.4f}    | LAT_ORGN : Organic N in lateral flow [mg/l]",
            # Line 16: LAT_ORGP (NEW)
            f"  {0.0:14.4f}    | LAT_ORGP : Organic P in lateral flow [mg/l]",
            # Line 17: ALPHA_BF_D (NEW)
            f"  {0.0:14.4f}    | ALPHA_BF_D : Baseflow alpha factor for deep aquifer [1/days]",
            # Line 18: GWNO3_AQ (NEW -- read with iostat=eof, optional)
            f"  {0.0:14.4f}    | GWNO3_AQ : NO3 concentration in groundwater [mg/l]",
        ]
        gw_path.write_text('\n'.join(gw_lines) + '\n', encoding='utf-8')

    def _generate_mgt_file(self) -> None:
        """Generate the management file (.mgt) -- readmgt.f reads exactly 30 lines.

        hruallo.f skips 30 lines before reading management operations.
        Line 1: title, then pairs of (data, title) or (data) reads.
        All data reads use free-format (read *).
        """
        mgt_path = self.pp.txtinout_dir / '000010001.mgt'
        props = self.pp._get_catchment_properties()
        mgt_lines = [
            # Line  1: title
            " Management parameters -- SYMFLUENCE generated",
            # Line  2: NMGT (management schedule number)
            f"  {0:14d}    | NMGT : Management schedule number",
            # Line  3: title
            " Plant growth:",
            # Line  4: IGRO (land cover status, 0=no crop growing)
            f"  {1:14d}    | IGRO : Land cover status (1=growing)",
            # Line  5: NCRP (land cover code -- ID in plant.dat)
            f"  {0:14d}    | NCRP : Land cover code from crop.dat",
            # Line  6: LAIDAY (leaf area index)
            f"  {0.0:14.4f}    | LAIDAY : Current leaf area index",
            # Line  7: BIO_MS (biomass, kg/ha)
            f"  {0.0:14.4f}    | BIO_MS : Biomass [kg/ha]",
            # Line  8: PHU_PLT (heat units to maturity)
            f"  {0.0:14.4f}    | PHU_PLT : Heat units to maturity",
            # Line  9: title
            " General management:",
            # Line 10: BIOMIX
            f"  {0.20:14.4f}    | BIOMIX : Biological mixing efficiency",
            # Line 11: CN2 (from the dominant land cover / hydrologic group)
            f"  {props['cn2']:14.4f}    | CN2 : Initial SCS CN for moisture condition II",
            # Line 12: USLE_P
            f"  {1.0:14.4f}    | USLE_P : USLE support practice factor",
            # Line 13: BIO_MIN
            f"  {0.0:14.4f}    | BIO_MIN : Min biomass for grazing [kg/ha]",
            # Line 14: FILTERW
            f"  {0.0:14.4f}    | FILTERW : Width of edge-of-field filter strip [m]",
            # Line 15: title
            " Urban:",
            # Line 16: IURBAN
            f"  {0:14d}    | IURBAN : Urban simulation code",
            # Line 17: URBLU
            f"  {0:14d}    | URBLU : Urban land type",
            # Line 18: title
            " Irrigation:",
            # Line 19: IRRSC
            f"  {0:14d}    | IRRSC : Irrigation code",
            # Line 20: IRRNO
            f"  {0:14d}    | IRRNO : Irrigation source code",
            # Line 21: FLOWMIN
            f"  {0.0:14.4f}    | FLOWMIN : Min in-stream flow for irrigation [m3/s]",
            # Line 22: DIVMAX
            f"  {0.0:14.4f}    | DIVMAX : Max irrigation diversion [mm]",
            # Line 23: FLOWFR
            f"  {0.0:14.4f}    | FLOWFR : Fraction of flow allowed for diversion",
            # Line 24: title
            " Tile drain:",
            # Line 25: DDRAIN
            f"  {0.0:14.4f}    | DDRAIN : Depth to subsurface drain [mm]",
            # Line 26: TDRAIN
            f"  {0.0:14.4f}    | TDRAIN : Time to drain to field capacity [hr]",
            # Line 27: GDRAIN
            f"  {0.0:14.4f}    | GDRAIN : Drain tile lag time [hr]",
            # Line 28: title
            " Management operations:",
            # Line 29: title (was NROT in older versions, now just title)
            " Rotation:",
            # Line 30: title
            " Schedule:",
        ]
        mgt_path.write_text('\n'.join(mgt_lines) + '\n', encoding='utf-8')

    def _generate_sol_file(self) -> None:
        """Generate the soil file (.sol) -- readsol.f reads with fixed-width formats.

        Line 1: title (a80)
        Line 2: SNAM -- format (12x,a16)
        Line 3: HYDGRP -- format (24x,a1)
        Line 4: SOL_ZMX -- format (28x,f12.2)
        Line 5: ANION_EXCL -- format (51x,f5.3)
        Line 6: title (skip)
        Lines 7+: data rows -- format (27x,15f12.2)
          Each row has a 27-char label then 12.2-float values per layer
        Variables: SOL_Z, SOL_BD, SOL_AWC, SOL_K, SOL_CBN, CLAY,
          SILT, SAND, ROCK, SOL_ALB, USLE_K, SOL_EC, PH, CACO3
        (14 data rows for 1 soil layer)
        """
        sol_path = self.pp.txtinout_dir / '000010001.sol'

        # Two-layer soil profile (mountain catchment defaults)
        # Layer 1: Topsoil (organic-rich, moderate conductivity)
        # Layer 2: Subsoil (denser, lower conductivity)
        sol_zmx = 1000.0    # Max rooting depth [mm]

        # Per-layer values: [layer1, layer2]
        sol_z =    [300.0,  1000.0]   # Depth to bottom of layer [mm]
        sol_bd =   [1.35,   1.55]     # Moist bulk density [Mg/m3]
        sol_awc =  [0.20,   0.14]     # Available water capacity [mm/mm]
        sol_k =    [18.0,   5.0]      # Saturated hydraulic conductivity [mm/hr]
        sol_cbn =  [2.5,    0.5]      # Organic carbon content [%]
        clay_pct = [12.0,   22.0]     # Clay content [%]
        silt_pct = [35.0,   33.0]     # Silt content [%]
        sand_pct = [53.0,   45.0]     # Sand content [%]
        rock_pct = [8.0,    15.0]     # Rock fragment content [%]
        sol_alb =  [0.10,   0.10]     # Moist soil albedo
        usle_k =   [0.28,   0.28]     # USLE soil erodibility factor
        sol_ec =   [0.0,    0.0]      # Electrical conductivity [dS/m]
        ph =       [6.0,    6.8]      # Soil pH
        caco3 =    [0.0,    0.0]      # CaCO3 content [%]

        def sol_data_row(label, values):
            """Format a .sol data row: 27-char label + 12.2 float per layer."""
            return f"{label:27s}" + ''.join(f"{v:12.2f}" for v in values)

        sol_lines = [
            # Line 1: title (a80)
            " Soil data -- SYMFLUENCE generated                                             ",
            # Line 2: SNAM -- format (12x,a16) -- 12 skip chars then 16-char name
            f"{'':12s}{'MountainSoil':16s}",
            # Line 3: HYDGRP -- format (24x,a1) -- 24 skip chars then 1 char
            f"{'':24s}{'B':1s}",
            # Line 4: SOL_ZMX -- format (28x,f12.2) -- 28 skip chars then f12.2
            f"{'':28s}{sol_zmx:12.2f}",
            # Line 5: ANION_EXCL -- format (51x,f5.3) -- 51 skip chars then f5.3
            f"{'':51s}{0.500:5.3f}",
            # Line 6: title (skip line)
            " Soil layer data:",
            # Lines 7+: data rows -- format (27x,15f12.2)
            sol_data_row(" SOL_Z(mm)               :", sol_z),
            sol_data_row(" SOL_BD(Mg/m**3)         :", sol_bd),
            sol_data_row(" SOL_AWC(mm/mm)           :", sol_awc),
            sol_data_row(" SOL_K(mm/hr)            :", sol_k),
            sol_data_row(" SOL_CBN(%)              :", sol_cbn),
            sol_data_row(" CLAY(%)                 :", clay_pct),
            sol_data_row(" SILT(%)                 :", silt_pct),
            sol_data_row(" SAND(%)                 :", sand_pct),
            sol_data_row(" ROCK(%)                 :", rock_pct),
            sol_data_row(" SOL_ALB                 :", sol_alb),
            sol_data_row(" USLE_K                  :", usle_k),
            sol_data_row(" SOL_EC(dS/m)            :", sol_ec),
            sol_data_row(" PH                      :", ph),
            sol_data_row(" CACO3                   :", caco3),
        ]
        sol_path.write_text('\n'.join(sol_lines) + '\n', encoding='utf-8')
