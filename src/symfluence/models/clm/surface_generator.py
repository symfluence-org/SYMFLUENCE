# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Surface Data and Parameter Generator

Generates the CLM5 surface data file (surfdata_clm.nc) and copies
the default parameter file (clm5_params.nc).
"""
import logging
import shutil

import numpy as np

logger = logging.getLogger(__name__)


class CLMSurfaceGenerator:
    """Generates CLM surface data and copies parameter files.

    Parameters
    ----------
    preprocessor : CLMPreProcessor
        Parent preprocessor instance providing config, paths, and
        geometry helpers.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    # MODIS IGBP → CLM5 natural PFT mapping (indices 0-14)
    IGBP_TO_CLM_PFT = {
        1:  {1: 80, 12: 20},
        2:  {4: 80, 13: 20},
        3:  {3: 80, 12: 20},
        4:  {7: 60, 13: 30, 0: 10},
        5:  {1: 30, 7: 30, 13: 30, 0: 10},
        6:  {10: 60, 13: 30, 0: 10},
        7:  {10: 30, 13: 30, 0: 40},
        8:  {7: 40, 13: 40, 0: 20},
        9:  {13: 50, 7: 20, 0: 30},
        10: {13: 70, 0: 30},
        11: {13: 60, 0: 20, 10: 20},
        12: {13: 40, 0: 10},
        13: {0: 60, 13: 40},
        14: {13: 45, 7: 20, 0: 10},
        15: {0: 100},
        16: {0: 100},
        17: {0: 100},
    }
    _BOREAL_LAT = 50.0
    _TROPICAL_LAT = 23.5

    def _get_pft_distribution(self, lat):
        """Derive CLM5 PFT fractions from MODIS IGBP land cover GeoTIFF."""
        try:
            import rasterio
            lc_dir = self.pp.project_dir / 'data' / 'attributes' / 'landclass'
            lc_files = list(lc_dir.glob('*.tif')) if lc_dir.exists() else []
            if not lc_files:
                raise FileNotFoundError("No land cover GeoTIFF found")

            with rasterio.open(lc_files[0]) as src:
                data = src.read(1)
                valid = data[data > 0]
                if len(valid) == 0:
                    raise ValueError("Empty land cover raster")
                unique, counts = np.unique(valid, return_counts=True)
                total = counts.sum()

            pft_pct = np.zeros(15)
            for igbp_class, count in zip(unique, counts):
                frac = count / total
                mapping = self.IGBP_TO_CLM_PFT.get(int(igbp_class), {0: 100})
                for pft_idx, pct in mapping.items():
                    pft_pct[pft_idx] += frac * pct

            abs_lat = abs(lat)
            if abs_lat < self._TROPICAL_LAT:
                pft_pct[4] += pft_pct[1]; pft_pct[1] = 0
                pft_pct[6] += pft_pct[7]; pft_pct[7] = 0
                pft_pct[14] += pft_pct[13] * 0.7; pft_pct[13] *= 0.3
            elif abs_lat > self._BOREAL_LAT:
                pft_pct[2] += pft_pct[1]; pft_pct[1] = 0
                pft_pct[11] += pft_pct[10]; pft_pct[10] = 0
            else:
                if abs_lat < 35:
                    pft_pct[14] += pft_pct[13] * 0.6; pft_pct[13] *= 0.4

            total_pct = pft_pct.sum()
            if total_pct > 0:
                pft_pct = pft_pct / total_pct * 100.0

            pft_names = ['bare', 'NET_temp', 'NET_boreal', 'NDB',
                         'BET_trop', 'BET_temp', 'BDT_trop', 'BDT_temp',
                         'BDT_boreal', 'BES_temp', 'BDS_temp', 'BDS_boreal',
                         'C3_arctic', 'C3_grass', 'C4_grass']
            active = [(pft_names[i], pft_pct[i]) for i in range(15) if pft_pct[i] > 0.5]
            logger.info(f"PFT distribution from MODIS (lat={lat:.1f}): "
                        + ", ".join(f"{n}={p:.0f}%" for n, p in active))
            return pft_pct
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not derive PFTs from MODIS: {e}; using defaults")
            return self._default_pft_distribution(lat)

    def _default_pft_distribution(self, lat):
        """Fallback PFT distribution when MODIS data is unavailable."""
        pft_pct = np.zeros(15)
        abs_lat = abs(lat)
        if abs_lat < self._TROPICAL_LAT:
            pft_pct[0] = 5; pft_pct[4] = 60; pft_pct[14] = 35
        elif abs_lat > self._BOREAL_LAT:
            pft_pct[0] = 5; pft_pct[2] = 60; pft_pct[12] = 35
        else:
            pft_pct[0] = 5; pft_pct[1] = 60; pft_pct[13] = 35
        return pft_pct

    def _get_soil_fractions(self):
        """Load sand/clay percentages from attributes store or config.

        Returns:
            Tuple of (pct_sand, pct_clay) as floats.
        """
        default_sand, default_clay = 45.0, 20.0

        # Try config overrides first
        pct_sand = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.pct_sand,
            default=None, dict_key='CLM_PCT_SAND'
        )
        pct_clay = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.pct_clay,
            default=None, dict_key='CLM_PCT_CLAY'
        )
        if pct_sand is not None and pct_clay is not None:
            return float(pct_sand), float(pct_clay)

        # Try model-ready attributes NetCDF
        try:
            import xarray as xr
            attrs_dir = self.pp.project_dir / 'data' / 'model_ready' / 'attributes'
            attrs_files = list(attrs_dir.glob('*_attributes.nc')) if attrs_dir.exists() else []
            if attrs_files:
                ds = xr.open_dataset(attrs_files[0])
                for sand_key in ['sand_frac', 'PCT_SAND', 'sand_0-5cm_mean']:
                    if sand_key in ds:
                        default_sand = float(ds[sand_key].values.mean())
                        break
                for clay_key in ['clay_frac', 'PCT_CLAY', 'clay_0-5cm_mean']:
                    if clay_key in ds:
                        default_clay = float(ds[clay_key].values.mean())
                        break
                ds.close()
                logger.info(f"Loaded soil fractions from attributes: sand={default_sand:.1f}%, clay={default_clay:.1f}%")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.debug(f"Could not load soil attributes: {e}")

        return default_sand, default_clay

    def generate_surface_data(self) -> None:
        """Generate CLM5 surface data file.

        Uses netCDF4 directly (not xarray) to ensure all required dimensions
        exist in the file even without attached variables.  CLM5.3 checks
        for dimension existence via ncd_inq_dimid at several init stages.

        Required dimensions: lsmlat, lsmlon, natpft(15), cft(2), lsmpft(17),
        nlevsoi(10), nlevurb(5), nglcec(10), nglcecp1(11), time(12).

        Required variables beyond basic land cover/soil:
        - PCT_CFT, PCT_GLC_MEC, TOPO_GLC_MEC, GLACIER_REGION
        - zbedrock, mxsoil_color, gdp, peatf, abm, LAKEDEPTH
        - EF1_BTR/FET/FDT/SHR/GRS/CRP (MEGAN emission factors)
        - MONTHLY_LAI/SAI/HEIGHT_TOP/HEIGHT_BOT (satellite phenology)

        Coordinates MUST match the ESMF mesh center exactly.
        """
        import netCDF4 as nc4

        lat, lon, _ = self.pp.domain_generator.get_catchment_centroid()
        pct_sand, pct_clay = self._get_soil_fractions()
        mean_elev = self.pp.domain_generator.get_mean_elevation()

        surfdata_file = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.surfdata_file,
            default='surfdata_clm.nc', dict_key='CLM_SURFDATA_FILE'
        )
        filepath = self.pp.params_dir / surfdata_file
        ds = nc4.Dataset(str(filepath), 'w', format='NETCDF4_CLASSIC')

        # -- Dimensions (all required by CLM5.3 init checks) --
        ds.createDimension('lsmlat', 1)
        ds.createDimension('lsmlon', 1)
        ds.createDimension('natpft', 15)
        ds.createDimension('cft', 2)
        ds.createDimension('lsmpft', 17)  # natpft + cft
        ds.createDimension('nlevsoi', 10)
        ds.createDimension('nlevurb', 5)
        ds.createDimension('nglcec', 10)
        ds.createDimension('nglcecp1', 11)
        ds.createDimension('time', 12)

        # -- Global attributes --
        ds.Dataset_Version = np.float32(5.3)
        ds.title = f'CLM surface data for {self.pp.domain_name}'
        ds.source = 'SYMFLUENCE'

        # Helper to create 2D scalar fields
        def add_2d(name, val, dtype='f8', **attrs):
            v = ds.createVariable(name, dtype, ('lsmlat', 'lsmlon'))
            v[:] = val
            for k, a in attrs.items():
                setattr(v, k, a)

        # -- Spatial coordinates (must match ESMF mesh center) --
        add_2d('LATIXY', lat, units='degrees_north')
        add_2d('LONGXY', lon, units='degrees_east')
        add_2d('LANDFRAC_PFT', 1.0)
        add_2d('PFTDATA_MASK', 1, dtype='i4')

        # -- Land cover fractions --
        add_2d('PCT_NATVEG', 100.0)
        add_2d('PCT_CROP', 0.0)
        add_2d('PCT_LAKE', 0.0)
        add_2d('PCT_WETLAND', 0.0)
        add_2d('PCT_GLACIER', 0.0)
        add_2d('PCT_OCEAN', 0.0)
        add_2d('PCT_URBAN', 0.0)

        # PFT distribution from MODIS land cover
        pft_dist = self._get_pft_distribution(lat)
        pct_nat = np.zeros((1, 1, 15))
        pct_nat[0, 0, :] = pft_dist
        v = ds.createVariable('PCT_NAT_PFT', 'f8', ('lsmlat', 'lsmlon', 'natpft'))
        v[:] = pct_nat

        # CFT distribution (must sum to 100 within crop landunit)
        pct_cft = np.zeros((1, 1, 2))
        pct_cft[0, 0, 0] = 100.0
        v = ds.createVariable('PCT_CFT', 'f8', ('lsmlat', 'lsmlon', 'cft'))
        v[:] = pct_cft

        # -- Topography (from DEM) --
        std_elev, mean_slope = self.pp.domain_generator.get_elevation_stats()
        add_2d('FMAX', 0.5)
        add_2d('TOPO', mean_elev, units='m')
        add_2d('STD_ELEV', std_elev, units='m')
        add_2d('SLOPE', mean_slope)

        # -- Soil properties (10 layers) --
        n_sl = 10
        for vname, val, kw in [
            ('PCT_SAND', pct_sand, {'units': '%'}),
            ('PCT_CLAY', pct_clay, {'units': '%'}),
            ('watsat', 0.45, {}),
            ('hksat', 0.005, {'units': 'mm/s'}),
            ('sucsat', 200.0, {'units': 'mm'}),
            ('bsw', 6.0, {}),
        ]:
            v = ds.createVariable(vname, 'f8', ('lsmlat', 'lsmlon', 'nlevsoi'))
            v[:] = np.full((1, 1, n_sl), val)
            for k, a in kw.items():
                setattr(v, k, a)

        # Organic matter -- decreases with depth
        org_vals = np.zeros((1, 1, n_sl))
        org_vals[0, 0, 0:3] = [30.0, 15.0, 5.0]
        v = ds.createVariable('ORGANIC', 'f8', ('lsmlat', 'lsmlon', 'nlevsoi'))
        v[:] = org_vals
        v.units = 'kg/m3'

        add_2d('SOIL_COLOR', 15, dtype='i4')
        add_2d('zbedrock', 2.0, units='m')

        # -- Glacier elevation classes (must sum to 100) --
        pct_glc = np.zeros((1, 1, 11))
        pct_glc[0, 0, 0] = 100.0
        v = ds.createVariable('PCT_GLC_MEC', 'f8', ('lsmlat', 'lsmlon', 'nglcecp1'))
        v[:] = pct_glc
        v = ds.createVariable('TOPO_GLC_MEC', 'f8', ('lsmlat', 'lsmlon', 'nglcecp1'))
        v[:] = np.zeros((1, 1, 11))
        add_2d('GLACIER_REGION', 0, dtype='i4')

        # -- Scalars and fire/VOC data --
        v = ds.createVariable('mxsoil_color', 'i4', ())
        v[:] = 20
        add_2d('gdp', 40.0)
        add_2d('peatf', 0.0)
        add_2d('abm', 7, dtype='i4')
        add_2d('LAKEDEPTH', 10.0, units='m')

        # MEGAN isoprene emission factors
        for ef_name, ef_val in [('EF1_BTR', 10000.), ('EF1_FET', 2000.),
                                 ('EF1_FDT', 2000.), ('EF1_SHR', 4000.),
                                 ('EF1_GRS', 800.), ('EF1_CRP', 1.)]:
            add_2d(ef_name, ef_val, units='ug/m2/hr')

        # -- Monthly vegetation data for satellite phenology (SP) --
        lai_net = [2.5, 2.5, 2.5, 3.0, 3.5, 4.0, 4.5, 4.5, 4.0, 3.5, 3.0, 2.5]
        lai_bet = [5.0, 5.0, 5.0, 5.2, 5.5, 5.8, 6.0, 6.0, 5.8, 5.5, 5.2, 5.0]
        lai_bdt = [0.5, 0.5, 1.0, 2.0, 3.5, 5.0, 5.5, 5.0, 4.0, 2.5, 1.0, 0.5]
        lai_shrub = [0.2, 0.2, 0.3, 0.5, 1.0, 1.5, 1.8, 1.8, 1.2, 0.6, 0.3, 0.2]
        lai_c3g = [0., 0., 0., 0.2, 0.5, 1.2, 1.5, 1.5, 0.8, 0.3, 0., 0.]
        lai_c4g = [0., 0., 0., 0.3, 0.8, 1.8, 2.5, 2.5, 1.5, 0.5, 0., 0.]

        pft_lai = {1: lai_net, 2: lai_net, 3: lai_net,
                   4: lai_bet, 5: lai_bet, 6: lai_bdt, 7: lai_bdt, 8: lai_bdt,
                   9: lai_shrub, 10: lai_shrub, 11: lai_shrub,
                   12: lai_c3g, 13: lai_c3g, 14: lai_c4g}
        pft_sai = {i: [1.0]*12 for i in range(1, 9)}
        pft_sai.update({i: [0.5]*12 for i in range(9, 15)})
        pft_htop = {1: 17., 2: 17., 3: 14., 4: 35., 5: 20., 6: 20., 7: 18.,
                    8: 14., 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5}
        pft_hbot = {k: v * 0.5 if v > 1.0 else 0.1 for k, v in pft_htop.items()}

        active_pfts = [i for i in range(15) if pft_dist[i] > 0.5]
        lai_map = {i: pft_lai.get(i, [0.]*12) for i in active_pfts if i > 0}
        sai_map = {i: pft_sai.get(i, [0.5]*12) for i in active_pfts if i > 0}
        htop_map = {i: [pft_htop.get(i, 0.5)]*12 for i in active_pfts if i > 0}
        hbot_map = {i: [pft_hbot.get(i, 0.1)]*12 for i in active_pfts if i > 0}

        for vname, pft_vals, attrs in [
            ('MONTHLY_LAI', lai_map,
             {'units': 'm^2/m^2', 'long_name': 'monthly leaf area index'}),
            ('MONTHLY_SAI', sai_map,
             {'units': 'm^2/m^2', 'long_name': 'monthly stem area index'}),
            ('MONTHLY_HEIGHT_TOP', htop_map,
             {'units': 'm', 'long_name': 'monthly vegetation height top'}),
            ('MONTHLY_HEIGHT_BOT', hbot_map,
             {'units': 'm', 'long_name': 'monthly vegetation height bottom'}),
        ]:
            data = np.zeros((12, 17, 1, 1))
            for pft_idx, monthly_vals in pft_vals.items():
                for m in range(12):
                    data[m, pft_idx, 0, 0] = monthly_vals[m]
            v = ds.createVariable(vname, 'f8',
                                  ('time', 'lsmpft', 'lsmlat', 'lsmlon'))
            v[:] = data
            for k, a in attrs.items():
                setattr(v, k, a)

        ds.close()
        logger.info(f"Generated CLM surface data: {filepath}")

    def copy_default_params(self) -> None:
        """Copy default clm5_params.nc to the project parameters directory.

        Resolution order:
        1. Bundled resource (symfluence/resources/base_settings/CLM/)
        2. CLM install path (<install>/share/clm5_params.nc)
        3. Error with download URL
        """
        from symfluence.resources import get_base_settings_dir

        params_file = self.pp._get_config_value(
            lambda: self.pp.config.model.clm.params_file,
            default='clm5_params.nc', dict_key='CLM_PARAMS_FILE'
        )
        params_dst = self.pp.params_dir / params_file

        # 1. Bundled resource
        try:
            bundled = get_base_settings_dir('CLM') / 'clm5_params.nc'
            if bundled.exists():
                shutil.copy2(bundled, params_dst)
                logger.info(f"Copied bundled CLM parameters: {params_dst}")
                return
        except FileNotFoundError:
            pass

        # 2. CLM install path
        install_path = self.pp._get_install_path()
        params_src = install_path / 'share' / 'clm5_params.nc'
        if params_src.exists():
            shutil.copy2(params_src, params_dst)
            logger.info(f"Copied CLM parameters from install: {params_dst}")
            return

        # 3. Not found
        logger.error(
            "clm5_params.nc not found in bundled resources or CLM install. "
            "Download from: https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/"
            "inputdata/lnd/clm2/paramdata/ and place in the CLM parameters directory."
        )
