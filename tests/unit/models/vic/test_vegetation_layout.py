"""The native image driver reserves the last vegetation class for bare soil."""
from __future__ import annotations

import numpy as np
import xarray as xr

from symfluence.models.vic.preprocessor import VICPreProcessor


def test_generated_library_reserves_bare_soil(tmp_path):
    pre = object.__new__(VICPreProcessor)
    pre.params_dir = tmp_path
    pre.domain_name = 'test'
    pre._get_config_value = lambda *args, default=None, **kwargs: default
    pre._get_catchment_properties = lambda: {'elev': 700.}
    pre._compute_elevation_bands = lambda n_bands: {'elevations': [600.,800.], 'area_fracs':[.5,.5], 'pfactors':[1.,1.]}
    xr.Dataset({'mask':(('lat','lon'),[[1]])},coords={'lat':[64.], 'lon':[-19.]}).to_netcdf(tmp_path/'vic_domain.nc')
    pre._generate_parameter_file()
    with xr.open_dataset(tmp_path/'vic_params.nc') as ds:
        assert ds.sizes['veg_class'] == 2
        np.testing.assert_array_equal(ds.Cv.values.ravel(), [1.,0.])
        assert ds.Nveg.item() == int((ds.Cv.isel(veg_class=slice(None,-1))>0).sum()) == 1
        assert (ds.LAI.isel(veg_class=0)>0).all()
        assert ds.LAI.isel(veg_class=-1).max() == 0
        assert ds.displacement.isel(veg_class=-1).max() == 0
        assert (ds.veg_rough.isel(veg_class=-1)>0).all()
        assert ds.root_fract.isel(veg_class=-1).sum() == 0
        assert ds.root_fract.isel(veg_class=0).sum() == 1
