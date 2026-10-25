"""Changing output experiment IDs must not replace real geometry with defaults."""
from __future__ import annotations

import logging

import geopandas as gpd
import pytest
from shapely.geometry import box

from symfluence.core.config.models.root import SymfluenceConfig
from symfluence.core.exceptions import SYMFLUENCEError
from symfluence.models.crhm.preprocessor import CRHMPreProcessor
from symfluence.models.gsflow.preprocessor import GSFLOWPreProcessor


@pytest.mark.parametrize('model,cls',[('GSFLOW',GSFLOWPreProcessor),('CRHM',CRHMPreProcessor)])
def test_geometry_survives_new_experiment_id(tmp_path,model,cls):
    config=SymfluenceConfig(**{'CATCHMENT_AREA':200.,'DOMAIN_NAME':'geometry_test','EXPERIMENT_ID':'new_calibration','SYMFLUENCE_DATA_DIR':str(tmp_path),'HYDROLOGICAL_MODEL':model,'FORCING_DATASET':'CARRA','DOMAIN_DEFINITION_METHOD':'lumped','SUB_GRID_DISCRETIZATION':'elevation','SYMFLUENCE_CODE_DIR':str(tmp_path),'EXPERIMENT_TIME_START':'2010-01-01','EXPERIMENT_TIME_END':'2010-12-31'})
    d=tmp_path/'domain_geometry_test/shapefiles/catchment/lumped/original';d.mkdir(parents=True)
    shp=d/'geometry_test_HRUs_elevation.shp'
    gpd.GeoDataFrame({'elev_mean':[700.]},geometry=[box(490000,7100000,510000,7110000)],crs='EPSG:32627').to_file(shp)
    p=cls(config,logging.getLogger('test'))
    assert not p.get_catchment_path().exists()
    props=p._get_catchment_properties()
    assert props['area_km2']==pytest.approx(200.,rel=1e-5)
    assert 63 < props['lat'] < 65
    assert -22 < props['lon'] < -18
    mismatched = config.model_copy(update={'domain':config.domain.model_copy(update={'catchment_area':100.})})
    with pytest.raises(SYMFLUENCEError,match='requires readable catchment geometry|disagrees with CATCHMENT_AREA'):
        cls(mismatched,logging.getLogger('test'))._get_catchment_properties()
    # Missing geometry must fail, not change back to a nominal 100 km2 basin.
    shp.unlink()
    with pytest.raises(SYMFLUENCEError,match='requires readable catchment geometry'):
        p._get_catchment_properties()
