"""Generated GSFLOW soil/cell mappings (moved from the PRMS soil tests)."""
from types import SimpleNamespace as NS
import pytest
from symfluence.models.gsflow.preprocessor import GSFLOWPreProcessor

PROPS=dict(lat=64.,lon=-19.,elev=700.,area_km2=1145.,slope=5.,aspect=0.)
def values(path):
    body=path.read_text().split('** Parameters **')[1]
    out={}
    for b in body.split('####'):
        x=b.strip().splitlines()
        if x:
            n=int(x[1]);out[x[0]]=[float(v) for v in x[4+n:]]
    return out

@pytest.mark.parametrize('distributed',[True,False])
def test_gsflow_mapping_covers_cells_and_partitions_hrus(tmp_path,distributed):
    p=object.__new__(GSFLOWPreProcessor)
    p.distributed_gw=distributed;p.gw_grid_n=5;p.setup_dir=tmp_path
    p._get_catchment_properties=lambda:PROPS
    p._generate_parameter_file();v=values(tmp_path/'params.dat')
    assert all(a==1 for a in v['gvr_cell_pct'])
    if distributed:
        assert sum(v['gvr_hru_pct'])==pytest.approx(1)
        assert len(v['gvr_hru_pct'])==25
        assert all(a>0 for a in v['gvr_hru_pct'])
    assert 'soil_rechr_max' not in v
    assert v['soil_rechr_max_frac'][0]==pytest.approx(1/3)

