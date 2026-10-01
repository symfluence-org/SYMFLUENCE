"""Check generated glacier setup and report paired daily discharge metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from symfluence.core.config.models.root import SymfluenceConfig
from symfluence.models.hype.cryosphere import read_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline', action='store_true', help='Check run_model output instead of final calibration')
    args = parser.parse_args()
    c = SymfluenceConfig.from_file(args.config)
    d = Path(c.system.data_dir) / f'domain_{c.domain.name}'
    calibrated = 'calibrate_model' in c.system.workflow_steps and not args.baseline
    out = (d/'optimization/HYPE'/('dds_'+c.domain.experiment_id)/'final_evaluation'
           if calibrated else d/'simulations'/c.domain.experiment_id/'HYPE')
    geo = pd.read_csv(d/'settings/HYPE/GeoData.txt', sep=r'\s+')
    classes = pd.read_csv(d/'settings/HYPE/GeoClass.txt', sep=r'\s+', comment='!', header=None)
    slc = [x for x in geo if x.startswith('SLC_')]
    assert np.allclose(geo[slc].sum(axis=1), 1., atol=1e-5), 'SLC fractions must sum to one'
    ice_classes = classes.loc[classes.iloc[:, 7] == 3].iloc[:, 0].astype(int)
    assert len(ice_classes), 'No native glacier class (special code 3)'
    ice = geo[[f'SLC_{i}' for i in ice_classes]].sum(axis=1)
    fraction = float(np.dot(ice, geo.area)/geo.area.sum())
    assert abs(fraction-c.model.hype.glacier_fraction)<1e-5
    outlet = geo.loc[geo.maindown == 0, 'subid'].astype(int).tolist()
    assert len(outlet) == 1, 'Expected one outlet'
    q = read_output(out, 'COUT')[str(outlet[0])]
    start = pd.Timestamp(c.domain.time_start).normalize()+pd.Timedelta(days=c.model.hype.spinup_days)
    end = pd.Timestamp(c.domain.time_end).normalize()
    assert q.index.equals(pd.date_range(start, end)), 'Unexpected daily output coverage'
    assert np.isfinite(q).all() and (q >= 0).all()
    for var in ['GLCA','GLCV']:
        # Native HYPE uses missing sentinels for glacier diagnostics on land-only bands.
        f = read_output(out, var)[geo.loc[ice > 0, 'subid'].astype(int).astype(str).tolist()]
        assert np.isfinite(f.to_numpy()).all() and (f.to_numpy() >= 0).all()
        assert (f.sum(axis=1) > 0).all()
    result = {'output':str(out),'days':len(q),'native_area_km2':float(geo.area.sum()/1e6),'initial_glacier_fraction':fraction,'periods':{}}
    if calibrated:
        obs = pd.read_csv(next((d/'data/observations/streamflow/preprocessed').glob('*.csv')), index_col=0, parse_dates=True).discharge_cms
        for name,a,b,n in [('calibration','2012-10-01','2016-09-30',1461),('evaluation','2016-10-01','2021-09-30',1826),('dry_transfer','2021-10-01','2023-09-30',730)]:
            pair = pd.concat([q.rename('s'),obs.rename('o')],axis=1).loc[a:b].dropna()
            assert len(pair)==n and np.isfinite(pair.to_numpy()).all()
            s,o=pair.s,pair.o
            result['periods'][name]={'days':n,'KGE':float(1-np.sqrt((s.corr(o)-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)),'bias_percent':float(100*(s.sum()/o.sum()-1))}
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
