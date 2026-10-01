# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Optional LamaH-Ice constraints for a single glacierised HYPE watershed.

Scores are maximized, with 1 for a perfect match. Error scales are objective
normalizations, not observational uncertainty estimates. Annual glacier balance
uses fixed Sep 30 boundaries as an approximation to field balance years.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError


def ensure_observations(project_dir, station_id, logger):
    """Request basin auxiliary observations through the registered provider."""
    from symfluence.core.registries import R
    handler = R.observation_handlers.get('lamah_ice_streamflow')
    return handler.ensure_cryosphere(project_dir, station_id, logger)


def read_output(directory, variable):
    frame = pd.read_csv(Path(directory) / f'time{variable}.txt', sep=r'\s+', skiprows=1)
    frame = frame.set_index(pd.to_datetime(frame.pop('DATE')))
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        raise ValidationError(f'Duplicate HYPE {variable} dates')
    return frame.astype(float).replace(-9999, np.nan)


def calculate_constraints(output_dir, project_dir, period, snow_scale=0.2, mb_scale=1.0):
    """Score calibration dates only; fail closed on missing or invalid inputs.

    CFSC excludes glacier ice but is normalized by HYPE land area. For the
    supported land/glacier-only class setup it is converted to snow fraction of
    the dynamically ice-free basin. GLCV is km3 ice; glacdens is fixed at 0.85.
    Only whole balance years with both endpoints inside the requested period
    are used, so no evaluation observations enter optimization.
    """
    if not period:
        raise ValidationError('Cryosphere constraints require an explicit calibration period')
    start, end = [pd.Timestamp(v.strip()) for v in str(period).split(',')]
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    obs_dir = project_dir / 'data/observations/cryosphere'
    setup = project_dir / 'settings/HYPE'
    geo = pd.read_csv(setup / 'GeoData.txt', sep=r'\s+').set_index('subid')
    geo.index = geo.index.astype(str)
    classes = pd.read_csv(setup / 'GeoClass.txt', sep=r'\s+', comment='!', header=None)
    if not set(classes.iloc[:, 7].astype(int)).issubset({0, 3}):
        raise ValidationError('Cryosphere snow-area normalization currently supports land and glacier classes only')
    if snow_scale <= 0 or mb_scale <= 0:
        raise ValidationError('Cryosphere error scales must be positive')
    area = geo['area'] / 1e6
    glacier_ids = geo.index[geo.glacier_fraction > 0].tolist()
    if not glacier_ids:
        raise ValidationError('No active glacier area for mass-balance constraint')
    snow = read_output(output_dir, 'CFSC').loc[start:end, area.index]
    ga = read_output(output_dir, 'GLCA').loc[snow.index, glacier_ids]
    gv = read_output(output_dir, 'GLCV').loc[start:end, glacier_ids]
    if not np.isfinite(snow.to_numpy()).all() or not np.isfinite(ga.to_numpy()).all() or not np.isfinite(gv.to_numpy()).all():
        raise ValidationError('Missing/nonfinite cryosphere model outputs')
    nonice_area = area.sum() - ga.sum(axis=1)
    if (nonice_area <= 0).any() or ((snow < 0) | (snow > 1.001)).any().any():
        raise ValidationError('Invalid HYPE snow or glacier area')
    modeled_snow = snow.mul(area, axis=1).sum(axis=1) / nonice_area
    modis = pd.read_csv(obs_dir / 'modis_fractional_snow_cover_and_glacier_albedo.csv', sep=';')
    dates = pd.to_datetime(dict(year=modis.YYYY, month=modis.MM, day=modis.DD))
    observed_snow = pd.Series(modis.fsca_outside_glaciers.to_numpy() / 100, index=dates)
    pair = pd.concat([modeled_snow.rename('sim'), observed_snow.loc[start:end].rename('obs')], axis=1).dropna()
    pair = pair.loc[pair.obs.between(0, 1)]
    if len(pair) < 30:
        raise ValidationError('Fewer than 30 matching calibration snow-cover observations')
    snow_rmse = float(np.sqrt(np.mean((pair.sim - pair.obs)**2)))
    observed_mb = pd.read_csv(obs_dir / 'glacier_timeseries.csv', sep=';', index_col=0)
    years = []
    volume = gv.sum(axis=1)
    for year, row in observed_mb.iterrows():
        left, right = pd.Timestamp(int(str(year))-1, 9, 30), pd.Timestamp(int(str(year)), 9, 30)
        if left < start or right > end:
            continue
        if left not in volume.index or right not in volume.index:
            raise ValidationError(f'Missing glacier balance endpoint for {year}')
        if not np.isfinite(row.annual_net_MB) or not np.isfinite(row.g_area_dyn) or row.g_area_dyn <= 0:
            continue
        # Water-equivalent volume change / observed glacier area -> m w.e.
        simulated = (volume.loc[right] - volume.loc[left]) * 0.85 * 1000 / row.g_area_dyn
        years.append(dict(year=int(str(year)), simulated_mwe=float(simulated), observed_mwe=float(row.annual_net_MB)))
    if not years:
        raise ValidationError('No complete calibration glacier balance year')
    mb_rmse = float(np.sqrt(np.mean([(r['simulated_mwe']-r['observed_mwe'])**2 for r in years])))
    metrics = dict(snow_score=1-snow_rmse/snow_scale, glacier_score=1-mb_rmse/mb_scale,
                   snow_cover_rmse=snow_rmse, glacier_mb_rmse_mwe=mb_rmse,
                   snow_observation_count=len(pair), glacier_balance_year_count=len(years))
    (output_dir / 'cryosphere_metrics.json').write_text(json.dumps(dict(
        period=str(period), metrics=metrics, annual_glacier_balance=years,
        balance_year_assumption='Fixed September 30 endpoints; approximate field balance years',
        objective_scales=dict(snow_fraction=snow_scale, glacier_mwe=mb_scale)), indent=2))
    return metrics
