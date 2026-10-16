# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Land-model adapters for MODFLOW groundwater coupling.

A coupled run splits the land model's runoff in two: deep drainage leaves the
land model as groundwater recharge and is routed through MODFLOW, and
everything else (surface, interflow and the land model's own river routing)
stays as the land model's *fast flow*. Total streamflow is fast flow plus
MODFLOW's groundwater discharge to the river.

Each adapter knows, for one land model, how to

* ``enable_recharge_export``: configure its settings so deep drainage leaves
  the model instead of returning to its rivers;
* ``recharge_zones``: give the spatial units that recharge is reported for
  (GeoDataFrame with ``zone_id`` and the model's own ``area_m2``);
* ``read_recharge``: read daily *local* recharge per zone in m/d;
* ``read_fast_flow``: read daily outlet fast flow in m3/s.

Register a new land model with ``@register_land_adapter('NAME')``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd

from symfluence.core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

_ADAPTERS: Dict[str, Type['LandRechargeAdapter']] = {}


def register_land_adapter(name: str):
    def deco(cls):
        _ADAPTERS[name.upper()] = cls
        return cls
    return deco


def available_land_adapters() -> list:
    return sorted(_ADAPTERS)


def get_land_adapter(name: str, options: Optional[Dict[str, Any]] = None) -> 'LandRechargeAdapter':
    try:
        cls = _ADAPTERS[str(name).upper()]
    except KeyError:
        raise ConfigurationError(
            f"No MODFLOW land adapter for '{name}'; available: {available_land_adapters()}") from None
    return cls(options or {})


def adapter_options(config) -> Dict[str, Any]:
    """Adapter options from a typed SymfluenceConfig or a flat ``MODFLOW_*`` dict."""
    keys = {'zone_shapefile': 'LAND_ZONE_SHAPEFILE', 'zone_id_field': 'LAND_ZONE_ID_FIELD',
            'outlet_id': 'LAND_OUTLET_ID', 'basin_shapefile': 'SUBBASIN_SHAPEFILE',
            'recharge_variable': 'RECHARGE_VARIABLE'}
    if isinstance(config, dict):
        return {k: config.get(f'MODFLOW_{v}') for k, v in keys.items()}
    mf = getattr(getattr(config, 'model', None), 'modflow', None)
    return {k: getattr(mf, v.lower(), None) for k, v in keys.items()} if mf is not None else {}


def resolve_land_model(config, default: str = 'SUMMA') -> str:
    """Land model feeding MODFLOW.

    LAND_SURFACE_MODEL, else the (first) HYDROLOGICAL_MODEL, else
    MODFLOW_COUPLING_SOURCE. HYDROLOGICAL_MODEL precedes the coupling source
    because a flattened typed config always carries the latter's default.
    """
    if not isinstance(config, dict):
        config = config.to_dict(flatten=True) if hasattr(config, 'to_dict') else {}
    hydro = str(config.get('HYDROLOGICAL_MODEL') or '').split(',')[0].strip()
    for name in (config.get('LAND_SURFACE_MODEL'), hydro, config.get('MODFLOW_COUPLING_SOURCE')):
        if name and str(name).strip().upper() not in ('MODFLOW', 'NONE', 'DEFAULT'):
            return str(name).strip().upper()
    return default


class LandRechargeAdapter(ABC):
    """Interface between one land model and the MODFLOW groundwater model.

    ``options`` carries adapter settings (``zone_shapefile``, ``zone_id_field``,
    ``outlet_id``, ...); unknown keys are ignored.
    """
    name = 'BASE'

    def __init__(self, options: Dict[str, Any]):
        self.options = dict(options)

    @abstractmethod
    def enable_recharge_export(self, settings_dir: Path) -> None:
        """Edit the land model's settings so deep drainage leaves the model."""

    @abstractmethod
    def recharge_zones(self, settings_dir: Path):
        """GeoDataFrame: ``zone_id``, ``area_m2`` (land-model area), geometry."""

    @abstractmethod
    def read_recharge(self, output_dir: Path, settings_dir: Path) -> pd.DataFrame:
        """Daily local recharge, m/d, one column per zone id."""

    @abstractmethod
    def read_fast_flow(self, output_dir: Path, settings_dir: Path) -> pd.Series:
        """Daily outlet streamflow that bypasses groundwater, m3/s."""

    def set_output_dir(self, settings_dir: Path, output_dir: Path) -> None:
        """Point the land model's output at ``output_dir`` (no-op when the
        caller configures it, e.g. SUMMA's fileManager)."""

    def _zone_shapes(self, settings_dir: Path):
        import geopandas as gpd
        path = self.options.get('zone_shapefile')
        if not path:
            raise ConfigurationError(f'{self.name} coupling needs MODFLOW_LAND_ZONE_SHAPEFILE')
        field = self.options.get('zone_id_field') or 'HRU_ID'
        gdf = gpd.read_file(path)
        if field not in gdf:
            raise ConfigurationError(f'Zone shapefile {path} has no column {field}')
        return gdf.rename(columns={field: 'zone_id'})[['zone_id', 'geometry']]


# ---------------------------------------------------------------------------
# HYPE
# ---------------------------------------------------------------------------

def _read_hype_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', skiprows=1, index_col=0, parse_dates=True)
    df.columns = [int(float(c)) for c in df.columns]
    return df


def _read_geodata(settings_dir: Path) -> pd.DataFrame:
    gd = pd.read_csv(Path(settings_dir) / 'GeoData.txt', sep='\t')
    gd.columns = [c.strip().lower() for c in gd.columns]
    return gd.set_index('subid')


@register_land_adapter('HYPE')
class HYPERechargeAdapter(LandRechargeAdapter):
    """HYPE with regional groundwater model 1 draining out of the domain.

    Deep percolation from the lowest soil layer (rate ``rcgrw``) leaves the
    model and is reported per subbasin by ``AQIN`` (m3/d, local). Fast flow is
    ``COUT`` at the outlet subbasin. Zones are HYPE subbasins; the zone
    shapefile's id field must hold the subid.

    HYPE replaces ``grwdown`` by ``maindown`` when *every* ``grwdown`` is 0,
    which would pass the water to the downstream subbasin's soil instead. So
    ``grwdown`` is set to a subid that does not exist, which HYPE resolves to
    "outside the model domain" for each subbasin.
    """
    name = 'HYPE'

    @property
    def OUT_OF_DOMAIN(self) -> int:
        from symfluence.models.hype.geodata_manager import GROUNDWATER_OUT_OF_DOMAIN_SUBID
        return GROUNDWATER_OUT_OF_DOMAIN_SUBID

    def enable_recharge_export(self, settings_dir: Path) -> None:
        """Edit existing HYPE settings in place (native runs set HYPE_GROUNDWATER_EXPORT instead)."""
        d = Path(settings_dir)
        info = d / 'info.txt'
        lines = info.read_text().splitlines()
        seen_dg = seen_out = False
        for i, line in enumerate(lines):
            parts = line.split()
            if parts[:2] == ['modeloption', 'deepground']:
                lines[i] = 'modeloption deepground\t1'
                seen_dg = True
            elif parts[:2] == ['timeoutput', 'variable']:
                names = [p.upper() for p in parts[2:]]
                for need in ('COUT', 'AQIN'):
                    if need not in names:
                        names.append(need)
                lines[i] = 'timeoutput variable ' + '\t'.join(names)
                seen_out = True
        if not seen_dg:
            lines.append('modeloption deepground\t1')
        if not seen_out:
            raise ConfigurationError(f'{info} has no "timeoutput variable" line')
        info.write_text('\n'.join(lines) + '\n')

        gd_path = d / 'GeoData.txt'
        gd = pd.read_csv(gd_path, sep='\t', dtype=str)
        col = next((c for c in gd.columns if c.strip().lower() == 'grwdown'), None)
        if col is None:
            raise ConfigurationError(f'{gd_path} has no grwdown column')
        if (gd.iloc[:, 0].astype(int) == self.OUT_OF_DOMAIN).any():
            raise ConfigurationError(f'subid {self.OUT_OF_DOMAIN} is reserved for out-of-domain groundwater')
        gd[col] = str(self.OUT_OF_DOMAIN)
        gd.to_csv(gd_path, sep='\t', index=False)

        par = d / 'par.txt'
        if par.exists() and not any(l.split()[:1] == ['rcgrw'] for l in par.read_text().splitlines()):
            logger.warning('HYPE par.txt has no rcgrw; HYPE default (0, no recharge) applies until calibrated')

    def set_output_dir(self, settings_dir: Path, output_dir: Path) -> None:
        info = Path(settings_dir) / 'info.txt'
        lines = info.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.split()[:1] == ['resultdir']:
                lines[i] = f"resultdir\t{str(output_dir).rstrip('/')}/"
                break
        else:
            raise ConfigurationError(f'{info} has no resultdir line')
        info.write_text('\n'.join(lines) + '\n')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def recharge_zones(self, settings_dir: Path):
        gd = _read_geodata(settings_dir)
        z = self._zone_shapes(settings_dir)
        z['zone_id'] = z.zone_id.astype(int)
        unknown = set(z.zone_id) - set(gd.index)
        if unknown or set(gd.index) - set(z.zone_id):
            raise ValidationError(f'HYPE subids {sorted(gd.index)} and zone ids {sorted(z.zone_id)} differ')
        z['area_m2'] = z.zone_id.map(gd['area'].astype(float))
        return z

    def read_recharge(self, output_dir: Path, settings_dir: Path) -> pd.DataFrame:
        gd = _read_geodata(settings_dir)
        if (gd['grwdown'].astype(int) != self.OUT_OF_DOMAIN).any():
            raise ConfigurationError('HYPE recharge export is not enabled (grwdown); run enable_recharge_export')
        aqin = _read_hype_table(Path(output_dir) / 'timeAQIN.txt')
        if (aqin < 0).any().any():
            raise ValidationError('HYPE AQIN contains negative recharge')
        return aqin.div(gd.loc[aqin.columns, 'area'].astype(float), axis=1)   # m3/d -> m/d

    def _outlet(self, settings_dir: Path) -> int:
        if self.options.get('outlet_id') is not None:
            return int(self.options['outlet_id'])
        gd = _read_geodata(settings_dir)
        outs = [int(s) for s in gd.index if int(gd.loc[s, 'maindown']) not in gd.index]
        if len(outs) != 1:
            raise ConfigurationError(f'HYPE outlet ambiguous ({outs}); set MODFLOW_LAND_OUTLET_ID')
        return outs[0]

    def read_fast_flow(self, output_dir: Path, settings_dir: Path) -> pd.Series:
        cout = _read_hype_table(Path(output_dir) / 'timeCOUT.txt')
        return cout[self._outlet(settings_dir)].rename('fast_flow_m3s')


# ---------------------------------------------------------------------------
# SUMMA
# ---------------------------------------------------------------------------

@register_land_adapter('SUMMA')
class SUMMARechargeAdapter(LandRechargeAdapter):
    """SUMMA with ``groundwatr noXplict``: soil drainage is recharge.

    Recharge is ``scalarSoilDrainage`` (m/s) per HRU; fast flow is
    ``scalarSurfaceRunoff`` (m/s) area-weighted over HRUs. Without a zone
    shapefile the basin is one zone (``zone_id`` 1).
    """
    name = 'SUMMA'

    def enable_recharge_export(self, settings_dir: Path) -> None:
        decisions = Path(settings_dir) / 'modelDecisions.txt'
        if not decisions.exists():
            raise ConfigurationError(f'{decisions} not found')
        lines = decisions.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.split()[:1] == ['groundwatr']:
                rest = line.split(None, 2)[2] if len(line.split()) > 2 else ''
                lines[i] = f'groundwatr                      noXplict        {rest}'.rstrip()
        decisions.write_text('\n'.join(lines) + '\n')

    def recharge_zones(self, settings_dir: Path):
        if self.options.get('zone_shapefile'):
            z = self._zone_shapes(settings_dir)
        else:
            import geopandas as gpd
            basin = self.options.get('basin_shapefile')
            if not basin:
                raise ConfigurationError('SUMMA coupling needs MODFLOW_LAND_ZONE_SHAPEFILE or a basin shapefile')
            g = gpd.read_file(basin)
            z = gpd.GeoDataFrame({'zone_id': [1]}, geometry=[g.geometry.union_all()], crs=g.crs)
        z['area_m2'] = z.to_crs(z.estimate_utm_crs()).geometry.area
        return z

    def _daily(self, output_dir: Path, var: str) -> pd.DataFrame:
        from .coupling import SUMMAToMODFLOWCoupler
        ds, _ = SUMMAToMODFLOWCoupler({}, logger)._open_summa_variable(Path(output_dir), var)
        try:
            da = ds[var]
            hru_dim = next((d for d in da.dims if d != 'time'), None)
            if hru_dim is None:
                df = da.to_series().to_frame(1)
            else:
                ids = ds['hruId'].values.astype(int) if 'hruId' in ds else range(1, da.sizes[hru_dim] + 1)
                df = pd.DataFrame(da.transpose('time', hru_dim).values, index=pd.to_datetime(ds['time'].values),
                                  columns=list(ids))
        finally:
            ds.close()
        df.index = pd.to_datetime(df.index).normalize()
        return df.groupby(level=0).mean()

    def read_recharge(self, output_dir: Path, settings_dir: Path) -> pd.DataFrame:
        var = self.options.get('recharge_variable') or 'scalarSoilDrainage'
        return self._daily(output_dir, var).clip(lower=0.0) * 86400.0

    def read_fast_flow(self, output_dir: Path, settings_dir: Path) -> pd.Series:
        q = self._daily(output_dir, 'scalarSurfaceRunoff')
        zones = self.recharge_zones(settings_dir).set_index('zone_id').area_m2
        if len(q.columns) == 1 and len(zones) == 1:
            return (q.iloc[:, 0] * float(zones.iloc[0])).rename('fast_flow_m3s')
        return (q[zones.index] * zones).sum(axis=1).rename('fast_flow_m3s')
