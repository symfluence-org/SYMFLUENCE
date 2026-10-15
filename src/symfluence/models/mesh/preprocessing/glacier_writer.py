"""Write a separate MESH native ice-surface input bundle from staged inputs.

The optional finite inventory requires the matching experimental native patch;
otherwise volume is provenance only. Explicit unit IDs follow CLASS/DDB column
order, never an inferred land-cover code. Forcing dimensions are unchanged.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from ..errors import MESHConfigurationError, MESHInputError


def _rewrite(line, marker, values):
    return '  ' + '  '.join(f'{x:.9g}' for x in values) + '  ' + marker


def ice_block(block, *, mid, albedo_visible, albedo_infrared, temperature_c):
    """Replace terrestrial canopy/soil states with explicit CLASS ice fields."""
    if not (0 < albedo_visible < 1 and 0 < albedo_infrared < 1
            and -100 < temperature_c <= 0):
        raise MESHInputError('Invalid glacier albedo or ice temperature')
    replacements = {
        '05 5xFCAN/4xLAMX': [0, 0, 0, 0, 1, 0, 0, 0, 0],
        '06 5xLNZ0/4xLAMN': [0, 0, 0, 0, np.log(.001), 0, 0, 0, 0],
        '07 5xALVC/4xCMAS': [0, 0, 0, 0, albedo_visible, 0, 0, 0, 0],
        '08 5xALIC/4xROOT': [0, 0, 0, 0, albedo_infrared, 0, 0, 0, 0],
        '09 4xRSMN/4xQA50': [0]*8,
        '10 4xVPDA/4xVPDB': [0]*8,
        '11 4xPSGA/4xPSGB': [0]*8,
        '14 3xSAND (or more)': [-4]*3,
        '15 3xCLAY (or more)': [0]*3,
        '16 3xORGM (or more)': [0]*3,
        '17 3xTBAR (or more)/TCAN/TSNO/TPND': [temperature_c]*5 + [0],
        # CLASSBG represents the bulk ice in HCPS with zero porosity. THIC
        # is pore ice; setting it to one double-counts heat capacity in CLASSZ.
        '18 3xTHLQ (or more)/3xTHIC (or more)/ZPND': [0]*7,
    }
    result, seen = [], set()
    for line in block:
        match = next((key for key in replacements if key in line), None)
        if match:
            # Reject extended-layer files rather than silently truncating them.
            numeric = line[:line.index(match)].split()
            if len(numeric) != len(replacements[match]):
                raise MESHConfigurationError(f'Unsupported CLASS field layout: {match}')
            result.append(_rewrite(line, match, replacements[match]))
            seen.add(match)
        elif '13 XSLP/XDRAINH/MANN/KSAT/MID' in line:
            values = [float(x) for x in line.split()[:4]] + [mid]
            result.append(_rewrite(line, '13 XSLP/XDRAINH/MANN/KSAT/MID', values))
        else:
            result.append(line)
    if seen != replacements.keys():
        raise MESHInputError('Incomplete CLASS glacier block')
    return result


def write_glacier_bundle(source, destination, manifest, *, unit_ids,
                         albedo_visible, albedo_infrared, temperature_c, finite_storage=False):
    """Split each spatial GRU into land/ice, cloning counted hydrology fields.

    Writes to a NEW directory only, after validating all transformations. The
    caller explicitly supplies unit IDs in native CLASS/DDB column order. This
    avoids guessing polygon identity from a generic land-cover MID. Forcing is
    kept on the same native subbasins; no second precipitation input is created.
    """
    source, destination = Path(source), Path(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise MESHConfigurationError('Glacier destination must be outside the source bundle')
    if destination.exists():
        raise MESHConfigurationError('Glacier destination must be new; never modify a live bundle')
    if manifest.get('schema_version') != 1:
        raise MESHConfigurationError('Unsupported glacier manifest')
    rows = {str(row['unit_id']): row for row in manifest['units']}
    ids = [str(uid) for uid in unit_ids]
    if len(rows) != len(manifest['units']) or len(set(ids)) != len(ids) or set(ids) != set(rows):
        raise MESHConfigurationError('Manifest and native spatial unit IDs must match exactly')
    fractions = np.array([rows[uid]['glacier_fraction'] for uid in ids])
    if not np.isfinite(fractions).all() or ((fractions < 0) | (fractions > 1)).any():
        raise MESHInputError('Invalid glacier fractions')
    for uid in ids:
        row = rows[uid]
        numeric = [row[key] for key in ('area_m2', 'glacier_area_m2', 'land_area_m2',
                                       'thickness_coverage', 'ice_volume_m3', 'ice_water_volume_m3')]
        if (not np.isfinite(numeric).all() or min(numeric) < 0 or
                row['area_m2'] <= 0 or not np.isclose(row['glacier_area_m2'],
                row['area_m2']*row['glacier_fraction']) or
                not np.isclose(row['land_area_m2']+row['glacier_area_m2'], row['area_m2']) or
                not np.isclose(row['thickness_coverage'], 1., rtol=0, atol=1e-9) or
                (row['glacier_area_m2'] > 0 and row['ice_volume_m3'] <= 0)):
            raise MESHInputError('Invalid or incomplete glacier geometry')
    class_name = 'MESH_parameters_CLASS.ini'
    lines = (source/class_name).read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line]
    ends = [i+1 for i, line in enumerate(lines) if '19 RCAN/SCAN/SNO/ALBS/RHOS/GRO' in line]
    if len(starts) != len(ids) or len(ends) != len(ids):
        raise MESHInputError('CLASS block count differs from geometry units')
    blocks = [lines[start:end] for start, end in zip(starts, ends)]
    header = next(i for i, line in enumerate(lines) if '04 DEGLAT/DEGLON' in line)
    with xr.open_dataset(source/'MESH_drainage_database.nc') as opened:
        ddb = opened.load()
    n = len(ids)
    if ddb.GRU.dims[-1] != 'NGRU' or ddb.GRU.ndim != 2:
        raise MESHConfigurationError('Unsupported native GRU dimensions')
    old = ddb.GRU.values
    dummy = old.shape[1] == n+1
    if not dummy or np.any(old[:, -1] != 0):
        raise MESHInputError('Unrecognized DDB dummy GRU layout')
    valid_rows = np.isclose(old.sum(axis=1), 1) | ((np.asarray(ddb.GridArea) == 0) & (old.sum(axis=1) == 0))
    if not np.isfinite(old).all() or (old < 0).any() or not valid_rows.all():
        raise MESHInputError('Invalid native GRU fractions')
    # Each manifest polygon must represent the exact spatial support of its
    # native GRU. A land-cover class spread over several cells is ambiguous.
    if np.any((old[:, :n] > 0).sum(axis=0) != 1):
        raise MESHConfigurationError('Each spatial GRU must occur in exactly one native subbasin')
    areas = np.asarray(ddb.GridArea.values).ravel()
    if str(ddb.GridArea.attrs.get('units', '')).replace(' ', '') not in ('m2', 'm^2', 'm**2'):
        raise MESHConfigurationError('GridArea must declare square metres')
    actual = (old[:, :n] * areas[:, None]).sum(axis=0)
    if not np.allclose(actual, [rows[uid]['area_m2'] for uid in ids], rtol=1e-5, atol=.01):
        raise MESHInputError('Geometry and native GRU areas disagree')
    columns, new_blocks, parents, surfaces, stores = [], [], [], [], []
    mids = [int(float(next(line for line in block if '13 XSLP/' in line).split()[4]))
            for block in blocks]
    next_mid = max(mids)+1
    for j, fraction in enumerate(fractions):
        for kind, weight in (('land', 1-fraction), ('ice', fraction)):
            if weight == 0:
                continue
            block = blocks[j]
            if kind == 'ice':
                stores.append((next_mid, rows[ids[j]]['ice_water_volume_m3'] / rows[ids[j]]['glacier_area_m2'] * 1000))
                block = ice_block(block, mid=next_mid, albedo_visible=albedo_visible,
                                  albedo_infrared=albedo_infrared, temperature_c=temperature_c)
                next_mid += 1
            columns.append(old[:, j]*weight)
            new_blocks.extend(block + [''])
            parents.append(j)
            surfaces.append(dict(unit_id=ids[j], surface=kind, fraction=float(weight)))
    if dummy:
        columns.append(np.zeros(old.shape[0]))
    # Any additional GRU-indexed state requires an explicit adapter.
    if any('NGRU' in var.dims for key, var in ddb.data_vars.items() if key != 'GRU'):
        raise MESHConfigurationError('Unsupported additional GRU-indexed DDB fields')
    new_ddb = ddb.drop_dims('NGRU')
    new_ddb['GRU'] = (ddb.GRU.dims, np.column_stack(columns), ddb.GRU.attrs)
    values = lines[header].split()[:9]
    if int(values[8]) != n:
        raise MESHInputError('CLASS NM does not match block count')
    values[8] = str(len(parents))
    new_header = lines[:starts[0]]
    new_header[header] = ' '.join(values) + '  ' + lines[header][lines[header].index('04 DEGLAT'):]
    new_class = '\n'.join(new_header + new_blocks + lines[ends[-1]:])+'\n'
    hydro = (source/'MESH_parameters_hydrology.ini').read_text().splitlines()
    section = next(i for i, line in enumerate(hydro) if 'GRU class dependent hydrologic parameters' in line)
    count_idx = next(i for i in range(section+1, len(hydro)) if re.match(r'^\s*\d+\s', hydro[i]))
    count, seen = int(hydro[count_idx].split()[0]), 0
    for i in range(count_idx+1, len(hydro)):
        if seen == count:
            break
        text, _, comment = hydro[i].partition('#')
        parts = text.split()
        if not parts or not re.match(r'^[A-Za-z_]\w*$', parts[0]):
            continue
        vals = parts[1:]
        if len(vals) not in (1, n):
            raise MESHInputError('Hydrology field has wrong GRU count')
        if len(vals) == 1:
            vals *= n
        expanded = [vals[j] for j in parents]
        if parts[0].upper() == 'IWF':
            expanded = ['1' if surface['surface'] == 'ice' else value
                        for value, surface in zip(expanded, surfaces)]
        hydro[i] = parts[0]+'  '+'  '.join(expanded) + (' #'+comment if comment else '')
        seen += 1
    if seen != count:
        raise MESHInputError('Incomplete counted hydrology section')
    report = dict(mode='finite_ice_guard' if finite_storage else 'native_ice_surface',
                  finite_ice_storage=finite_storage, depletion_behavior='stop before uncommitted exhaustion',
                  geometry=manifest, native_columns=surfaces,
                  initial_ice_temperature_c=temperature_c,
                  ice_albedo_visible=albedo_visible, ice_albedo_infrared=albedo_infrared)
    # Publish only a complete bundle. Interrupted writes leave no destination.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.mesh-glacier-') as temporary:
        staged = Path(temporary)/'inputs'
        shutil.copytree(source, staged)
        if np.any(fractions):
            (staged/class_name).write_text(new_class)
            (staged/'MESH_parameters_hydrology.ini').write_text('\n'.join(hydro)+'\n')
            new_ddb.to_netcdf(staged/'MESH_drainage_database.nc', mode='w')
        (staged/'glacier_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        if finite_storage:
            (staged/'MESH_glacier_storage.ini').write_text(str(len(stores))+'\n' +
                ''.join(f'{mid} {mass:.17g}\n' for mid, mass in stores))
        staged.rename(destination)
    return report
