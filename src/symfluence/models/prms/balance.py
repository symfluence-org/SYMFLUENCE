"""Native PRMS basin balance, including the fixed-extent glacier convention."""
from pathlib import Path
import numpy as np


def check_water_balance(statvar: Path, parameters: Path, tolerance_mm: float,
                        glacier: bool = False) -> dict:
    """Raise for missing/nonfinite diagnostics or nonconserving simulations.

    Glacier storage is a cumulative change already containing snow on ice.
    Remove that snow from the separately reported basin snowpack storage,
    and include ice evaporation omitted by native basin_actet. Dynamic
    geometry is not certified by this fixed-extent accounting.
    """
    lines = statvar.read_text().splitlines()
    n = int(lines[0])
    values = np.loadtxt(statvar, skiprows=n + 1, ndmin=2)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Nonfinite or insufficient PRMS balance output')
    columns = {'_'.join(line.split()): values[:, 7 + i] for i, line in enumerate(lines[1:n + 1])}
    storage = columns['basin_storage_1'].copy()
    ice_evap = 0.
    if glacier:
        blocks = parameters.read_text().split('** Parameters **', 1)[1].split('####')
        area = None
        for block in blocks:
            rows = block.strip().splitlines()
            if rows and rows[0] == 'hru_area':
                area = np.asarray(rows[4 + int(rows[1]):], dtype=float)
                break
        if area is None or not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError('Missing or invalid PRMS HRU areas')
        weights = area / area.sum()
        for i, weight in enumerate(weights, 1):
            fraction = columns[f'glacier_frac_{i}']
            if np.ptp(fraction) > 1.e-6 or not np.isin(fraction, [0., 1.]).all():
                raise ValueError('Glacier balance gate requires fixed pure glacier/land HRUs')
            storage -= columns[f'pkwater_equiv_{i}'] * fraction * weight
        ice_evap = columns['basin_glacrevap_1']
    flux = (columns['basin_ppt_1'] + columns['basin_gwstor_minarea_wb_1']
            - columns['basin_actet_1'] - ice_evap
            - columns['basin_stflow_out_1'] - columns['basin_gwsink_1'])
    residual = (flux[1:] - np.diff(storage)) * 25.4
    result = {'days': len(values), 'max_abs_daily_residual_mm': float(np.abs(residual).max()),
              'cumulative_residual_mm': float(residual.sum())}
    if result['max_abs_daily_residual_mm'] > tolerance_mm or abs(result['cumulative_residual_mm']) > 10 * tolerance_mm:
        raise ValueError(f'PRMS water balance failed: {result}')
    return result
