"""Narrow compatibility for EASYMORE point grids with current NumPy."""
import easymore
import numpy as np

_Base = getattr(easymore, 'Easymore', None) or getattr(easymore, 'easymore', None)
if _Base is None:
    raise ImportError('EASYMORE does not expose a supported remapper class')


class NativePointEasymore(_Base):
    """Keep structured-grid behavior; return scalar indices for point grids."""

    def create_row_col_df(self, lat_source, lon_source, lat_target_int, lon_target_int):
        if self.case != 3:
            return super().create_row_col_df(lat_source, lon_source, lat_target_int, lon_target_int)
        lat, lon = np.asarray(lat_source), np.asarray(lon_source)
        if lat.ndim != 1 or lon.shape != lat.shape or lat.size == 0 or not np.isfinite([lat, lon]).all():
            raise ValueError('Invalid EASYMORE point coordinates')
        if len(lat_target_int) != len(lon_target_int):
            raise ValueError("Mismatched EASYMORE target coordinates")
        rows = np.empty(len(lat_target_int), dtype=int)
        for i, (target_lat, target_lon) in enumerate(zip(lat_target_int, lon_target_int)):
            distance = np.abs(target_lat-lat) + np.abs(target_lon-lon)
            if not np.isfinite(distance).all():
                raise ValueError('Invalid EASYMORE target coordinates')
            matches = np.flatnonzero(distance == distance.min())
            if len(matches) != 1:
                raise ValueError('Ambiguous EASYMORE source point')
            rows[i] = int(matches[0])
        # EASYMORE's case-3 convention stores the point index in both columns.
        return rows, rows.copy()
