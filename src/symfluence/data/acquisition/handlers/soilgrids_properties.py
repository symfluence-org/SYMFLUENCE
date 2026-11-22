# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SoilGrids Continuous Properties Acquisition Handler

Cloud-based acquisition of SoilGrids v2 continuous soil property maps via
OGC Web Coverage Service (WCS).

SoilGrids Properties Overview:
    Data Type: Continuous soil property predictions (mean, Q0.05, Q0.95, uncertainty)
    Resolution: 250m global
    Coverage: Global
    Properties: sand, clay, silt, bdod, ocd, soc, ocs, phh2o, cec, cfvo, nitrogen
    Depths: 0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm
    Source: ISRIC SoilGrids v2.0 (Poggio et al., 2021)

Data Access:
    WCS: https://maps.isric.org/mapserv?map=/map/{property}.map
    No authentication required

Unit Conversions:
    sand/clay/silt: g/kg -> %  (divide by 10)
    bdod: cg/cm3 -> g/cm3  (divide by 100)
    phh2o: pH*10 -> pH  (divide by 10)
    cec: mmol(c)/kg -> cmol(c)/kg  (divide by 10)
    ocd: dg/kg -> g/kg  (divide by 10)
    soc: dg/kg -> g/kg  (divide by 10)
    ocs: t/ha*10 -> t/ha  (divide by 10)
    cfvo: cm3/dm3 -> cm3/100cm3  (divide by 10)
    nitrogen: cg/kg -> g/kg  (divide by 100)

References:
    Poggio, L., et al. (2021). SoilGrids 2.0: producing soil information for
    the globe with quantified spatial uncertainty. SOIL, 7, 217-240.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from symfluence.core.registries import R
from symfluence.data.acquisition.utils import atomic_write

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..utils import create_robust_session

# WCS endpoint template
_WCS_BASE = "https://maps.isric.org/mapserv"

# Available properties and their WCS map paths
_PROPERTY_MAPS = {
    'sand': '/map/sand.map',
    'clay': '/map/clay.map',
    'silt': '/map/silt.map',
    'bdod': '/map/bdod.map',
    'ocd': '/map/ocd.map',
    'soc': '/map/soc.map',
    'ocs': '/map/ocs.map',
    'phh2o': '/map/phh2o.map',
    'cec': '/map/cec.map',
    'cfvo': '/map/cfvo.map',
    'nitrogen': '/map/nitrogen.map',
}

# Unit conversion divisors (raw -> standard units)
_UNIT_CONVERSIONS = {
    'sand': 10.0,      # g/kg -> %
    'clay': 10.0,      # g/kg -> %
    'silt': 10.0,      # g/kg -> %
    'bdod': 100.0,     # cg/cm3 -> g/cm3
    'phh2o': 10.0,     # pH*10 -> pH
    'cec': 10.0,       # mmol(c)/kg -> cmol(c)/kg
    'ocd': 10.0,       # dg/kg -> g/kg
    'soc': 10.0,       # dg/kg -> g/kg
    'ocs': 10.0,       # t/ha*10 -> t/ha
    'cfvo': 10.0,      # cm3/dm3 -> cm3/100cm3
    'nitrogen': 100.0,  # cg/kg -> g/kg
}

# Default properties and depths
_DEFAULT_PROPERTIES = ['sand', 'clay', 'silt', 'bdod']
_ALL_DEPTHS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
_DEFAULT_DEPTHS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']


@R.acquisition_handlers.add('SOILGRIDS_PROPERTIES')
class SoilGridsPropertiesAcquirer(BaseAcquisitionHandler, RetryMixin):
    """SoilGrids v2 continuous soil property acquisition via WCS.

    Downloads quantitative soil property maps from ISRIC SoilGrids v2 using
    the OGC Web Coverage Service. Each property+depth is retrieved as a
    separate GeoTIFF with spatial subsetting to the domain bounding box.

    Acquisition Strategy:
        1. For each property+depth combination
        2. Build WCS GetCoverage request with bbox subsetting
        3. Validate response (GeoTIFF magic bytes)
        4. Apply unit conversion
        5. Save as compressed GeoTIFF

    Configuration:
        SOILGRIDS_PROPERTIES: List of soil properties
            (default: ['sand', 'clay', 'silt', 'bdod'])
            Options: sand, clay, silt, bdod, ocd, soc, ocs, phh2o, cec, cfvo, nitrogen
        SOILGRIDS_DEPTHS: List of depth layers
            (default: ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm'])
        SOILGRIDS_QUANTILE: Statistical quantile
            (default: 'mean', options: mean, Q0.05, Q0.5, Q0.95, uncertainty)
        SOILGRIDS_CONVERT_UNITS: Apply unit conversions (default: True)

    Output:
        Per-property+depth GeoTIFF files in project_dir/attributes/soilclass/soilgrids/
        e.g., domain_{domain_name}_soilgrids_sand_0-5cm_mean.tif

    References:
        Poggio et al. (2021). SoilGrids 2.0. SOIL, 7, 217-240.
    """

    def download(self, output_dir: Path) -> Path:
        sg_dir = self._attribute_dir("soilclass") / "soilgrids"
        sg_dir.mkdir(parents=True, exist_ok=True)

        properties = self._get_config_value(lambda: None, default=_DEFAULT_PROPERTIES, dict_key='SOILGRIDS_PROPERTIES')
        properties = [p for p in properties if p in _PROPERTY_MAPS]
        depths = self._get_config_value(lambda: None, default=_DEFAULT_DEPTHS, dict_key='SOILGRIDS_DEPTHS')
        depths = [d for d in depths if d in _ALL_DEPTHS]
        quantile = self._get_config_value(lambda: None, default='mean', dict_key='SOILGRIDS_QUANTILE')
        convert_units = self._get_config_value(lambda: None, default=True, dict_key='SOILGRIDS_CONVERT_UNITS')

        if not properties:
            raise ValueError(
                f"No valid SoilGrids properties. Choose from: {list(_PROPERTY_MAPS.keys())}"
            )
        if not depths:
            raise ValueError(f"No valid SoilGrids depths. Choose from: {_ALL_DEPTHS}")

        self.logger.info(
            f"Acquiring SoilGrids properties for bbox: {self.bbox}, "
            f"properties: {properties}, depths: {depths}, quantile: {quantile}"
        )

        session = create_robust_session(max_retries=3, backoff_factor=2.0)
        output_paths = {}
        total_combos = len(properties) * len(depths)
        completed = 0

        for prop in properties:
            wcs_map = _PROPERTY_MAPS[prop]

            for depth in depths:
                completed += 1
                combo_key = f"{prop}_{depth}"
                out_path = sg_dir / (
                    f"domain_{self.domain_name}_soilgrids_{prop}_{depth}_{quantile}.tif"
                )

                if self._skip_if_exists(out_path):
                    output_paths[combo_key] = out_path
                    continue

                self.logger.info(
                    f"[{completed}/{total_combos}] Downloading SoilGrids {prop} {depth} {quantile}"
                )

                # Build coverage ID: e.g., sand_0-5cm_mean
                coverage_id = f"{prop}_{depth}_{quantile}"

                try:
                    raw_path = self._download_wcs_coverage(
                        session, wcs_map, coverage_id, sg_dir
                    )

                    # Apply unit conversion if enabled
                    if convert_units and prop in _UNIT_CONVERSIONS:
                        self._apply_unit_conversion(raw_path, out_path, prop)
                        if raw_path != out_path:
                            raw_path.unlink(missing_ok=True)
                    else:
                        if raw_path != out_path:
                            raw_path.rename(out_path)

                    output_paths[combo_key] = out_path
                    self.logger.info(f"Saved: {out_path}")

                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(
                        f"Failed to download SoilGrids {prop} {depth}: {e}"
                    , exc_info=True)
                    continue

        if not output_paths:
            raise RuntimeError("No SoilGrids property data could be downloaded")

        self.logger.info(
            f"SoilGrids properties acquisition complete: {len(output_paths)} files"
        )
        return sg_dir

    def _download_wcs_coverage(
        self, session, wcs_map: str, coverage_id: str, work_dir: Path
    ) -> Path:
        """Download a single WCS coverage as GeoTIFF."""
        raw_path = work_dir / f"raw_{coverage_id}.tif"

        params = [
            ("map", wcs_map),
            ("SERVICE", "WCS"),
            ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage_id),
            ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("SUBSET", f"Lat({self.bbox['lat_min']},{self.bbox['lat_max']})"),
            ("SUBSET", f"Lon({self.bbox['lon_min']},{self.bbox['lon_max']})"),
        ]

        def do_request():
            resp = session.get(
                _WCS_BASE, params=params, stream=True, timeout=120
            )
            resp.raise_for_status()

            content_type = (resp.headers.get("Content-Type") or "").lower()
            chunks = resp.iter_content(chunk_size=65536)
            first_chunk = next(chunks, b"")

            # Validate response is GeoTIFF
            if "text/html" in content_type or first_chunk.lstrip().startswith(b"<"):
                snippet = first_chunk[:200].decode("utf-8", errors="ignore")
                raise ValueError(f"WCS returned HTML response: {snippet}")

            if not first_chunk.startswith((b"II*\x00", b"MM\x00*")):
                snippet = first_chunk[:200].decode("utf-8", errors="ignore")
                raise ValueError(f"WCS returned unexpected content: {snippet}")

            with atomic_write(raw_path) as _part, open(_part, 'wb') as f:
                f.write(first_chunk)
                for chunk in chunks:
                    f.write(chunk)

            return raw_path

        return self.execute_with_retry(
            do_request,
            max_retries=3,
            base_delay=10.0,
            backoff_factor=2.0,
        )

    def _apply_unit_conversion(self, src_path: Path, dst_path: Path, prop: str):
        """Apply unit conversion to a raw SoilGrids GeoTIFF."""
        divisor = _UNIT_CONVERSIONS[prop]

        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata

            # Preserve nodata pixels
            if nodata is not None:
                mask = data == nodata
                data = data / divisor
                data[mask] = nodata
            else:
                data = data / divisor

            meta = src.meta.copy()
            meta.update({
                "dtype": "float32",
                "compress": "lzw",
            })

            with rasterio.open(dst_path, "w", **meta) as dst:
                dst.write(data, 1)
