# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GRACE Data Acquisition Handler

Provides cloud acquisition for GRACE/GRACE-FO Terrestrial Water Storage anomaly data.
Retrieves data from NASA PO.DAAC or similar cloud-hosted repositories.
"""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

import requests
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@lru_cache(maxsize=1)
def _ca_bundle() -> str:
    """Path to a CA bundle of certifi roots plus vendored intermediates.

    Some GRACE hosts (notably download.csr.utexas.edu) serve an incomplete
    certificate chain that omits a valid Sectigo intermediate. Rather than
    disabling TLS verification, we append the missing intermediate(s) — each
    of which chains to a root already in certifi — so verification still
    succeeds against a complete trust path. Falls back to certifi alone if the
    vendored certs cannot be read.
    """
    import certifi

    certs_dir = Path(__file__).resolve().parents[3] / "resources" / "certs"
    try:
        with open(certifi.where(), encoding="utf-8") as f:
            bundle = f.read()
        extra = "".join(
            p.read_text(encoding="utf-8") for p in sorted(certs_dir.glob("*.pem"))
        )
        if not extra:
            return certifi.where()
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".pem", prefix="symfluence_ca_", delete=False, encoding="utf-8"
        )
        tmp.write(bundle + "\n" + extra)
        tmp.flush()
        tmp.close()
        return tmp.name
    except OSError:
        return certifi.where()


@R.acquisition_handlers.add('GRACE')
class GRACEAcquirer(BaseAcquisitionHandler):
    """
    Handles GRACE/GRACE-FO data acquisition.
    Currently focuses on the JPL/CSR/GSFC Mascon solutions.
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download GRACE data (JPL, CSR, and GSFC Mascon RL06v02).
        """
        self.logger.info("Starting GRACE data acquisition (JPL, CSR, GSFC)")
        output_dir.mkdir(parents=True, exist_ok=True)

        subset_enabled = self._parse_bool(self._get_config_value(lambda: None, default=False, dict_key='GRACE_SUBSET'))
        force_download = self._parse_bool(self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'))

        datasets = {
            'jpl': {
                'filename': 'GRCTellus.JPL.200204_202511.GLO.RL06.3M.MSCNv04CRI.nc',
                'url': 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/tellus/L3/grace/nasajpl/RL06.3_v04/GRCTellus.JPL.200204_202511.GLO.RL06.3M.MSCNv04CRI.nc'
            },
            'csr': {
                'filename': 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc',
                'url': 'https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
            },
            'gsfc': {
                'filename': 'gsfc.glb_.200204_202505_rl06v2.0_obp-ice6gd_halfdegree.nc',
                'url': 'https://earth.gsfc.nasa.gov/sites/default/files/geo/gsfc.glb_.200204_202505_rl06v2.0_obp-ice6gd_halfdegree.nc'
            }
        }

        success_count = 0

        for center, info in datasets.items():
            target_file = output_dir / info['filename']
            subset_file = target_file.with_name(f"{target_file.stem}_subset.nc")
            url = info['url']

            if subset_enabled and center == 'jpl' and subset_file.exists() and not force_download:
                self.logger.info(f"GRACE {center.upper()} subset already exists: {subset_file}")
                success_count += 1
                continue
            if target_file.exists() and not force_download:
                self.logger.info(f"GRACE {center.upper()} file already exists: {target_file}")
                success_count += 1
                continue

            try:
                # JPL is auth-gated on NASA Earthdata and is resolved dynamically
                # via CMR (the legacy podaac-opendap.jpl.nasa.gov host has been
                # decommissioned, so there is no static-URL fallback).
                if center == 'jpl':
                    if subset_enabled:
                        if self._download_jpl_subset(url, subset_file):
                            self.logger.info(f"Successfully downloaded GRACE JPL subset to {subset_file}")
                            success_count += 1
                            continue
                        self.logger.warning("JPL subset download failed; falling back to CMR.")

                    cmr_path = self._download_jpl_from_cmr(output_dir, force_download)
                    if cmr_path is None:
                        raise RuntimeError(
                            "JPL GRACE download via CMR failed — verify NASA Earthdata "
                            "credentials (~/.netrc for urs.earthdata.nasa.gov, or "
                            "EARTHDATA_TOKEN / EARTHDATA_USERNAME+PASSWORD)."
                        )
                    self.logger.info(f"Successfully downloaded GRACE JPL data to {cmr_path}")
                    success_count += 1
                    continue

                # CSR / GSFC: public hosts, no Earthdata auth required.
                # TLS verification stays enabled. Some hosts (e.g. CSR) serve an
                # incomplete chain, so we verify against certifi plus the vendored
                # intermediate(s) rather than disabling verification globally.
                self.logger.info(f"Downloading {center.upper()} from {url}")
                session = requests.Session()
                session.verify = _ca_bundle()
                self._stream_to_file(session, url, target_file, timeout=120)
                self.logger.info(f"Successfully downloaded GRACE {center.upper()} data to {target_file}")
                success_count += 1
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.error(f"Failed to download GRACE {center.upper()} data: {e}", exc_info=True)
                self.logger.warning(f"Please manually download the {center.upper()} Mascon NetCDF file and place it in the observation directory if automatic download fails.")

        if success_count == 0:
            raise RuntimeError("Failed to acquire any GRACE data.")

        return output_dir

    def _stream_to_file(self, session: requests.Session, url: str, target_file: Path, timeout: int) -> None:
        """Stream a download to a temp file, then atomically rename into place.

        Writing directly to the target leaves a truncated NetCDF behind if the
        transfer is interrupted, which later "file exists" checks would treat as
        a valid cached download. Staging to a .part file and renaming on success
        makes the on-disk artifact all-or-nothing.
        """
        tmp = target_file.with_name(target_file.name + ".part")
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            tmp.replace(target_file)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass  # best-effort cleanup of the partial download

    def _parse_bool(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {'true', '1', 'yes', 'y'}
        return bool(value)

    def _download_jpl_subset(self, url: str, target_file: Path) -> bool:
        if not self.bbox:
            self.logger.warning("GRACE_SUBSET requested but BOUNDING_BOX_COORDS not set.")
            return False

        try:
            ds = xr.open_dataset(url)
        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to open JPL OPeNDAP dataset: {exc}", exc_info=True)
            return False

        try:
            subset = self._subset_grace_dataset(ds)
            subset.to_netcdf(target_file)
            return True
        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to subset JPL dataset: {exc}", exc_info=True)
            return False
        finally:
            try:
                ds.close()
            except (OSError, AttributeError):
                pass  # Dataset may already be closed or invalid

    def _subset_grace_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        lat_name = self._get_coord_name(ds, ("lat", "latitude"))
        lon_name = self._get_coord_name(ds, ("lon", "longitude"))
        if not lat_name or not lon_name:
            raise ValueError("GRACE dataset missing expected lat/lon coordinates")

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        lon_vals = ds[lon_name].values
        if lon_vals.max() > 180 and (lon_min < 0 or lon_max < 0):
            lon_min = lon_min % 360
            lon_max = lon_max % 360

        if lon_min <= lon_max:
            lon_subset = ds.sel({lon_name: slice(lon_min, lon_max)})
        else:
            lon_subset = xr.concat(
                [
                    ds.sel({lon_name: slice(lon_min, lon_vals.max())}),
                    ds.sel({lon_name: slice(lon_vals.min(), lon_max)}),
                ],
                dim=lon_name,
            )

        lat_vals = lon_subset[lat_name].values
        if lat_vals[0] > lat_vals[-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        subset = lon_subset.sel({lat_name: lat_slice})
        if "time" in subset.coords:
            subset = subset.sel(time=slice(self.start_date, self.end_date))

        return subset

    @staticmethod
    def _get_coord_name(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
        for name in candidates:
            if name in ds.coords:
                return name
        return None

    def _download_jpl_from_cmr(
        self,
        output_dir: Path,
        force_download: bool,
    ) -> Optional[Path]:
        collection_id = self._get_config_value(
            lambda: None,
            default='C3195527175-POCLOUD',
            dict_key='GRACE_JPL_COLLECTION_ID',
        )

        self.logger.info(
            "Querying CMR for JPL GRACE mascon (collection %s)",
            collection_id,
        )

        # Redirect-aware session: the granule download on archive.podaac is
        # 302-redirected to urs.earthdata.nasa.gov for OAuth, and the stock
        # requests session strips the Authorization header on that cross-host
        # hop (→ 401). EarthdataSession re-applies the token / credentials at
        # the URS host so the redirect chain authenticates correctly.
        session = self._earthdata_auth().session(origin_auth=False, purpose="JPL GRACE (PO.DAAC)")

        try:
            resp = session.get(
                "https://cmr.earthdata.nasa.gov/search/granules.json",
                params={
                    "collection_concept_id": collection_id,
                    "page_size": 1,
                    "sort_key": "-start_date",
                },
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"CMR query failed: {exc}", exc_info=True)
            return None

        entries = resp.json().get("feed", {}).get("entry", [])
        if not entries:
            self.logger.warning("CMR query returned no granules for JPL collection.")
            return None

        links = entries[0].get("links", [])
        data_links = [
            link.get("href")
            for link in links
            if link.get("rel", "").endswith("data#")
            and link.get("href", "").endswith(".nc")
        ]
        if not data_links:
            self.logger.warning("CMR granule did not include a NetCDF data link.")
            return None

        download_url = next(
            (href for href in data_links if "archive.podaac.earthdata.nasa.gov" in href),
            data_links[0],
        )
        target_file = output_dir / Path(download_url).name
        if target_file.exists() and not force_download:
            return target_file

        try:
            self._stream_to_file(session, download_url, target_file, timeout=600)
            return target_file
        except Exception as exc:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"CMR download failed: {exc}", exc_info=True)
            return None
