# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Sentinel-1 SAR Soil Moisture Data Acquisition Handler

Provides acquisition for Sentinel-1 SAR-derived soil moisture products.
Sentinel-1 provides high-resolution soil moisture estimates that complement
passive microwave sensors like SMAP.

Sentinel-1 SM features:
- ~1 km spatial resolution (much higher than SMAP/SMOS)
- 6-12 day revisit time
- C-band SAR backscatter
- Works under clouds (active radar)
- Available through Copernicus Data Space

Data access via Copernicus Data Space Ecosystem:
https://dataspace.copernicus.eu/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler


@R.acquisition_handlers.add('SENTINEL1_SM')
@R.acquisition_handlers.add('S1_SM')
class Sentinel1SMAcquirer(BaseAcquisitionHandler):
    """
    Handles Sentinel-1 soil moisture data acquisition.

    Downloads Sentinel-1 SAR soil moisture products from Copernicus
    Data Space or processes raw backscatter data.

    Configuration:
        SENTINEL1_CLIENT_ID: Copernicus Data Space client ID
        SENTINEL1_CLIENT_SECRET: Copernicus Data Space client secret
        SENTINEL1_PRODUCT: Product type ('SM', 'backscatter')
        SENTINEL1_POLARIZATION: 'VV', 'VH', or 'dual' (default: VV)
    """

    CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"  # nosec B105
    CDSE_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    def download(self, output_dir: Path) -> Path:
        """
        Download Sentinel-1 soil moisture data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data directory
        """
        self.logger.info("Starting Sentinel-1 soil moisture data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get credentials
        client_id, client_secret = self._get_credentials()
        if not client_id or not client_secret:
            raise ValueError(
                "Copernicus Data Space credentials required. "
                "Set SENTINEL1_CLIENT_ID and SENTINEL1_CLIENT_SECRET "
                "or register at https://dataspace.copernicus.eu/"
            )

        # Get access token
        token = self._get_access_token(client_id, client_secret)
        if not token:
            raise RuntimeError("Failed to authenticate with Copernicus Data Space")

        # Search for products
        products = self._search_products(token)
        if not products:
            self.logger.warning("No Sentinel-1 products found for specified criteria")
            return output_dir

        self.logger.info(f"Found {len(products)} Sentinel-1 products")

        # Download products
        force_download = self._get_config_value(lambda: self.config.data.force_download, dict_key='FORCE_DOWNLOAD')
        downloaded = 0

        for product in products:
            product_id = product.get('Id')
            product_name = product.get('Name', f'S1_{product_id}')

            output_file = output_dir / f"{product_name}.zip"
            if output_file.exists() and not force_download:
                downloaded += 1
                continue

            if self._download_product(token, product_id, output_file):
                downloaded += 1

        self.logger.info(f"Sentinel-1 download complete: {downloaded} products")
        return output_dir

    def _get_credentials(self):
        """Get Copernicus Data Space credentials."""
        client_id = (
            os.environ.get('SENTINEL1_CLIENT_ID') or
            os.environ.get('CDSE_CLIENT_ID') or
            self._get_config_value(lambda: None, default=None, dict_key='SENTINEL1_CLIENT_ID')
        )
        client_secret = (
            os.environ.get('SENTINEL1_CLIENT_SECRET') or
            os.environ.get('CDSE_CLIENT_SECRET') or
            self._get_config_value(lambda: None, default=None, dict_key='SENTINEL1_CLIENT_SECRET')
        )
        return client_id, client_secret

    def _get_access_token(self, client_id: str, client_secret: str) -> Optional[str]:
        """Get OAuth2 access token from Copernicus Data Space."""
        try:
            response = requests.post(
                self.CDSE_TOKEN_URL,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': client_id,
                    'client_secret': client_secret,
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get('access_token')
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to get access token: {e}", exc_info=True)
            return None

    def _search_products(self, token: str) -> List[Dict[str, Any]]:
        """Search for Sentinel-1 products in the catalog."""
        headers = {'Authorization': f'Bearer {token}'}

        # Build OData filter
        filters = [
            "Collection/Name eq 'SENTINEL-1'",
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD')",
        ]

        # Date filter
        start_str = self.start_date.strftime('%Y-%m-%dT00:00:00.000Z')
        end_str = self.end_date.strftime('%Y-%m-%dT23:59:59.999Z')
        filters.append(f"ContentDate/Start ge {start_str}")
        filters.append(f"ContentDate/Start le {end_str}")

        # Spatial filter
        if self.bbox:
            # Create WKT polygon
            wkt = (
                f"POLYGON(("
                f"{self.bbox['lon_min']} {self.bbox['lat_min']},"
                f"{self.bbox['lon_max']} {self.bbox['lat_min']},"
                f"{self.bbox['lon_max']} {self.bbox['lat_max']},"
                f"{self.bbox['lon_min']} {self.bbox['lat_max']},"
                f"{self.bbox['lon_min']} {self.bbox['lat_min']}))"
            )
            filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')")

        filter_str = ' and '.join(filters)

        try:
            response = requests.get(
                self.CDSE_CATALOG_URL,
                headers=headers,
                params={
                    '$filter': filter_str,
                    '$top': 100,
                    '$orderby': 'ContentDate/Start asc',
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get('value', [])
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Product search failed: {e}", exc_info=True)
            return []

    def _download_product(self, token: str, product_id: str, output_file: Path) -> bool:
        """Download a single product."""
        headers = {'Authorization': f'Bearer {token}'}

        download_url = f"{self.CDSE_CATALOG_URL}({product_id})/$value"

        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=600) as response:
                response.raise_for_status()

                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            self.logger.debug(f"Downloaded: {output_file.name}")
            return True

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.warning(f"Failed to download {product_id}: {e}", exc_info=True)
            return False
