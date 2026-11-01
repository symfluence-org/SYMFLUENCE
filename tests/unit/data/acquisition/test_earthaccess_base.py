"""
Unit Tests for BaseEarthaccessAcquirer.

Tests the shared NASA Earthdata Cloud functionality:
- CMR granule search
- Download URL extraction
- Earthaccess download orchestration
- Granule counting utility
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from fixtures.acquisition_fixtures import MockConfigFactory

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def concrete_acquirer_class():
    """
    Create a concrete implementation of BaseEarthaccessAcquirer for testing.

    Returns the class, not an instance.
    """
    from symfluence.data.acquisition.earthaccess_base import (
        BaseEarthaccessAcquirer,
    )

    class ConcreteEarthaccessAcquirer(BaseEarthaccessAcquirer):
        """Concrete acquirer for testing base class functionality."""

        def download(self, output_dir: Path) -> Path:
            output_file = output_dir / "test_output.nc"
            output_file.touch()
            return output_file

    return ConcreteEarthaccessAcquirer


@pytest.fixture
def acquirer_instance(concrete_acquirer_class, mock_config, mock_logger):
    """Create an instance of the concrete acquirer."""
    return concrete_acquirer_class(mock_config, mock_logger)


@pytest.fixture
def sample_cmr_response():
    """Create a sample CMR API JSON response."""
    return {
        "feed": {
            "entry": [
                {
                    "id": "granule_1",
                    "title": "MOD10A1.A2020001.h09v05.061",
                    "links": [
                        {
                            "href": "https://data.nsidc.org/MOD10A1/h09v05.hdf",
                            "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                        },
                        {
                            "href": "https://data.nsidc.org/MOD10A1/h09v05.xml",
                            "rel": "http://esipfed.org/ns/fedsearch/1.1/metadata#",
                        },
                    ],
                },
                {
                    "id": "granule_2",
                    "title": "MOD10A1.A2020002.h09v05.061",
                    "links": [
                        {
                            "href": "https://data.nsidc.org/MOD10A1/h09v05_2.hdf",
                            "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                        }
                    ],
                },
            ]
        }
    }


@pytest.fixture
def empty_cmr_response():
    """CMR response with no granules."""
    return {"feed": {"entry": []}}


# =============================================================================
# CMR URL constant
# =============================================================================

class TestBaseEarthaccessConstants:
    """Test class-level constants."""

    def test_cmr_url_is_set(self, concrete_acquirer_class):
        """CMR_URL should point to NASA CMR."""
        assert "cmr.earthdata.nasa.gov" in concrete_acquirer_class.CMR_URL


# =============================================================================
# CMR Search Tests
# =============================================================================

@pytest.mark.acquisition
class TestSearchGranulesCMR:
    """Tests for _search_granules_cmr method."""

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_search_returns_granules(
        self, mock_get, acquirer_instance, sample_cmr_response
    ):
        """Search should return granule entries from CMR."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_cmr_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = acquirer_instance._search_granules_cmr("MOD10A1", version="61")

        assert len(result) == 2
        assert result[0]["id"] == "granule_1"

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_search_empty_response(
        self, mock_get, acquirer_instance, empty_cmr_response
    ):
        """Search should return empty list when no granules found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = empty_cmr_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = acquirer_instance._search_granules_cmr("MOD10A1")

        assert result == []

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_search_uses_instance_dates(self, mock_get, acquirer_instance):
        """Search should use instance start_date/end_date by default."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"feed": {"entry": []}}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        acquirer_instance._search_granules_cmr("MOD10A1")

        call_params = mock_get.call_args
        params = call_params[1].get("params", call_params[0][1] if len(call_params[0]) > 1 else {})
        if not params:
            params = call_params.kwargs.get("params", {})

        assert "temporal" in params
        assert "bounding_box" in params

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_search_handles_network_error(self, mock_get, acquirer_instance):
        """Search should handle network errors gracefully."""
        mock_get.side_effect = ConnectionError("Network error")

        result = acquirer_instance._search_granules_cmr("MOD10A1")

        assert result == []

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_search_paginates(self, mock_get, acquirer_instance):
        """Search should paginate when page is full."""
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.raise_for_status = Mock()
        # Return exactly page_size entries to trigger next page
        page1_entries = [{"id": f"g_{i}", "links": []} for i in range(10)]
        page1_response.json.return_value = {"feed": {"entry": page1_entries}}

        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.raise_for_status = Mock()
        page2_response.json.return_value = {"feed": {"entry": [{"id": "g_last", "links": []}]}}

        mock_get.side_effect = [page1_response, page2_response]

        result = acquirer_instance._search_granules_cmr("MOD10A1", page_size=10)

        assert len(result) == 11
        assert mock_get.call_count == 2

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_search_includes_version_when_provided(self, mock_get, acquirer_instance):
        """Search should include version in CMR params if specified."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"feed": {"entry": []}}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        acquirer_instance._search_granules_cmr("MOD10A1", version="61")

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params", {})
        assert params.get("version") == "61"


# =============================================================================
# Download URL Extraction Tests
# =============================================================================

@pytest.mark.acquisition
class TestGetDownloadURLs:
    """Tests for _get_download_urls method."""

    def test_extracts_hdf_urls(self, acquirer_instance, sample_cmr_response):
        """Should extract .hdf URLs from granule entries."""
        granules = sample_cmr_response["feed"]["entry"]
        urls = acquirer_instance._get_download_urls(granules)

        assert len(urls) == 2
        assert all(url.endswith(".hdf") for url in urls)

    def test_respects_extensions_filter(self, acquirer_instance):
        """Should only return URLs matching given extensions."""
        granules = [
            {
                "links": [
                    {"href": "https://example.com/file.nc"},
                    {"href": "https://example.com/file.hdf"},
                ]
            }
        ]

        urls = acquirer_instance._get_download_urls(granules, extensions=(".nc",))

        assert len(urls) == 1
        assert urls[0].endswith(".nc")

    def test_empty_granules_returns_empty(self, acquirer_instance):
        """Should return empty list for empty input."""
        urls = acquirer_instance._get_download_urls([])
        assert urls == []

    def test_skips_non_http_links(self, acquirer_instance):
        """Should skip links that do not contain 'http'."""
        granules = [
            {
                "links": [
                    {"href": "ftp://example.com/file.hdf"},
                    {"href": "s3://bucket/file.hdf"},
                ]
            }
        ]

        urls = acquirer_instance._get_download_urls(granules)
        assert urls == []

    def test_only_first_matching_link_per_granule(self, acquirer_instance):
        """Should take only the first matching link per granule."""
        granules = [
            {
                "links": [
                    {"href": "https://example.com/file_a.hdf"},
                    {"href": "https://example.com/file_b.hdf"},
                ]
            }
        ]

        urls = acquirer_instance._get_download_urls(granules)
        assert len(urls) == 1
        assert urls[0].endswith("file_a.hdf")


# =============================================================================
# Download with Earthaccess Tests
# =============================================================================

@pytest.mark.acquisition
class TestDownloadWithEarthaccess:
    """Tests for _download_with_earthaccess method."""

    def test_empty_urls_returns_empty(self, acquirer_instance, tmp_path):
        """Should return empty list when no URLs given."""
        result = acquirer_instance._download_with_earthaccess([], tmp_path)
        assert result == []

    def test_skips_existing_files(self, acquirer_instance, tmp_path):
        """Should skip already-downloaded files when skip_existing=True."""
        pytest.importorskip("earthaccess")
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()

        # Pre-create a file
        existing = output_dir / "file1.hdf"
        existing.write_bytes(b"data")

        urls = ["https://example.com/path/file1.hdf"]

        with patch("earthaccess.login", return_value=True), \
             patch("earthaccess.get_requests_https_session") as mock_session:
            mock_session.return_value = MagicMock()

            result = acquirer_instance._download_with_earthaccess(
                urls, output_dir, skip_existing=True
            )

        assert len(result) == 1
        assert result[0] == existing

    def test_creates_output_directory(self, acquirer_instance, tmp_path):
        """Should create output directory if missing."""
        pytest.importorskip("earthaccess")
        output_dir = tmp_path / "new_dir"
        assert not output_dir.exists()

        # Use a non-empty URL list so we get past the early return,
        # but mock the session to avoid actual downloads
        urls = ["https://example.com/path/file1.hdf"]

        with patch("earthaccess.login", return_value=True), \
             patch("earthaccess.get_requests_https_session") as mock_session:
            mock_sess = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"fake data"
            mock_resp.raise_for_status = Mock()
            mock_sess.get.return_value = mock_resp
            mock_session.return_value = mock_sess

            acquirer_instance._download_with_earthaccess(urls, output_dir)

        assert output_dir.exists()


# =============================================================================
# Download Granules (convenience) Tests
# =============================================================================

@pytest.mark.acquisition
class TestDownloadGranulesEarthaccess:
    """Tests for _download_granules_earthaccess convenience method."""

    def test_returns_empty_when_no_granules(self, acquirer_instance, tmp_path):
        """Should return empty list when no granules found."""
        with patch.object(
            acquirer_instance, "_search_granules_cmr", return_value=[]
        ):
            result = acquirer_instance._download_granules_earthaccess(
                "MOD10A1", tmp_path
            )

        assert result == []

    def test_returns_empty_when_no_urls(self, acquirer_instance, tmp_path):
        """Should return empty list when granules have no matching URLs."""
        fake_granules = [{"id": "g1", "links": []}]

        with patch.object(
            acquirer_instance, "_search_granules_cmr", return_value=fake_granules
        ), patch.object(
            acquirer_instance, "_get_download_urls", return_value=[]
        ):
            result = acquirer_instance._download_granules_earthaccess(
                "MOD10A1", tmp_path
            )

        assert result == []

    def test_chains_search_and_download(self, acquirer_instance, tmp_path):
        """Should chain CMR search, URL extraction, and download."""
        fake_granules = [{"id": "g1", "links": [{"href": "https://x.com/f.hdf"}]}]
        fake_urls = ["https://x.com/f.hdf"]
        fake_paths = [tmp_path / "f.hdf"]

        with patch.object(
            acquirer_instance, "_search_granules_cmr", return_value=fake_granules
        ) as mock_search, patch.object(
            acquirer_instance, "_get_download_urls", return_value=fake_urls
        ) as mock_urls, patch.object(
            acquirer_instance, "_download_with_earthaccess", return_value=fake_paths
        ) as mock_download:
            result = acquirer_instance._download_granules_earthaccess(
                "MOD10A1", tmp_path, version="61"
            )

        mock_search.assert_called_once_with("MOD10A1", version="61")
        mock_urls.assert_called_once()
        mock_download.assert_called_once_with(fake_urls, tmp_path)
        assert result == fake_paths


# =============================================================================
# Count Available Granules Tests
# =============================================================================

@pytest.mark.acquisition
class TestCountAvailableGranules:
    """Tests for _count_available_granules method."""

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_returns_count_from_header(self, mock_get, acquirer_instance):
        """Should return count from CMR-Hits header."""
        mock_response = Mock()
        mock_response.headers = {"CMR-Hits": "42"}
        mock_get.return_value = mock_response

        count = acquirer_instance._count_available_granules("MOD10A1")

        assert count == 42

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_returns_zero_on_error(self, mock_get, acquirer_instance):
        """Should return 0 on network error."""
        mock_get.side_effect = ConnectionError("fail")

        count = acquirer_instance._count_available_granules("MOD10A1")

        assert count == 0

    @patch("symfluence.data.acquisition.earthaccess_base.requests.get")
    def test_returns_zero_when_header_missing(self, mock_get, acquirer_instance):
        """Should return 0 when CMR-Hits header is absent."""
        mock_response = Mock()
        mock_response.headers = {}
        mock_get.return_value = mock_response

        count = acquirer_instance._count_available_granules("MOD10A1")

        assert count == 0
