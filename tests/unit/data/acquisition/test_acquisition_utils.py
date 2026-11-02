"""
Unit Tests for Acquisition Utility Functions.

Tests the utility functions:
- create_robust_session(): session creation with retry adapter
- resolve_credentials(): config, environment, netrc fallback chain
- download_file_streaming(): successful download, atomic write cleanup
- atomic_write(): context manager for safe file writes
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
from fixtures.acquisition_fixtures import MockResponseFactory, MockSessionFactory

# =============================================================================
# Create Robust Session Tests
# =============================================================================

@pytest.mark.acquisition
class TestCreateRobustSession:
    """Tests for create_robust_session function."""

    def test_returns_session(self):
        """Should return a requests.Session object."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session()

        assert session is not None
        assert hasattr(session, 'get')
        assert hasattr(session, 'post')

    def test_session_has_retry_adapter(self):
        """Session should have retry adapter mounted."""
        from requests.adapters import HTTPAdapter

        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session()

        # Check adapters are mounted
        assert 'https://' in session.adapters
        assert 'http://' in session.adapters

        # Check adapters are HTTPAdapter (which includes retry)
        assert isinstance(session.adapters['https://'], HTTPAdapter)
        assert isinstance(session.adapters['http://'], HTTPAdapter)

    def test_custom_max_retries(self):
        """Should respect custom max_retries parameter."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session(max_retries=10)

        # Session should be created successfully
        assert session is not None

    def test_custom_status_forcelist(self):
        """Should respect custom status_forcelist."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session(status_forcelist=[500, 502])

        assert session is not None

    def test_custom_backoff_factor(self):
        """Should respect custom backoff_factor."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session(backoff_factor=2.0)

        assert session is not None


# =============================================================================
# Download File Streaming Tests
# =============================================================================

@pytest.mark.acquisition
class TestDownloadFileStreaming:
    """Tests for download_file_streaming function."""

    def test_successful_download(self, tmp_path):
        """Should successfully download and save file."""
        from symfluence.data.acquisition.utils import download_file_streaming

        content = b"test file content"
        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=content)
        )

        target = tmp_path / "downloaded.txt"

        bytes_downloaded = download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session
        )

        assert target.exists()
        assert target.read_bytes() == content
        assert bytes_downloaded == len(content)

    def test_creates_parent_directory(self, tmp_path):
        """Should create parent directories if needed."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"data")
        )

        # Nested path that doesn't exist
        target = tmp_path / "a" / "b" / "c" / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session
        )

        assert target.exists()

    def test_atomic_write_with_temp_file(self, tmp_path):
        """Should use .part file for atomic write."""
        from symfluence.data.acquisition.utils import download_file_streaming

        content = b"test content"
        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=content)
        )

        target = tmp_path / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session,
            use_temp_file=True
        )

        # Target should exist, .part should not
        assert target.exists()
        assert not Path(str(target) + '.part').exists()

    def test_cleanup_on_error(self, tmp_path):
        """Should clean up .part file on error."""
        from requests.exceptions import HTTPError

        from symfluence.data.acquisition.utils import download_file_streaming

        mock_response = MockResponseFactory.error(500, "Server Error")
        mock_session = MockSessionFactory.create(default_response=mock_response)

        target = tmp_path / "file.txt"

        with pytest.raises(HTTPError):
            download_file_streaming(
                "https://example.com/file.txt",
                target,
                session=mock_session
            )

        # Neither target nor .part should exist
        assert not target.exists()
        assert not Path(str(target) + '.part').exists()

    def test_verifies_content_length(self, tmp_path):
        """Should verify download matches Content-Length."""
        from symfluence.data.acquisition.utils import download_file_streaming

        # Response claims 1000 bytes but only has 100
        content = b"x" * 100
        mock_response = MockResponseFactory.success(content=content)
        mock_response.headers['content-length'] = '1000'

        mock_session = MockSessionFactory.create(default_response=mock_response)

        target = tmp_path / "file.txt"

        with pytest.raises(IOError) as exc_info:
            download_file_streaming(
                "https://example.com/file.txt",
                target,
                session=mock_session
            )

        assert "Incomplete download" in str(exc_info.value)

    def test_passes_auth(self, tmp_path):
        """Should pass auth credentials to request."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"data")
        )

        target = tmp_path / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session,
            auth=("user", "pass")
        )

        # Verify auth was passed
        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs['auth'] == ("user", "pass")

    def test_passes_headers(self, tmp_path):
        """Should pass custom headers to request."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"data")
        )

        target = tmp_path / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session,
            headers={"X-Custom": "value"}
        )

        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs['headers'] == {"X-Custom": "value"}


# =============================================================================
# Atomic Write Tests
# =============================================================================

@pytest.mark.acquisition
class TestAtomicWrite:
    """Tests for atomic_write context manager."""

    def test_successful_write(self, tmp_path):
        """Should write to target on success."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"

        with atomic_write(target) as temp_path:
            temp_path.write_text("test content")

        assert target.exists()
        assert target.read_text() == "test content"
        assert not Path(str(target) + '.part').exists()

    def test_creates_parent_directory(self, tmp_path):
        """Should create parent directories."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "new" / "nested" / "dir" / "file.txt"

        with atomic_write(target) as temp_path:
            temp_path.write_text("data")

        assert target.exists()

    def test_cleanup_on_error(self, tmp_path):
        """Should clean up .part file on error."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"

        with pytest.raises(RuntimeError):
            with atomic_write(target) as temp_path:
                temp_path.write_text("partial")
                raise RuntimeError("Simulated error")

        # Target and .part should not exist
        assert not target.exists()
        assert not Path(str(target) + '.part').exists()

    def test_yields_part_path(self, tmp_path):
        """Should yield .part path for writing."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"

        with atomic_write(target) as temp_path:
            assert str(temp_path).endswith('.part')
            temp_path.write_text("data")

    def test_overwrites_existing_target(self, tmp_path):
        """Should overwrite existing target file."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"
        target.write_text("old content")

        with atomic_write(target) as temp_path:
            temp_path.write_text("new content")

        assert target.read_text() == "new content"


# =============================================================================
# Resolve Credentials Tests
# =============================================================================

@pytest.mark.acquisition
class TestResolveCredentials:
    """Tests for resolve_credentials function."""

    def test_from_environment_variables(self, clean_environment):
        """Should get credentials from environment variables."""
        from symfluence.data.acquisition.utils import resolve_credentials

        os.environ['TEST_USERNAME'] = 'env_user'
        os.environ['TEST_PASSWORD'] = 'env_pass'

        try:
            username, password = resolve_credentials(
                hostname='example.com',
                env_prefix='TEST'
            )

            assert username == 'env_user'
            assert password == 'env_pass'
        finally:
            os.environ.pop('TEST_USERNAME', None)
            os.environ.pop('TEST_PASSWORD', None)

    def test_from_config_dict(self, clean_environment):
        """Should get credentials from config dictionary."""
        from symfluence.data.acquisition.utils import resolve_credentials

        config = {
            'TEST_USERNAME': 'config_user',
            'TEST_PASSWORD': 'config_pass',
        }

        username, password = resolve_credentials(
            hostname='example.com',
            env_prefix='TEST',
            config=config
        )

        assert username == 'config_user'
        assert password == 'config_pass'

    def test_config_takes_priority_over_env(self, clean_environment):
        """Config should take priority over environment variables."""
        from symfluence.data.acquisition.utils import resolve_credentials

        os.environ['TEST_USERNAME'] = 'env_user'
        os.environ['TEST_PASSWORD'] = 'env_pass'

        config = {
            'TEST_USERNAME': 'config_user',
            'TEST_PASSWORD': 'config_pass',
        }

        try:
            username, password = resolve_credentials(
                hostname='example.com',
                env_prefix='TEST',
                config=config
            )

            # Config should win
            assert username == 'config_user'
            assert password == 'config_pass'
        finally:
            os.environ.pop('TEST_USERNAME', None)
            os.environ.pop('TEST_PASSWORD', None)

    def test_returns_none_when_not_found(self, clean_environment):
        """Should return (None, None) when credentials not found."""
        from symfluence.data.acquisition.utils import resolve_credentials

        username, password = resolve_credentials(
            hostname='example.com',
            env_prefix='NONEXISTENT'
        )

        assert username is None
        assert password is None

    def test_partial_credentials_from_env(self, clean_environment):
        """Should handle partial credentials (only username or password)."""
        from symfluence.data.acquisition.utils import resolve_credentials

        os.environ['TEST_USERNAME'] = 'only_user'
        # No password set

        try:
            username, password = resolve_credentials(
                hostname='example.com',
                env_prefix='TEST'
            )

            # Partial credentials should return None, None
            # (requires both to be set)
            assert username is None or password is None
        finally:
            os.environ.pop('TEST_USERNAME', None)

    def test_from_netrc(self, clean_environment, tmp_path):
        """Should get credentials from .netrc file."""
        from symfluence.data.acquisition.utils import resolve_credentials

        # Create a mock .netrc file
        netrc_content = "machine example.com login netrc_user password netrc_pass"
        netrc_path = tmp_path / '.netrc'
        netrc_path.write_text(netrc_content)

        with patch('pathlib.Path.home', return_value=tmp_path):
            username, password = resolve_credentials(
                hostname='example.com'
            )

        assert username == 'netrc_user'
        assert password == 'netrc_pass'

    def test_alt_hostnames_checked(self, clean_environment, tmp_path):
        """Should check alternative hostnames in .netrc."""
        from symfluence.data.acquisition.utils import resolve_credentials

        netrc_content = "machine alt.example.com login alt_user password alt_pass"
        netrc_path = tmp_path / '.netrc'
        netrc_path.write_text(netrc_content)

        with patch('pathlib.Path.home', return_value=tmp_path):
            username, password = resolve_credentials(
                hostname='example.com',
                alt_hostnames=['alt.example.com']
            )

        assert username == 'alt_user'
        assert password == 'alt_pass'


# =============================================================================
# Get Earthdata Credentials Tests
# =============================================================================

@pytest.mark.acquisition
class TestGetEarthdataCredentials:
    """Tests for get_earthdata_credentials convenience function."""

    def test_uses_earthdata_env_prefix(self, clean_environment, tmp_path):
        """Should use EARTHDATA_ environment variable prefix."""
        from symfluence.data.acquisition.utils import get_earthdata_credentials

        os.environ['EARTHDATA_USERNAME'] = 'earth_user'
        os.environ['EARTHDATA_PASSWORD'] = 'earth_pass'

        try:
            # Patch home to avoid reading system .netrc
            with patch('pathlib.Path.home', return_value=tmp_path):
                username, password = get_earthdata_credentials()

            assert username == 'earth_user'
            assert password == 'earth_pass'
        finally:
            os.environ.pop('EARTHDATA_USERNAME', None)
            os.environ.pop('EARTHDATA_PASSWORD', None)

    def test_checks_urs_earthdata_hostname(self, clean_environment, tmp_path):
        """Should check urs.earthdata.nasa.gov in .netrc."""
        from symfluence.data.acquisition.utils import get_earthdata_credentials

        netrc_content = "machine urs.earthdata.nasa.gov login urs_user password urs_pass"
        netrc_path = tmp_path / '.netrc'
        netrc_path.write_text(netrc_content)

        with patch('pathlib.Path.home', return_value=tmp_path):
            username, password = get_earthdata_credentials()

        assert username == 'urs_user'
        assert password == 'urs_pass'


# =============================================================================
# Resolve Earthdata Token Tests
# =============================================================================

@pytest.mark.acquisition
class TestResolveEarthdataToken:
    """Tests for resolve_earthdata_token function."""

    def test_from_environment(self, clean_environment):
        """Should get token from EARTHDATA_TOKEN env var."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        os.environ['EARTHDATA_TOKEN'] = 'my-test-token-123'

        try:
            token = resolve_earthdata_token()
            assert token == 'my-test-token-123'
        finally:
            os.environ.pop('EARTHDATA_TOKEN', None)

    def test_from_config(self, clean_environment):
        """Should get token from config dictionary."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        config = {'EARTHDATA_TOKEN': 'config-token-456'}
        token = resolve_earthdata_token(config=config)
        assert token == 'config-token-456'

    def test_config_takes_precedence_over_env(self, clean_environment):
        """Config (explicit, per-project) should take precedence over the environment."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        os.environ['EARTHDATA_TOKEN'] = 'env-token'
        config = {'EARTHDATA_TOKEN': 'config-token'}

        try:
            token = resolve_earthdata_token(config=config)
            assert token == 'config-token'
        finally:
            os.environ.pop('EARTHDATA_TOKEN', None)

    def test_returns_none_when_not_found(self, clean_environment):
        """Should return None when no token is available."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        token = resolve_earthdata_token()
        assert token is None

    def test_returns_none_for_empty_config(self, clean_environment):
        """Should return None when config has no token."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        token = resolve_earthdata_token(config={'OTHER_KEY': 'value'})
        assert token is None


# =============================================================================
# Get CDS Credentials Tests
# =============================================================================

@pytest.mark.acquisition
class TestGetCdsCredentials:
    """Tests for get_cds_credentials convenience function."""

    def test_from_environment(self, clean_environment):
        """Should get CDS credentials from environment."""
        from symfluence.data.acquisition.utils import get_cds_credentials

        os.environ['CDSAPI_URL'] = 'https://cds.example.com/api'
        os.environ['CDSAPI_KEY'] = '12345:abcdef'

        try:
            url, key = get_cds_credentials()

            assert url == 'https://cds.example.com/api'
            assert key == '12345:abcdef'
        finally:
            os.environ.pop('CDSAPI_URL', None)
            os.environ.pop('CDSAPI_KEY', None)

    def test_from_config(self, clean_environment):
        """Should get CDS credentials from config."""
        from symfluence.data.acquisition.utils import get_cds_credentials

        config = {
            'CDSAPI_URL': 'https://config.cds.example.com/api',
            'CDSAPI_KEY': '99999:config_key',
        }

        url, key = get_cds_credentials(config=config)

        assert url == 'https://config.cds.example.com/api'
        assert key == '99999:config_key'

    def test_returns_none_when_not_found(self, clean_environment):
        """Should return (None, None) when not found."""
        from symfluence.data.acquisition.utils import get_cds_credentials

        url, key = get_cds_credentials()

        assert url is None
        assert key is None


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.acquisition
class TestUtilsEdgeCases:
    """Edge case tests for utility functions."""

    def test_download_zero_byte_file(self, tmp_path):
        """Should handle zero-byte file download."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"")
        )

        target = tmp_path / "empty.txt"

        bytes_downloaded = download_file_streaming(
            "https://example.com/empty.txt",
            target,
            session=mock_session
        )

        assert target.exists()
        assert bytes_downloaded == 0

    def test_download_large_file_chunked(self, tmp_path):
        """Should handle large file in chunks."""
        from symfluence.data.acquisition.utils import download_file_streaming

        # 1MB of data
        content = b"x" * (1024 * 1024)
        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=content)
        )

        target = tmp_path / "large.bin"

        bytes_downloaded = download_file_streaming(
            "https://example.com/large.bin",
            target,
            session=mock_session,
            chunk_size=65536
        )

        assert target.exists()
        assert bytes_downloaded == len(content)


# =============================================================================
# Resumable Download Tests
# =============================================================================


class _FakeRangeResp:
    """A minimal streaming response that honours Range and can break mid-stream.

    ``chunks`` are yielded in order by ``iter_content``; if ``raise_at`` is set,
    the exception ``exc`` is raised *after* yielding that many chunks, simulating
    a connection dropped part-way through the body.
    """

    def __init__(self, status_code, headers, chunks, raise_at=None, exc=None):
        self.status_code = status_code
        self.headers = headers
        self._chunks = list(chunks)
        self._raise_at = raise_at
        self._exc = exc

    def raise_for_status(self):
        from requests.exceptions import HTTPError
        if self.status_code >= 400 and self.status_code != 416:
            raise HTTPError(f"{self.status_code} Error")

    def iter_content(self, chunk_size=65536):
        for i, chunk in enumerate(self._chunks):
            if self._raise_at is not None and i == self._raise_at:
                raise self._exc
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _range_session(get_fn):
    """Wrap a get(url, **kwargs) callable into an object with a .get method."""
    session = MagicMock()
    session.get.side_effect = get_fn
    return session


@pytest.mark.acquisition
class TestDownloadFileResumable:
    """Tests for download_file_resumable (HTTP range-based resume)."""

    def test_resume_after_midstream_break(self, tmp_path):
        """A dropped connection should resume from the .part offset, not restart."""
        from requests.exceptions import ChunkedEncodingError

        from symfluence.data.acquisition.utils import download_file_resumable

        full = bytes((i % 256) for i in range(2048))
        seen_ranges = []
        state = {"n": 0}

        def get(url, stream=True, timeout=None, headers=None, auth=None):
            state["n"] += 1
            seen_ranges.append((headers or {}).get("Range"))
            if state["n"] == 1:
                # First attempt: no range, deliver half then drop the connection.
                return _FakeRangeResp(
                    200,
                    {"content-length": str(len(full))},
                    chunks=[full[:1024], full[1024:]],
                    raise_at=1,
                    exc=ChunkedEncodingError("connection broken"),
                )
            # Second attempt: honour the range request, deliver the remainder.
            return _FakeRangeResp(
                206,
                {
                    "content-length": "1024",
                    "Content-Range": f"bytes 1024-2047/{len(full)}",
                },
                chunks=[full[1024:]],
            )

        target = tmp_path / "soil.zip"
        n = download_file_resumable(
            "https://example.com/soil.zip",
            target,
            session=_range_session(get),
            base_delay=0,
        )

        assert target.read_bytes() == full
        assert n == len(full)
        assert state["n"] == 2
        # First attempt sends no Range; the resume attempt asks for bytes=1024-.
        assert seen_ranges == [None, "bytes=1024-"]
        assert not (tmp_path / "soil.zip.part").exists()

    def test_server_ignoring_range_restarts_cleanly(self, tmp_path):
        """If the server returns 200 to a Range request, restart (no corruption)."""
        from symfluence.data.acquisition.utils import download_file_resumable

        full = b"COMPLETE-PAYLOAD" * 8
        # Pre-seed a stale .part so the function sends a Range header.
        part = tmp_path / "f.bin.part"
        part.write_bytes(b"GARBAGE-PARTIAL")

        def get(url, stream=True, timeout=None, headers=None, auth=None):
            # Ignore the Range: reply 200 with the full body.
            return _FakeRangeResp(
                200, {"content-length": str(len(full))}, chunks=[full]
            )

        target = tmp_path / "f.bin"
        download_file_resumable(
            "https://example.com/f.bin",
            target,
            session=_range_session(get),
            base_delay=0,
        )

        # Must equal the full payload, not garbage+payload.
        assert target.read_bytes() == full

    def test_416_treated_as_complete(self, tmp_path):
        """A 416 on an already-complete .part should finalise, not error."""
        from symfluence.data.acquisition.utils import download_file_resumable

        full = b"already-here" * 4
        part = tmp_path / "d.bin.part"
        part.write_bytes(full)

        def get(url, stream=True, timeout=None, headers=None, auth=None):
            return _FakeRangeResp(416, {}, chunks=[])

        target = tmp_path / "d.bin"
        n = download_file_resumable(
            "https://example.com/d.bin",
            target,
            session=_range_session(get),
            base_delay=0,
        )
        assert target.read_bytes() == full
        assert n == len(full)

    def test_custom_part_path_is_used_and_resumes(self, tmp_path):
        """A caller-supplied part_path (e.g. .part.<pid>) resumes and is honoured."""
        from requests.exceptions import ConnectionError as ReqConnErr

        from symfluence.data.acquisition.utils import download_file_resumable

        full = bytes((i % 251) for i in range(3000))
        custom_part = tmp_path / "tile.tif.part.9999"
        seen_ranges = []
        state = {"n": 0}

        def get(url, stream=True, timeout=None, headers=None, auth=None):
            state["n"] += 1
            seen_ranges.append((headers or {}).get("Range"))
            if state["n"] == 1:
                return _FakeRangeResp(
                    200,
                    {"content-length": str(len(full))},
                    chunks=[full[:2000], full[2000:]],
                    raise_at=1,
                    exc=ReqConnErr("reset"),
                )
            return _FakeRangeResp(
                206,
                {"content-length": "1000", "Content-Range": f"bytes 2000-2999/{len(full)}"},
                chunks=[full[2000:]],
            )

        target = tmp_path / "tile.tif"
        download_file_resumable(
            "https://example.com/tile.tif",
            target,
            session=_range_session(get),
            base_delay=0,
            part_path=custom_part,
        )

        assert target.read_bytes() == full
        # The resume used the custom part file, not the default <target>.part.
        assert seen_ranges == [None, "bytes=2000-"]
        assert not custom_part.exists()
        assert not (tmp_path / "tile.tif.part").exists()

    def test_exhausts_attempts_and_retains_part(self, tmp_path):
        """After max_attempts of persistent failures, raise and keep the .part."""
        from requests.exceptions import ChunkedEncodingError

        from symfluence.data.acquisition.utils import download_file_resumable

        def get(url, stream=True, timeout=None, headers=None, auth=None):
            return _FakeRangeResp(
                200,
                {"content-length": "4096"},
                chunks=[b"x" * 512, b"y" * 512],
                raise_at=1,
                exc=ChunkedEncodingError("still broken"),
            )

        target = tmp_path / "e.bin"
        with pytest.raises((ChunkedEncodingError, OSError)):
            download_file_resumable(
                "https://example.com/e.bin",
                target,
                session=_range_session(get),
                max_attempts=3,
                base_delay=0,
            )
        # The partial file is retained for a future resume (not deleted).
        assert (tmp_path / "e.bin.part").exists()
        assert not target.exists()
