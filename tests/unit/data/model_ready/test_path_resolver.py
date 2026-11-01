"""Tests for model-ready path resolver with fallback logic."""
from __future__ import annotations

from pathlib import Path

import pytest

from symfluence.core.modeling.model_ready.path_resolver import resolve_model_ready_path


class TestResolveModelReadyPath:
    """Tests for resolve_model_ready_path fallback behaviour."""

    def test_new_path_preferred(self, tmp_path):
        """When new-style dir exists, it should be returned."""
        new_dir = tmp_path / 'data' / 'model_ready' / 'forcings'
        new_dir.mkdir(parents=True)
        result = resolve_model_ready_path(tmp_path, 'forcings')
        assert result == new_dir

    def test_fallback_to_legacy(self, tmp_path):
        """When new-style dir absent but legacy exists, return legacy."""
        legacy = tmp_path / 'forcing' / 'basin_averaged_data'
        legacy.mkdir(parents=True)
        result = resolve_model_ready_path(tmp_path, 'forcings')
        assert result == legacy

    def test_fallback_observations(self, tmp_path):
        legacy = tmp_path / 'observations'
        legacy.mkdir(parents=True)
        result = resolve_model_ready_path(tmp_path, 'observations')
        assert result == legacy

    def test_fallback_attributes(self, tmp_path):
        legacy = tmp_path / 'shapefiles' / 'catchment_intersection'
        legacy.mkdir(parents=True)
        result = resolve_model_ready_path(tmp_path, 'attributes')
        assert result == legacy

    def test_no_fallback(self, tmp_path):
        """With fallback=False, always return new canonical path."""
        result = resolve_model_ready_path(tmp_path, 'forcings', fallback=False)
        assert result == tmp_path / 'data' / 'model_ready' / 'forcings'
        assert not result.exists()

    def test_nothing_exists_returns_new(self, tmp_path):
        """When nothing exists, return new canonical path."""
        result = resolve_model_ready_path(tmp_path, 'forcings')
        assert result == tmp_path / 'data' / 'model_ready' / 'forcings'

    def test_new_preferred_over_legacy(self, tmp_path):
        """Both exist — new path takes precedence."""
        new_dir = tmp_path / 'data' / 'model_ready' / 'forcings'
        new_dir.mkdir(parents=True)
        legacy = tmp_path / 'forcing' / 'basin_averaged_data'
        legacy.mkdir(parents=True)
        result = resolve_model_ready_path(tmp_path, 'forcings')
        assert result == new_dir

    def test_invalid_data_type(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown data_type"):
            resolve_model_ready_path(tmp_path, 'invalid')
