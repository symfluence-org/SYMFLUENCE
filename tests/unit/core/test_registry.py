"""Comprehensive tests for the unified Registry[T] class."""

from __future__ import annotations

import warnings

import pytest

from symfluence.core.registry import Registry, _LazyEntry

# ======================================================================
# Fixtures
# ======================================================================


class _DummyRunner:
    model_name = "DUMMY"

    def run(self, **kw):
        return None


class _DummyPreprocessor:
    MODEL_NAME = "DUMMY"

    def run_preprocessing(self):
        return True


@pytest.fixture()
def reg():
    """Fresh registry with UPPERCASE normalization (default)."""
    return Registry("test")


@pytest.fixture()
def lower_reg():
    """Fresh registry with lowercase normalization."""
    return Registry("test_lower", normalize=str.lower)


# ======================================================================
# add / get / __getitem__ / __contains__
# ======================================================================


class TestAddAndLookup:
    def test_add_direct(self, reg):
        reg.add("summa", _DummyRunner)
        assert reg.get("summa") is _DummyRunner
        assert reg.get("SUMMA") is _DummyRunner
        assert "SUMMA" in reg

    def test_add_decorator(self, reg):
        @reg.add("fuse")
        class FuseRunner:
            pass

        assert reg["FUSE"] is FuseRunner

    def test_get_returns_none_on_miss(self, reg):
        assert reg.get("NONEXISTENT") is None

    def test_get_returns_custom_default(self, reg):
        sentinel = object()
        assert reg.get("NONEXISTENT", sentinel) is sentinel

    def test_getitem_raises_keyerror(self, reg):
        with pytest.raises(KeyError, match="unknown key"):
            reg["NONEXISTENT"]

    def test_contains_false_on_miss(self, reg):
        assert "MISSING" not in reg

    def test_key_normalization_uppercase(self, reg):
        reg.add("lowercase_key", _DummyRunner)
        assert reg.get("LOWERCASE_KEY") is _DummyRunner

    def test_key_normalization_lowercase(self, lower_reg):
        lower_reg.add("ERA5", _DummyRunner)
        assert lower_reg.get("era5") is _DummyRunner
        assert "era5" in lower_reg


# ======================================================================
# Metadata
# ======================================================================


class TestMetadata:
    def test_meta_stored_on_add(self, reg):
        reg.add("summa", _DummyRunner, runner_method="run_summa", version=3)
        meta = reg.meta("summa")
        assert meta["runner_method"] == "run_summa"
        assert meta["version"] == 3

    def test_meta_empty_when_none(self, reg):
        reg.add("summa", _DummyRunner)
        assert reg.meta("summa") == {}

    def test_meta_missing_key(self, reg):
        assert reg.meta("MISSING") == {}


# ======================================================================
# Aliases
# ======================================================================


class TestAliases:
    def test_alias_resolves_to_canonical(self, reg):
        reg.add("SACSMA", _DummyRunner)
        reg.alias("SAC-SMA", "SACSMA")
        assert reg.get("SAC-SMA") is _DummyRunner
        assert reg["SAC-SMA"] is _DummyRunner
        assert "SAC-SMA" in reg

    def test_alias_with_normalization(self, lower_reg):
        lower_reg.add("semidistributed", _DummyRunner)
        lower_reg.alias("subset", "semidistributed")
        assert lower_reg.get("SUBSET") is _DummyRunner  # normalized to lowercase

    def test_alias_before_canonical_exists(self, reg):
        # Alias can be created before the canonical key is registered
        reg.alias("ALIAS", "CANON")
        assert reg.get("ALIAS") is None  # canonical not yet present
        reg.add("CANON", _DummyRunner)
        assert reg.get("ALIAS") is _DummyRunner

    @pytest.mark.parametrize("alias_first", [True, False])
    def test_real_entry_wins_over_alias_of_same_name(self, reg, alias_first):
        # model_manifest() declares aliases without checking reg.keys() (which
        # would force the lazy seeders), so precedence must hold at lookup
        # time whichever was declared first.
        if alias_first:
            reg.alias("XINANJIANG", "XAJ")
        reg.add("XAJ", _DummyPreprocessor)
        reg.add("XINANJIANG", _DummyRunner)
        if not alias_first:
            reg.alias("XINANJIANG", "XAJ")
        assert reg.get("XINANJIANG") is _DummyRunner
        assert reg["XINANJIANG"] is _DummyRunner
        assert reg.get("XAJ") is _DummyPreprocessor


# ======================================================================
# Lazy imports
# ======================================================================


class TestLazyImports:
    def test_add_lazy_resolves_on_get(self, reg):
        # Use a class that definitely exists
        reg.add_lazy("MATH_LOG", "math.log")
        import math
        assert reg.get("MATH_LOG") is math.log

    def test_add_lazy_resolves_on_getitem(self, reg):
        reg.add_lazy("MATH_LOG", "math.log")
        import math
        assert reg["MATH_LOG"] is math.log

    def test_add_lazy_caches_after_resolve(self, reg):
        reg.add_lazy("MATH_LOG", "math.log")
        _ = reg.get("MATH_LOG")
        # Second access should not be a _LazyEntry
        assert not isinstance(reg._entries.get("MATH_LOG"), _LazyEntry)

    def test_add_lazy_bad_path_raises(self, reg):
        reg.add_lazy("BAD", "nonexistent.module.Class")
        with pytest.raises((ImportError, ModuleNotFoundError)):
            reg["BAD"]

    def test_add_lazy_with_metadata(self, reg):
        reg.add_lazy("MATH_LOG", "math.log", kind="function")
        assert reg.meta("MATH_LOG") == {"kind": "function"}


# ======================================================================
# Discovery
# ======================================================================


class TestDiscovery:
    def test_keys(self, reg):
        reg.add("B", _DummyRunner)
        reg.add("A", _DummyPreprocessor)
        assert reg.keys() == ["A", "B"]

    def test_items(self, reg):
        reg.add("A", _DummyRunner)
        items = reg.items()
        assert items == [("A", _DummyRunner)]

    def test_len(self, reg):
        assert len(reg) == 0
        reg.add("X", _DummyRunner)
        assert len(reg) == 1

    def test_iter(self, reg):
        reg.add("C", _DummyRunner)
        reg.add("A", _DummyRunner)
        assert list(reg) == ["A", "C"]

    def test_repr(self, reg):
        assert "test" in repr(reg)
        assert "0 entries" in repr(reg)

    def test_bool_always_true(self, reg):
        assert bool(reg) is True

    def test_summary(self, reg):
        reg.add("A", _DummyRunner)
        reg.alias("B", "A")
        s = reg.summary()
        assert s["name"] == "test"
        assert s["entries"] == 1
        assert s["aliases"] == 1
        assert "A" in s["keys"]


# ======================================================================
# Lifecycle: clear, freeze, remove
# ======================================================================


class TestLifecycle:
    def test_clear(self, reg):
        reg.add("A", _DummyRunner)
        reg.alias("B", "A")
        reg.clear()
        assert len(reg) == 0
        assert reg.get("A") is None
        assert reg.get("B") is None

    def test_freeze_blocks_add(self, reg):
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.add("A", _DummyRunner)

    def test_freeze_blocks_add_lazy(self, reg):
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.add_lazy("A", "math.log")

    def test_freeze_blocks_alias(self, reg):
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.alias("A", "B")

    def test_freeze_blocks_remove(self, reg):
        reg.add("A", _DummyRunner)
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.remove("A")

    def test_freeze_allows_read(self, reg):
        reg.add("A", _DummyRunner)
        reg.freeze()
        assert reg.get("A") is _DummyRunner
        assert reg["A"] is _DummyRunner
        assert "A" in reg

    def test_clear_unfreezes(self, reg):
        reg.freeze()
        reg.clear()
        # Should not raise
        reg.add("A", _DummyRunner)

    def test_remove(self, reg):
        reg.add("A", _DummyRunner, x=1)
        reg.remove("A")
        assert reg.get("A") is None
        assert reg.meta("A") == {}

    def test_remove_nonexistent_is_noop(self, reg):
        reg.remove("MISSING")  # should not raise


# ======================================================================
# Protocol validation (advisory)
# ======================================================================


class TestProtocolValidation:
    def test_protocol_warns_on_missing_attrs(self):
        from symfluence.core.modeling.base.protocols import ModelRunner

        reg = Registry("runners", protocol=ModelRunner)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class BadRunner:
                pass

            reg.add("BAD", BadRunner)
            # Should have warned
            protocol_warnings = [x for x in w if "may not satisfy" in str(x.message)]
            assert len(protocol_warnings) >= 1

    def test_protocol_no_warn_on_conformant(self):
        from symfluence.core.modeling.base.protocols import ModelRunner

        reg = Registry("runners", protocol=ModelRunner)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.add("GOOD", _DummyRunner)
            protocol_warnings = [x for x in w if "may not satisfy" in str(x.message)]
            assert len(protocol_warnings) == 0

    def test_no_protocol_no_validation(self, reg):
        # reg has no protocol — should never warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class Anything:
                pass

            reg.add("X", Anything)
            protocol_warnings = [x for x in w if "may not satisfy" in str(x.message)]
            assert len(protocol_warnings) == 0


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_overwrite_entry(self, reg):
        reg.add("A", _DummyRunner)
        reg.add("A", _DummyPreprocessor)
        assert reg["A"] is _DummyPreprocessor

    def test_alias_chains_are_single_hop(self, reg):
        reg.add("REAL", _DummyRunner)
        reg.alias("A1", "REAL")
        reg.alias("A2", "A1")
        # A2 -> A1, but A1 is not in _entries (it's only in _aliases)
        # So A2 resolves to A1 which is an alias that points to REAL,
        # but _resolve_alias only does one hop.
        assert reg.get("A2") is None  # single-hop: A2 -> A1 (not in entries)
        assert reg.get("A1") is _DummyRunner  # A1 -> REAL (in entries)

    def test_add_returns_value_in_direct_form(self, reg):
        result = reg.add("A", _DummyRunner)
        assert result is _DummyRunner

    def test_items_resolves_lazy(self, reg):
        reg.add_lazy("MATH_LOG", "math.log")
        items = reg.items()
        import math
        assert items[0] == ("MATH_LOG", math.log)


# ======================================================================
# Plugin discovery (_discover_plugins)
# ======================================================================


class TestPluginDiscovery:
    """Tests for entry-point–based plugin loading."""

    def test_discover_calls_plugin(self, monkeypatch):
        """A healthy plugin callable is invoked during discovery."""
        from unittest.mock import MagicMock

        from symfluence.core import _bootstrap

        called = MagicMock()

        # Fake entry point
        class _FakeEP:
            name = "fake_plugin"
            value = "fake_package:register"

            def load(self):
                return called

        monkeypatch.setattr(
            _bootstrap,
            "entry_points",
            lambda group: [_FakeEP()],
            raising=False,
        )
        # Patch at module level since _discover_plugins uses a local import
        import importlib.metadata

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [_FakeEP()])
        _bootstrap._discover_plugins()
        called.assert_called_once()

    def test_discover_declines_incompatible_plugin_before_call(self, monkeypatch, caplog):
        """Contract skew is rejected before a plugin mutates registries."""
        import importlib.metadata
        import logging as _logging
        from unittest.mock import MagicMock

        from symfluence.core import _bootstrap
        from symfluence.core.contracts import plugin_contracts

        called = MagicMock()
        incompatible = plugin_contracts(models="0.99.0")(called)

        class _FakeEP:
            name = "future_model_plugin"
            value = "future_package:register"

            def load(self):
                return incompatible

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [_FakeEP()])
        with caplog.at_level(_logging.WARNING, logger="symfluence.core._bootstrap"):
            _bootstrap._discover_plugins()

        called.assert_not_called()
        assert any(
            "incompatible SYMFLUENCE contract" in record.getMessage()
            and "Registration was not attempted" in record.getMessage()
            for record in caplog.records
        )

    def test_broken_plugin_does_not_raise(self, monkeypatch):
        """A plugin that raises is logged and skipped, not propagated."""
        from symfluence.core import _bootstrap

        class _BrokenEP:
            name = "broken_plugin"
            value = "broken_package:register"

            def load(self):
                def boom():
                    raise RuntimeError("plugin exploded")
                return boom

        import importlib.metadata

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [_BrokenEP()])
        # Should not raise
        _bootstrap._discover_plugins()

    def test_no_plugins_is_fine(self, monkeypatch):
        """No entry points registered — discovery is a no-op."""
        import importlib.metadata

        from symfluence.core import _bootstrap

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])
        _bootstrap._discover_plugins()  # should not raise

    def test_plugin_can_register_into_R(self, monkeypatch):
        """End-to-end: a plugin registers a class into R and it's discoverable."""
        from symfluence.core import _bootstrap
        from symfluence.core.registries import R

        class _PluginRunner:
            pass

        def fake_register():
            R.runners.add("PLUGIN_TEST_MODEL", _PluginRunner)

        class _FakeEP:
            name = "test_model_plugin"
            value = "test_package:register"

            def load(self):
                return fake_register

        import importlib.metadata

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [_FakeEP()])

        _bootstrap._discover_plugins()

        assert R.runners.get("PLUGIN_TEST_MODEL") is _PluginRunner
        # Clean up
        R.runners.remove("PLUGIN_TEST_MODEL")

    @staticmethod
    def _ep_raising(exc):
        """Build a fake entry point whose register() raises *exc*."""

        class _EP:
            name = "failing_plugin"
            value = "failing_package:register"

            def load(self):
                def register():
                    raise exc
                return register

        return _EP()

    def test_missing_thirdparty_dep_stays_quiet(self, monkeypatch, caplog):
        """A genuinely missing third-party module is debug-logged, not warned.

        e.g. an MPI/GPU-only model on a laptop without the dependency.
        """
        import importlib.metadata
        import logging as _logging

        from symfluence.core import _bootstrap

        exc = ModuleNotFoundError("No module named 'jax'", name="jax")
        monkeypatch.setattr(
            importlib.metadata, "entry_points", lambda group: [self._ep_raising(exc)]
        )
        with caplog.at_level(_logging.DEBUG, logger="symfluence.core._bootstrap"):
            _bootstrap._discover_plugins()

        warnings_ = [r for r in caplog.records if r.levelno >= _logging.WARNING
                     and "failing_plugin" in r.getMessage()]
        assert not warnings_, "missing optional dependency should stay at DEBUG"
        assert any("optional dependency missing" in r.getMessage()
                   for r in caplog.records)

    def test_api_drift_import_error_warns(self, monkeypatch, caplog):
        """An ImportError caused by SYMFLUENCE API drift is warned loudly.

        Regression: installed jxaj/jsacsma/jhbv plugins importing the removed
        ``StandardModelPostprocessor`` alias were debug-logged as an "optional
        dependency missing", silently removing those models from the registry
        and voiding whole calibration runs.
        """
        import importlib.metadata
        import logging as _logging

        from symfluence.core import _bootstrap

        exc = ImportError(
            "cannot import name 'StandardModelPostprocessor' from "
            "'symfluence.core.modeling.base.standard_postprocessor'",
            name="symfluence.core.modeling.base.standard_postprocessor",
        )
        monkeypatch.setattr(
            importlib.metadata, "entry_points", lambda group: [self._ep_raising(exc)]
        )
        with caplog.at_level(_logging.DEBUG, logger="symfluence.core._bootstrap"):
            _bootstrap._discover_plugins()

        warnings_ = [r for r in caplog.records if r.levelno >= _logging.WARNING
                     and "incompatible" in r.getMessage()]
        assert warnings_, "API-drift ImportError must surface as a WARNING"

    def test_missing_symfluence_module_warns(self, monkeypatch, caplog):
        """A ModuleNotFoundError for a symfluence.* module is API drift, not
        an optional dependency, and must be warned."""
        import importlib.metadata
        import logging as _logging

        from symfluence.core import _bootstrap

        exc = ModuleNotFoundError(
            "No module named 'symfluence.core.modeling.base.standard_postprocessor'",
            name="symfluence.core.modeling.base.standard_postprocessor",
        )
        monkeypatch.setattr(
            importlib.metadata, "entry_points", lambda group: [self._ep_raising(exc)]
        )
        with caplog.at_level(_logging.DEBUG, logger="symfluence.core._bootstrap"):
            _bootstrap._discover_plugins()

        warnings_ = [r for r in caplog.records if r.levelno >= _logging.WARNING
                     and "incompatible" in r.getMessage()]
        assert warnings_
