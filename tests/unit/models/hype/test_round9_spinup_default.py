"""HYPE_SPINUP_DAYS: unset/null defers to SPINUP_PERIOD, else one year."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.hype.preprocessor import DEFAULT_HYPE_SPINUP_DAYS, HYPEPreProcessor

TEMPLATE = (Path(__file__).resolve().parents[4] / "src/symfluence/resources/config_templates"
            / "config_template_comprehensive.yaml")


def _cfg(**extra):
    base = dict(
        SYMFLUENCE_DATA_DIR="/tmp/x", SYMFLUENCE_CODE_DIR="/tmp/y", DOMAIN_NAME="d",
        EXPERIMENT_ID="e", EXPERIMENT_TIME_START="2000-01-01 00:00",
        EXPERIMENT_TIME_END="2002-12-31 23:00", DOMAIN_DEFINITION_METHOD="lumped",
        SUB_GRID_DISCRETIZATION="GRUs", HYDROLOGICAL_MODEL="HYPE", FORCING_DATASET="ERA5",
    )
    base.update(extra)
    return SymfluenceConfig(**base)


def test_schema_default_is_none():
    assert _cfg().model.hype.spinup_days is None


def test_explicit_null_is_not_an_override():
    pp = SimpleNamespace(config=_cfg(SPINUP_PERIOD="2000-01-01, 2000-12-31", HYPE_SPINUP_DAYS=None))
    assert HYPEPreProcessor._hype_spinup_days_explicit(pp) is False


def test_explicit_zero_is_an_override():
    pp = SimpleNamespace(config=_cfg(SPINUP_PERIOD="2000-01-01, 2000-12-31", HYPE_SPINUP_DAYS=0))
    assert HYPEPreProcessor._hype_spinup_days_explicit(pp) is True


def test_template_leaves_hype_spinup_days_unset():
    # A config copied from the template must not silently override SPINUP_PERIOD.
    tmpl = yaml.safe_load(TEMPLATE.read_text())
    assert tmpl["HYPE_SPINUP_DAYS"] is None
    assert DEFAULT_HYPE_SPINUP_DAYS == 365
