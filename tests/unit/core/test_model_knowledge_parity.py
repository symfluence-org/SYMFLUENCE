# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The behavioural specification for per-model knowledge moved out of ``core``.

Service-decomposition phase 0 moved per-model declarations into the packages
that own them (bounds -> ``register_model_bounds``; runoff / spatial-mode /
routing metadata -> the registered ``ModelConfigSchema``). This module is what
made that safe, and it now serves two distinct purposes — do not conflate them
when editing:

**Parity.** The bound values are pinned against
``data/model_bounds_snapshot.json``: every one of the 438 entries must survive
the split across model packages unchanged. Regenerate the snapshot ONLY when a
bound value is intentionally changed, never to make a failing test pass.

**Specification.** The runoff, routing and spatial-mode assertions started as
characterization — pinning stale values verbatim so the refactor was a provable
no-op — and several were then deliberately flipped to the CORRECTED behaviour
once the underlying bugs were approved for fixing (the routing-integration
table, the spatial-mode lookup, the unknown-source error, HYPE's removal as a
routable source). Each such test's docstring says what changed and why.

So an assertion here is a decision, not a mirror of the implementation. If one
fails, establish which of the two roles it is playing before touching it: a
parity failure means a value moved that should not have; a specification
failure means behaviour diverged from an agreed contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from symfluence.core.calibration.parameters import get_model_bounds
from symfluence.core.calibration.parameters import parameter_bounds_registry as pbr
from symfluence.core.constants import SupportedModels
from symfluence.core.modeling.mixins.spatial_mode_mixin import SpatialModeDetectionMixin
from symfluence.core.modeling.utilities.routing_decider import RoutingDecider
from symfluence.core.modeling.utilities.runoff_loader import get_model_config

pytestmark = [pytest.mark.unit]

_SNAPSHOT = Path(__file__).parent / "data" / "model_bounds_snapshot.json"


# ---------------------------------------------------------------------------
# Calibration bounds
# ---------------------------------------------------------------------------

def _load_snapshot() -> dict:
    with _SNAPSHOT.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_snapshot_covers_every_servable_model():
    """The snapshot must not silently stop covering a model.

    Pinned against ``registered_bound_models()`` — everything
    ``get_model_bounds`` can serve — rather than against
    ``_BUILTIN_MODEL_BOUNDS``. The built-in table is only one of the two
    sources: a model that leaves it for ``register_model_bounds()`` (the whole
    direction of the decomposition) would otherwise pass this test while being
    covered by no parity assertion at all, which is precisely the coverage the
    snapshot exists to provide.
    """
    snapshot = _load_snapshot()
    assert set(snapshot) == set(pbr.registered_bound_models()), (
        "model_bounds_snapshot.json is out of sync with the models "
        "get_model_bounds() can serve. A model may leave the built-in table "
        "only by registering equivalent bounds through register_model_bounds() "
        "— and either way it must stay in the snapshot. Regenerate the snapshot "
        "only when the bound VALUES are intentionally changed."
    )


@pytest.mark.parametrize("model", sorted(_load_snapshot()))
def test_model_bounds_match_snapshot(model):
    """Every bound value survives the move to per-package registration.

    This is the guard that lets the bounds catalogue be split across model
    packages without changing a single calibration result.
    """
    expected = _load_snapshot()[model]
    actual = get_model_bounds(model)

    assert set(actual) == set(expected), (
        f"{model}: parameter set changed "
        f"(added: {sorted(set(actual) - set(expected))}, "
        f"removed: {sorted(set(expected) - set(actual))})"
    )
    for name, bounds in expected.items():
        assert actual[name]["min"] == pytest.approx(bounds["min"]), f"{model}.{name} min"
        assert actual[name]["max"] == pytest.approx(bounds["max"]), f"{model}.{name} max"


# Served bound names are model-local: a model's catalogue entry may be
# namespaced (``fuse_MBASE``) and stripped on the way out, so one served name
# can legitimately mean different physics in different model families. These
# are the eight that legitimately do today. A NEW entry here means a model
# registered a name that collides with another model's without namespacing —
# the exact defect #368 fixed, where SACSMA's definitions silently overrode
# FUSE's and FUSE calibrated against Snow-17 melt bounds.
_KNOWN_NAME_COLLISIONS = {
    "K", "MBASE", "MFMAX", "MFMIN", "PWR", "R2N", "m", "soil_depth",
}


def test_no_new_shared_parameter_collisions():
    """A name serving different bounds in different models must be deliberate.

    Built from the LIVE registry (``get_model_bounds`` over
    ``registered_bound_models``), not from the JSON snapshot. Reading the
    snapshot made this test touch no production code at all: it compared one
    static file against one hardcoded set, so a model registering a colliding
    name could not fail it — only regenerating the snapshot could, and that is
    a deliberate act. The defect this guards (#368, SACSMA's ``MBASE``
    silently overriding FUSE's) happens at *registration* time, so registration
    is what has to be inspected.
    """
    by_name: dict[str, dict[str, tuple]] = {}
    for model in pbr.registered_bound_models():
        for name, bounds in get_model_bounds(model).items():
            by_name.setdefault(name, {})[model] = (bounds["min"], bounds["max"])

    collisions = {
        name: per_model
        for name, per_model in by_name.items()
        if len(set(per_model.values())) > 1
    }

    unexpected = set(collisions) - _KNOWN_NAME_COLLISIONS
    assert not unexpected, (
        "new parameter-name collisions across models: "
        f"{ {n: collisions[n] for n in sorted(unexpected)} } — namespace the "
        "catalogue entry (e.g. 'mymodel_PARAM' + strip_prefix) instead of "
        "letting two models' definitions fight over one name"
    )

    resolved = _KNOWN_NAME_COLLISIONS - set(collisions)
    assert not resolved, (
        f"collisions {sorted(resolved)} no longer occur — remove them from "
        "_KNOWN_NAME_COLLISIONS so the guard keeps its teeth"
    )


# ---------------------------------------------------------------------------
# Runoff-source metadata (runoff_loader.MODEL_CONFIGS)
# ---------------------------------------------------------------------------

# The four routable sources, field for field. These values now come from each
# model's registered ``ModelConfigSchema.runoff``; core no longer tabulates
# them. ``runoff`` is deliberately a sibling of ``output``, not the same thing:
# for models whose calibration path converts output the two describe different
# files (see ParallelCalibrationConfig).
#
# CHANGED: HYPE was removed. It declared a ``runoff`` artifact
# ``{experiment_id}_timestep.nc`` that nothing in the HYPE adapter writes —
# ``config_manager.py`` asks info.txt for ``timeoutput variable COUT EVAP
# SNOW``, so HYPE produces timeCOUT.txt / timeEVAP.txt / timeSNOW.txt. Writing
# a converter was rejected rather than deferred: HYPE's ``cout`` is already
# routed discharge at subbasin outlets, so routing it through mizuRoute would
# route it twice. HYPE is therefore not a routable source at all, and the
# declaration's absence is what says so.
_EXPECTED_RUNOFF = {
    "SUMMA": dict(
        output_dir_key="EXPERIMENT_OUTPUT_SUMMA", output_dir_name="SUMMA",
        default_var="averageRoutedRunoff", default_units="m/s", default_dt="3600",
        output_file_pattern="{experiment_id}_timestep.nc",
        # CORRECTED from hru/hruId. averageRoutedRunoff is a basin variable:
        # SUMMA registers it in bvar_meta and defines every bvar with needGRU,
        # so it is always (time, gru) — confirmed on real output files, which
        # carry BOTH dimensions, which is what made 'hru' look plausible. The
        # old value was patched at runtime by
        # MizuRouteRunner.sync_control_file_dimensions, so the declaration being
        # wrong never surfaced.
        hru_dim="gru", hru_var="gruId", comment_name="SUMMA",
    ),
    "FUSE": dict(
        output_dir_key="EXPERIMENT_OUTPUT_FUSE", output_dir_name="FUSE",
        default_var="q_routed", default_units="m/s", default_dt="86400",
        output_file_pattern="{domain_name}_{experiment_id}_runs_def.nc",
        hru_dim="gru", hru_var="gruId", comment_name="FUSE",
    ),
    "GR": dict(
        output_dir_key="EXPERIMENT_OUTPUT_GR", output_dir_name="GR",
        default_var="q_routed", default_units="m/s", default_dt="86400",
        output_file_pattern="{domain_name}_{experiment_id}_runs_def.nc",
        hru_dim="gru", hru_var="gruId", comment_name="GR4J",
    ),
    "NGEN": dict(
        output_dir_key="EXPERIMENT_OUTPUT_NGEN", output_dir_name="NGEN",
        default_var="runoff", default_units="m/s", default_dt="3600",
        output_file_pattern="{experiment_id}_runoff.nc",
        hru_dim="hru", hru_var="hruId", comment_name="NGEN",
    ),
}


@pytest.mark.parametrize("model", sorted(_EXPECTED_RUNOFF))
def test_runoff_config_fields(model):
    cfg = get_model_config(model)
    for field, expected in _EXPECTED_RUNOFF[model].items():
        assert getattr(cfg, field) == expected, f"{model}.{field}"


@pytest.mark.parametrize("variant", ["GR4J", "GR5J", "GR6J"])
def test_gr_variants_normalize_to_gr(variant):
    assert get_model_config(variant).output_dir_name == "GR"


def test_unknown_model_raises():
    """An unregistered source model is an error, not a silent SUMMA run.

    The old fallback returned SUMMA's runoff layout for any unrecognised name,
    hiding config typos and unroutable models behind plausible SUMMA paths
    until someone noticed the routed hydrograph was wrong.
    """
    from symfluence.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="NOT_A_MODEL"):
        get_model_config("NOT_A_MODEL")


def test_routable_sources_are_exactly_the_declared_ones():
    """The set of models that can feed a routing model, stated once.

    CHANGED: HYPE left this set. Its runoff declaration pointed at a file the
    adapter never writes, and its ``cout`` is already routed discharge at
    subbasin outlets — so there is no honest artifact to route and building one
    would double-route. Membership here is the *only* thing that makes a model
    routable, so dropping the declaration is the whole change.
    """
    from symfluence.core.modeling.utilities.runoff_loader import MODEL_CONFIGS

    assert {name.upper() for name in MODEL_CONFIGS} == {"SUMMA", "FUSE", "GR", "NGEN"}


def test_hype_as_a_routing_source_fails_early_and_says_why():
    """Configuring HYPE as a source must fail here, not at a missing file.

    Before, ``get_model_config('HYPE')`` returned a layout for
    ``{experiment_id}_timestep.nc`` and the run proceeded until mizuRoute (or
    the time-precision fix) went looking for a file HYPE never produces. The
    error now names the model, lists what *is* routable, and says a model
    without a declaration cannot feed a routing model.
    """
    from symfluence.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        get_model_config("HYPE")

    message = str(excinfo.value)
    assert "HYPE" in message
    assert "cannot feed a routing model" in message
    for routable in ("SUMMA", "FUSE", "GR", "NGEN"):
        assert routable in message


def test_runoff_lookup_is_case_insensitive():
    assert get_model_config("summa").output_dir_name == "SUMMA"
    assert get_model_config("  FuSe  ").output_dir_name == "FUSE"


# ---------------------------------------------------------------------------
# Routing decisions (RoutingDecider)
# ---------------------------------------------------------------------------

def test_spatial_mode_keys_table():
    """Which models' spatial mode the routing DECISION is allowed to read.

    ``spatial_orchestrator`` derives ``f"{MODEL}_SPATIAL_MODE"`` by convention
    for every model and acts on it; the decision read a six-entry table. This
    is the same asymmetry just fixed for routing-integration keys, but the
    naive fix — one entry per declaring model — is wrong three ways, and the
    spec below is what handling all three produces.

    1. **A schema default must not decide.** ``config_dict`` is
       ``flatten_nested_config``'s output, and that dumps with
       ``exclude_none=True``, not ``exclude_unset=True``: a config that never
       mentions ``SWAT_SPATIAL_MODE`` still yields
       ``config_dict['SWAT_SPATIAL_MODE'] == 'lumped'``. At this seam an unset
       key is literally indistinguishable from an explicit one, so pydantic's
       ``model_fields_set`` is not reachable and no value-based trick recovers
       it (FUSE's *explicit* lumped and SWAT's *defaulted* lumped are the same
       string). Adding a ``lumped``-defaulting model would therefore let a
       value nobody wrote fire the veto branch, which sits ahead of
       ``ROUTING_MODEL`` and of the routing-integration check — silently
       turning routing OFF for users of the shipped template, which sets
       ``ROUTING_MODEL: mizuRoute`` alongside a lumped domain.

       So the answer to "should the key participate only when explicitly set?"
       is yes, and the way to get that property without provenance is to admit
       only models whose declared default is a *deferral* (``auto``). For those,
       an untouched config resolves to exactly the
       ``DOMAIN_DEFINITION_METHOD`` decision the model gets today: inclusion is
       a provable no-op until a user writes the key. Models whose typed default
       is a concrete mode join only by declaring ``spatial_mode_key`` on their
       ``ModelConfigSchema`` — a reviewed, per-model decision, which is what
       FUSE is.

    2. **``auto``/``default`` are understood.** The comprehensive template ships
       ``VIC_SPATIAL_MODE: auto`` and GR/MESH/VIC default to it. The decider
       used to compare that as a literal string, matching neither the lumped
       test nor the distributed one. It now resolves it from
       ``DOMAIN_DEFINITION_METHOD`` the way ``spatial_orchestrator`` does.

    3. **SUMMA stays an explicit exception.** There is no ``SUMMA_SPATIAL_MODE``
       field anywhere (``SUMMAConfig`` has no ``spatial_mode``); SUMMA's mode is
       ``DOMAIN_DEFINITION_METHOD``. It is in this table only because its
       schema says so, never by convention.

    Membership changes, all deliberate:

    * ``VIC`` joins (typed default ``auto``) — inert until ``VIC_SPATIAL_MODE``
      is set, and then it agrees with the orchestrator.
    * ``HYPE`` and ``NGEN`` leave. Both were dead: neither ``HYPEConfig`` nor
      ``NGENConfig`` declares a ``spatial_mode`` field, so neither
      ``HYPE_SPATIAL_MODE`` nor ``NGEN_SPATIAL_MODE`` exists as a config key and
      no template, test config or example sets one. They could only ever have
      matched a hand-built raw dict.
    * ``GR`` and ``MESH`` stay, and would now also qualify automatically.
    """
    assert RoutingDecider.SPATIAL_MODE_KEYS == {
        "SUMMA": "DOMAIN_DEFINITION_METHOD",
        "FUSE": "FUSE_SPATIAL_MODE",
        "GR": "GR_SPATIAL_MODE",
        "MESH": "MESH_SPATIAL_MODE",
        "VIC": "VIC_SPATIAL_MODE",
    }


def test_summa_has_no_spatial_mode_key_of_its_own():
    """Hazard 3, pinned: nothing may invent ``SUMMA_SPATIAL_MODE``.

    SUMMA maps to ``DOMAIN_DEFINITION_METHOD``. If a ``SUMMA_SPATIAL_MODE``
    field ever appears, the convention-based half of the table would pick it up
    and the most-used model's routing decision would change silently.
    """
    from symfluence.core.registries import R

    summa_schema = R.config_schemas.get("SUMMA")
    assert summa_schema is not None
    assert "spatial_mode" not in summa_schema.model_fields
    assert RoutingDecider.SPATIAL_MODE_KEYS["SUMMA"] == "DOMAIN_DEFINITION_METHOD"


def test_a_stray_summa_spatial_mode_key_is_ignored():
    """This is not hypothetical: the wizard writes ``SUMMA_SPATIAL_MODE``.

    ``cli/wizard/questions.py`` asks "What spatial mode should SUMMA use?" with
    default ``lumped`` and ``project_wizard.py`` writes the answer into the
    generated config, next to ``ROUTING_MODEL`` whose default is ``mizuRoute``.
    Nothing reads the key — ``SUMMAConfig`` has no such field — but a table
    derived by the ``f"{MODEL}_SPATIAL_MODE"`` convention would have picked it
    up and made every wizard-generated SUMMA project veto the routing it just
    asked the user to enable.
    """
    decider = RoutingDecider()
    wizard_like = {
        "SUMMA_SPATIAL_MODE": "lumped",
        "ROUTING_MODEL": "mizuRoute",
        "DOMAIN_DEFINITION_METHOD": "distributed",
        "ROUTING_DELINEATION": "lumped",
    }

    assert decider.needs_routing(wizard_like, "SUMMA") is True


# Every model whose typed config declares a ``<MODEL>_SPATIAL_MODE`` alias,
# with the default a user who never set it receives. Only the ``auto`` rows may
# join SPATIAL_MODE_KEYS automatically; the rest are excluded *because* of that
# default, and this is the evidence for the exclusion.
_TYPED_SPATIAL_MODE_DEFAULTS = {
    "CLM": "lumped", "CLMPARFLOW": "lumped", "CRHM": "lumped",
    "CWATM": "distributed", "FUSE": "lumped", "GR": "auto",
    "GSFLOW": "semi_distributed", "LISFLOOD": "lumped", "MESH": "auto",
    "MHM": "lumped", "MODFLOW": "lumped", "PARFLOW": "lumped",
    "PCRGLOBWB": "distributed", "PIHM": "lumped", "PRMS": "semi_distributed",
    "SWAT": "lumped", "VIC": "auto", "WATFLOOD": "distributed",
    "WFLOW": "lumped", "WRFHYDRO": "distributed",
}


def test_only_deferring_defaults_join_the_table_automatically():
    """The admission rule, stated as a rule rather than as a list.

    A model with a concrete typed default may still be in the table — FUSE is —
    but only via an explicit ``ModelConfigSchema.spatial_mode_key``. What must
    never happen is a model being admitted *by the default alone*.
    """
    from symfluence.core.registries import R

    declared = {
        name: R.config_schemas[name].model_fields["spatial_mode"].default
        for name in R.config_schemas
        if "spatial_mode" in getattr(R.config_schemas[name], "model_fields", {})
        and R.config_schemas[name].model_fields["spatial_mode"].alias
    }
    assert declared == _TYPED_SPATIAL_MODE_DEFAULTS

    explicit = {
        name for name, schema in _registered_model_schemas().items()
        if schema.spatial_mode_key is not None
    }
    for model, default in declared.items():
        in_table = model in RoutingDecider.SPATIAL_MODE_KEYS
        if default == "auto":
            assert in_table, f"{model} defers by default and must be consulted"
        else:
            assert in_table == (model in explicit), (
                f"{model} declares the concrete default '{default}'; it may only "
                f"be in SPATIAL_MODE_KEYS via an explicit spatial_mode_key"
            )


def _registered_model_schemas():
    from symfluence.core.modeling.config_schema import REGISTERED_SCHEMAS

    return dict(REGISTERED_SCHEMAS)


def test_deferred_spatial_mode_resolves_from_the_domain_method():
    """Hazard 2, pinned: ``auto`` is resolved, not compared as a string.

    Same resolution ``spatial_orchestrator.get_spatial_config`` applies, so the
    decision and the thing that acts on it can no longer disagree.
    """
    decider = RoutingDecider()

    # auto + lumped domain == the decision a model with no key would get.
    assert decider.needs_routing(
        {"VIC_SPATIAL_MODE": "auto", "DOMAIN_DEFINITION_METHOD": "lumped",
         "ROUTING_DELINEATION": "lumped"}, "VIC") is False
    # auto + distributed domain routes; the literal 'auto' used to match nothing.
    assert decider.needs_routing(
        {"VIC_SPATIAL_MODE": "auto", "DOMAIN_DEFINITION_METHOD": "distributed",
         "ROUTING_DELINEATION": "lumped"}, "VIC") is True
    # 'default' is the same sentinel, and the legacy 'delineate' spelling the
    # typed config normalises away still resolves.
    assert decider.needs_routing(
        {"GR_SPATIAL_MODE": "default", "DOMAIN_DEFINITION_METHOD": "delineate",
         "ROUTING_DELINEATION": "lumped"}, "GR") is True


def test_an_untouched_config_decides_the_same_with_or_without_the_key():
    """Hazard 1, pinned: joining the table cannot change an unset model.

    ``VIC_SPATIAL_MODE`` carries its ``auto`` default in every flattened typed
    config, so this is the case that would have regressed had a concrete
    default been admitted: the shipped template pairs ``ROUTING_MODEL:
    mizuRoute`` with a lumped domain, and a defaulted ``lumped`` would have
    vetoed it.
    """
    decider = RoutingDecider()
    template_like = {
        "ROUTING_MODEL": "mizuRoute",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "ROUTING_DELINEATION": "river_network",
    }

    without_key = decider.needs_routing(dict(template_like), "VIC")
    with_default = decider.needs_routing(
        dict(template_like, VIC_SPATIAL_MODE="auto"), "VIC")

    assert without_key == with_default is True


def test_routing_integration_keys_table():
    """Every model defining a routing-integration key has it consulted.

    Was FUSE-only: GR, VIC, SWAT, MHM and CRHM each define a
    ``<MODEL>_ROUTING_INTEGRATION`` key, and ``spatial_orchestrator`` already
    derived the same key by convention — so the orchestrator honoured those
    settings while the routing *decision* ignored them. Fixed deliberately;
    routing decisions change for those five models.
    """
    assert RoutingDecider.ROUTING_INTEGRATION_KEYS == {
        "CRHM": "CRHM_ROUTING_INTEGRATION",
        "FUSE": "FUSE_ROUTING_INTEGRATION",
        "GR": "GR_ROUTING_INTEGRATION",
        "MHM": "MHM_ROUTING_INTEGRATION",
        "SWAT": "SWAT_ROUTING_INTEGRATION",
        "VIC": "VIC_ROUTING_INTEGRATION",
    }


# ---------------------------------------------------------------------------
# Spatial-mode config resolution (SpatialModeDetectionMixin)
# ---------------------------------------------------------------------------

class _StubModelConfig:
    """Stands in for ``config.model`` with one attribute per model section."""

    def __init__(self, **sections):
        for name, value in sections.items():
            setattr(self, name, value)


class _StubSection:
    def __init__(self, spatial_mode):
        self.spatial_mode = spatial_mode


class _StubHost(SpatialModeDetectionMixin):
    def __init__(self, model_sections):
        self.config = type("Cfg", (), {"model": _StubModelConfig(**model_sections)})()


# Every model core can currently resolve a configured spatial mode for.
_SPATIAL_MODE_MODELS = [
    "HBV", "GR", "FUSE", "CFUSE", "JFUSE", "LSTM",
    "GNN", "SUMMA", "HYPE", "MESH", "NGEN", "RHESSYS",
]


@pytest.mark.parametrize("model", _SPATIAL_MODE_MODELS)
def test_configured_spatial_mode_resolves(model):
    host = _StubHost({model.lower(): _StubSection("distributed")})
    assert host._get_configured_spatial_mode(model) == "distributed"


@pytest.mark.parametrize("model", ["SWAT", "VIC", "CLM", "MODFLOW", "PRMS",
                                   "CRHM", "WFLOW", "PIHM", "GSFLOW"])
def test_previously_unlisted_models_resolve_spatial_mode(model):
    """Any model with a config section resolves its configured spatial mode.

    Was a hand-maintained 12-entry allow-list whose every entry was just
    ``model_name.lower()``, so ~20 models silently lost their configured
    spatial mode. A plain lowercase lookup already returns None for a model
    with no config section, so the allow-list contributed nothing but the bug.
    """
    host = _StubHost({model.lower(): _StubSection("distributed")})
    assert host._get_configured_spatial_mode(model) == "distributed"


def test_unknown_model_still_resolves_no_spatial_mode():
    """A model with no config section resolves None rather than raising."""
    host = _StubHost({"fuse": _StubSection("lumped")})
    assert host._get_configured_spatial_mode("NOT_A_MODEL") is None


def test_missing_config_section_resolves_none():
    host = _StubHost({})
    assert host._get_configured_spatial_mode("FUSE") is None


# ---------------------------------------------------------------------------
# Registry aliases and model capabilities
# ---------------------------------------------------------------------------

_EXPECTED_ALIASES = {
    "HEC-HMS": "HECHMS",
    "SAC-SMA": "SACSMA",
    "CLM-PARFLOW": "CLMPARFLOW",
    "RHESS": "RHESSYS",
}


def test_summa_modflow_alias_removed():
    """SUMMA-MODFLOW used to alias COUPLED_GW, which has no runner (dangling)."""
    from symfluence.core.registries import R

    assert "SUMMA-MODFLOW" not in R.runners


@pytest.mark.parametrize("alias,canonical", sorted(_EXPECTED_ALIASES.items()))
def test_model_aliases_resolve(alias, canonical):
    """Hyphenated / alternate spellings keep resolving after aliases move.

    These are declared in ``core._bootstrap``; the target state is a manifest
    field owned by the package that registers the canonical key.
    """
    from symfluence.core.registries import R

    for registry in (R.runners, R.preprocessors, R.postprocessors,
                     R.optimizers, R.workers):
        resolved = registry.get(alias)
        canonical_value = registry.get(canonical)
        if canonical_value is not None:
            assert resolved is canonical_value, f"{alias} -> {canonical}"


def test_self_training_models():
    """Models whose calibration is internal training, not a parameter search."""
    assert SupportedModels.SELF_TRAINING == frozenset({"LSTM", "GNN"})
