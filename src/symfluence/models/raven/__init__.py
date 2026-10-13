"""Native Raven HBV-EC adapter, including explicit glacier HRUs."""

from __future__ import annotations

from symfluence.core.registry import model_manifest

from .config import RavenConfigAdapter


def register():
    from symfluence.core.registries import R

    model_manifest(
        "RAVEN",
        config_adapter=RavenConfigAdapter,
        build_instructions_module="symfluence.models.raven.build_instructions",
    )
    base = "symfluence.models.raven"
    R.result_extractors.add_lazy("RAVEN", f"{base}.extractor.RavenResultExtractor")
    for registry, name in [
        (R.preprocessors, "RavenPreProcessor"),
        (R.postprocessors, "RavenPostProcessor"),
        (R.runners, "RavenRunner"),
        (R.optimizers, "RavenOptimizer"),
        (R.workers, "RavenWorker"),
        (R.parameter_managers, "RavenParameterManager"),
    ]:
        module = "postprocessor" if registry is R.postprocessors else "adapter"
        registry.add_lazy("RAVEN", f"{base}.{module}.{name}")
