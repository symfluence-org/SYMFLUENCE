# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Regression tests for the round-6 audit fixes."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# --- metric direction -------------------------------------------------------


@pytest.mark.parametrize("name", ["bias", "Bias", "BIAS", "PBIAS", "pbias", "PBias", "Pbias"])
def test_signed_bias_metrics_penalise_both_signs(name):
    """+20 and -20 bias are equally bad; neither may score above a perfect 0."""
    from symfluence.core.metrics import MetricTransformer

    assert MetricTransformer.get_direction(name) == "minimize"
    assert MetricTransformer.transform_for_maximization(name, 20.0) == -20.0
    assert MetricTransformer.transform_for_maximization(name, -20.0) == -20.0
    assert MetricTransformer.transform_for_maximization(name, 0.0) == 0.0


# --- month lists for chunked downloads ---------------------------------------


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2020-11-15", "2021-02-28", [(2020, 11), (2020, 12), (2021, 1), (2021, 2)]),
        ("2020-11-15", "2020-12-10", [(2020, 11), (2020, 12)]),
        ("2020-01-01 01:00", "2020-03-10", [(2020, 1), (2020, 2), (2020, 3)]),
        ("2020-05-03", "2020-05-20", [(2020, 5)]),
    ],
)
def test_year_month_list_keeps_mid_month_start(start, end, expected):
    from symfluence.data.acquisition.mixins.chunked import ChunkedDownloadMixin

    mixin = ChunkedDownloadMixin.__new__(ChunkedDownloadMixin)
    assert mixin.generate_year_month_list(pd.Timestamp(start), pd.Timestamp(end)) == expected


# --- TWS handlers use the domain bbox -----------------------------------------


@pytest.mark.parametrize("module,cls_name", [
    ("symfluence.data.acquisition.handlers.gldas_tws", "GLDASAcquirer"),
    ("symfluence.data.acquisition.handlers.cnes_grgs_tws", "CNESGRGSAcquirer"),
])
def test_tws_handlers_use_domain_bbox_not_globe(module, cls_name):
    import importlib

    cls = getattr(importlib.import_module(module), cls_name)
    handler = cls.__new__(cls)
    handler.bbox = {"lat_min": 51.2, "lat_max": 51.8, "lon_min": -1.0, "lon_max": 0.5}
    assert handler._get_bounding_box() == (-1.0, 51.2, 0.5, 51.8)

    handler.bbox = None
    with pytest.raises(ValueError):
        handler._get_bounding_box()


# --- GA does not re-run its elites --------------------------------------------


def test_ga_evaluates_only_offspring():
    from symfluence.core.calibration.optimizers.algorithms.ga import GAAlgorithm

    pop_size, iterations, elites = 10, 4, 2
    sizes = []

    def evaluate_population(pop, iteration=0):
        sizes.append(len(pop))
        return -np.sum((np.atleast_2d(pop) - 0.5) ** 2, axis=1)

    config = {
        "OPTIMIZATION_METRIC": "KGE",
        "NUMBER_OF_ITERATIONS": iterations,
        "POPULATION_SIZE": pop_size,
        "GA_ELITISM_COUNT": elites,
        "RANDOM_SEED": 7,
    }
    result = GAAlgorithm(config, logging.getLogger("ga-test")).optimize(
        n_params=3,
        evaluate_solution=lambda x, i: 0.0,
        evaluate_population=evaluate_population,
        denormalize_params=lambda x: {f"p{i}": float(v) for i, v in enumerate(x)},
        record_iteration=lambda *a, **k: None,
        update_best=lambda *a, **k: None,
        log_progress=lambda *a, **k: None,
    )
    assert sizes == [pop_size] + [pop_size - elites] * iterations
    assert np.isfinite(result.get("best_score", result.get("best_fitness")))


# --- process pool keeps finished results when the pool breaks -----------------


def test_broken_pool_reruns_only_unfinished_tasks(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    from symfluence.core.calibration.mixins.parallel.execution_strategies import process_pool

    calls = []

    def worker(task):
        calls.append(task["individual_id"])
        return {"individual_id": task["individual_id"], "score": 1.0}

    class FakePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, wave):
            yield {"individual_id": wave[0]["individual_id"], "score": 0.5}
            raise BrokenProcessPool("worker died")

    monkeypatch.setattr(process_pool, "ProcessPoolExecutor", FakePool)
    strategy = process_pool.ProcessPoolExecutionStrategy(logging.getLogger("pp-test"))
    tasks = [{"individual_id": i} for i in range(4)]
    results = strategy.execute(tasks, worker, max_workers=2)

    assert [r["individual_id"] for r in results] == [0, 1, 2, 3]
    assert results[0]["score"] == 0.5  # kept, not recomputed
    assert calls == [1, 2, 3]


# --- multi-gauge minimum keeps each model's own fallback ----------------------


def test_hype_multi_gauge_min_gauges_falls_back_to_three(tmp_path):
    """A schema default for MULTI_GAUGE_MIN_GAUGES would override HYPE's 3."""
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.models.hype.calibration.worker import HYPEWorker

    cfg = SymfluenceConfig.from_minimal(
        "t", "HYPE", "ERA5",
        EXPERIMENT_TIME_START="2000-01-01 00:00", EXPERIMENT_TIME_END="2001-01-01 00:00",
        SYMFLUENCE_DATA_DIR=str(tmp_path), SYMFLUENCE_CODE_DIR=str(tmp_path),
    )
    flat = cfg.to_dict(flatten=True)
    worker = HYPEWorker.__new__(HYPEWorker)
    worker.config = flat
    assert int(worker._resolve_config_value("MULTI_GAUGE_MIN_GAUGES", flat, 3)) == 3


# --- acquisition downloads are written atomically ------------------------------


def test_acquisition_handlers_do_not_stream_into_final_paths():
    """A raw `open(path, 'wb')` leaves a truncated file at `path` on failure,
    and the handlers' `if path.exists()` cache checks then reuse it as complete.
    Write through `atomic_write(path)` (or to an explicit .part/tmp path)."""
    import re
    from pathlib import Path

    import symfluence.data.acquisition as acq

    root = Path(acq.__file__).parent
    raw = re.compile(r"open\((?P<tgt>[^)]*?),\s*['\"]wb['\"]\)")
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "utils.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            m = raw.search(line)
            if m and not re.search(r"_part|part|tmp|temp", m["tgt"], re.I):
                offenders.append(f"{path.relative_to(root)}:{lineno}")
    assert not offenders, "non-atomic download writes: " + ", ".join(offenders)
