"""`import symfluence` must not eagerly load every model's calibration stack.

Declaring a model alias used to call ``registry.keys()``, which fired the lazy
optimizer/worker seeders; those import every ``models/*/calibration/optimizer``
module, and GR's runner then started the embedded R interpreter via rpy2 --
about 1 s (plus jax, geopandas, ...) on every CLI command.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def test_import_does_not_seed_calibration_registries():
    code = textwrap.dedent("""
        import sys
        import symfluence  # noqa: F401
        from symfluence.core.registries import R
        print(R.optimizers._seeder is not None, R.workers._seeder is not None,
              'symfluence.models.gr.runner' in sys.modules, 'rpy2' in sys.modules)
    """)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    ).stdout.split()
    optimizers_lazy, workers_lazy, gr_runner_loaded, rpy2_loaded = out
    assert optimizers_lazy == "True", "R.optimizers was seeded during import"
    assert workers_lazy == "True", "R.workers was seeded during import"
    assert gr_runner_loaded == "False"
    assert rpy2_loaded == "False"


def test_gr_runner_import_does_not_start_r():
    code = textwrap.dedent("""
        import sys
        import symfluence.models.gr.runner as runner
        print(runner.HAS_RPY2 is None, 'rpy2' in sys.modules)
    """)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    ).stdout.split()
    assert out == ["True", "False"]
