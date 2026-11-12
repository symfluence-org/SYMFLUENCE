"""A missing catchment area is an error, not 2210 km2 (the Bow at Banff)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import symfluence
from symfluence.core.exceptions import ConfigurationError
from symfluence.models.wrfhydro.calibration.worker import WRFHydroWorker
from symfluence.models.wrfhydro.extractor import WRFHydroResultExtractor

SRC = Path(symfluence.__file__).parent
# 2210 as a value a read falls back to; prose and the worked examples in
# comments ("~47 km for 2210 km2") are not fallbacks.
FALLBACK = re.compile(r"(default\s*=\s*|,\s*|\bor\s+|return\s+)2210(\.0)?\b")


def test_no_code_path_falls_back_to_2210_km2():
    offenders = []
    for path in SRC.rglob('*.py'):
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            code = line.split('#', 1)[0]
            if FALLBACK.search(code) and "'area_km2': 2210.0" not in code:
                offenders.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_wrfhydro_ldasout_streamflow_needs_an_area(tmp_path):
    (tmp_path / '2001010100.LDASOUT_DOMAIN1').write_bytes(b'')
    with pytest.raises(ConfigurationError, match='catchment_area'):
        WRFHydroResultExtractor('WRFHYDRO').extract_streamflow(tmp_path)


def test_wrfhydro_worker_does_not_invent_an_area(tmp_path):
    config = {'DOMAIN_NAME': 'x', 'SYMFLUENCE_DATA_DIR': str(tmp_path)}
    with pytest.raises(ValueError, match='catchment area'):
        WRFHydroWorker(config)._extract_ldasout_streamflow([], config)
