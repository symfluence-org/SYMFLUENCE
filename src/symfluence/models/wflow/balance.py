"""Native Wflow balance gate for distributed experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ValidationError


def check(output):
    f = pd.read_csv(Path(output) / "output.csv")
    if not np.isfinite(f.select_dtypes("number")).all().all():
        raise ValidationError("Nonfinite Wflow outputs")
    if (f.Q < 0).any():
        raise ValidationError("Negative outlet discharge")
    cols = [k for k in f if k.startswith("WB_")]
    error = float(f[cols].abs().to_numpy().max())
    if error > 0.05:
        raise ValidationError(f"Wflow land balance error {error} mm exceeds 0.05 mm")
    result = dict(days=len(f), max_land_balance_error_mm=error)
    for prefix in ["RiverWB", "SSF_WB"]:
        cols = [k for k in f if k.startswith(prefix)]
        if cols:
            error = float(f[cols].abs().to_numpy().max())
            unit = 'm3/s' if prefix == 'RiverWB' else 'm3/day'
            suffix = '_max_error_cms' if prefix == 'RiverWB' else '_max_error_m3day'
            result[prefix + suffix] = error
            if error > 0.01:
                raise ValidationError(f"Wflow {prefix} routing balance error exceeds 0.01 {unit}")
    return result
