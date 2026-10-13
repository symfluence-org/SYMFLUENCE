# PRMS 5.2.1.1 glacier conservation patch

This is a **local SYMFLUENCE patch, not an upstream USGS release**. The source is
[USGS PRMS 5.2.1.1, October 2023](https://water.usgs.gov/water-resources/software/PRMS/prms_5.2.1.1_linux.zip).
The current nhm-usgs/prms master installer is not a substitute: the installed
master build used in this experiment did not include glacier processes.

The patch preserves runoff from ice-free portions of glacier-capable HRUs and
separates basal ice melt from the snowmelt condition in infiltration. Previously,
positive basal melt could make the snow-free rainfall branch unreachable.

It also adds `glacier_dynamics` (default 1, preserving dynamic geometry). Setting
it to 0 bypasses the annual geometry/ELA update while retaining daily glacier
melt, routing and cumulative glacier water storage. This is a **fixed-geometry
baseline**, not a validated glacier-retreat or finite-ice-inventory model. Use
separate pure glacier and land HRUs. Native mixed-HRU annual snow reset still
requires a fuller upstream land/ice state separation; this patch does not claim
to repair that dynamic-geometry case.

Apply to an extracted source tree containing `mmf/` and `prms/`:

```sh
python src/symfluence/models/prms/patches/apply_glacier_patch.py /path/to/prms_5.2.1.1
```

The script verifies normalized source hashes, refuses unknown source versions,
and supports an already-patched tree. Rebuild both `mmf` and `prms`; existing
modern-gfortran ABI fixes (`ftnlen` and hidden string length prototypes) and
platform compiler/linker settings remain necessary. The Tungnaá reproducible
build recipe and tested source tree are in `output/tungnaa_mariufossar_physics/`.
Set `model.prms.install_path` to the resulting binary directory and
`glacier_dynamics: false` for the fixed-extent configuration. Preprocessing
checks that the selected binary recognizes this control, and refuses fractional
initial glacier coverage in this mode.

Tungnaá full-period checks: 5,386 days, 6 HRUs, fixed 11.304% glacier coverage.
After removing duplicated snow-on-ice storage and including ice evaporation,
the baseline scored-period residual is -0.00127 mm. The paired no-glacier
baseline residual is -0.00074 mm. The optional calibration water-balance gate
rejects non-conserving candidates rather than fitting their errors.
