# Tungnaá at Maríufossar: ten-year glacier HYPE experiment

Run the normal SYMFLUENCE workflow from the repository:

```sh
symfluence workflow run --config examples/02_watershed_modelling/config_tungnaa_mariufossar_hype_belgingur_2009_2018.yaml --force-rerun
```

The configuration uses default installation/data roots and LamaH-Ice station 86.
It needs the Belgingur adapter and HYPE glacier/cryosphere changes in this working
tree; these changes have not yet been published as a develop release. Sharing
the YAML alone with a stock installation is therefore insufficient at present.

Output domain: `SYMFLUENCE_data/domain_Tungnaa_Mariufossar_HYPE_2009_2018`.
The local launch stages all 120 cached monthly Belgingur files; a fresh workflow
can acquire them through the same adapter. No CDS acquisition is requested.

| Purpose | Dates |
|---|---|
| Spinup | 2009 |
| Calibration | 2010–2015 |
| Held-out evaluation | 2016–2018 |

The workflow creates elevation bands, remaps the three-hourly forcing, aggregates
daily HYPE inputs, runs the baseline, and launches 1,000 single-process DDS
evaluations with seed 42. Prior experiments remain in their own domains.

## Glacier initialization

The initial glacier fraction is 0.11304, from the 2009 LamaH-Ice dynamic glacier
fraction, rather than the later static fraction 0.102. Ice occupies the highest
bands first. `preserve_icecap_volume` writes band-specific `logvolcor` values so
splitting one parent ice cap does not reduce its total initial volume. The
parent volume is still HYPE's empirical estimate, not measured ice thickness.
Type 1, coefficient 1.701, exponent 1.25 and ice/water density ratio 0.85 are fixed.

## Calibration objective

The objective is **COMPOSITE**, not KGE:

`0.6 * KGE + 0.2 * SNOW_SCORE + 0.2 * GLACIER_SCORE`

- `SNOW_SCORE = 1 - RMSE(non-glacier snow fraction) / 0.2`.
- `GLACIER_SCORE = 1 - RMSE(annual glacier balance in m w.e.) / 1.0`.

The weights and scales are explicit exploratory choices, not estimates of
measurement uncertainty. Raw KGE, NSE, snow RMSE and glacier-balance RMSE remain
separate metrics. Missing model outputs or insufficient observations penalize a
trial rather than silently dropping the extra constraints.

The preprocessor fetches the station's glacier and MODIS series from the official
[LamaH-Ice archive](https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/)
if absent. Locally, the already-extracted official series were copied into
`data/observations/cryosphere`. The [dataset supplement](https://essd.copernicus.org/articles/16/2741/2024/essd-16-2741-2024-supplement.pdf)
describes the interpolated glacier mass-balance observations and their limits.

HYPE `CFSC` represents snow outside glacier ice, normalized by land area; it is
converted to a fraction of the dynamically ice-free basin using `GLCA`. This
implementation requires land/glacier classes only and rejects lake classes.
`GLCV` changes are converted from km³ ice to water-equivalent change, divided by
the observed year's glacier area. Fixed September 30 boundaries approximate the
field balance years. Only complete balance intervals entirely within calibration
are scored: 2011–2015. MODIS comparisons use available calibration dates only.
Held-out observations do not influence optimization.

Each trial writes `cryosphere_metrics.json` beside its model outputs with the
period, component errors, observation counts and annual glacier comparisons.

## Interpretation and remaining limitations

This is an experiment to address the excessive 2012 melt peak, not a validated
forecasting model. The earlier elevation run achieved evaluation KGE 0.248 for
2012–2013; the new evaluation period is different, so the two aggregate scores
must not be presented as a direct improvement comparison.

The model still uses fixed temperature-index glacier melt. It does not represent
year-specific volcanic ash/albedo effects. Glacier area allocation is an elevation
approximation. Routing retains the native lumped/elevation topology and placeholder
reach lengths; the diagnostic 106.6 km longest path has not been adopted as a
measured routing length. A physically supported channel/storage treatment remains
a separate improvement if the constrained run retains distorted melt peaks.

Before accepting the model, examine daily runoff, annual glacier balances,
snow-cover timing, parameter bounds, and the untouched 2016–2018 evaluation.
