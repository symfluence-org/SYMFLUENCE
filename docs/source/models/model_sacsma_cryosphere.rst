SAC-SMA cryosphere research replay
=================================

This optional interface replays ``SACSMA_CRYOSPHERE_UNIFORM_HRU_V2`` with
jSACSMA and jSnow17 0.3.0. It is an explicit elevation/finite-ice extension,
not standard lumped SAC-SMA or a physically validated glacier model. Install
the corrected model packages before running this interface. The draft PRs must
land in order: Snow17, SAC-SMA, then this replay interface.

Create a bundle with a JSON ``ReplaySpec`` and relative CSV inputs. Validate
with ``ReplaySpec.model_validate_json``. Use ``runtime_fingerprints()`` to record
the exact source hashes and package versions after choosing an implementation;
do not silently update fingerprints on an existing scientific replay.
``ReplaySpec.model_json_schema()`` provides the full configuration schema.
No absolute input paths, unknown fields or escaping symlinks are accepted.

Forcing columns are ``datetime``, and ``precip_<HRU>``, ``temp_<HRU>``,
``pet_<HRU>`` for each ordered HRU ID. Depths are mm per declared input interval,
temperature degrees C. Spacing must equal ``config.timestep_days``. Native
subdaily forcing is preferable to repeated daily means. Geometry, initial ice
in water-equivalent mm, maximum substep, corrections and routing are explicit.
The required geometry reference and qualifications describe their provenance.
Observations use ``datetime,discharge_cms``. Set
``observation_aggregation="daily_mean"`` to compare native subdaily simulations
with daily observations. Complete days are required and observations are labelled
at midnight; metrics use daily averaged discharge, not midnight samples.
Date-only period endpoints include the full calendar day, including glacier
storage endpoints at its last subdaily forcing label. Monthly temperature corrections
are recorded separately from the global model offset. PET is an explicit input;
its derivation belongs in the bundle's qualifications/provenance.

Run::

   python -m symfluence.models.sacsma_cryosphere.replay spec.json --output results

The output directory must be empty. Output is self-contained: input copies,
portable specification, source/version fingerprints, checksums, netCDF and CSV
discharge, water-balance ledger, final state and metrics. Balance gates apply
per step and cumulatively. ``initial_ice_mm`` is honored rather than hard-coded.
No results index or ensemble vote is modified.

Audit integration
-----------------

``load_audited_candidate(results)`` verifies hashes, identities, discharge units,
netCDF/CSV/depth consistency and water-balance gates, returning discharge and
provenance. ``audit_candidate(results, observations, expected_periods)`` also
requires exactly matching ensemble observations and scoring periods. Periods
map names to ``(start, end, role)``. Treat the returned candidate as family
``SACSMA``: alternatives are comparators or share the existing family vote.
Loading historical results does not require the original engine; replaying
them requires the exact pinned implementation.

Optional snow targets require ``datetime,observed_snow_fraction`` and explicit
``geometric_uniform_hru`` or ``effective_area_proxy`` interpretation. Neither
establishes an optical observation operator. Glacier targets require
``start_label,end_label,observed_mb_mwe`` and fixed-model-area specific balance.
Convert dynamic-observed-area targets to this basis explicitly before use.
Storage labels are end-of-interval states labelled by the forcing timestamp;
complete balance windows only are scored. Composite weights and component
errors are recorded, with target input checksums.

Scientific qualification
------------------------

Use measured geometry or georeferenced thickness estimates to define explicit
per-HRU ice stocks. Record the estimate epoch, ice-to-water density conversion,
spatial coverage and mapping assumptions. A thickness estimate is a prior, not
proof of the exact inventory at initialization. Profile stock and area changes;
unchanged discharge under thick stocks means the initial stock is unidentifiable.

For geometric snow-area evaluation, jSACSMA's optional
``compiled_snow_observer(config)`` exposes interval-end per-HRU snow-water
histories. ``jsacsma.observations.snow_covered_area`` aggregates them with an
explicit surface mask and detection threshold. Resolve elevation/surface
heterogeneity in HRUs and renormalize weights within the observed domain.
Match observation masks and temporal support; gap-filled MODIS daily products
are not independent clear-sky scenes. This external observation operator is
distinct from the replay schema's legacy effective-area proxy targets.

Use actual basin-relevant survey dates when available. Stratigraphic glacier
balances need an observation operator consistent with their summer-surface
reference; chronological storage changes remain an approximation. Winter and
summer errors can cancel in annual net balance. Archive externally computed
seasonal objectives and targets explicitly when native replay metrics cannot
represent them. Keep unused independent observations outside parameter selection.

The variant uses binary snow presence within uniform HRUs to gate exposed ice,
separate from Snow17 effective melt area. Ice is finite. V2 estimates covered versus bare time from Snow17 melt-energy
and heat-deficit diagnostics within each computational substep; only the
remaining bare time melts ice after snow exhaustion. Tiny transient snowfall
cannot shield a whole interval. This constant-rate approximation requires
substep sensitivity checks. V1 historical bundles remain readable under their
original identity and fingerprints; replay requires their original source,
and V2 requires recalibration. Fixed area, no retreat/firn/redistribution, no frozen
ground and simplified Snow17 liquid drainage limit physical interpretation.

Previous prototype parameters must use ``uncalibrated_replay`` after these
kernel changes. Recalibration and forcing/geometry/storage sensitivity remain
necessary before promotion. Later periods inspected during development must
use ``development_validation`` or ``transfer`` roles. The schema intentionally
has no untouched-holdout role. Numerical integrity alone is not physical
validation. This interface does not promote the Maríufossar prototype.

Validation
----------

Further process variants must retain their own model identity and cannot be
replayed by silently changing this V2 engine. Separate glacier and land forcing
corrections; reserve AWS sites/years before inspecting target values. Statistical
snow patches need normalized snowfall redistribution, unchanged rainfall,
separate snow states and geometric observation support. Test patch resolution
and distribution priors independently of optical-data fitting.

For frozen soil, preserve original SAC capacities and distinguish total from
liquid water. Ice occupies available storage; only liquid supplies fluxes.
Retain pre-redistribution latent enthalpy when advancing the heat component,
including heat released by freezing new water. Equation checks and conservation
do not establish operational SAC-HT equivalence or physical soil-state validity.
Require explicit thermal/texture assumptions, reference cases, timestep and
property profiles, and spring/summer flood counterchecks. Keep chronological and
stratigraphic glacier-balance observation definitions distinct. A source-complete
research replay remains ineligible for the native ensemble adapter until its
identity, schema, outputs and physical qualifications are reviewed.

The unit suite includes schema/hash/path checks, initial-ice effects, moving and
rerunning a bundle, native output agreement, balance gates and rejection of
changed ensemble observations/windows. Engine integration tests skip when the
optional corrected packages are unavailable; run with both correction PRs to
exercise those tests. Upstream references: NOAA SAC1 compiled fixtures and
`Anderson's Snow17 description <https://www.weather.gov/media/owp/oh/hrl/docs/22snow17.pdf>`_.
