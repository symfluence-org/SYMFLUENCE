# SPDX-License-Identifier: GPL-3.0-or-later
"""Schema-validated, hash-verified replay of the uniform-HRU research variant.

Run with python -m symfluence.models.sacsma_cryosphere.replay SPEC --output DIR.
An output directory is a self-contained data/spec/results bundle; implementation
source hashes pin the two optional model packages. The adapter reads historical
bundles without silently replaying them through a newer implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveFloat, StrictBool, model_validator

from symfluence.core.exceptions import FileOperationError
from symfluence.core.exceptions import ValidationError as SYMFLUENCEValidationError


class ReplayValidationError(SYMFLUENCEValidationError, ValueError):
    """Replay contract failure compatible with Pydantic validator handling."""


class ReplayOutputExistsError(FileOperationError, FileExistsError):
    """Refuse to overwrite a scientific replay bundle."""


MODEL_ID = "SACSMA_CRYOSPHERE_UNIFORM_HRU_V2"
SUPPORTED_MODEL_IDS = {"SACSMA_CRYOSPHERE_UNIFORM_HRU_V1", MODEL_ID}
MODULES = (
    "jsacsma.cryosphere",
    "jsacsma.nws",
    "jsacsma.sacsma",
    "jsacsma.parameters",
    "jsnow17.model",
    "jsnow17.parameters",
    "jsnow17.depletion",
    "symfluence.models.sacsma_cryosphere.replay",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FileRef(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def relative(self):
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ReplayValidationError("Bundle input paths must be relative and remain inside the bundle")
        return self


class CryosphereSettings(StrictModel):
    area_km2: PositiveFloat
    weights: tuple[PositiveFloat, ...]
    glacier: tuple[StrictBool, ...]
    latitudes: tuple[FiniteFloat, ...]
    elevations_m: tuple[FiniteFloat, ...]
    initial_ice_mm: FiniteFloat | tuple[FiniteFloat, ...] = 100000.0
    si: PositiveFloat = 600.0
    ice_degree_day_factor: PositiveFloat = 5.0
    routing_tau_days: PositiveFloat = 1.0
    timestep_days: PositiveFloat = 1.0
    max_substep_days: PositiveFloat = Field(default=0.125, le=0.25)
    precipitation_scale: PositiveFloat = 1.0
    temperature_offset_c: FiniteFloat = 0.0

    @model_validator(mode="after")
    def geometry(self):
        n = len(self.weights)
        if not n or any(len(v) != n for v in [self.glacier, self.latitudes, self.elevations_m]):
            raise ReplayValidationError("Geometry vector lengths differ")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-10, rtol=0):
            raise ReplayValidationError("Weights must sum to one")
        if any(abs(v) > 90 for v in self.latitudes) or any(v < -500 or v > 9000 for v in self.elevations_m):
            raise ReplayValidationError("Invalid latitude/elevation")
        ice = np.asarray(self.initial_ice_mm)
        if (ice < 0).any() or (ice.ndim and ice.shape != (n,)):
            raise ReplayValidationError("Invalid initial ice inventory")
        return self


class Period(StrictModel):
    start: str
    end: str
    role: Literal["calibration", "development_validation", "transfer"]

    @model_validator(mode="after")
    def order(self):
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ReplayValidationError("Period ends before it starts")
        return self


class GlacierConstraint(StrictModel):
    targets: FileRef
    area_basis: Literal["model_fixed_area_specific"] = "model_fixed_area_specific"
    units: Literal["m_water_equivalent"] = "m_water_equivalent"
    weight: FiniteFloat = Field(default=0.1, ge=0)


class SnowConstraint(StrictModel):
    targets: FileRef
    interpretation: Literal["geometric_uniform_hru", "effective_area_proxy"] = "geometric_uniform_hru"
    weight: FiniteFloat = Field(default=0.2, ge=0)


class ReplaySpec(StrictModel):
    schema_version: Literal[1] = 1
    model: Literal["SACSMA_CRYOSPHERE_UNIFORM_HRU_V1", "SACSMA_CRYOSPHERE_UNIFORM_HRU_V2"] = MODEL_ID
    family: Literal["SACSMA"] = "SACSMA"
    backend: Literal["numpy", "jax"] = "numpy"
    forcing: FileRef
    observations: FileRef | None = None
    observation_aggregation: Literal["native", "daily_mean"] = "native"
    glacier_constraint: GlacierConstraint | None = None
    snow_constraint: SnowConstraint | None = None
    forcing_units: Literal["mm_per_input_timestep_and_degC"] = "mm_per_input_timestep_and_degC"
    hru_ids: tuple[str, ...]
    geometry_reference: str = Field(min_length=1)
    config: CryosphereSettings
    parameters: dict[str, FiniteFloat] = Field(default_factory=dict)
    monthly_temperature_offsets_c: tuple[FiniteFloat, ...] = Field(default=(0.0,) * 12, min_length=12, max_length=12)
    implementation_hashes: dict[str, str]
    package_versions: dict[str, str]
    periods: dict[str, Period] = Field(default_factory=dict)
    state_time_reference: Literal["end_of_interval_labeled_by_forcing_timestamp"] = (
        "end_of_interval_labeled_by_forcing_timestamp"
    )
    parameter_status: Literal["uncalibrated_replay", "calibrated"] = "uncalibrated_replay"
    qualifications: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def contract(self):
        if len(self.hru_ids) != len(self.config.weights) or len(set(self.hru_ids)) != len(self.hru_ids):
            raise ReplayValidationError("HRU IDs must be unique and match geometry")
        if set(self.implementation_hashes) != set(MODULES):
            raise ReplayValidationError("All model source fingerprints are required")
        if any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in self.implementation_hashes.values()
        ):
            raise ReplayValidationError("Invalid implementation SHA256")
        if set(self.package_versions) != {"jsacsma", "jsnow17"}:
            raise ReplayValidationError("Both model package versions are required")
        if any(not name or any(c in name for c in "/\\") for name in self.hru_ids):
            raise ReplayValidationError("Invalid HRU ID")
        return self


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_fingerprints():
    hashes = {name: sha(importlib.import_module(name).__file__) for name in MODULES}
    versions = {name: importlib.import_module(name).__version__ for name in ["jsacsma", "jsnow17"]}
    return hashes, versions


def _read_input(root, ref):
    path = (root / ref.path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ReplayValidationError("Input symlink escapes bundle")
    if sha(path) != ref.sha256:
        raise ReplayValidationError(f"Input checksum mismatch: {ref.path}")
    return path


def _frame(path):
    df = pd.read_csv(path, float_precision="round_trip")
    if "datetime" not in df:
        raise ReplayValidationError("Input must contain datetime")
    index = pd.DatetimeIndex(pd.to_datetime(df.pop("datetime"), utc=True)).tz_localize(None)
    if index.has_duplicates or not index.is_monotonic_increasing or len(index) == 0:
        raise ReplayValidationError("Time axis must be nonempty, unique and increasing")
    df.index = index
    return df


def _period_end(period):
    # Date-only endpoints include the complete calendar day at subdaily dt.
    end = pd.Timestamp(period.end)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period.end):
        end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return end


def _metrics(s, o):
    valid = np.isfinite(o)
    s = s[valid]
    o = o[valid]
    if len(o) < 2 or o.std() == 0:
        return {"samples": len(o), "KGE": None, "NSE": None}
    r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else 0.0
    a = s.std() / o.std()
    b = s.mean() / o.mean()
    return {
        "samples": len(o),
        "KGE": float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)),
        "NSE": float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)),
        "r": float(r),
        "alpha": float(a),
        "beta": float(b),
        "PBIAS": float(100 * (b - 1)),
    }


def replay_bundle(spec_path, output_dir):
    """Verify inputs/implementation, replay, and freeze data and result hashes.

    Output must be empty. No selection index or ensemble vote is modified.
    """
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ReplayOutputExistsError("Replay output directory must be empty")
    spec_path = Path(spec_path).resolve()
    root = spec_path.parent
    spec = ReplaySpec.model_validate_json(spec_path.read_text())
    forcing_path = _read_input(root, spec.forcing)
    observation_path = _read_input(root, spec.observations) if spec.observations else None
    glacier_path = _read_input(root, spec.glacier_constraint.targets) if spec.glacier_constraint else None
    snow_path = _read_input(root, spec.snow_constraint.targets) if spec.snow_constraint else None
    hashes, versions = runtime_fingerprints()
    if hashes != spec.implementation_hashes or versions != spec.package_versions:
        raise ReplayValidationError("Model implementation/version differs from the frozen specification")
    from jsacsma.cryosphere import MODEL_ID as engine_model_id
    from jsacsma.cryosphere import CryosphereConfig, simulate_cryosphere

    if spec.model != engine_model_id:
        raise ReplayValidationError("Model identity differs from replay engine")

    data = _frame(forcing_path)
    if len(data) > 1 and not np.all(
        np.diff(data.index.values) == np.timedelta64(round(spec.config.timestep_days * 86400 * 1e9), "ns")
    ):
        raise ReplayValidationError("Forcing time axis does not match declared timestep")
    values = []
    for prefix in ["precip", "temp", "pet"]:
        columns = [f"{prefix}_{hru}" for hru in spec.hru_ids]
        if any(c not in data for c in columns):
            raise ReplayValidationError(f"Missing forcing columns: {columns}")
        values.append(data[columns].to_numpy(float))
    obs = None
    if observation_path:
        observation = _frame(observation_path)
        if "discharge_cms" not in observation:
            raise ReplayValidationError("Observation file requires discharge_cms")
        obs = observation.discharge_cms.reindex(data.index).to_numpy(float)
        if np.isinf(obs).any() or (obs[np.isfinite(obs)] < 0).any():
            raise ReplayValidationError("Observed discharge must be nonnegative")
    for name, period in spec.periods.items():
        if pd.Timestamp(period.start) < data.index[0] or _period_end(period) >= data.index[-1] + pd.Timedelta(
            days=spec.config.timestep_days
        ):
            raise ReplayValidationError(f"Period outside replay: {name}")
    glacier_targets = pd.read_csv(glacier_path) if glacier_path else None
    if glacier_targets is not None:
        if not {"start_label", "end_label", "observed_mb_mwe"}.issubset(glacier_targets):
            raise ReplayValidationError("Glacier targets require start_label, end_label, observed_mb_mwe")
        glacier_targets["start_label"] = pd.to_datetime(glacier_targets.start_label)
        glacier_targets["end_label"] = pd.to_datetime(glacier_targets.end_label)
        if (glacier_targets.start_label >= glacier_targets.end_label).any() or not np.isfinite(
            glacier_targets.observed_mb_mwe
        ).all():
            raise ReplayValidationError("Invalid glacier balance targets")
        if not any(spec.config.glacier):
            raise ReplayValidationError("Glacier targets require glacier HRUs")
        if any(stamp not in data.index for stamp in [*glacier_targets.start_label, *glacier_targets.end_label]):
            raise ReplayValidationError("Glacier target endpoints must match model state labels")
    snow_targets = _frame(snow_path) if snow_path else None
    if snow_targets is not None:
        if "observed_snow_fraction" not in snow_targets:
            raise ReplayValidationError("Snow targets require observed_snow_fraction")
        finite = snow_targets.observed_snow_fraction.dropna()
        if not np.isfinite(finite).all() or ((finite < 0) | (finite > 1)).any():
            raise ReplayValidationError("Snow fractions must be in [0,1] or missing")
        if all(spec.config.glacier):
            raise ReplayValidationError("Land-snow targets require land HRUs")
    if spec.observation_aggregation == "daily_mean":
        steps = 1 / spec.config.timestep_days
        if not np.isclose(steps, round(steps)):
            raise ReplayValidationError("Daily aggregation requires an interval dividing a day")
        counts = pd.Series(1, index=data.index).resample("D").sum()
        if not (counts == round(steps)).all():
            raise ReplayValidationError("Daily aggregation requires complete days")
        if observation_path and not observation.index.equals(observation.index.normalize()):
            raise ReplayValidationError("Daily observations must be labelled at midnight")
    offsets = np.asarray(spec.monthly_temperature_offsets_c)[data.index.month - 1]
    values[1] = values[1] + offsets[:, None]
    if spec.backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
    records, state = simulate_cryosphere(
        *values,
        data.index.dayofyear.to_numpy(),
        spec.parameters,
        CryosphereConfig(**spec.config.model_dump()),
        spec.backend == "jax",
    )
    if any(not np.isfinite(v).all() for v in records.values()) or (records["runoff_mm"] < -1e-10).any():
        raise ReplayValidationError("Replay produced invalid outputs")
    residual = records["balance_residual_mm"]
    if np.max(np.abs(residual)) > 1e-6 or np.max(np.abs(np.cumsum(residual))) > 1e-5:
        raise ReplayValidationError("Water-balance gate failed")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ReplayOutputExistsError("Replay output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir()
    archived = spec.model_copy(deep=True)
    for name, path in [("forcing", forcing_path), ("observations", observation_path)]:
        if path is None:
            continue
        dest = output / "inputs" / f"{name}.csv"
        dest.write_bytes(path.read_bytes())
        setattr(archived, name, FileRef(path=f"inputs/{name}.csv", sha256=sha(dest)))
    for name, path in [("glacier_constraint", glacier_path), ("snow_constraint", snow_path)]:
        if path is None:
            continue
        dest = output / "inputs" / f"{name}.csv"
        dest.write_bytes(path.read_bytes())
        getattr(archived, name).targets = FileRef(path=f"inputs/{name}.csv", sha256=sha(dest))
    (output / "replay_spec.json").write_text(archived.model_dump_json(indent=2) + "\n")
    flow = records["runoff_mm"] * spec.config.area_km2 * 1000 / (spec.config.timestep_days * 86400)
    series = pd.DataFrame({"streamflow_cms": flow}, index=data.index)
    series.index.name = "datetime"
    if obs is not None:
        series["observed_cms"] = obs
    series.to_csv(output / "streamflow.csv")
    balance = pd.DataFrame(records, index=data.index)
    balance.index.name = "datetime"
    balance.to_csv(output / "water_balance.csv")
    import xarray as xr

    xr.Dataset(
        {"streamflow": ("time", flow, {"units": "m3 s-1"})},
        coords={"time": data.index.to_numpy()},
        attrs={"model": MODEL_ID, "family": "SACSMA", "parameter_status": spec.parameter_status},
    ).to_netcdf(output / "streamflow_output.nc")
    snow, soil, ice, routing = state
    final = {
        "snow": {k: np.asarray(v).tolist() for k, v in snow._asdict().items()},
        "soil": {k: np.asarray(v).tolist() for k, v in soil._asdict().items()},
        "ice_mm": np.asarray(ice).tolist(),
        "routing_mm": float(routing),
    }
    (output / "final_state.json").write_text(json.dumps(final, indent=2) + "\n")
    metric_index = data.index
    metric_flow = flow
    metric_obs = obs
    if spec.observation_aggregation == "daily_mean":
        grouped = pd.Series(flow, index=data.index).resample("D")
        averaged = grouped.mean()
        metric_index = averaged.index
        metric_flow = averaged.to_numpy()
        if observation_path:
            metric_obs = observation.discharge_cms.reindex(metric_index).to_numpy(float)
    metrics = {}
    for name, period in spec.periods.items():
        mask = (data.index >= pd.Timestamp(period.start)) & (data.index <= _period_end(period))
        metric_mask = (metric_index >= pd.Timestamp(period.start)) & (metric_index <= _period_end(period))
        if (
            not mask.any()
            or pd.Timestamp(period.start) < data.index[0]
            or _period_end(period) >= data.index[-1] + pd.Timedelta(days=spec.config.timestep_days)
        ):
            raise ReplayValidationError(f"Period outside replay: {name}")
        metrics[name] = {
            "role": period.role,
            "observation_aggregation": spec.observation_aggregation,
            **(
                _metrics(metric_flow[metric_mask], metric_obs[metric_mask])
                if metric_obs is not None
                else {"samples": 0, "KGE": None, "NSE": None}
            ),
        }
        score = metrics[name]["KGE"]
        if glacier_targets is not None:
            targets = glacier_targets[
                (
                    glacier_targets.start_label
                    >= pd.Timestamp(period.start) - pd.Timedelta(days=spec.config.timestep_days)
                )
                & (glacier_targets.end_label <= _period_end(period))
            ]
            deltas = []
            cryo = pd.Series(records["glacier_cryo_storage_mm"], index=data.index)
            for row in targets.itertuples():
                deltas.append((cryo.loc[row.end_label] - cryo.loc[row.start_label]) / 1000 - row.observed_mb_mwe)
            rmse = float(np.sqrt(np.mean(np.square(deltas)))) if deltas else None
            metrics[name].update(
                glacier_balance_samples=len(deltas),
                glacier_balance_RMSE_mwe=rmse,
                glacier_area_basis=spec.glacier_constraint.area_basis,
            )
            if score is not None and rmse is not None:
                score -= spec.glacier_constraint.weight * rmse
        if snow_targets is not None:
            key = (
                "geometric_land_snow_fraction"
                if spec.snow_constraint.interpretation == "geometric_uniform_hru"
                else "effective_land_snow_area"
            )
            targets = snow_targets.observed_snow_fraction.reindex(data.index).to_numpy(float)
            valid = mask & np.isfinite(targets)
            rmse = float(np.sqrt(np.mean((records[key][valid] - targets[valid]) ** 2))) if valid.any() else None
            metrics[name].update(
                snow_samples=int(valid.sum()),
                snow_RMSE_fraction=rmse,
                snow_interpretation=spec.snow_constraint.interpretation,
            )
            if score is not None and rmse is not None:
                score -= spec.snow_constraint.weight * rmse
        metrics[name]["declared_composite_score"] = score
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    artifact_names = [
        "replay_spec.json",
        "streamflow.csv",
        "water_balance.csv",
        "streamflow_output.nc",
        "final_state.json",
        "metrics.json",
        "inputs/forcing.csv",
    ]
    if observation_path:
        artifact_names.append("inputs/observations.csv")
    if glacier_path:
        artifact_names.append("inputs/glacier_constraint.csv")
    if snow_path:
        artifact_names.append("inputs/snow_constraint.csv")
    manifest = {
        "schema_version": 1,
        "model": MODEL_ID,
        "family": "SACSMA",
        "parameter_status": spec.parameter_status,
        "qualifications": list(spec.qualifications),
        "implementation_hashes": hashes,
        "package_versions": versions,
        "artifacts_sha256": {name: sha(output / name) for name in artifact_names},
        "balance": {
            "max_daily_or_step_residual_mm": float(np.max(np.abs(residual))),
            "max_cumulative_residual_mm": float(np.max(np.abs(np.cumsum(residual)))),
        },
        "ensemble_vote_rule": "One SACSMA family vote; alternative configurations are comparators or share that vote.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_audited_candidate(output_dir):
    """Load an immutable replay bundle as a typed SAC-SMA family candidate."""
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("model") not in SUPPORTED_MODEL_IDS or manifest.get("family") != "SACSMA":
        raise ReplayValidationError("Wrong model/family identity")
    required = {
        "replay_spec.json",
        "streamflow.csv",
        "water_balance.csv",
        "streamflow_output.nc",
        "final_state.json",
        "metrics.json",
        "inputs/forcing.csv",
    }
    if not required.issubset(manifest.get("artifacts_sha256", {})):
        raise ReplayValidationError("Incomplete replay manifest")
    for name, digest in manifest["artifacts_sha256"].items():
        _read_input(root, FileRef(path=name, sha256=digest))
    spec = ReplaySpec.model_validate_json((root / "replay_spec.json").read_text())
    if (
        spec.model != manifest.get("model")
        or spec.implementation_hashes != manifest.get("implementation_hashes")
        or spec.package_versions != manifest.get("package_versions")
        or spec.parameter_status != manifest.get("parameter_status")
        or list(spec.qualifications) != manifest.get("qualifications")
    ):
        raise ReplayValidationError("Manifest/spec identity mismatch")
    _read_input(root, spec.forcing)
    if spec.observations:
        _read_input(root, spec.observations)
    if spec.glacier_constraint:
        _read_input(root, spec.glacier_constraint.targets)
    if spec.snow_constraint:
        _read_input(root, spec.snow_constraint.targets)
    frame = _frame(root / "streamflow.csv")
    balance = _frame(root / "water_balance.csv")
    flow = frame.streamflow_cms
    if not np.isfinite(flow).all() or (flow < 0).any() or not frame.index.equals(balance.index):
        raise ReplayValidationError("Invalid audited series/time axis")
    residual = balance.balance_residual_mm.to_numpy(float)
    if not np.isfinite(residual).all() or np.max(np.abs(residual)) > 1e-6 or np.max(np.abs(np.cumsum(residual))) > 1e-5:
        raise ReplayValidationError("Audited balance gate failed")
    forcing = _frame(_read_input(root, spec.forcing))
    if not forcing.index.equals(frame.index):
        raise ReplayValidationError("Replay/forcing time axes differ")
    expected_flow = balance.runoff_mm * spec.config.area_km2 * 1000 / (spec.config.timestep_days * 86400)
    if not np.allclose(flow, expected_flow, rtol=1e-12, atol=1e-10):
        raise ReplayValidationError("Discharge/depth conversion mismatch")
    import xarray as xr

    with xr.open_dataset(root / "streamflow_output.nc") as ds:
        if (
            ds.attrs.get("model") != spec.model
            or ds.attrs.get("family") != "SACSMA"
            or ds.streamflow.attrs.get("units") != "m3 s-1"
        ):
            raise ReplayValidationError("NetCDF model/family/units mismatch")
        nc = ds.streamflow.to_series()
        if not nc.index.equals(frame.index) or not np.allclose(nc, flow, rtol=1e-12, atol=1e-10):
            raise ReplayValidationError("NetCDF/CSV discharge mismatch")
    return flow, manifest


def audit_candidate(output_dir, observations, expected_periods):
    """Audit a candidate against the ensemble's observations and scoring windows.

    Returns native discharge and provenance; selection remains the caller's job.
    The candidate belongs to SACSMA and must not earn an additional family vote.
    expected_periods maps period names to (start, end, role). Inspected later
    windows must use development_validation or transfer, not independent holdout.
    """
    flow, manifest = load_audited_candidate(output_dir)
    root = Path(output_dir).resolve()
    spec = ReplaySpec.model_validate_json((root / "replay_spec.json").read_text())
    actual = {name: (p.start, p.end, p.role) for name, p in spec.periods.items()}
    if actual != expected_periods:
        raise ReplayValidationError("Candidate/ensemble scoring periods differ")
    if spec.observations is None:
        raise ReplayValidationError("Ensemble audit requires recorded observations")
    recorded = _frame(_read_input(root, spec.observations)).discharge_cms
    if not isinstance(observations.index, pd.DatetimeIndex) or not observations.index.is_unique:
        raise ReplayValidationError("Ensemble observations require a unique datetime axis")
    aligned = recorded.reindex(observations.index)
    if aligned.isna().any() or not np.array_equal(aligned.to_numpy(), observations.to_numpy()):
        raise ReplayValidationError("Candidate/ensemble observations differ")
    return flow, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = replay_bundle(args.spec, args.output)
    print(
        json.dumps(
            {"model": manifest["model"], "output": str(Path(args.output).resolve()), "balance": manifest["balance"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
