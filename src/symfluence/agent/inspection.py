# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Read-only platform inspection behind the agent's MCP tools.

These helpers answer the questions a modelling session asks constantly — where
are the logs, how far along is the calibration, what did the experiment score —
by reading the ``domain_{NAME}`` tree directly (cheap file reads, stdlib CSV
parsing; nothing imports the heavy scientific stack). ``update_config`` is the
one writer: a guarded, validated, line-preserving edit of the user's own
experiment YAML.

Path conventions read here (see ``project/logging_manager.py`` and
``optimization/optimization_results_manager.py``):

- domain dir:      ``{SYMFLUENCE_DATA_DIR}/domain_{DOMAIN_NAME}``
- run logs:        ``{domain}/_workLog_{DOMAIN_NAME}/symfluence_{DOMAIN_NAME}[_{EXPERIMENT_ID}]_{ts}.log``
  (legacy runs: ``symfluence_general_{DOMAIN_NAME}_{ts}.log``)
- optimization:    ``{domain}/optimization/{algorithm}_{experiment_id}/``
  with ``best_parameters.csv``, ``optimization_history.csv``,
  ``optimization_metadata.csv``, and ``final_evaluation/``.
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

from .context import _find_value


def _load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    import yaml
    data = yaml.safe_load(path.read_text(encoding='utf-8', errors='replace'))
    if not isinstance(data, dict):
        raise ValueError(f"Not a SYMFLUENCE config (not a YAML mapping): {path}")
    return data


def resolve_domain(config_path: str | Path) -> tuple[str, Path]:
    """The (domain_name, domain_dir) a config points at."""
    data = _load_config(config_path)
    domain = _find_value(data, 'DOMAIN_NAME')
    data_dir = _find_value(data, 'SYMFLUENCE_DATA_DIR')
    if not domain:
        raise ValueError(f"No DOMAIN_NAME in {config_path}")
    if not data_dir:
        raise ValueError(f"No SYMFLUENCE_DATA_DIR in {config_path}")
    return str(domain), Path(str(data_dir)).expanduser() / f'domain_{domain}'


def read_run_log(
    config_path: str | Path,
    log_type: str = 'general',
    tail_lines: int = 100,
    grep: str | None = None,
) -> dict:
    """Tail (and optionally filter) the newest run log for a config's domain."""
    domain, domain_dir = resolve_domain(config_path)
    log_dir = domain_dir / f'_workLog_{domain}'
    if not log_dir.is_dir():
        raise ValueError(f"No log directory yet: {log_dir}")

    if log_type == 'general':
        # New protocol: symfluence_{domain}[_{experiment_id}]_{ts}.log;
        # keep matching the legacy symfluence_general_{domain}_*.log too.
        candidates = (
            set(log_dir.glob(f'symfluence_{domain}_*.log'))
            | set(log_dir.glob(f'symfluence_general_{domain}_*.log'))
        )
    else:
        candidates = set(log_dir.glob(f'symfluence_{log_type}_{domain}_*.log'))
    # Tie-break equal mtimes (coarse filesystem timestamps, e.g. Windows) by
    # name — the filenames embed their creation timestamp.
    logs = sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))
    if not logs:
        raise ValueError(f"No '{log_type}' logs in {log_dir}")
    log_path = logs[-1]

    lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    if grep:
        lines = [line for line in lines if grep in line]
    return {'log_path': str(log_path), 'lines': lines[-max(0, tail_lines):]}


def list_domains(root_path: str | Path | None = None,
                 config_path: str | Path | None = None) -> dict:
    """The ``domain_*`` directories under a data root, with a quick shape summary.

    ``root_path`` wins; otherwise the root comes from the config's
    ``SYMFLUENCE_DATA_DIR``.
    """
    if root_path is not None:
        root = Path(root_path).expanduser()
    elif config_path is not None:
        data = _load_config(config_path)
        data_dir = _find_value(data, 'SYMFLUENCE_DATA_DIR')
        if not data_dir:
            raise ValueError(f"No SYMFLUENCE_DATA_DIR in {config_path}")
        root = Path(str(data_dir)).expanduser()
    else:
        raise ValueError("Provide root_path or config_path")

    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    domains = []
    for path in sorted(root.glob('domain_*')):
        if not path.is_dir():
            continue
        optimization = path / 'optimization'
        experiments = sorted(
            p.name for p in optimization.iterdir() if p.is_dir()
        ) if optimization.is_dir() else []
        domains.append({
            'name': path.name.removeprefix('domain_'),
            'path': str(path),
            'experiments': experiments,
            'has_optimization': bool(experiments),
            'has_simulations': (path / 'simulations').is_dir(),
        })
    return {'root': str(root), 'domains': domains}


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding='utf-8', errors='replace', newline='') as fh:
        return list(csv.DictReader(fh))


def _pick_experiment_dir(domain_dir: Path, experiment_id: str | None) -> Path:
    """The optimization run directory to inspect (newest unless pinned)."""
    optimization = domain_dir / 'optimization'
    if not optimization.is_dir():
        raise ValueError(f"No optimization directory yet: {optimization}")
    candidates = [p for p in optimization.iterdir() if p.is_dir()]
    if experiment_id:
        candidates = [p for p in candidates if p.name.endswith(f'_{experiment_id}')
                      or p.name == experiment_id]
    if not candidates:
        raise ValueError(
            f"No optimization runs{f' for {experiment_id!r}' if experiment_id else ''} "
            f"in {optimization}"
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calibration_status(config_path: str | Path,
                       experiment_id: str | None = None) -> dict:
    """Progress of the newest (or named) calibration run for a config's domain."""
    _, domain_dir = resolve_domain(config_path)
    run_dir = _pick_experiment_dir(domain_dir, experiment_id)

    result: dict = {'run_dir': str(run_dir), 'run_name': run_dir.name}

    metadata_csv = run_dir / 'optimization_metadata.csv'
    if metadata_csv.is_file():
        rows = _read_csv_rows(metadata_csv)
        if rows:
            meta = rows[0]
            result.update({
                'algorithm': meta.get('algorithm'),
                'experiment_id': meta.get('experiment_id'),
                'metric': meta.get('target_metric'),
                'best_score': _float_or_none(meta.get('best_score')),
                'completed_at': meta.get('completed_at'),
            })

    history_csv = run_dir / 'optimization_history.csv'
    if history_csv.is_file():
        rows = _read_csv_rows(history_csv)
        result['iterations'] = len(rows)
        best_score, best_iteration = None, None
        for row in rows:
            score = _float_or_none(row.get('best_score'))
            if score is not None and (best_score is None or score > best_score):
                best_score, best_iteration = score, row.get('generation')
        result.setdefault('best_score', best_score)
        result['best_iteration'] = best_iteration
        result['in_progress'] = not metadata_csv.is_file()

    params_csv = run_dir / 'best_parameters.csv'
    if params_csv.is_file():
        result['best_parameters_path'] = str(params_csv)

    if 'iterations' not in result and 'algorithm' not in result:
        raise ValueError(f"No optimization records yet in {run_dir}")
    return result


def get_results_summary(config_path: str | Path,
                        experiment_id: str | None = None) -> dict:
    """Headline metrics and artifact locations for an experiment."""
    _, domain_dir = resolve_domain(config_path)
    run_dir = _pick_experiment_dir(domain_dir, experiment_id)

    metrics: dict = {}
    sources: list[str] = []

    metadata_csv = run_dir / 'optimization_metadata.csv'
    if metadata_csv.is_file():
        rows = _read_csv_rows(metadata_csv)
        if rows:
            meta = rows[0]
            metric_name = meta.get('target_metric') or 'score'
            score = _float_or_none(meta.get('best_score'))
            if score is not None:
                metrics[str(metric_name)] = score
            sources.append(str(metadata_csv))

    final_evaluation = run_dir / 'final_evaluation'
    if final_evaluation.is_dir():
        for path in sorted(final_evaluation.rglob('*.csv')):
            sources.append(str(path))
            for row in _read_csv_rows(path):
                for key, value in row.items():
                    score = _float_or_none(value)
                    if score is not None and key and any(
                        token in key.upper()
                        for token in ('KGE', 'NSE', 'RMSE', 'PBIAS', 'MAE', 'R2')
                    ):
                        metrics.setdefault(key, score)

    plots_dir = run_dir / 'plots'
    plots = sorted(str(p) for p in plots_dir.glob('*.png')) if plots_dir.is_dir() else []

    if not metrics and not sources:
        raise ValueError(
            f"No results found in {run_dir} — has the calibration finished? "
            f"Try calibration_status."
        )
    return {
        'run_dir': str(run_dir),
        'run_name': run_dir.name,
        'metrics': metrics,
        'sources': sources,
        'plots': plots,
    }


def get_plot_paths(config_path: str | Path,
                   experiment_id: str | None = None) -> dict:
    """Figures produced for the newest (or named) experiment run."""
    _, domain_dir = resolve_domain(config_path)
    run_dir = _pick_experiment_dir(domain_dir, experiment_id)
    plots = sorted(
        str(p) for pattern in ('*.png', '*.pdf', '*.svg')
        for p in run_dir.rglob(pattern)
    )
    return {'run_dir': str(run_dir), 'run_name': run_dir.name, 'plots': plots}


def compare_experiments(config_path: str | Path) -> dict:
    """One row per optimization run of the config's domain, best score first."""
    _, domain_dir = resolve_domain(config_path)
    optimization = domain_dir / 'optimization'
    if not optimization.is_dir():
        raise ValueError(f"No optimization directory yet: {optimization}")

    rows = []
    for run_dir in sorted(p for p in optimization.iterdir() if p.is_dir()):
        row: dict = {'run_name': run_dir.name}
        metadata_csv = run_dir / 'optimization_metadata.csv'
        if metadata_csv.is_file():
            meta_rows = _read_csv_rows(metadata_csv)
            if meta_rows:
                meta = meta_rows[0]
                row.update({
                    'algorithm': meta.get('algorithm'),
                    'experiment_id': meta.get('experiment_id'),
                    'metric': meta.get('target_metric'),
                    'best_score': _float_or_none(meta.get('best_score')),
                    'completed_at': meta.get('completed_at'),
                })
        history_csv = run_dir / 'optimization_history.csv'
        if history_csv.is_file():
            row['iterations'] = len(_read_csv_rows(history_csv))
            row.setdefault('best_score', max(
                (score for r in _read_csv_rows(history_csv)
                 if (score := _float_or_none(r.get('best_score'))) is not None),
                default=None,
            ))
        row['in_progress'] = history_csv.is_file() and not metadata_csv.is_file()
        rows.append(row)

    rows.sort(key=lambda r: (r.get('best_score') is None,
                             -(r.get('best_score') or 0.0)))
    return {'domain_dir': str(domain_dir), 'experiments': rows}


def update_config(config_path: str | Path, changes: dict,
                  dry_run: bool = False) -> dict:
    """Apply flat key changes to an experiment config, validated and backed up.

    The edit is line-preserving: existing ``KEY: value`` lines are rewritten in
    place (comments and ordering survive), missing keys are appended. The
    changed file must pass typed ``SymfluenceConfig`` validation before it
    replaces the original; the original is backed up next to itself first.
    """
    path = Path(config_path)
    _load_config(path)  # existence + YAML-mapping check
    if not isinstance(changes, dict) or not changes:
        raise ValueError("'changes' must be a non-empty mapping of config keys")

    import yaml

    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    changed_keys, appended_keys = [], []
    for key, value in changes.items():
        rendered = yaml.safe_dump({key: value}, default_flow_style=True,
                                  width=10_000).strip()
        # safe_dump of a one-key mapping gives '{KEY: value}'; unwrap it.
        rendered = rendered.removeprefix('{').removesuffix('}')
        replaced = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(f'{key}:') and len(line) - len(stripped) == 0:
                lines[i] = rendered
                replaced = True
                break
        if replaced:
            changed_keys.append(key)
        else:
            lines.append(rendered)
            appended_keys.append(key)

    new_text = '\n'.join(lines) + '\n'

    # Validate the result with the typed config system before touching the file.
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False,
                                     encoding='utf-8') as tmp:
        tmp.write(new_text)
        tmp_path = Path(tmp.name)
    try:
        from symfluence.core.config.models import SymfluenceConfig
        try:
            SymfluenceConfig.from_file(tmp_path)
        except Exception as e:  # noqa: BLE001 — any validation failure blocks the edit
            raise ValueError(f"Change rejected — config would become invalid: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    result = {
        'config_path': str(path),
        'changed_keys': changed_keys,
        'appended_keys': appended_keys,
        'dry_run': dry_run,
    }
    if dry_run:
        return result

    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    path.write_text(new_text, encoding='utf-8')
    result['backup_path'] = str(backup)
    return result
