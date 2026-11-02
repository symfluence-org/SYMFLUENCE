# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Forcings store builder for the model-ready data store.

Symlinks (or copies) basin-averaged forcing NetCDF files into
``data/model_ready/forcings/`` and enriches them with CF-1.8 global
attributes and per-variable source metadata.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.modeling.model_ready.cf_conventions import (
    CANONICAL_FORCING_ALIASES,
    CF_STANDARD_NAMES,
    build_global_attrs,
)

from .source_metadata import SourceMetadata

logger = logging.getLogger(__name__)


def _staging_suffix() -> str:
    """Suffix for a private staging path.

    Unique per writer — process *and* thread — so that two builders publishing
    the same shared artefact never share a staging path.
    """
    return f".staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"


def _publish(tmp: Path, dst: Path) -> None:
    """Atomically publish a staged path over its destination.

    POSIX rename replaces a destination another process holds open; Windows
    MoveFileEx fails with EACCES until the handle closes. Retrying briefly
    keeps the store's contract — concurrent readers see the old file or the
    new one, never a missing or half-written one — cross-platform.
    """
    deadline = time.monotonic() + 5.0
    delay = 0.01
    while True:
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if os.name != 'nt' or time.monotonic() >= deadline:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 0.25)


class ForcingsStoreBuilder:
    """Build the forcings section of the model-ready data store.

    Parameters
    ----------
    project_dir : Path
        Root of the SYMFLUENCE domain directory.
    domain_name : str
        Name of the hydrological domain.
    forcing_dataset : str
        Name of the forcing dataset (e.g. ``'ERA5'``, ``'RDRS'``).
    strategy : str
        Either ``'symlink'`` (default) or ``'copy'``.  Symlinks are
        preferred; copies are needed on HPC file-systems that forbid
        symlinks across partitions.
    """

    def __init__(
        self,
        project_dir: Path,
        domain_name: str,
        forcing_dataset: str = 'ERA5',
        strategy: str = 'symlink',
    ) -> None:
        """Initialise the forcings builder.

        Args:
            project_dir: Root of the SYMFLUENCE domain directory.
            domain_name: Name of the hydrological domain.
            forcing_dataset: Forcing dataset identifier (e.g. ``'ERA5'``).
            strategy: ``'symlink'`` (default) or ``'copy'``.
        """
        self.project_dir = project_dir
        self.domain_name = domain_name
        self.forcing_dataset = forcing_dataset
        self.strategy = strategy

        self.source_dir = resolve_data_subdir(project_dir, 'forcing') / 'basin_averaged_data'
        self.target_dir = project_dir / 'data' / 'model_ready' / 'forcings'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Optional[Path]:
        """Build the forcings store.

        Returns the target directory on success, or ``None`` if the source
        directory does not exist or contains no NetCDF files.
        """
        if not self.source_dir.exists():
            logger.info(
                "Skipping forcings store: source dir does not exist: %s",
                self.source_dir,
            )
            return None

        nc_files = list(self.source_dir.glob('*.nc'))
        if not nc_files:
            logger.info("No NetCDF files in %s, skipping forcings store", self.source_dir)
            return None

        # A store may legitimately hold more than one discretization of the same
        # domain — a lumped hru=1 forcing beside a 12-band elevation hru=12
        # forcing HYPE built — provided each lives under its own namespaced
        # {domain}_{forcing}_remapped_{token}_* filename so readers can select
        # the one matching their discretization (issue #339). Verify each
        # namespace is internally size-consistent, and fail loudly at the write
        # boundary — naming the stray file — only on a genuine collision WITHIN a
        # namespace (including two legacy untokened files of different sizes, the
        # original #339 signature), so a corrupt namespace never reaches a model.
        self._assert_consistent_within_discretization(nc_files)

        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Enrich before linking so that a 'copy' strategy store carries the
        # metadata contract too.
        self._enrich_metadata(nc_files)
        self._create_links(nc_files)

        logger.info(
            "Forcings store built: %d files in %s", len(nc_files), self.target_dir
        )
        return self.target_dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_consistent_within_discretization(self, nc_files: list) -> None:
        """Reject a source set that collides WITHIN a single discretization.

        Delegates to the reader's namespace-aware check so the writer and the
        reader enforce one contract: within each ``_remapped_{token}_``
        namespace every file shares the same ``hru`` (and ``gru``) size, while
        distinct namespaces (a lumped hru=1 forcing beside an elevation hru=12
        forcing) coexist. Raises ``ValueError`` naming the offending file(s) on a
        genuine intra-namespace collision.
        """
        from symfluence.core.modeling.model_ready.forcing_reader import assert_consistent_within_discretization

        assert_consistent_within_discretization([Path(f) for f in nc_files])

    def _create_links(self, nc_files: list) -> None:
        """Create symlinks or copies from source to target directory.

        Prunes stale target entries first: a store built from an earlier, smaller source set (e.g.
        before basin-averaging extended the forcing to the full period) must reconcile against the
        current source on rebuild, not just add missing links -- otherwise it keeps serving the old,
        truncated file set.
        """
        current = {src.name for src in nc_files}
        for existing in self.target_dir.glob('*.nc'):
            if existing.name in current:
                continue
            # Re-check against the live source dir before unlinking: another
            # process on this domain may have published a new remapped file
            # after our snapshot was taken. Pruning on the stale snapshot alone
            # would delete a link that is both valid and in use.
            if (self.source_dir / existing.name).exists():
                logger.debug("Keeping forcing link %s (present in source)", existing.name)
                continue
            existing.unlink(missing_ok=True)
            logger.debug("Pruned stale forcing link %s", existing.name)
        for src_file in nc_files:
            dst = self.target_dir / src_file.name
            resolved = src_file.resolve()

            if self.strategy == 'copy':
                # Stage the copy then rename over the destination: concurrent
                # readers (parallel calibration workers) see old or new,
                # never a missing file.
                tmp = dst.with_name(f"{dst.name}{_staging_suffix()}")
                shutil.copy2(str(src_file), str(tmp))
                _publish(tmp, dst)
                logger.debug("Copied %s -> %s", src_file.name, dst)
            else:
                # Fast path: link already correct — touch nothing, no window.
                if dst.is_symlink() and os.readlink(str(dst)) == str(resolved):
                    continue
                tmp = dst.with_name(f"{dst.name}{_staging_suffix()}")
                if tmp.is_symlink() or tmp.exists():
                    tmp.unlink()
                os.symlink(resolved, tmp)
                _publish(tmp, dst)
                logger.debug("Symlinked %s -> %s", src_file.name, dst)

    def _forcing_timestep_seconds(self, nc_files: list) -> Optional[float]:
        """Determine the forcing timestep (seconds) from a file's time axis.

        Validates the file's entire axis, not just the first interval: writing a
        ``timestep_seconds`` attribute derived from an irregular axis would let
        downstream resampling declare gappy forcing "regular" and skip fixing it.
        Returns ``None`` (no attribute written) when no file has a regular axis.
        """
        try:
            import xarray as xr

            from symfluence.core.modeling.model_ready.forcing_reader import (
                _has_datetime_axis,
                validated_timestep_seconds,
            )
            for f in nc_files:
                with xr.open_dataset(f) as ds:
                    if _has_datetime_axis(ds):
                        try:
                            return validated_timestep_seconds(ds['time'].values, context=Path(f).name)
                        except ValueError as e:
                            logger.warning("Not writing timestep_seconds: %s", e)
                            return None
        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            logger.warning("Could not determine forcing timestep: %s", e, exc_info=True)
        return None

    @staticmethod
    def _needs_enrichment(src_file: Path, dt_seconds: Optional[float]) -> bool:
        """Report whether *src_file* still lacks the store's metadata contract.

        Enrichment writes into the basin-averaged files, which are a
        domain-shared product that other processes may be reading.  A rebuild
        must therefore leave already-enriched files untouched — not reopen them
        in append mode — so the shared forcing is immutable in steady state.
        """
        import netCDF4

        try:
            with netCDF4.Dataset(str(src_file), 'r') as ds:
                attrs = set(ds.ncattrs())
                if 'Conventions' not in attrs:
                    return True
                vocabulary = ds.getncattr('forcing_vocabulary') if 'forcing_vocabulary' in attrs else None
                if vocabulary != 'SYMFLUENCE-canonical':
                    return True
                return dt_seconds is not None and 'timestep_seconds' not in attrs
        except (OSError, RuntimeError, AttributeError, KeyError) as e:
            # An unreadable/incomplete file (e.g. a concurrent writer mid-swap)
            # is treated as needing enrichment: the caller then works on a copy
            # and republishes atomically, never mutating the shared original.
            logger.debug("Could not inspect metadata for %s: %s", src_file.name, e)
            return True

    def _enrich_metadata(self, nc_files: list) -> None:
        """Add CF-1.8 global attrs and per-variable source attrs.

        Operates on the *original* files so that metadata is visible
        through symlinks and persists across rebuilds.  Files that already
        carry the contract are skipped entirely (see ``_needs_enrichment``).
        """
        try:
            import netCDF4
        except ImportError:
            logger.warning("netCDF4 not available; skipping metadata enrichment")
            return

        global_attrs = build_global_attrs(
            domain_name=self.domain_name,
            title=f'{self.domain_name} forcing data',
            history=f'Enriched by ForcingsStoreBuilder from {self.forcing_dataset}',
        )
        # Declare the canonical forcing timestep on the store so downstream
        # model adapters never have to guess it (the recurring "rate x 3600 over
        # a 3-hourly step" bug). Computed from the time axis of the file.
        global_attrs['forcing_vocabulary'] = 'SYMFLUENCE-canonical'
        dt_seconds = self._forcing_timestep_seconds(nc_files)
        if dt_seconds is not None:
            global_attrs['timestep_seconds'] = float(dt_seconds)

        source_meta = SourceMetadata(
            source=self.forcing_dataset,
            processing='EASMORE area-weighted remapping',
        )
        var_attrs = source_meta.to_netcdf_attrs()

        for src_file in nc_files:
            if not self._needs_enrichment(src_file, dt_seconds):
                logger.debug("Forcing %s already enriched; leaving untouched", src_file.name)
                continue
            # Enrich a private copy and publish it with an atomic rename. The
            # basin-averaged forcing is shared across every model in the domain;
            # mutating it in place would expose concurrent readers to a
            # half-updated NetCDF (out-of-range values, truncated variables).
            staged = src_file.with_name(f"{src_file.name}.enrich.{os.getpid()}")
            try:
                shutil.copy2(str(src_file), str(staged))
                with netCDF4.Dataset(str(staged), 'a') as ds:
                    # Global attributes — refresh canonical contract markers.
                    if 'Conventions' not in ds.ncattrs():
                        ds.setncatts({k: v for k, v in global_attrs.items()
                                      if k not in ('timestep_seconds', 'forcing_vocabulary')})
                    ds.setncattr('forcing_vocabulary', 'SYMFLUENCE-canonical')
                    if dt_seconds is not None and 'timestep_seconds' not in ds.ncattrs():
                        ds.setncattr('timestep_seconds', float(dt_seconds))

                    # Per-variable CF attributes, resolving canonical aliases so
                    # CARRA/SUMMA-vocabulary names (pptrate/airtemp/...) carry
                    # declared standard_name + units, not just the CF keys.
                    for var_name in ds.variables:
                        var = ds.variables[var_name]
                        cf = CF_STANDARD_NAMES.get(var_name) or CF_STANDARD_NAMES.get(
                            CANONICAL_FORCING_ALIASES.get(var_name, ''))
                        if cf:
                            for attr_key, attr_val in cf.items():
                                if attr_key not in var.ncattrs():
                                    var.setncattr(attr_key, attr_val)

                        # Source provenance (skip coordinate variables)
                        if var_name not in ('time', 'hru', 'gru', 'lat', 'lon'):
                            for ak, av in var_attrs.items():
                                if ak not in var.ncattrs():
                                    var.setncattr(ak, av)

                _publish(staged, src_file)
                logger.debug("Enriched forcing metadata for %s", src_file.name)

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                logger.warning("Could not enrich metadata for %s: %s", src_file.name, e, exc_info=True)
            finally:
                if staged.exists():
                    staged.unlink()
