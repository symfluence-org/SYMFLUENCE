"""Portable replay and audit interface for the SAC-SMA cryosphere variant."""

from __future__ import annotations

from importlib import import_module

__all__ = ["ReplaySpec", "replay_bundle", "load_audited_candidate", "audit_candidate"]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module(".replay", __name__), name)
    raise AttributeError(name)
