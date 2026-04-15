"""Shared helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def normalize_cutoff_end(cutoff: pd.Timestamp) -> pd.Timestamp:
    """End-of-day inclusive upper bound for observation features."""
    if not isinstance(cutoff, pd.Timestamp):
        cutoff = pd.Timestamp(cutoff)
    return cutoff.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def observation_bounds(
    cutoff: pd.Timestamp, observation_days: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    obs_end = normalize_cutoff_end(cutoff)
    obs_start = obs_end.normalize() - pd.Timedelta(days=observation_days)
    return obs_start, obs_end
