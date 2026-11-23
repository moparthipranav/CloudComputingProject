"""Synthetic physical-server generation, sized to the VM workload."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ServerSet:
    names: list[str]
    capacity: np.ndarray    # (S, 3) cpu cores, ram GB, bw Mbps
    background: np.ndarray  # (S, 3) pre-existing ("current") utilisation, absolute

    @property
    def num_servers(self) -> int:
        return len(self.names)

    def to_dataframe(self) -> pd.DataFrame:
        c, b = self.capacity, self.background
        return pd.DataFrame({
            "server_id": np.arange(self.num_servers), "name": self.names,
            "cpu_capacity": c[:, 0], "ram_capacity_gb": c[:, 1], "bw_capacity_mbps": c[:, 2],
            "cpu_used": b[:, 0], "ram_used_gb": b[:, 1], "bw_used_mbps": b[:, 2]})

    def save_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


def generate_servers(names: list[str], demand: np.ndarray, rng: np.random.Generator,
                     cfg: dict[str, Any]) -> ServerSet:
    """Create heterogeneous servers so that aggregate demand ~ target_utilization of
    aggregate free capacity, rounded up to realistic units (2 cores, 4 GB, 100 Mbps)."""
    s = len(names)
    target = cfg["target_utilization"]
    bg_lo, bg_hi = cfg["background_utilization"]
    bg_frac = rng.uniform(bg_lo, bg_hi, size=(s, 3))
    mult = rng.choice(cfg["capacity_heterogeneity"], size=s).astype(float)
    mult /= mult.mean()
    total = demand.sum(axis=0)
    mean_cap = total / (target * s * (1.0 - bg_frac.mean(axis=0)))
    cap = mean_cap[None, :] * mult[:, None]
    # every server must be able to host the largest single VM
    cap = np.maximum(cap, demand.max(axis=0)[None, :] / (1.0 - bg_hi))
    unit = np.array([2.0, 4.0, 100.0])
    cap = np.ceil(cap / unit) * unit
    return ServerSet(names=list(names), capacity=cap, background=cap * bg_frac)
