"""Synthetic VM workload generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class VMSet:
    cpu: np.ndarray   # cores
    ram: np.ndarray   # GB
    bw: np.ndarray    # Mbps (reserved)

    @property
    def num_vms(self) -> int:
        return int(self.cpu.size)

    @property
    def demand(self) -> np.ndarray:
        """(V, 3) matrix: cpu, ram, bw."""
        return np.column_stack([self.cpu, self.ram, self.bw]).astype(np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"vm_id": np.arange(self.num_vms), "cpu_cores": self.cpu,
                             "ram_gb": self.ram, "bw_mbps": self.bw})

    def save_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


def generate_vms(num_vms: int, rng: np.random.Generator, cfg: dict[str, Any],
                 traffic_mbps: np.ndarray | None = None) -> VMSet:
    """Generate VM resource requirements.

    Reserved bandwidth = random base + ``bw_traffic_factor`` * the VM's own traffic,
    so a VM's reservation always covers the traffic it can send.
    """
    if num_vms < 1:
        raise ValueError("num_vms must be >= 1")
    probs = np.asarray(cfg["cpu_probs"], dtype=float)
    cpu = rng.choice(cfg["cpu_choices"], size=num_vms, p=probs / probs.sum()).astype(float)
    ram = cpu * rng.choice(cfg["ram_per_cpu_gb"], size=num_vms)
    lo, hi = cfg["bw_base_mbps"]
    bw = rng.uniform(lo, hi, size=num_vms)
    if traffic_mbps is not None:
        if traffic_mbps.shape != (num_vms,):
            raise ValueError("traffic_mbps must have one entry per VM")
        bw = bw + cfg.get("bw_traffic_factor", 1.0) * traffic_mbps
    return VMSet(cpu=cpu, ram=ram, bw=np.round(bw, 1))
