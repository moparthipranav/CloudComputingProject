"""Configuration loading and light validation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration file and validate it."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping")
    validate_config(cfg)
    return cfg


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``base`` updated recursively with ``override``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ``ValueError`` for obviously invalid settings."""
    for section in ("topology", "servers", "vms", "communication", "objective",
                    "simulated_annealing", "experiments"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    w = cfg["objective"]["weights"]
    for name in ("alpha", "beta", "gamma", "delta"):
        if w.get(name, 0) < 0:
            raise ValueError(f"Objective weight {name} must be >= 0")
    if sum(w.get(n, 0) for n in ("alpha", "beta", "gamma", "delta")) <= 0:
        raise ValueError("At least one objective weight must be positive")
    sa = cfg["simulated_annealing"]
    if not 0 < sa["cooling_rate"] < 1:
        raise ValueError("cooling_rate must be in (0, 1)")
    if sa["cooling_schedule"] not in ("geometric", "linear", "fast"):
        raise ValueError("cooling_schedule must be geometric, linear or fast")
    if not 0 < cfg["servers"]["target_utilization"] < 1:
        raise ValueError("servers.target_utilization must be in (0, 1)")
