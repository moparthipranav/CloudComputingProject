"""Hard resource constraints (CPU, RAM, bandwidth) for VM placement.

Placement vectors are ``int`` arrays of length V holding the server index of each
VM (-1 = unassigned). Nothing in this module ever silently accepts a violation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RESOURCES = ("cpu", "ram", "bw")
EPS = 1e-9


class ConstraintViolationError(RuntimeError):
    """Raised when a placement would exceed a server capacity."""


class InfeasiblePlacementError(RuntimeError):
    """Raised when no server can host a VM."""


def compute_usage(demand: np.ndarray, placement: np.ndarray, num_servers: int) -> np.ndarray:
    """(S, 3) VM resource usage per server for a complete placement."""
    if placement.shape != (demand.shape[0],):
        raise ValueError("placement must have one entry per VM")
    if placement.min(initial=0) < 0 or placement.max(initial=0) >= num_servers:
        raise ValueError("placement contains unassigned or out-of-range servers")
    return np.column_stack([np.bincount(placement, weights=demand[:, r], minlength=num_servers)
                            for r in range(demand.shape[1])])


def find_violations(placement: np.ndarray, demand: np.ndarray, capacity: np.ndarray,
                    background: np.ndarray) -> list[tuple[int, str, float]]:
    """Return (server, resource, excess) for every violated capacity."""
    used = compute_usage(demand, placement, capacity.shape[0])
    excess = used + background - capacity
    bad = np.argwhere(excess > EPS)
    return [(int(s), RESOURCES[r], float(excess[s, r])) for s, r in bad]


@dataclass
class ResourceManager:
    """Tracks free capacity per server and enforces constraints on every change."""
    capacity: np.ndarray    # (S, 3)
    background: np.ndarray  # (S, 3)
    demand: np.ndarray      # (V, 3)
    placement: np.ndarray = field(init=False)
    free: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity.shape != self.background.shape:
            raise ValueError("capacity and background shapes differ")
        if np.any(self.background > self.capacity + EPS):
            raise ValueError("background utilisation exceeds capacity")
        self.placement = np.full(self.demand.shape[0], -1, dtype=np.int64)
        self.free = self.capacity - self.background

    @property
    def num_servers(self) -> int:
        return self.capacity.shape[0]

    @property
    def num_vms(self) -> int:
        return self.demand.shape[0]

    def fits(self, vm: int, server: int) -> bool:
        return bool((self.free[server] - self.demand[vm]).min() >= -EPS)

    def feasible_mask(self, vm: int) -> np.ndarray:
        return (self.free - self.demand[vm] >= -EPS).all(axis=1)

    def assign(self, vm: int, server: int) -> None:
        if self.placement[vm] != -1:
            raise ValueError(f"VM {vm} is already placed; use move()")
        if not self.fits(vm, server):
            raise ConstraintViolationError(f"VM {vm} does not fit on server {server}")
        self.placement[vm] = server
        self.free[server] -= self.demand[vm]

    def release(self, vm: int) -> None:
        server = self.placement[vm]
        if server < 0:
            raise ValueError(f"VM {vm} is not placed")
        self.free[server] += self.demand[vm]
        self.placement[vm] = -1

    def move(self, vm: int, new_server: int) -> None:
        old = int(self.placement[vm])
        if old == new_server:
            return
        if not self.fits(vm, new_server):
            raise ConstraintViolationError(f"VM {vm} does not fit on server {new_server}")
        if old >= 0:
            self.free[old] += self.demand[vm]
        self.free[new_server] -= self.demand[vm]
        self.placement[vm] = new_server

    def load_placement(self, placement: np.ndarray) -> None:
        """Replace the state with a complete placement; raises on any violation."""
        viol = find_violations(placement, self.demand, self.capacity, self.background)
        if viol:
            s, r, e = viol[0]
            raise ConstraintViolationError(
                f"{len(viol)} violated capacities, e.g. server {s} {r} exceeded by {e:.3f}")
        used = compute_usage(self.demand, placement, self.num_servers)
        self.placement = placement.astype(np.int64).copy()
        self.free = self.capacity - self.background - used

    def utilization(self, include_background: bool = True) -> np.ndarray:
        """(S, 3) utilisation fractions of the current placement."""
        used = self.capacity - self.free
        if not include_background:
            used = used - self.background
        return used / self.capacity

    def copy(self) -> "ResourceManager":
        rm = ResourceManager(self.capacity, self.background, self.demand)
        rm.placement = self.placement.copy()
        rm.free = self.free.copy()
        return rm
