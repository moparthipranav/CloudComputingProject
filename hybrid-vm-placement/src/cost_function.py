"""Multi-objective cost function with exact incremental (delta) evaluation.

Total = alpha * C_comm/ref_comm + beta * C_lat/ref_lat + gamma * Imb/0.5 + delta * Mig

  C_comm = sum_{i<j} w_comm(i,j) * hops(P_i, P_j)         (traffic-weighted distance)
  C_lat  = sum_{i<j} w_lat(i,j)  * latency_ms(P_i, P_j)    (Dijkstra latency)
  Imb    = population std-dev of server load (mean of CPU & RAM utilisation)
  Mig    = RAM-weighted fraction of VMs that differ from the reference placement

The reference values are properties of the problem instance (worst case: every
communicating pair at maximum distance), so each normalised component lies in
[0, 1] and the normalisation never changes while optimising - which keeps the
incremental and full computations identical.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .communication_model import CommunicationModel
from .resource_manager import compute_usage

IMBALANCE_REF = 0.5  # max possible std-dev of values in [0, 1]


@dataclass(frozen=True)
class CostWeights:
    alpha: float = 0.4   # communication cost
    beta: float = 0.2    # network latency cost
    gamma: float = 0.3   # server load imbalance
    delta: float = 0.1   # migration cost

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma, self.delta) < 0:
            raise ValueError("Objective weights must be non-negative")

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> "CostWeights":
        return cls(**{k: float(d[k]) for k in ("alpha", "beta", "gamma", "delta") if k in d})


@dataclass
class CostBreakdown:
    comm_raw: float
    latency_raw: float
    imbalance_raw: float
    migration_raw: float
    comm_norm: float
    latency_norm: float
    imbalance_norm: float
    migration_norm: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class CostModel:
    """Static, placement-independent data needed to evaluate the objective."""

    def __init__(self, comm: CommunicationModel, hops: np.ndarray, latency: np.ndarray,
                 capacity: np.ndarray, background: np.ndarray, demand: np.ndarray,
                 weights: CostWeights, comm_weight: str = "frequency",
                 latency_weight: str = "intensity",
                 reference: np.ndarray | None = None) -> None:
        self.comm = comm
        self.hops = hops
        self.latency = latency
        self.capacity = capacity
        self.background = background
        self.demand = demand
        self.weights = weights
        self.n = comm.num_vms
        self.num_servers = capacity.shape[0]
        self.nbrs, self.wc = comm.neighbor_lists(comm_weight)
        _, self.wl = comm.neighbor_lists(latency_weight)
        self.ps, self.pd, self.pc = comm.pair_arrays(comm_weight)
        _, _, self.pl = comm.pair_arrays(latency_weight)
        self.comm_ref = max(float(self.pc.sum()) * float(hops.max()), 1e-12)
        self.lat_ref = max(float(self.pl.sum()) * float(latency.max()), 1e-12)
        # scalar look-up tables (python floats are faster than numpy scalars in the SA loop)
        self._dcpu = demand[:, 0].tolist()
        self._dram = demand[:, 1].tolist()
        self._icpu = (1.0 / capacity[:, 0]).tolist()
        self._iram = (1.0 / capacity[:, 1]).tolist()
        self.total_ram = float(demand[:, 1].sum())
        self.mig_w = (demand[:, 1] / self.total_ram).tolist()
        self.reference: np.ndarray | None = None
        self.set_reference(reference)

    def set_reference(self, reference: np.ndarray | None) -> None:
        """Set the placement migrations are measured against (None disables the term)."""
        if reference is not None:
            reference = np.asarray(reference, dtype=np.int64)
            if reference.shape != (self.n,):
                raise ValueError("reference placement has wrong shape")
        self.reference = reference
        self._ref_list = None if reference is None else reference.tolist()

    def vm_load(self, v: int, s: int) -> float:
        """Load contributed by VM ``v`` to server ``s`` (mean of CPU & RAM fractions)."""
        return 0.5 * (self._dcpu[v] * self._icpu[s] + self._dram[v] * self._iram[s])

    def server_loads(self, placement: np.ndarray) -> np.ndarray:
        used = compute_usage(self.demand, placement, self.num_servers)
        frac = (used + self.background)[:, :2] / self.capacity[:, :2]
        return frac.mean(axis=1)

    def migration_raw(self, placement: np.ndarray) -> float:
        if self.reference is None:
            return 0.0
        moved = placement != self.reference
        return float(self.demand[moved, 1].sum() / self.total_ram)

    def combine(self, comm_n: float, lat_n: float, imb_n: float, mig_n: float) -> float:
        w = self.weights
        return w.alpha * comm_n + w.beta * lat_n + w.gamma * imb_n + w.delta * mig_n

    def full_cost(self, placement: np.ndarray) -> CostBreakdown:
        """Complete O(E + V) evaluation; the ground truth for incremental updates."""
        placement = np.asarray(placement, dtype=np.int64)
        if placement.shape != (self.n,) or placement.min() < 0:
            raise ValueError("full_cost requires a complete placement")
        sa, sb = placement[self.ps], placement[self.pd]
        comm = float((self.pc * self.hops[sa, sb]).sum())
        lat = float((self.pl * self.latency[sa, sb]).sum())
        imb = float(self.server_loads(placement).std())
        mig = self.migration_raw(placement)
        cn, ln, im = comm / self.comm_ref, lat / self.lat_ref, imb / IMBALANCE_REF
        return CostBreakdown(comm, lat, imb, mig, cn, ln, im, mig, self.combine(cn, ln, im, mig))


class IncrementalCost:
    """Maintains the objective for a complete placement under single-VM moves.

    ``delta_move`` touches only the moved VM's neighbours (O(degree)) and the two
    affected servers (O(1)); no full recalculation is performed.
    """

    def __init__(self, model: CostModel, placement: np.ndarray) -> None:
        self.model = model
        self.placement = np.asarray(placement, dtype=np.int64).copy()
        self.resync()

    def resync(self) -> float:
        """Recompute all running sums from scratch; returns the drift that was removed."""
        m = self.model
        old_total = getattr(self, "total", None)
        bd = m.full_cost(self.placement)
        self.loads = m.server_loads(self.placement)
        self.s1 = float(self.loads.sum())
        self.s2 = float((self.loads ** 2).sum())
        self.comm = bd.comm_raw
        self.lat = bd.latency_raw
        self.mig = bd.migration_raw
        self.imb = bd.imbalance_raw
        self.total = bd.total
        return 0.0 if old_total is None else abs(old_total - bd.total)

    def delta_move(self, v: int, b: int) -> tuple[float, tuple]:
        """Cost change if VM ``v`` moved to server ``b`` (state is not modified)."""
        m = self.model
        a = int(self.placement[v])
        if a == b:
            return 0.0, (0.0, 0.0, self.s1, self.s2, self.imb, 0.0, 0.0, 0.0)
        nb = m.nbrs[v]
        if nb.size:
            ps = self.placement[nb]
            d_comm = float(m.wc[v] @ (m.hops[b][ps] - m.hops[a][ps]))
            d_lat = float(m.wl[v] @ (m.latency[b][ps] - m.latency[a][ps]))
        else:
            d_comm = d_lat = 0.0
        dla, dlb = -m.vm_load(v, a), m.vm_load(v, b)
        la, lb = float(self.loads[a]), float(self.loads[b])
        s1 = self.s1 + dla + dlb
        s2 = self.s2 + (la + dla) ** 2 - la * la + (lb + dlb) ** 2 - lb * lb
        ns = m.num_servers
        var = s2 / ns - (s1 / ns) ** 2
        imb = var ** 0.5 if var > 0 else 0.0
        d_mig = 0.0
        if m._ref_list is not None:
            r = m._ref_list[v]
            if a == r and b != r:
                d_mig = m.mig_w[v]
            elif a != r and b == r:
                d_mig = -m.mig_w[v]
        w = m.weights
        delta = (w.alpha * d_comm / m.comm_ref + w.beta * d_lat / m.lat_ref
                 + w.gamma * (imb - self.imb) / IMBALANCE_REF + w.delta * d_mig)
        return delta, (d_comm, d_lat, s1, s2, imb, d_mig, dla, dlb)

    def apply_move(self, v: int, b: int, delta: float, parts: tuple) -> None:
        """Commit a move previously evaluated with ``delta_move``."""
        a = int(self.placement[v])
        if a == b:
            return
        d_comm, d_lat, s1, s2, imb, d_mig, dla, dlb = parts
        self.placement[v] = b
        self.loads[a] += dla
        self.loads[b] += dlb
        self.s1, self.s2, self.imb = s1, s2, imb
        self.comm += d_comm
        self.lat += d_lat
        self.mig += d_mig
        self.total += delta
