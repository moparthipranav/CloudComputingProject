"""Greedy, communication-aware initial placement.

1. Group VMs by communication intensity (heavy-edge union-find clustering, with a
   resource cap so a group can roughly fit on one server).
2. Visit groups from the most to the least traffic-heavy; inside a group, visit VMs
   from the highest to the lowest traffic strength.
3. Put each VM on the feasible server that minimises (normalised distance to its
   already-placed partners) + load_balance_weight * (server utilisation after).
   Partners already on the same server cost 0, so groups fill a server first and
   then spill to topologically nearby servers.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .communication_model import CommunicationModel
from .resource_manager import InfeasiblePlacementError, ResourceManager


def group_vms(comm: CommunicationModel, demand: np.ndarray, limit: np.ndarray,
              max_group_size: int, kind: str = "intensity") -> np.ndarray:
    """Heavy-edge clustering. Returns a group id per VM."""
    n = comm.num_vms
    parent = list(range(n))
    size = [1] * n
    dem = demand[:, :2].copy()  # cpu, ram drive the group size cap

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    src, dst, w = comm.pair_arrays(kind)
    for e in np.argsort(-w, kind="stable"):
        ru, rv = find(int(src[e])), find(int(dst[e]))
        if ru == rv or size[ru] + size[rv] > max_group_size:
            continue
        merged = dem[ru] + dem[rv]
        if np.any(merged > limit):
            continue
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        dem[ru] = merged
    roots = np.array([find(v) for v in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def greedy_placement(comm: CommunicationModel, hops: np.ndarray, resources: ResourceManager,
                     cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Return a feasible complete placement; raises InfeasiblePlacementError otherwise."""
    cfg = cfg or {}
    rm = resources.copy()
    n, ns = rm.num_vms, rm.num_servers
    eff = rm.capacity - rm.background
    limit = cfg.get("group_capacity_fraction", 1.0) * eff[:, :2].mean(axis=0)
    gid = group_vms(comm, rm.demand, limit, int(cfg.get("max_group_size", 64)))
    strength = comm.strength("intensity")
    n_groups = gid.max() + 1
    group_traffic = np.bincount(gid, weights=strength, minlength=n_groups)
    # order: heavy groups first, heavy VMs first inside a group
    order = np.lexsort((-strength, -group_traffic[gid], gid))
    lam = float(cfg.get("load_balance_weight", 0.05))
    hmax = float(hops.max()) or 1.0
    nbrs, wts = comm.neighbor_lists("frequency")
    base = rm.background[:, :2] / rm.capacity[:, :2]
    inv_cap = 1.0 / rm.capacity[:, :2]
    util = base.mean(axis=1)  # running mean(cpu, ram) utilisation per server

    for v in order:
        mask = rm.feasible_mask(v)
        if not mask.any():
            raise InfeasiblePlacementError(f"No server can host VM {v} (greedy placement)")
        placed = rm.placement[nbrs[v]] >= 0
        if placed.any():
            ps = rm.placement[nbrs[v][placed]]
            w = wts[v][placed]
            cost = (hops[:, ps] @ w) / (w.sum() * hmax)
        else:
            cost = np.zeros(ns)
        util_after = util + 0.5 * (rm.demand[v, :2] * inv_cap).sum(axis=1)
        score = np.where(mask, cost + lam * util_after, np.inf)
        s = int(np.argmin(score))
        rm.assign(v, s)
        util[s] = util_after[s]
    return rm.placement.copy()
