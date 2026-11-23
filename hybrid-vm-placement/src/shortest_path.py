"""All-pairs server distances via Dijkstra, computed once and cached.

Two matrices are produced (both S x S over *servers* only):
  * ``latency`` - Dijkstra shortest path using edge latency (ms) as the weight.
  * ``hops``    - Dijkstra with unit weights (number of links traversed).
They are computed once before optimisation; the optimiser only does look-ups.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .topology import Topology


@dataclass
class DistanceMatrices:
    latency: np.ndarray  # (S, S) milliseconds
    hops: np.ndarray     # (S, S) link counts

    @property
    def num_servers(self) -> int:
        return self.latency.shape[0]

    def validate(self) -> None:
        for name, m in (("latency", self.latency), ("hops", self.hops)):
            if not np.all(np.isfinite(m)):
                raise ValueError(f"{name} matrix contains unreachable pairs")
            if not np.allclose(m, m.T):
                raise ValueError(f"{name} matrix is not symmetric")
            if not np.allclose(np.diag(m), 0):
                raise ValueError(f"{name} matrix diagonal must be zero")

    @property
    def nbytes(self) -> int:
        return int(self.latency.nbytes + self.hops.nbytes)


def compute_distance_matrices(topology: Topology) -> DistanceMatrices:
    """Run Dijkstra from every server and keep the server-to-server block."""
    g = topology.graph
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    rows, cols, lat = [], [], []
    for u, v, d in g.edges(data=True):
        rows.append(idx[u]); cols.append(idx[v]); lat.append(d["latency"])
    n = len(nodes)
    lat_graph = csr_matrix((lat, (rows, cols)), shape=(n, n))
    hop_graph = csr_matrix((np.ones(len(lat)), (rows, cols)), shape=(n, n))
    src = np.array([idx[s] for s in topology.server_nodes], dtype=np.int64)
    d_lat = dijkstra(lat_graph, directed=False, indices=src)[:, src]
    d_hop = dijkstra(hop_graph, directed=False, indices=src)[:, src]
    out = DistanceMatrices(latency=np.ascontiguousarray(d_lat),
                           hops=np.ascontiguousarray(d_hop))
    out.validate()
    return out


def _topology_key(topology: Topology) -> str:
    edges = sorted((*sorted((u, v)), round(d["latency"], 9))
                   for u, v, d in topology.graph.edges(data=True))
    payload = repr((edges, topology.server_nodes)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def get_distance_matrices(topology: Topology,
                          cache_dir: str | Path | None = None) -> DistanceMatrices:
    """Return cached distances if available, otherwise compute (and cache) them."""
    if cache_dir is None:
        return compute_distance_matrices(topology)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"dist_{_topology_key(topology)}.npz"
    if path.is_file():
        try:
            with np.load(path) as z:
                dm = DistanceMatrices(latency=z["latency"], hops=z["hops"])
            if dm.num_servers == topology.num_servers:
                return dm
        except (OSError, KeyError, ValueError):
            pass  # corrupt cache: fall through and recompute
    dm = compute_distance_matrices(topology)
    np.savez_compressed(path, latency=dm.latency, hops=dm.hops)
    return dm
