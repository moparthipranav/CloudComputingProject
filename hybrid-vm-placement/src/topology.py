"""Data-center network topology models (Fat-tree, Tree, custom).

The network is a weighted undirected NetworkX graph. Nodes are physical servers
or switches; the edge attribute ``latency`` (milliseconds) is the edge weight.
Every node carries ``kind`` (server/edge/agg/core/tor/switch) and ``layer``
(0 = server, increasing towards the core) for drawing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml


class TopologyError(ValueError):
    """Raised for malformed or disconnected topologies."""


@dataclass
class Topology:
    name: str
    graph: nx.Graph
    server_nodes: list[str]   # order defines the server index used everywhere else
    switch_nodes: list[str]

    @property
    def num_servers(self) -> int:
        return len(self.server_nodes)

    @property
    def server_index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.server_nodes)}

    def validate(self) -> None:
        if not self.server_nodes:
            raise TopologyError("Topology has no servers")
        missing = [n for n in self.server_nodes if n not in self.graph]
        if missing:
            raise TopologyError(f"Server nodes missing from graph: {missing[:3]}")
        if not nx.is_connected(self.graph):
            raise TopologyError("Topology graph is not connected")
        for u, v, d in self.graph.edges(data=True):
            lat = d.get("latency")
            if lat is None or not lat > 0:
                raise TopologyError(f"Edge ({u},{v}) needs a positive 'latency'")


def fat_tree_k_for_servers(min_servers: int) -> int:
    """Smallest even k >= 4 whose fat-tree (k^3/4 servers) has >= ``min_servers``."""
    k = 4
    while k ** 3 // 4 < min_servers:
        k += 2
    return k


def build_fat_tree(k: int, server_edge: float = 0.02, edge_agg: float = 0.05,
                   agg_core: float = 0.10) -> Topology:
    """Standard k-ary fat-tree: k pods, (k/2)^2 core switches, k^3/4 servers."""
    if k < 2 or k % 2:
        raise TopologyError("Fat-tree k must be an even integer >= 2")
    half = k // 2
    g = nx.Graph()
    cores = [f"core_{i}_{j}" for i in range(half) for j in range(half)]
    for c in cores:
        g.add_node(c, kind="core", layer=3)
    servers: list[str] = []
    switches: list[str] = list(cores)
    for pod in range(k):
        edges = [f"edge_{pod}_{e}" for e in range(half)]
        aggs = [f"agg_{pod}_{a}" for a in range(half)]
        for n in edges:
            g.add_node(n, kind="edge", layer=1)
        for n in aggs:
            g.add_node(n, kind="agg", layer=2)
        switches += edges + aggs
        for e in edges:
            for a in aggs:
                g.add_edge(e, a, latency=edge_agg)
        for a_idx, a in enumerate(aggs):
            for j in range(half):
                g.add_edge(a, cores[a_idx * half + j], latency=agg_core)
        for e_idx, e in enumerate(edges):
            for h in range(half):
                s = f"srv_{pod}_{e_idx}_{h}"
                g.add_node(s, kind="server", layer=0)
                g.add_edge(s, e, latency=server_edge)
                servers.append(s)
    topo = Topology(f"fat_tree_k{k}", g, servers, switches)
    topo.validate()
    return topo


def build_tree(num_servers: int, servers_per_rack: int = 16, racks_per_agg: int = 4,
               server_tor: float = 0.02, tor_agg: float = 0.05,
               agg_core: float = 0.10) -> Topology:
    """Classic 3-tier tree: core -> aggregation -> top-of-rack -> servers."""
    if num_servers < 1 or servers_per_rack < 1 or racks_per_agg < 1:
        raise TopologyError("Tree parameters must be positive")
    g = nx.Graph()
    g.add_node("core_0", kind="core", layer=3)
    servers: list[str] = []
    switches = ["core_0"]
    n_racks = math.ceil(num_servers / servers_per_rack)
    n_aggs = math.ceil(n_racks / racks_per_agg)
    for a in range(n_aggs):
        g.add_node(f"agg_{a}", kind="agg", layer=2)
        g.add_edge(f"agg_{a}", "core_0", latency=agg_core)
        switches.append(f"agg_{a}")
    for r in range(n_racks):
        tor = f"tor_{r}"
        g.add_node(tor, kind="tor", layer=1)
        g.add_edge(tor, f"agg_{r // racks_per_agg}", latency=tor_agg)
        switches.append(tor)
    for i in range(num_servers):
        s = f"srv_{i}"
        g.add_node(s, kind="server", layer=0)
        g.add_edge(s, f"tor_{i // servers_per_rack}", latency=server_tor)
        servers.append(s)
    topo = Topology(f"tree_{num_servers}", g, servers, switches)
    topo.validate()
    return topo


def load_custom_topology(path: str | Path) -> Topology:
    """Load a custom topology from YAML.

    Format::
        servers: [s1, s2]
        switches: [sw1]
        links:            # [u, v, latency_ms]
          - [s1, sw1, 0.02]
    """
    p = Path(path)
    if not p.is_file():
        raise TopologyError(f"Custom topology file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    try:
        servers = [str(s) for s in spec["servers"]]
        switches = [str(s) for s in spec.get("switches", [])]
        links = spec["links"]
    except (KeyError, TypeError) as exc:
        raise TopologyError(f"Invalid custom topology file: {exc}") from exc
    g = nx.Graph()
    for s in servers:
        g.add_node(s, kind="server", layer=0)
    for s in switches:
        g.add_node(s, kind="switch", layer=1)
    for link in links:
        if len(link) != 3:
            raise TopologyError(f"Link must be [u, v, latency]: {link}")
        u, v, lat = str(link[0]), str(link[1]), float(link[2])
        if u not in g or v not in g:
            raise TopologyError(f"Link references unknown node: {link}")
        g.add_edge(u, v, latency=lat)
    topo = Topology(f"custom_{p.stem}", g, servers, switches)
    topo.validate()
    return topo


def build_topology(topo_cfg: dict[str, Any], num_vms: int) -> Topology:
    """Build the topology described by the config, sized for ``num_vms``."""
    kind = topo_cfg.get("type", "fat_tree")
    target = max(1, math.ceil(num_vms / topo_cfg.get("target_vms_per_server", 8)))
    if kind == "fat_tree":
        ft = topo_cfg.get("fat_tree", {})
        k = ft.get("k", "auto")
        k = fat_tree_k_for_servers(target) if k == "auto" else int(k)
        lat = ft.get("latency_ms", {})
        return build_fat_tree(k, **lat)
    if kind == "tree":
        tr = topo_cfg.get("tree", {})
        lat = tr.get("latency_ms", {})
        return build_tree(target, tr.get("servers_per_rack", 16),
                          tr.get("racks_per_agg", 4), **lat)
    if kind == "custom":
        path = topo_cfg.get("custom", {}).get("path")
        if not path:
            raise TopologyError("topology.custom.path must be set for custom topologies")
        return load_custom_topology(path)
    raise TopologyError(f"Unknown topology type: {kind}")
