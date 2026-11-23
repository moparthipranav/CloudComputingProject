"""A complete problem instance: topology + distances + VMs + servers + traffic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .communication_model import CommunicationModel, generate_communication
from .cost_function import CostModel, CostWeights
from .resource_manager import ResourceManager
from .server_generator import ServerSet, generate_servers
from .shortest_path import DistanceMatrices, get_distance_matrices
from .topology import Topology, build_topology
from .vm_generator import VMSet, generate_vms


@dataclass
class ProblemInstance:
    num_vms: int
    seed: int
    config: dict[str, Any]
    topology: Topology
    distances: DistanceMatrices
    vms: VMSet
    servers: ServerSet
    comm: CommunicationModel

    def resource_manager(self) -> ResourceManager:
        return ResourceManager(self.servers.capacity, self.servers.background, self.vms.demand)

    def cost_model(self, reference: np.ndarray | None = None) -> CostModel:
        obj = self.config["objective"]
        return CostModel(self.comm, self.distances.hops, self.distances.latency,
                         self.servers.capacity, self.servers.background, self.vms.demand,
                         CostWeights.from_config(obj["weights"]),
                         obj.get("comm_weight", "frequency"),
                         obj.get("latency_weight", "intensity"), reference)

    def footprint_mb(self) -> dict[str, float]:
        """Analytic size of the main data structures (MB)."""
        mb = 1024 ** 2
        return {
            "distance_matrices_mb": self.distances.nbytes / mb,
            "sparse_traffic_mb": self.comm.nbytes / mb,
            "dense_traffic_equivalent_mb": self.num_vms ** 2 * 8 / mb,
        }

    def save(self, out_dir: str | Path) -> None:
        """Write the dataset (VMs, servers, traffic, topology) as CSV files."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        tag = self.num_vms
        self.vms.save_csv(d / f"vms_{tag}.csv")
        self.servers.save_csv(d / f"servers_{tag}.csv")
        s, t, f = self.comm.pair_arrays("frequency")
        _, _, v = self.comm.pair_arrays("volume")
        _, _, i = self.comm.pair_arrays("intensity")
        pd.DataFrame({"vm_a": s, "vm_b": t, "frequency_msgs_s": f, "volume_MBps": v,
                      "intensity": i}).to_csv(d / f"communication_{tag}.csv", index=False)
        pd.DataFrame([(u, w, x["latency"]) for u, w, x in self.topology.graph.edges(data=True)],
                     columns=["node_a", "node_b", "latency_ms"]
                     ).to_csv(d / f"topology_{tag}.csv", index=False)


def build_problem(num_vms: int, config: dict[str, Any], seed: int | None = None,
                  cache_dir: str | Path | None = None) -> ProblemInstance:
    """Generate a reproducible problem instance for ``num_vms`` VMs."""
    seed = int(config.get("seed", 0) if seed is None else seed)
    ss = np.random.SeedSequence([seed, num_vms])
    r_comm, r_vm, r_srv = (np.random.default_rng(s) for s in ss.spawn(3))
    c = config["communication"]
    comm = generate_communication(
        num_vms, r_comm, num_relationships=c.get("num_relationships"),
        relationships_per_vm=c.get("relationships_per_vm", 5.0),
        group_size_range=tuple(c["group_size_range"]),
        intra_group_fraction=c["intra_group_fraction"],
        freq_median_intra=c["frequency_median"]["intra"],
        freq_median_inter=c["frequency_median"]["inter"],
        freq_sigma=c["frequency_sigma"], msg_kb_median=c["message_kb_median"],
        msg_kb_sigma=c["message_kb_sigma"])
    traffic_mbps = 8.0 * np.bincount(
        np.repeat(np.arange(num_vms), np.diff(comm.indptr)), weights=comm.volume,
        minlength=num_vms)
    vms = generate_vms(num_vms, r_vm, config["vms"], traffic_mbps)
    topo = build_topology(config["topology"], num_vms)
    dist = get_distance_matrices(topo, cache_dir)
    servers = generate_servers(topo.server_nodes, vms.demand, r_srv, config["servers"])
    return ProblemInstance(num_vms, seed, config, topo, dist, vms, servers, comm)
