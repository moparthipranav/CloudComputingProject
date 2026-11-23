"""Export a problem instance and placements for CloudSim Plus validation.

Files written to ``out_dir`` (all plain CSV/JSON, consumed by cloudsim/CloudSimValidator.java):
  hosts.csv, vms.csv, traffic.csv, topology_edges.csv, host_latency_ms.csv,
  mapping_<algorithm>.csv (vm_id,host_id), scenario.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_algorithms import PlacementResult
from .problem import ProblemInstance

# Conversion assumptions for CloudSim (documented in the README)
MIPS_PER_CORE = 2500
HOST_STORAGE_MB = 1_000_000
VM_STORAGE_MB = 10_000


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def export_cloudsim_scenario(problem: ProblemInstance, results: dict[str, PlacementResult],
                             out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s, v = problem.servers, problem.vms
    cap, bg = s.capacity, s.background
    free = cap - bg  # the background load is modelled as reserved capacity

    pd.DataFrame({
        "host_id": np.arange(s.num_servers), "name": s.names,
        "pes": np.floor(free[:, 0]).astype(int), "mips_per_pe": MIPS_PER_CORE,
        "ram_mb": (free[:, 1] * 1024).astype(int), "bw_mbps": free[:, 2].astype(int),
        "storage_mb": HOST_STORAGE_MB}).to_csv(out / "hosts.csv", index=False)
    pd.DataFrame({
        "vm_id": np.arange(v.num_vms), "pes": v.cpu.astype(int), "mips_per_pe": MIPS_PER_CORE,
        "ram_mb": (v.ram * 1024).astype(int), "bw_mbps": np.ceil(v.bw).astype(int),
        "storage_mb": VM_STORAGE_MB}).to_csv(out / "vms.csv", index=False)
    src, dst, f = problem.comm.pair_arrays("frequency")
    vol = problem.comm.pair_arrays("volume")[2]
    pd.DataFrame({"vm_a": src, "vm_b": dst, "frequency_msgs_s": f,
                  "volume_MBps": vol}).to_csv(out / "traffic.csv", index=False)
    pd.DataFrame([(a, b, d["latency"]) for a, b, d in problem.topology.graph.edges(data=True)],
                 columns=["node_a", "node_b", "latency_ms"]
                 ).to_csv(out / "topology_edges.csv", index=False)
    np.savetxt(out / "host_latency_ms.csv", problem.distances.latency, delimiter=",", fmt="%.6f")
    for name, r in results.items():
        pd.DataFrame({"vm_id": np.arange(v.num_vms), "host_id": r.placement}
                     ).to_csv(out / f"mapping_{_slug(name)}.csv", index=False)
    (out / "scenario.json").write_text(json.dumps({
        "num_vms": problem.num_vms, "num_hosts": s.num_servers, "seed": problem.seed,
        "topology": problem.topology.name, "mips_per_core": MIPS_PER_CORE,
        "mappings": [f"mapping_{_slug(n)}.csv" for n in results],
        "note": "Optimisation metrics come from the Python model; CloudSim results are "
                "produced separately by CloudSimValidator."}, indent=2), encoding="utf-8")
    return out
