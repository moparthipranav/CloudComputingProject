import json

import numpy as np
import pandas as pd

from src.baseline_algorithms import run_all_algorithms
from src.cloudsim_export import export_cloudsim_scenario


def test_dataset_save_and_export(small_problem, tiny_config, tmp_path):
    p = small_problem
    p.config = tiny_config
    p.save(tmp_path / "data")
    for stem in ("vms", "servers", "communication", "topology"):
        assert (tmp_path / "data" / f"{stem}_{p.num_vms}.csv").is_file()
    res = run_all_algorithms(p, 1)
    out = export_cloudsim_scenario(p, res, tmp_path / "cs")
    assert len(pd.read_csv(out / "hosts.csv")) == p.servers.num_servers
    assert len(pd.read_csv(out / "vms.csv")) == p.num_vms
    maps = sorted(out.glob("mapping_*.csv"))
    assert len(maps) == 3
    m = pd.read_csv(maps[0])
    assert m["host_id"].between(0, p.servers.num_servers - 1).all()
    assert json.loads((out / "scenario.json").read_text())["num_vms"] == p.num_vms
    assert np.loadtxt(out / "host_latency_ms.csv", delimiter=",").shape == (
        p.servers.num_servers, p.servers.num_servers)
