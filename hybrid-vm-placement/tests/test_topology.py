import networkx as nx
import numpy as np
import pytest
import yaml

from src.shortest_path import compute_distance_matrices, get_distance_matrices
from src.topology import (TopologyError, Topology, build_fat_tree, build_tree,
                          fat_tree_k_for_servers, load_custom_topology)


def test_fat_tree_counts():
    t = build_fat_tree(4)
    assert t.num_servers == 16 and len(t.switch_nodes) == 20
    assert build_fat_tree(6).num_servers == 54
    assert fat_tree_k_for_servers(100) == 8 and fat_tree_k_for_servers(16) == 4


def test_fat_tree_distances_known_values():
    t = build_fat_tree(4, 0.02, 0.05, 0.10)
    d = compute_distance_matrices(t)
    idx = t.server_index
    same_edge = d.latency[idx["srv_0_0_0"], idx["srv_0_0_1"]]
    same_pod = d.latency[idx["srv_0_0_0"], idx["srv_0_1_0"]]
    other_pod = d.latency[idx["srv_0_0_0"], idx["srv_1_0_0"]]
    assert same_edge == pytest.approx(0.04)
    assert same_pod == pytest.approx(0.14)
    assert other_pod == pytest.approx(0.34)
    assert d.hops[idx["srv_0_0_0"], idx["srv_0_0_1"]] == 2
    assert d.hops[idx["srv_0_0_0"], idx["srv_0_1_0"]] == 4
    assert d.hops[idx["srv_0_0_0"], idx["srv_1_0_0"]] == 6
    assert np.all(np.diag(d.latency) == 0)
    assert np.allclose(d.latency, d.latency.T)


def test_scipy_dijkstra_matches_networkx():
    t = build_fat_tree(4)
    d = compute_distance_matrices(t)
    rng = np.random.default_rng(0)
    for _ in range(25):
        i, j = rng.integers(0, t.num_servers, 2)
        ref = nx.dijkstra_path_length(t.graph, t.server_nodes[i], t.server_nodes[j],
                                      weight="latency")
        assert d.latency[i, j] == pytest.approx(ref)


def test_tree_topology():
    t = build_tree(40, servers_per_rack=16, racks_per_agg=2)
    assert t.num_servers == 40
    d = compute_distance_matrices(t)
    assert d.hops[0, 1] == 2          # same rack
    assert d.hops[0, 39] == 6         # different aggregation switch


def test_custom_topology_and_errors(tmp_path):
    spec = {"servers": ["a", "b", "c"], "switches": ["sw"],
            "links": [["a", "sw", 0.1], ["b", "sw", 0.2], ["c", "sw", 0.3]]}
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(spec))
    t = load_custom_topology(p)
    d = compute_distance_matrices(t)
    assert d.latency[0, 1] == pytest.approx(0.3)
    assert d.latency[1, 2] == pytest.approx(0.5)
    spec["links"] = spec["links"][:2]  # server c disconnected
    p.write_text(yaml.safe_dump(spec))
    with pytest.raises(TopologyError):
        load_custom_topology(p)


def test_invalid_fat_tree():
    with pytest.raises(TopologyError):
        build_fat_tree(5)


def test_distance_cache_roundtrip(tmp_path):
    t = build_fat_tree(4)
    d1 = get_distance_matrices(t, tmp_path)
    assert list(tmp_path.glob("dist_*.npz"))
    d2 = get_distance_matrices(t, tmp_path)  # loaded from cache
    assert np.array_equal(d1.latency, d2.latency) and np.array_equal(d1.hops, d2.hops)


def test_disconnected_graph_rejected():
    g = nx.Graph()
    g.add_edge("s1", "sw", latency=0.1)
    g.add_node("s2")
    with pytest.raises(TopologyError):
        Topology("bad", g, ["s1", "s2"], ["sw"]).validate()
