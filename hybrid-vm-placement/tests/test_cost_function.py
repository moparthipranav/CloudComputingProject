import numpy as np
import pytest

from src.communication_model import CommunicationModel, generate_communication
from src.cost_function import CostModel, CostWeights, IncrementalCost
from src.baseline_algorithms import random_placement


def brute_force(comm_pairs, hops, lat, placement):
    c = sum(f * hops[placement[i], placement[j]] for i, j, f in comm_pairs)
    l = sum(w * lat[placement[i], placement[j]] for i, j, w in comm_pairs)
    return c, l


def test_communication_cost_hand_computed():
    # 3 VMs, 2 servers 3 hops apart. pairs: (0,1) f=10, (1,2) f=5
    comm = CommunicationModel.from_pairs(3, [0, 1], [1, 2], [10, 5], [1, 1], [0.5, 0.5])
    hops = np.array([[0, 3], [3, 0]], float)
    lat = hops * 0.1
    cap = np.full((2, 3), 100.0)
    cm = CostModel(comm, hops, lat, cap, np.zeros((2, 3)), np.ones((3, 3)), CostWeights())
    bd = cm.full_cost(np.array([0, 1, 1]))   # only pair (0,1) crosses servers
    assert bd.comm_raw == pytest.approx(10 * 3)
    assert bd.latency_raw == pytest.approx(0.5 * 0.3)
    assert cm.full_cost(np.array([0, 0, 0])).comm_raw == 0


def test_full_cost_matches_brute_force(small_problem):
    p = small_problem
    cm = p.cost_model()
    pl = random_placement(p.resource_manager(), np.random.default_rng(1))
    s, d, f = p.comm.pair_arrays("frequency")
    _, _, w = p.comm.pair_arrays("intensity")
    pairs_c = list(zip(s, d, f))
    pairs_l = list(zip(s, d, w))
    c, _ = brute_force(pairs_c, p.distances.hops, p.distances.latency, pl)
    _, l = brute_force(pairs_l, p.distances.hops, p.distances.latency, pl)
    bd = cm.full_cost(pl)
    assert bd.comm_raw == pytest.approx(c) and bd.latency_raw == pytest.approx(l)


def test_normalised_components_in_unit_interval(small_problem):
    p = small_problem
    cm = p.cost_model()
    for seed in range(5):
        pl = random_placement(p.resource_manager(), np.random.default_rng(seed))
        bd = cm.full_cost(pl)
        for v in (bd.comm_norm, bd.latency_norm, bd.imbalance_norm, bd.migration_norm):
            assert 0.0 <= v <= 1.0


def test_incremental_matches_full_over_many_moves(small_problem):
    """Core requirement: incremental deltas reproduce the full cost after every move."""
    p = small_problem
    rng = np.random.default_rng(3)
    start = random_placement(p.resource_manager(), rng)
    cm = p.cost_model(reference=start)  # includes the migration term
    inc = IncrementalCost(cm, start)
    ns = p.servers.num_servers
    for step in range(800):
        v, b = int(rng.integers(p.num_vms)), int(rng.integers(ns))
        delta, parts = inc.delta_move(v, b)
        before = inc.total
        inc.apply_move(v, b, delta, parts)       # capacity is irrelevant to cost maths
        assert inc.total == pytest.approx(before + delta, abs=1e-12)
        if step % 50 == 0:
            full = cm.full_cost(inc.placement)
            assert inc.total == pytest.approx(full.total, abs=1e-9)
            assert inc.comm == pytest.approx(full.comm_raw, rel=1e-9)
            assert inc.lat == pytest.approx(full.latency_raw, rel=1e-9)
            assert inc.imb == pytest.approx(full.imbalance_raw, abs=1e-9)
            assert inc.mig == pytest.approx(full.migration_raw, abs=1e-9)
    assert inc.total == pytest.approx(cm.full_cost(inc.placement).total, abs=1e-9)


def test_delta_move_does_not_mutate_state(small_problem):
    p = small_problem
    start = random_placement(p.resource_manager(), np.random.default_rng(0))
    inc = IncrementalCost(p.cost_model(), start)
    snap, total = inc.placement.copy(), inc.total
    inc.delta_move(0, (int(start[0]) + 1) % p.servers.num_servers)
    assert np.array_equal(inc.placement, snap) and inc.total == total


def test_migration_cost(small_problem):
    p = small_problem
    ref = random_placement(p.resource_manager(), np.random.default_rng(0))
    cm = p.cost_model(reference=ref)
    assert cm.full_cost(ref).migration_raw == 0
    moved = ref.copy()
    moved[0] = (ref[0] + 1) % p.servers.num_servers
    expected = p.vms.ram[0] / p.vms.ram.sum()
    assert cm.full_cost(moved).migration_raw == pytest.approx(expected)


def test_weights_validation_and_scaling(small_problem):
    with pytest.raises(ValueError):
        CostWeights(alpha=-1)
    p = small_problem
    pl = random_placement(p.resource_manager(), np.random.default_rng(0))
    cm = p.cost_model()
    bd = cm.full_cost(pl)
    w = cm.weights
    assert bd.total == pytest.approx(w.alpha * bd.comm_norm + w.beta * bd.latency_norm
                                     + w.gamma * bd.imbalance_norm + w.delta * bd.migration_norm)


def test_traffic_generator_properties():
    rng = np.random.default_rng(5)
    comm = generate_communication(200, rng, num_relationships=700)
    assert comm.num_pairs == 700
    m = comm.csr("frequency")
    assert (m != m.T).nnz == 0                      # symmetric
    assert m.diagonal().sum() == 0                   # no self loops
    assert np.all(comm.frequency > 0) and np.all(comm.intensity <= 1.0 + 1e-12)
    # configurable number of relationships, and capped by the maximum possible
    assert generate_communication(10, rng, num_relationships=10_000).num_pairs == 45
