import numpy as np
import pytest

from src.baseline_algorithms import random_placement, run_all_algorithms
from src.greedy_placement import greedy_placement
from src.resource_manager import (ConstraintViolationError, InfeasiblePlacementError,
                                  ResourceManager, find_violations)


def make_rm(cap, dem, bg=None):
    cap = np.array(cap, float)
    return ResourceManager(cap, np.zeros_like(cap) if bg is None else np.array(bg, float),
                           np.array(dem, float))


def test_cpu_ram_bw_each_enforced():
    # one server: cpu 4, ram 8, bw 100
    for demand in ([5, 1, 1], [1, 9, 1], [1, 1, 101]):
        rm = make_rm([[4, 8, 100]], [demand])
        assert not rm.fits(0, 0)
        with pytest.raises(ConstraintViolationError):
            rm.assign(0, 0)
    rm = make_rm([[4, 8, 100]], [[4, 8, 100]])
    rm.assign(0, 0)  # exactly full is allowed
    assert rm.free[0].tolist() == [0, 0, 0]


def test_background_utilisation_reduces_capacity():
    rm = make_rm([[4, 8, 100]], [[3, 1, 1]], bg=[[2, 0, 0]])
    assert not rm.fits(0, 0)


def test_move_and_release_bookkeeping():
    rm = make_rm([[4, 8, 100], [4, 8, 100]], [[3, 1, 1], [3, 1, 1]])
    rm.assign(0, 0)
    rm.assign(1, 1)
    with pytest.raises(ConstraintViolationError):
        rm.move(0, 1)               # server 1 is full on cpu
    assert rm.placement[0] == 0 and rm.free[0, 0] == 1  # failed move leaves state intact
    rm.release(1)
    rm.move(0, 1)
    assert rm.placement[0] == 1 and rm.free[0, 0] == 4


def test_find_violations_detects_overload():
    dem = np.array([[3, 1, 1], [3, 1, 1]], float)
    cap = np.array([[4, 8, 100]], float)
    v = find_violations(np.array([0, 0]), dem, cap, np.zeros_like(cap))
    assert v and v[0][1] == "cpu"
    with pytest.raises(ConstraintViolationError):
        make_rm(cap, dem).load_placement(np.array([0, 0]))


def test_all_algorithms_produce_valid_complete_placements(small_problem, tiny_config):
    p = small_problem
    p.config = tiny_config
    res = run_all_algorithms(p, seed=1)
    rm = p.resource_manager()
    for name, r in res.items():
        assert r.placement.shape == (p.num_vms,), name
        assert r.placement.min() >= 0 and r.placement.max() < p.servers.num_servers, name
        assert not find_violations(r.placement, rm.demand, rm.capacity, rm.background), name
        rm.copy().load_placement(r.placement)  # raises on any violation


def test_random_and_greedy_respect_bandwidth(small_problem):
    p = small_problem
    for pl in (random_placement(p.resource_manager(), np.random.default_rng(2)),
               greedy_placement(p.comm, p.distances.hops, p.resource_manager(), {})):
        used_bw = np.bincount(pl, weights=p.vms.bw, minlength=p.servers.num_servers)
        assert np.all(used_bw + p.servers.background[:, 2] <= p.servers.capacity[:, 2] + 1e-6)


def test_infeasible_problem_raises(small_problem):
    p = small_problem
    tiny = ResourceManager(p.servers.capacity * 0.01, p.servers.background * 0.01,
                           p.vms.demand)
    with pytest.raises(InfeasiblePlacementError):
        random_placement(tiny, np.random.default_rng(0), max_attempts=2)
    with pytest.raises(InfeasiblePlacementError):
        greedy_placement(p.comm, p.distances.hops, tiny, {})


def test_greedy_beats_random_on_communication(small_problem):
    p = small_problem
    cm = p.cost_model()
    rnd = random_placement(p.resource_manager(), np.random.default_rng(0))
    gr = greedy_placement(p.comm, p.distances.hops, p.resource_manager(), {})
    assert cm.full_cost(gr).comm_raw < cm.full_cost(rnd).comm_raw
