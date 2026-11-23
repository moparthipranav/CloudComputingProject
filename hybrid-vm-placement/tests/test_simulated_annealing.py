import math
import time
import tracemalloc

import numpy as np
import pytest

from src.baseline_algorithms import run_greedy, run_simulated_annealing
from src.config import deep_update
from src.problem import build_problem
from src.resource_manager import find_violations
from src.simulated_annealing import (SAConfig, SimulatedAnnealing, acceptance_probability,
                                     next_temperature)


def _run(problem, seed, **over):
    g = run_greedy(problem)
    cfg = dict(problem.config["simulated_annealing"], max_iterations=6000,
               iterations_per_temperature=300)
    cfg.update(over)
    cm = problem.cost_model(reference=g.placement)
    sa = SimulatedAnnealing(cm, problem.resource_manager(), SAConfig.from_config(cfg, seed))
    return g, sa.run(g.placement), cm


def test_acceptance_probability_formula():
    assert acceptance_probability(-1.0, 0.5) == 1.0
    assert acceptance_probability(0.0, 0.5) == 1.0
    assert acceptance_probability(2.0, 4.0) == pytest.approx(math.exp(-0.5))
    assert acceptance_probability(1.0, 0.1) < acceptance_probability(1.0, 1.0)  # hotter accepts more
    assert acceptance_probability(1.0, 0.0) == 0.0


@pytest.mark.parametrize("schedule", ["geometric", "linear", "fast"])
def test_cooling_schedules_decrease(schedule):
    t0, tmin, temps = 1.0, 1e-3, [1.0]
    for level in range(1, 20):
        temps.append(next_temperature(schedule, t0, temps[-1], 0.9, level, tmin, 40))
    assert all(b < a for a, b in zip(temps, temps[1:]))


def test_sa_improves_and_stays_feasible(small_problem, tiny_config):
    p = small_problem
    p.config = tiny_config
    g, res, cm = _run(p, 5)
    assert res.best_cost <= res.initial_cost + 1e-12
    rm = p.resource_manager()
    assert not find_violations(res.best_placement, rm.demand, rm.capacity, rm.background)
    assert res.best_cost == pytest.approx(cm.full_cost(res.best_placement).total)
    assert res.max_drift < 1e-9           # incremental cost stayed consistent with full cost
    assert res.accepted > 0 and res.iterations <= 6000


def test_reproducible_with_same_seed(small_problem, tiny_config):
    p = small_problem
    p.config = tiny_config
    _, a, _ = _run(p, 11)
    _, b, _ = _run(p, 11)
    _, c, _ = _run(p, 12)
    assert np.array_equal(a.best_placement, b.best_placement)
    assert [h["best_cost"] for h in a.history] == [h["best_cost"] for h in b.history]
    assert [h["current_cost"] for h in a.history] == [h["current_cost"] for h in b.history]
    # a different seed explores a different trajectory
    assert [h["current_cost"] for h in a.history] != [h["current_cost"] for h in c.history]


def test_problem_generation_reproducible(config):
    a, b = build_problem(40, config, 3), build_problem(40, config, 3)
    assert np.array_equal(a.vms.demand, b.vms.demand)
    assert np.array_equal(a.comm.indices, b.comm.indices)
    assert np.array_equal(a.servers.capacity, b.servers.capacity)


def test_convergence_and_temperature_stop(small_problem, tiny_config):
    p = small_problem
    p.config = tiny_config
    _, res, _ = _run(p, 1, max_iterations=200_000, convergence_patience=2,
                     convergence_min_temperature_ratio=1.0)
    assert res.stop_reason in ("converged", "min_temperature")
    assert res.iterations < 200_000
    _, res2, _ = _run(p, 1, max_iterations=200_000, convergence_patience=10_000,
                      min_temperature_ratio=0.5)
    assert res2.stop_reason == "min_temperature"


def test_explicit_temperature_config(small_problem, tiny_config):
    p = small_problem
    p.config = tiny_config
    _, res, _ = _run(p, 1, initial_temperature=0.01, min_temperature=1e-4)
    assert res.initial_temperature == 0.01


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        SAConfig(cooling_rate=1.5).validate()
    with pytest.raises(ValueError):
        SAConfig(cooling_schedule="nope").validate()


@pytest.mark.parametrize("n", [50, 100, 250, 500])
def test_increasing_vm_populations(config, n):
    """Valid result, plus measured time and (Python-heap) memory for growing sizes."""
    cfg = deep_update(config, {"simulated_annealing": {"max_iterations": 8000}})
    p = build_problem(n, cfg, 1)
    g = run_greedy(p)
    tracemalloc.start()
    t0 = time.perf_counter()
    r = run_simulated_annealing(p, g.placement, 1, g.runtime_s)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rm = p.resource_manager()
    assert not find_violations(r.placement, rm.demand, rm.capacity, rm.background)
    print(f"\n{n} VMs: {elapsed:.2f}s, peak traced memory {peak / 1e6:.1f} MB")
    assert peak < 500e6


def test_scales_to_1000_vms(config):
    cfg = deep_update(config, {"simulated_annealing": {"max_iterations": 30000}})
    p = build_problem(1000, cfg, 1)
    g = run_greedy(p)
    r = run_simulated_annealing(p, g.placement, 1, g.runtime_s)
    rm = p.resource_manager()
    assert not find_violations(r.placement, rm.demand, rm.capacity, rm.background)
    sa = r.extra["sa_result"]
    assert sa.best_cost <= sa.initial_cost and sa.max_drift < 1e-9
