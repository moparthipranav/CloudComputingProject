"""Random, Greedy and Simulated-Annealing placement behind one interface.

All three receive the same ProblemInstance (same VMs, servers, topology, traffic)
and the same ResourceManager constraints.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil

from .greedy_placement import greedy_placement
from .problem import ProblemInstance
from .resource_manager import InfeasiblePlacementError, ResourceManager
from .simulated_annealing import SAConfig, SAResult, SimulatedAnnealing

ALGORITHMS = ("Random", "Greedy", "Simulated Annealing")


@dataclass
class PlacementResult:
    name: str
    placement: np.ndarray
    runtime_s: float            # time spent by this algorithm itself
    runtime_incl_init_s: float  # for SA: includes the greedy initial placement
    rss_delta_mb: float = 0.0   # process RSS change while running (coarse)
    extra: dict[str, Any] = field(default_factory=dict)


def random_placement(resources: ResourceManager, rng: np.random.Generator,
                     max_attempts: int = 50) -> np.ndarray:
    """Uniformly random feasible server for each VM (VMs visited in random order)."""
    for _ in range(max_attempts):
        rm = resources.copy()
        ok = True
        for v in rng.permutation(rm.num_vms):
            cand = np.flatnonzero(rm.feasible_mask(v))
            if cand.size == 0:
                ok = False
                break
            rm.assign(int(v), int(rng.choice(cand)))
        if ok:
            return rm.placement.copy()
    raise InfeasiblePlacementError("Random placement failed to find a feasible assignment")


def _timed(fn):
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0, (proc.memory_info().rss - rss0) / 1024 ** 2


def run_random(problem: ProblemInstance, seed: int) -> PlacementResult:
    rng = np.random.default_rng(seed)
    pl, t, mem = _timed(lambda: random_placement(problem.resource_manager(), rng))
    return PlacementResult("Random", pl, t, t, mem)


def run_greedy(problem: ProblemInstance) -> PlacementResult:
    cfg = problem.config.get("greedy", {})
    pl, t, mem = _timed(lambda: greedy_placement(
        problem.comm, problem.distances.hops, problem.resource_manager(), cfg))
    return PlacementResult("Greedy", pl, t, t, mem)


def run_simulated_annealing(problem: ProblemInstance, initial: np.ndarray, seed: int,
                            init_time_s: float = 0.0,
                            sa_overrides: dict[str, Any] | None = None) -> PlacementResult:
    sa_cfg = dict(problem.config["simulated_annealing"])
    if sa_overrides:
        sa_cfg.update(sa_overrides)
    cm = problem.cost_model(reference=initial)  # migrations measured vs. the greedy start

    def go() -> SAResult:
        sa = SimulatedAnnealing(cm, problem.resource_manager(), SAConfig.from_config(sa_cfg, seed))
        return sa.run(initial)

    res, t, mem = _timed(go)
    return PlacementResult("Simulated Annealing", res.best_placement, t, t + init_time_s, mem,
                           extra={"sa_result": res})


def run_all_algorithms(problem: ProblemInstance, seed: int,
                       sa_overrides: dict[str, Any] | None = None) -> dict[str, PlacementResult]:
    """Run Random, Greedy and SA on the identical instance."""
    rnd = run_random(problem, seed)
    greedy = run_greedy(problem)
    sa = run_simulated_annealing(problem, greedy.placement, seed, greedy.runtime_s, sa_overrides)
    return {r.name: r for r in (rnd, greedy, sa)}
