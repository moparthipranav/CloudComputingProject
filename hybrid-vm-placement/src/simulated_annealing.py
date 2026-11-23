"""Simulated Annealing for network- and topology-aware VM placement.

State  : a feasible VM -> server mapping.
Move   : relocate one random VM to a feasible candidate server. Candidates are
         biased towards servers hosting (or near) the VM's communication partners,
         with the rest drawn uniformly so the whole space stays reachable.
Cost   : IncrementalCost (O(degree) per move, never a full recalculation).
Accept : improvements always; worse moves with probability exp(-delta / T).
Cool   : geometric / linear / fast schedules; stop at T_min, max_iterations or when
         the best cost has not improved for ``convergence_patience`` levels.
"""
from __future__ import annotations

import bisect
import math
import random
import time
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

import numpy as np

from .cost_function import CostModel, IncrementalCost
from .resource_manager import ResourceManager, find_violations, ConstraintViolationError


@dataclass
class SAConfig:
    initial_temperature: float | None = None   # None -> calibrate automatically
    auto_initial_acceptance: float = 0.3
    calibration_samples: int = 500
    min_temperature: float | None = None       # None -> T0 * min_temperature_ratio
    min_temperature_ratio: float = 1e-4
    cooling_schedule: str = "geometric"
    cooling_rate: float = 0.95
    iterations_per_temperature: int | None = None  # None -> max(500, num_vms)
    max_iterations: int = 400_000
    convergence_patience: int = 25
    convergence_tolerance: float = 1e-6
    convergence_min_temperature_ratio: float = 0.1  # only test convergence once T < ratio * T0
    neighbor_move_probability: float = 0.7
    nearest_servers_k: int = 12
    max_candidate_tries: int = 10
    seed: int = 0

    @classmethod
    def from_config(cls, d: dict[str, Any], seed: int = 0) -> "SAConfig":
        d = dict(d)
        t0 = d.get("initial_temperature", "auto")
        d["initial_temperature"] = None if t0 in (None, "auto") else float(t0)
        ipt = d.get("iterations_per_temperature", "auto")
        d["iterations_per_temperature"] = None if ipt in (None, "auto") else int(ipt)
        fields = cls.__dataclass_fields__
        kwargs = {k: v for k, v in d.items() if k in fields and k != "seed"}
        return cls(seed=seed, **kwargs)

    def validate(self) -> None:
        if not 0 < self.cooling_rate < 1:
            raise ValueError("cooling_rate must be in (0, 1)")
        if self.cooling_schedule not in ("geometric", "linear", "fast"):
            raise ValueError(f"Unknown cooling schedule {self.cooling_schedule!r}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.initial_temperature is not None and self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")


@dataclass
class SAResult:
    best_placement: np.ndarray
    best_cost: float
    initial_cost: float
    final_cost: float
    iterations: int
    accepted: int
    uphill_accepted: int
    no_candidate: int
    initial_temperature: float
    final_temperature: float
    stop_reason: str
    elapsed_s: float
    max_drift: float
    history: list[dict[str, float]] = field(default_factory=list)


def acceptance_probability(delta: float, temperature: float) -> float:
    """Metropolis criterion: 1 for improvements, exp(-delta / T) otherwise."""
    if delta <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta / temperature)


def next_temperature(schedule: str, t0: float, t_prev: float, rate: float, level: int,
                     t_min: float, total_levels: int) -> float:
    """Temperature for cooling level ``level`` (1-based)."""
    if schedule == "geometric":
        return t_prev * rate
    if schedule == "linear":
        return t0 - level * (t0 - t_min) / max(total_levels, 1)
    if schedule == "fast":
        return t0 / (1.0 + level)
    raise ValueError(f"Unknown cooling schedule {schedule!r}")


class SimulatedAnnealing:
    def __init__(self, cost_model: CostModel, resources: ResourceManager, config: SAConfig,
                 latency: np.ndarray | None = None) -> None:
        config.validate()
        self.cm = cost_model
        self.resources = resources
        self.cfg = config
        lat = cost_model.latency if latency is None else latency
        k = min(config.nearest_servers_k, lat.shape[0])
        self._near = np.argsort(lat, axis=1, kind="stable")[:, :k].tolist()
        # cumulative traffic weights per VM for traffic-proportional partner sampling
        self._cum = [list(accumulate(w.tolist())) for w in cost_model.wc]

    # -- candidate generation -------------------------------------------------------
    def _propose(self, v: int, rnd: random.Random, placement: np.ndarray,
                 rm: ResourceManager) -> int | None:
        cfg = self.cfg
        ns = rm.num_servers
        a = int(placement[v])
        nb = self.cm.nbrs[v]
        cum = self._cum[v]
        for _ in range(cfg.max_candidate_tries):
            if cum and rnd.random() < cfg.neighbor_move_probability:
                j = bisect.bisect_left(cum, rnd.random() * cum[-1])
                s = int(placement[nb[min(j, len(cum) - 1)]])
                b = s if rnd.random() < 0.5 else self._near[s][rnd.randrange(len(self._near[s]))]
            else:
                b = rnd.randrange(ns)
            if b != a and rm.fits(v, b):
                return b
        return None

    # -- temperature calibration ---------------------------------------------------
    def _calibrate(self, inc: IncrementalCost, rm: ResourceManager,
                   rnd: random.Random) -> float:
        ups: list[float] = []
        n = self.cm.n
        for _ in range(self.cfg.calibration_samples):
            v = rnd.randrange(n)
            b = self._propose(v, rnd, inc.placement, rm)
            if b is None:
                continue
            delta, _ = inc.delta_move(v, b)
            if delta > 0:
                ups.append(delta)
        if not ups:
            return 1e-4
        p0 = min(max(self.cfg.auto_initial_acceptance, 1e-3), 0.999)
        return float(np.median(ups)) / -math.log(p0)

    # -- main loop --------------------------------------------------------------------
    def run(self, initial_placement: np.ndarray) -> SAResult:
        cfg = self.cfg
        t_start = time.perf_counter()
        rnd = random.Random(cfg.seed)
        rm = self.resources.copy()
        rm.load_placement(np.asarray(initial_placement, dtype=np.int64))  # validates
        inc = IncrementalCost(self.cm, initial_placement)
        n = self.cm.n

        t0 = cfg.initial_temperature or self._calibrate(inc, rm, rnd)
        t_min = cfg.min_temperature if cfg.min_temperature is not None \
            else t0 * cfg.min_temperature_ratio
        if t_min >= t0:
            raise ValueError("minimum temperature must be below the initial temperature")
        ipt = cfg.iterations_per_temperature or max(500, n)
        total_levels = max(1, math.ceil(cfg.max_iterations / ipt))

        initial_cost = best_cost = inc.total
        best = inc.placement.copy()
        temp, it, accepted, uphill, no_cand, level = t0, 0, 0, 0, 0, 0
        max_drift, stale, stop = 0.0, 0, "max_iterations"
        cool_levels = 0
        history = [dict(iteration=0, temperature=t0, current_cost=inc.total,
                        best_cost=best_cost, acceptance_rate=float("nan"))]

        while it < cfg.max_iterations:
            if temp < t_min:
                stop = "min_temperature"
                break
            level += 1
            level_best_before = best_cost
            acc_level = tries_level = 0
            for _ in range(ipt):
                if it >= cfg.max_iterations:
                    break
                it += 1
                v = rnd.randrange(n)
                b = self._propose(v, rnd, inc.placement, rm)
                if b is None:
                    no_cand += 1
                    continue
                tries_level += 1
                delta, parts = inc.delta_move(v, b)
                if delta <= 0 or rnd.random() < math.exp(-delta / temp):
                    rm.move(v, b)  # raises if infeasible (it cannot be: _propose checked)
                    inc.apply_move(v, b, delta, parts)
                    accepted += 1
                    acc_level += 1
                    if delta > 0:
                        uphill += 1
                    if inc.total < best_cost - 1e-15:
                        best_cost = inc.total
                        best = inc.placement.copy()
            # remove floating point drift and record how large it was
            max_drift = max(max_drift, inc.resync())
            history.append(dict(iteration=it, temperature=temp, current_cost=inc.total,
                                best_cost=best_cost,
                                acceptance_rate=acc_level / tries_level if tries_level else 0.0))
            gain = (level_best_before - best_cost) / max(abs(level_best_before), 1e-12)
            stale = 0 if gain > cfg.convergence_tolerance else stale + 1
            if temp < cfg.convergence_min_temperature_ratio * t0:
                cool_levels += 1
                stale_cool = stale if cool_levels > 1 else 0
            else:
                stale_cool = 0
            if stale_cool >= cfg.convergence_patience:
                stop = "converged"
                break
            temp = next_temperature(cfg.cooling_schedule, t0, temp, cfg.cooling_rate, level,
                                    t_min, total_levels)

        # final safety net: the returned placement must satisfy every hard constraint
        if find_violations(best, rm.demand, rm.capacity, rm.background):
            raise ConstraintViolationError("SA produced an infeasible placement")
        final_bd = self.cm.full_cost(best)
        return SAResult(
            best_placement=best, best_cost=final_bd.total, initial_cost=initial_cost,
            final_cost=inc.total, iterations=it, accepted=accepted, uphill_accepted=uphill,
            no_candidate=no_cand, initial_temperature=t0, final_temperature=temp,
            stop_reason=stop, elapsed_s=time.perf_counter() - t_start, max_drift=max_drift,
            history=history)
