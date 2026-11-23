"""Metric computation, comparison tables and report helpers.

Every number is computed from an actual placement; nothing is hard-coded.
"""
from __future__ import annotations

try:
    import resource as _resource

    def peak_rss_mb() -> float:
        """Peak resident set size of this process (MB; Linux reports KiB)."""
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    import psutil

    def peak_rss_mb() -> float:
        """Peak resident set size on Windows using psutil."""
        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_algorithms import PlacementResult
from .problem import ProblemInstance
from .resource_manager import compute_usage, find_violations


def evaluate_placement(problem: ProblemInstance, placement: np.ndarray,
                       reference: np.ndarray | None, name: str = "",
                       runtime_s: float = float("nan"),
                       runtime_incl_init_s: float = float("nan")) -> dict[str, float | str | int]:
    """Compute all evaluation metrics for one placement."""
    cm = problem.cost_model(reference)
    bd = cm.full_cost(placement)
    d = problem.distances
    cap, bg, dem = problem.servers.capacity, problem.servers.background, problem.vms.demand
    n_srv = cap.shape[0]

    sa, sb = placement[cm.ps], placement[cm.pd]
    f = problem.comm.pair_arrays("frequency")[2]
    comm_cost = float((f * d.hops[sa, sb]).sum())            # sum freq * hops
    avg_latency = float((f * d.latency[sa, sb]).sum() / f.sum())  # frequency-weighted ms
    total_dist = float(d.hops[sa, sb].sum())                 # sum of hops over pairs
    colocated = float(f[sa == sb].sum() / f.sum())           # traffic kept on one server

    used = compute_usage(dem, placement, n_srv)
    util = (used + bg) / cap                                  # includes background load
    # actual egress traffic per server (Mbps): volume to partners on other servers
    vol = problem.comm.pair_arrays("volume")[2] * 8.0
    cross = sa != sb
    egress = (np.bincount(sa[cross], weights=vol[cross], minlength=n_srv)
              + np.bincount(sb[cross], weights=vol[cross], minlength=n_srv))
    net_util = egress / cap[:, 2]
    migrations = int((placement != reference).sum()) if reference is not None else 0
    violations = len(find_violations(placement, dem, cap, bg))

    return {
        "algorithm": name,
        "num_vms": problem.num_vms,
        "num_servers": n_srv,
        "total_objective": bd.total,
        "comm_cost": comm_cost,
        "comm_cost_norm": bd.comm_norm,
        "latency_cost_norm": bd.latency_norm,
        "avg_latency_ms": avg_latency,
        "total_distance_hops": total_dist,
        "colocated_traffic_share": colocated,
        "cpu_util_mean": float(util[:, 0].mean()), "cpu_util_max": float(util[:, 0].max()),
        "cpu_util_std": float(util[:, 0].std()),
        "ram_util_mean": float(util[:, 1].mean()), "ram_util_max": float(util[:, 1].max()),
        "ram_util_std": float(util[:, 1].std()),
        "bw_reserved_util_max": float(util[:, 2].max()),
        "net_bw_util_mean": float(net_util.mean()), "net_bw_util_max": float(net_util.max()),
        "load_imbalance": bd.imbalance_raw,
        "servers_used": int((np.bincount(placement, minlength=n_srv) > 0).sum()),
        "migrations": migrations,
        "migration_cost": bd.migration_raw,
        "runtime_s": runtime_s,
        "runtime_incl_init_s": runtime_incl_init_s,
        "constraint_violations": violations,
    }


def evaluate_all(problem: ProblemInstance, results: dict[str, PlacementResult],
                 reference_name: str = "Greedy") -> pd.DataFrame:
    """One row per algorithm; migrations are counted against ``reference_name``."""
    ref = results[reference_name].placement
    rows = []
    for name, r in results.items():
        row = evaluate_placement(problem, r.placement, ref, name, r.runtime_s,
                                 r.runtime_incl_init_s)
        row["rss_delta_mb"] = r.rss_delta_mb
        sa = r.extra.get("sa_result")
        if sa is not None:
            row.update(sa_iterations=sa.iterations, sa_accepted=sa.accepted,
                       sa_stop_reason=sa.stop_reason, sa_max_drift=sa.max_drift)
        rows.append(row)
    return pd.DataFrame(rows)


def improvement_table(df: pd.DataFrame,
                      columns: tuple[str, ...] = ("total_objective", "comm_cost",
                                                  "avg_latency_ms", "load_imbalance")
                      ) -> pd.DataFrame:
    """Percentage change of each algorithm vs Random and vs Greedy (negative = lower/better
    for cost-like metrics). Computed from the measured values."""
    base = df.set_index("algorithm")
    out = []
    for algo in base.index:
        row = {"algorithm": algo}
        for ref in ("Random", "Greedy"):
            if ref in base.index and algo != ref:
                for c in columns:
                    row[f"{c} vs {ref} (%)"] = 100.0 * (base.loc[algo, c] - base.loc[ref, c]) \
                        / base.loc[ref, c]
        out.append(row)
    return pd.DataFrame(out)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """Minimal markdown table writer (avoids the optional 'tabulate' dependency)."""
    cols = list(df.columns)

    def fmt(x: object) -> str:
        if isinstance(x, (float, np.floating)):
            return format(float(x), floatfmt)
        return str(x)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, out_dir: str | Path, stem: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(df_to_markdown(df) + "\n", encoding="utf-8")


def convergence_table(history: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(history)
