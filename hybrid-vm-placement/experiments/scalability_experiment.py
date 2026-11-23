"""Scalability experiment: 100/250/500/750/1000 VMs, all three algorithms.

Run directly:  python experiments/scalability_experiment.py [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.baseline_algorithms import PlacementResult, run_all_algorithms  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation import evaluate_all, peak_rss_mb, save_table  # noqa: E402
from src.problem import ProblemInstance, build_problem  # noqa: E402

METRICS = ["total_objective", "comm_cost", "avg_latency_ms", "total_distance_hops",
           "load_imbalance", "cpu_util_max", "ram_util_max", "net_bw_util_mean", "migrations",
           "runtime_s", "runtime_incl_init_s", "servers_used", "colocated_traffic_share"]


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over repeats for every (num_vms, algorithm)."""
    g = runs.groupby(["num_vms", "algorithm"])
    out = g[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["repeats"] = g.size()
    return out.reset_index()


def run_scalability(config: dict[str, Any], vm_counts: list[int] | None = None,
                    repeats: int | None = None, data_dir: str | Path | None = None,
                    cache_dir: str | Path | None = None, verbose: bool = True,
                    sa_overrides: dict[str, Any] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Returns (per-run table, summary table, details of the largest scale / first repeat)."""
    exp = config["experiments"]
    vm_counts = vm_counts or exp["vm_counts"]
    repeats = repeats or exp["repeats"]
    base_seed = int(config.get("seed", 0))
    rows: list[pd.DataFrame] = []
    details: dict[str, Any] = {}
    for n in vm_counts:
        for rep in range(repeats):
            seed = base_seed + rep
            problem = build_problem(n, config, seed, cache_dir)
            if rep == 0 and data_dir is not None:
                problem.save(data_dir)
            t0 = time.perf_counter()
            results = run_all_algorithms(problem, seed, sa_overrides)
            df = evaluate_all(problem, results)
            df["repeat"] = rep
            df["seed"] = seed
            df["peak_rss_mb"] = peak_rss_mb()
            rows.append(df)
            if verbose:
                sa = df[df.algorithm == "Simulated Annealing"].iloc[0]
                gr = df[df.algorithm == "Greedy"].iloc[0]
                print(f"[{n:>5} VMs rep {rep}] servers={problem.servers.num_servers} "
                      f"greedy cost={gr.total_objective:.4f} SA cost={sa.total_objective:.4f} "
                      f"SA time={sa.runtime_s:.1f}s ({time.perf_counter() - t0:.1f}s total)",
                      flush=True)
            if n == max(vm_counts) and rep == 0:
                details = {"problem": problem, "results": results, "metrics": df}
    runs = pd.concat(rows, ignore_index=True)
    return runs, summarise(runs), details


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--vms", type=int, nargs="*", help="override VM counts")
    ap.add_argument("--repeats", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    runs, summary, _ = run_scalability(cfg, args.vms, args.repeats,
                                       ROOT / paths["data_generated"], ROOT / paths["cache"])
    rep = ROOT / paths["reports"]
    save_table(runs, rep, "scalability_runs")
    save_table(summary, rep, "scalability_summary")
    from src.visualization import generate_scalability_plots
    generate_scalability_plots(summary, ROOT / paths["graphs"])
    print(f"Saved results to {rep}")


if __name__ == "__main__":
    main()
