"""Single entry point: runs the whole pipeline.

    python main.py                    # full run: 100..1000 VMs, 3 repeats, all figures
    python main.py --quick            # small/fast smoke run (fewer VMs and iterations)
    python main.py --vms 200 400      # custom VM counts
    python main.py --config my.yaml

Pipeline: dataset generation -> Random / Greedy / Simulated Annealing on identical
instances -> evaluation tables -> figures 1-10 -> CloudSim export -> markdown report.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.scalability_experiment import run_scalability  # noqa: E402
from src.cloudsim_export import export_cloudsim_scenario  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation import df_to_markdown, improvement_table, save_table  # noqa: E402
from src.visualization import generate_scalability_plots, generate_single_scenario_plots  # noqa: E402


def write_report(path: Path, config: dict, runs, summary, details: dict, elapsed: float) -> None:
    """Markdown report containing only numbers produced by this run."""
    df = details["metrics"]
    n = int(df["num_vms"].iloc[0])
    sa = details["results"]["Simulated Annealing"].extra["sa_result"]
    cols = ["algorithm", "total_objective", "comm_cost", "avg_latency_ms",
            "total_distance_hops", "load_imbalance", "cpu_util_mean", "cpu_util_max",
            "ram_util_mean", "ram_util_max", "net_bw_util_mean", "net_bw_util_max",
            "migrations", "runtime_s", "constraint_violations"]
    sc = ["num_vms", "algorithm", "total_objective_mean", "total_objective_std",
          "comm_cost_mean", "avg_latency_ms_mean", "load_imbalance_mean", "runtime_s_mean"]
    lines = [
        "# VM Placement Experiment Report", "",
        "All values below were produced by actual algorithm execution in this run. "
        "They are **Python-model results**, not CloudSim simulation results.", "",
        f"* Total wall-clock time: {elapsed:.1f} s",
        f"* Seed: {config.get('seed')}, repeats per size: {runs['repeat'].nunique()}",
        f"* Objective weights: {config['objective']['weights']}",
        f"* Migrations are counted against the Greedy placement (the SA starting point).", "",
        f"## Detailed comparison - {n} VMs (seed {int(df['seed'].iloc[0])})", "",
        df_to_markdown(df[cols]), "",
        "### Change relative to Random / Greedy (percent, computed from the table above)", "",
        df_to_markdown(improvement_table(df)), "",
        "### Simulated Annealing run statistics", "",
        f"* Iterations: {sa.iterations}, accepted moves: {sa.accepted} "
        f"(uphill accepted: {sa.uphill_accepted})",
        f"* Initial temperature {sa.initial_temperature:.3g}, final {sa.final_temperature:.3g}, "
        f"stop reason: {sa.stop_reason}",
        f"* Cost: start {sa.initial_cost:.5f} -> best {sa.best_cost:.5f}",
        f"* Largest incremental-vs-full cost drift observed: {sa.max_drift:.2e}", "",
        f"## Scalability (mean over {runs['repeat'].nunique()} repeats)", "",
        df_to_markdown(summary[sc]), "",
        "Figures are in `results/graphs/`; raw tables in `results/reports/*.csv`.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--vms", type=int, nargs="*", help="VM counts (default from config)")
    ap.add_argument("--repeats", type=int, help="repeats per VM count (default from config)")
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    vm_counts, repeats, overrides = args.vms, args.repeats, None
    if args.quick:
        vm_counts = vm_counts or [50, 100, 200]
        repeats = repeats or 1
        overrides = {"max_iterations": 20000}

    t0 = time.perf_counter()
    np.random.seed(int(cfg.get("seed", 0)))
    runs, summary, details = run_scalability(cfg, vm_counts, repeats, paths["data_generated"],
                                             paths["cache"], True, overrides)
    save_table(runs, paths["reports"], "scalability_runs")
    save_table(summary, paths["reports"], "scalability_summary")
    save_table(details["metrics"], paths["reports"], "comparison_largest_scenario")

    problem, results = details["problem"], details["results"]
    for name, r in results.items():  # placements as CSV
        import pandas as pd
        out = paths["data_results"] / f"placement_{problem.num_vms}_{name.lower().replace(' ', '_')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"vm_id": np.arange(problem.num_vms), "server_id": r.placement}).to_csv(out, index=False)
    export_cloudsim_scenario(problem, results, paths["data_results"] / "cloudsim")

    if not args.no_plots:
        generate_single_scenario_plots(problem, results, details["metrics"], paths["graphs"])
        generate_scalability_plots(summary, paths["graphs"])
    elapsed = time.perf_counter() - t0
    paths["reports"].mkdir(parents=True, exist_ok=True)
    write_report(paths["reports"] / "experiment_report.md", cfg, runs, summary, details, elapsed)
    print(f"\nDone in {elapsed:.1f}s. Report: {paths['reports'] / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
