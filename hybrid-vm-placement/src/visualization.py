"""Matplotlib / Seaborn figures (saved as PNG; headless-safe)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: E402

from .baseline_algorithms import ALGORITHMS, PlacementResult  # noqa: E402
from .problem import ProblemInstance  # noqa: E402
from .resource_manager import compute_usage  # noqa: E402

# categorical slots 1-3 of the reference palette (blue, orange, aqua), fixed per algorithm
COLORS = {"Random": "#2a78d6", "Greedy": "#eb6834", "Simulated Annealing": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#fcfcfb"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    })


def _save(fig: plt.Figure, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _layout(topo) -> dict:
    g = topo.graph
    pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal")
    return pos


def plot_topology(problem: ProblemInstance, path: str | Path) -> Path:
    """1. Network topology graph (servers at the bottom, core at the top)."""
    _style()
    g = problem.topology.graph
    pos = _layout(problem.topology)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.grid(False)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#c9c8c2", width=0.5)
    kinds = nx.get_node_attributes(g, "kind")
    sw = [n for n in g if kinds[n] != "server"]
    nx.draw_networkx_nodes(g, pos, nodelist=problem.topology.server_nodes, node_size=14,
                           node_color=COLORS["Random"], ax=ax, label="Server")
    nx.draw_networkx_nodes(g, pos, nodelist=sw, node_size=38, node_color=COLORS["Greedy"],
                           node_shape="s", ax=ax, label="Switch")
    ax.set_title(f"Network topology: {problem.topology.name} "
                 f"({problem.topology.num_servers} servers, {len(sw)} switches)")
    ax.legend(loc="upper right")
    ax.axis("off")
    return _save(fig, path)


def plot_placement(problem: ProblemInstance, placement: np.ndarray, title: str,
                   path: str | Path) -> Path:
    """2/3. Placement on the topology: server colour = load, size = number of VMs."""
    _style()
    g = problem.topology.graph
    pos = _layout(problem.topology)
    s = problem.servers
    used = compute_usage(problem.vms.demand, placement, s.num_servers)
    load = (((used + s.background)[:, :2]) / s.capacity[:, :2]).mean(axis=1)
    counts = np.bincount(placement, minlength=s.num_servers)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.grid(False)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#dcdbd5", width=0.4)
    sw = [n for n in g if g.nodes[n]["kind"] != "server"]
    nx.draw_networkx_nodes(g, pos, nodelist=sw, node_size=14, node_color="#b9b8b0",
                           node_shape="s", ax=ax)
    sc = nx.draw_networkx_nodes(g, pos, nodelist=problem.topology.server_nodes,
                                node_size=20 + 14 * counts, node_color=load, cmap="Blues",
                                vmin=0, vmax=1, edgecolors="#184f95", linewidths=0.4, ax=ax)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Server load (mean of CPU & RAM utilisation)")
    ax.set_title(f"{title}  (node size = VMs hosted, {int((counts > 0).sum())} servers used)")
    ax.axis("off")
    return _save(fig, path)


def plot_comm_heatmap(problem: ProblemInstance, path: str | Path) -> Path:
    """4. Communication intensity heatmap (VMs reordered with reverse Cuthill-McKee)."""
    _style()
    m = problem.comm.csr("intensity")
    order = reverse_cuthill_mckee(m, symmetric_mode=True)
    dense = m.toarray()[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.grid(False)
    sns.heatmap(dense, ax=ax, cmap="Blues", cbar_kws={"label": "Communication intensity"},
                xticklabels=False, yticklabels=False, square=True)
    ax.set_title(f"Communication intensity ({problem.num_vms} VMs, "
                 f"{problem.comm.num_pairs} communicating pairs)")
    ax.set_xlabel("VM (reordered)")
    ax.set_ylabel("VM (reordered)")
    return _save(fig, path)


def _bar(df: pd.DataFrame, col: str, ylabel: str, title: str, path: str | Path,
         pct_vs: str | None = "Random") -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    algos = [a for a in ALGORITHMS if a in set(df["algorithm"])]
    vals = [float(df.loc[df["algorithm"] == a, col].iloc[0]) for a in algos]
    bars = ax.bar(algos, vals, color=[COLORS[a] for a in algos], width=0.55)
    ax.bar_label(bars, labels=[f"{v:.4g}" for v in vals], padding=3, color=INK, fontsize=9)
    if pct_vs in algos:
        base = vals[algos.index(pct_vs)]
        for b, a, v in zip(bars, algos, vals):
            if a != pct_vs and base:
                ax.annotate(f"{100 * (v - base) / base:+.1f}% vs {pct_vs}",
                            (b.get_x() + b.get_width() / 2, v / 2), ha="center", color="white",
                            fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=0.12)
    return _save(fig, path)


def plot_cost_reduction(df: pd.DataFrame, path: str | Path) -> Path:
    """5. Cost comparison (weighted objective)."""
    n = int(df["num_vms"].iloc[0])
    return _bar(df, "total_objective", "Total weighted cost (normalised)",
                f"Total cost by algorithm ({n} VMs, lower is better)", path)


def plot_latency(df: pd.DataFrame, path: str | Path) -> Path:
    """6. Average latency comparison."""
    n = int(df["num_vms"].iloc[0])
    return _bar(df, "avg_latency_ms", "Frequency-weighted latency (ms)",
                f"Average network latency ({n} VMs, lower is better)", path)


def plot_utilization(problem: ProblemInstance, results: dict[str, PlacementResult],
                     path: str | Path) -> Path:
    """7. Per-server CPU and RAM utilisation distribution for each algorithm."""
    _style()
    s = problem.servers
    rows = []
    for name, r in results.items():
        used = compute_usage(problem.vms.demand, r.placement, s.num_servers)
        util = (used + s.background) / s.capacity
        for i in range(s.num_servers):
            rows.append((name, "CPU", util[i, 0] * 100))
            rows.append((name, "RAM", util[i, 1] * 100))
    d = pd.DataFrame(rows, columns=["algorithm", "resource", "utilisation (%)"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, res in zip(axes, ("CPU", "RAM")):
        sub = d[d["resource"] == res]
        sns.boxplot(data=sub, x="algorithm", y="utilisation (%)", hue="algorithm",
                    order=[a for a in ALGORITHMS if a in results], palette=COLORS, ax=ax,
                    width=0.5, linewidth=1.0, fliersize=2, legend=False)
        ax.set_title(f"{res} utilisation per server")
        ax.set_xlabel("")
    axes[0].set_ylabel("Utilisation (%)")
    return _save(fig, path)


def plot_convergence(sa_result, path: str | Path) -> Path:
    """8. SA convergence: current and best cost, temperature on a secondary panel."""
    _style()
    h = pd.DataFrame(sa_result.history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    ax1.plot(h["iteration"], h["current_cost"], color="#9bd3bd", lw=1.4, label="Current cost")
    ax1.plot(h["iteration"], h["best_cost"], color=COLORS["Simulated Annealing"], lw=2.2,
             label="Best cost")
    ax1.axhline(h["best_cost"].iloc[0], color=COLORS["Greedy"], ls="--", lw=1.2,
                label="Greedy start")
    ax1.set_ylabel("Total weighted cost")
    ax1.set_title(f"Simulated Annealing convergence (stopped: {sa_result.stop_reason})")
    ax1.legend()
    ax2.plot(h["iteration"], h["temperature"], color=MUTED, lw=1.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("Temperature")
    ax2.set_xlabel("Iteration")
    return _save(fig, path)


def plot_runtime_vs_vms(summary: pd.DataFrame, path: str | Path) -> Path:
    """9. Execution time versus number of VMs."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for a in ALGORITHMS:
        d = summary[summary["algorithm"] == a].sort_values("num_vms")
        ax.errorbar(d["num_vms"], d["runtime_s_mean"], yerr=d["runtime_s_std"].fillna(0),
                    marker="o", ms=6, lw=2, color=COLORS[a], label=a, capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of VMs")
    ax.set_ylabel("Execution time (s, log scale)")
    ax.set_title("Execution time vs. problem size")
    ax.legend()
    return _save(fig, path)


def plot_scalability(summary: pd.DataFrame, path: str | Path) -> Path:
    """10. Scalability comparison: quality metrics versus number of VMs."""
    _style()
    metrics = [("total_objective", "Total weighted cost"), ("comm_cost", "Communication cost"),
               ("avg_latency_ms", "Avg latency (ms)"), ("load_imbalance", "Load imbalance (std)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, label) in zip(axes.ravel(), metrics):
        for a in ALGORITHMS:
            d = summary[summary["algorithm"] == a].sort_values("num_vms")
            m, sd = d[f"{col}_mean"].to_numpy(), d[f"{col}_std"].fillna(0).to_numpy()
            ax.plot(d["num_vms"], m, marker="o", lw=2, color=COLORS[a], label=a)
            ax.fill_between(d["num_vms"], m - sd, m + sd, color=COLORS[a], alpha=0.15, lw=0)
        ax.set_xlabel("Number of VMs")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0, 0].legend()
    return _save(fig, path)


def generate_single_scenario_plots(problem: ProblemInstance, results: dict[str, PlacementResult],
                                   df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Figures 1-8 for one scenario."""
    out = Path(out_dir)
    n = problem.num_vms
    sa = results["Simulated Annealing"].extra["sa_result"]
    return [
        plot_topology(problem, out / "01_network_topology.png"),
        plot_placement(problem, results["Greedy"].placement,
                       f"Initial placement (Greedy, {n} VMs)", out / "02_initial_placement.png"),
        plot_placement(problem, results["Simulated Annealing"].placement,
                       f"Optimised placement (Simulated Annealing, {n} VMs)",
                       out / "03_optimized_placement.png"),
        plot_comm_heatmap(problem, out / "04_communication_heatmap.png"),
        plot_cost_reduction(df, out / "05_cost_comparison.png"),
        plot_latency(df, out / "06_latency_comparison.png"),
        plot_utilization(problem, results, out / "07_cpu_ram_utilization.png"),
        plot_convergence(sa, out / "08_sa_convergence.png"),
    ]


def generate_scalability_plots(summary: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Figures 9-10 from the scalability summary table."""
    out = Path(out_dir)
    return [plot_runtime_vs_vms(summary, out / "09_runtime_vs_vms.png"),
            plot_scalability(summary, out / "10_scalability_comparison.png")]
