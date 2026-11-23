# Network-Aware and Topology-Aware VM Placement with Simulated Annealing

A modular Python implementation that places VMs on physical servers so that
communicating VMs end up close together in the data-center network, while keeping
server load balanced and never violating CPU, RAM or bandwidth capacity. It compares
**Random**, **Greedy** and **Simulated Annealing (SA)** placement on identical inputs
and scales to 1,000+ VMs.

> **Honesty note.** Every number produced by this project comes from actually running
> the algorithms (`results/reports/`). No result is hard-coded. All metrics are from the
> **Python model**; CloudSim validation is provided as an export + an *untested* Java
> template (see [CloudSim integration](#cloudsim-integration)).

## 1. Problem statement

Given V VMs (CPU/RAM/bandwidth demands), S servers connected by a data-center network
(servers and switches, latency-weighted links) and a sparse VM-to-VM traffic matrix, find
a mapping `P: VM -> server` that minimises a weighted sum of communication cost, network
latency, load imbalance and migration cost, subject to hard per-server capacity limits.

## 2. Research motivation

Inter-VM traffic that crosses aggregation/core switches costs latency and consumes shared
bandwidth. Placement that ignores topology (or traffic) scatters tightly-coupled
applications across racks and pods. The problem is a variant of the NP-hard quadratic
assignment / multi-dimensional bin-packing problem, so heuristics are needed. SA is a
natural fit: moves are cheap, the cost change of a move is local (O(degree)), and
probabilistic uphill moves let it escape the local optima that a greedy pass gets stuck in.

## 3. Methodology and architecture

```
topology.py ──► shortest_path.py ──► (hops, latency matrices, cached)
vm_generator.py / server_generator.py / communication_model.py  ──► problem.py (ProblemInstance)
resource_manager.py (hard constraints)        cost_function.py (full + incremental cost)
greedy_placement.py ─► simulated_annealing.py   baseline_algorithms.py (Random/Greedy/SA)
evaluation.py ─► visualization.py ─► experiments/scalability_experiment.py ─► main.py
cloudsim_export.py ─► cloudsim/CloudSimValidator.java
```

```
vm-placement-optimization/
├── config/config.yaml        all parameters and objective weights
├── src/                      library modules (listed above, plus config.py, problem.py,
│                             cloudsim_export.py - small additions to the requested layout)
├── experiments/scalability_experiment.py
├── tests/                    40 unit tests (pytest)
├── cloudsim/CloudSimValidator.java   untested template
├── data/generated/           synthetic datasets (CSV) for 100..1000 VMs
├── data/results/             placements (CSV) and CloudSim export
├── results/graphs/           figures 01-10
├── results/reports/          tables (CSV/MD) and experiment_report.md
└── main.py                   single entry point
```

**Topology.** NetworkX graph, edge attribute `latency` (ms). Fat-tree (k-ary, k³/4 servers),
3-tier Tree, and custom (YAML: `servers`, `switches`, `links: [[u, v, latency_ms], ...]`).
Distances use **Dijkstra** (`scipy.sparse.csgraph.dijkstra`, validated against
`networkx.dijkstra_path_length` in the tests) run once from every server, cached in memory
and on disk (`data/generated/cache/*.npz`, keyed by a hash of the topology). The optimiser only
does array look-ups. Two server×server matrices are kept: `latency` (ms-weighted shortest path)
and `hops` (unit-weight shortest path).

**Traffic.** Sparse symmetric CSR: per pair a frequency (msg/s, log-normal), volume (MB/s) and
intensity = ½·f/f_max + ½·v/v_max. Synthetic traffic is organised in application groups
(85 % of pairs inside a group by default); the number of relationships is configurable
(`relationships_per_vm` or `num_relationships`).

**Resources.** Each server has CPU, RAM and NIC capacity plus pre-existing ("current")
background utilisation. VM bandwidth reservation = random base + the VM's own traffic.
Capacity is generated so aggregate demand ≈ `target_utilization` (default 60 %) of free
capacity, with heterogeneous servers. Capacity is a **hard constraint** everywhere:
`ResourceManager` raises `ConstraintViolationError` / `InfeasiblePlacementError`; nothing is
silently clipped, and SA re-verifies its final answer.

## 4. Mathematical formulation

Let `P_i` be the server of VM i, `E` the set of communicating pairs, `f_ij` frequency,
`w_ij` intensity, `H(s,t)` hop distance, `D(s,t)` Dijkstra latency (ms).

Minimise `Cost(P) = α·Ĉ_comm + β·Ĉ_lat + γ·Î + δ·M̂` subject to, for every server s and
resource r ∈ {CPU, RAM, BW}: `bg_{s,r} + Σ_{i: P_i = s} d_{i,r} ≤ cap_{s,r}`.

| Term | Raw definition | Normalisation |
|---|---|---|
| Communication | `C_comm = Σ_{(i,j)∈E} f_ij · H(P_i,P_j)` | `/ (Σ f_ij · H_max)` ∈ [0,1] |
| Latency | `C_lat = Σ_{(i,j)∈E} w_ij · D(P_i,P_j)` | `/ (Σ w_ij · D_max)` ∈ [0,1] |
| Load imbalance | `I = std_s(u_s)`, `u_s` = mean of CPU & RAM utilisation (incl. background) | `/ 0.5` (max std on [0,1]) |
| Migration | `M = Σ_{i: P_i ≠ P_ref,i} ram_i / Σ ram_i` | already in [0,1] |

Normalisers are properties of the instance (worst case: every pair at max distance), so they
never change during optimisation. Co-located VMs have distance 0. Defaults: α=0.4, β=0.2,
γ=0.3, δ=0.1 (`config.yaml → objective.weights`). Communication weights are configurable
(`comm_weight`, `latency_weight` ∈ frequency | volume | intensity); by default the
communication term uses *frequency* (as specified) and the latency term uses *intensity*
(so data volume also matters).

The **migration reference** is the placement SA starts from (the Greedy placement), so SA
pays for relocating VMs away from it. In the evaluation tables, migrations of every algorithm
are counted against the Greedy placement (Random's number therefore says "how different it is
from Greedy", not a real migration plan).

### Incremental cost

Moving VM v from server a to b changes only terms that involve v:

```
ΔC_comm = Σ_{j∈N(v)} f_vj · (H(b,P_j) − H(a,P_j))      (same form for C_lat)
ΔI      = std(new loads) − std(old loads)  via running Σu and Σu² (only servers a and b change)
ΔM      = ±ram_v/Σram  only if the move leaves/returns to the reference host
```
Cost per move is O(deg(v)); no V×V or full-edge recomputation. After each temperature level the
state is re-synchronised with the full cost and the largest drift is recorded (typically ~1e-14);
the tests compare incremental and full cost after hundreds of random moves.

## 5. Algorithms

**Random.** Random VM order, uniformly random *feasible* server per VM.

**Greedy** (also the SA start). (1) Heavy-edge union-find clustering groups VMs by
communication intensity, capped so a group roughly fits on one server; (2) groups are
processed from the heaviest traffic down, VMs by traffic strength; (3) each VM goes to the
feasible server minimising `normalised distance to already-placed partners +
load_balance_weight · utilisation_after`. Groups fill a server then spill to nearby servers.

**Simulated Annealing.**
1. T0: configured, or calibrated automatically so a median uphill move is accepted with
   probability `auto_initial_acceptance` (default 0.3).
2. Start from the Greedy placement.
3. Pick a random VM; propose a feasible server - with probability 0.7 one hosting (or near,
   by latency) a traffic-weighted random communication partner, otherwise uniformly random.
4. Compute ΔCost incrementally. Accept if ΔCost ≤ 0, else with `P = exp(−ΔCost / T)`.
5. Cool: `geometric` (T←rate·T), `linear` or `fast` (T0/(1+k)); `iterations_per_temperature`
   moves per level.
6. Track and return the best placement found. Stop at `min_temperature`, `max_iterations`, or
   when the best cost hasn't improved for `convergence_patience` levels (only checked once
   T < `convergence_min_temperature_ratio`·T0, so early heating can't trigger a false stop).
7. Seeded with `random.Random(seed)` → reproducible.

## 6. Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements.txt
```
(Tested here with Python 3.13, NumPy 2.5, SciPy 1.18, NetworkX 3.7, pandas 3.0.)

## 7. Execution

```bash
python main.py                  # everything: 100/250/500/750/1000 VMs × 3 repeats, figures, report
python main.py --quick          # fast smoke run
python main.py --vms 200 400 --repeats 2
python experiments/scalability_experiment.py     # scalability only
python -m pytest -q             # unit tests
```
Outputs: `results/graphs/01..10_*.png`, `results/reports/experiment_report.md` (+CSV/MD tables),
`data/generated/*_<n>.csv` (datasets), `data/results/placement_*.csv`, `data/results/cloudsim/`.

Figures: 01 topology · 02 initial (Greedy) placement · 03 optimised (SA) placement ·
04 communication heatmap · 05 cost comparison · 06 latency comparison · 07 CPU/RAM utilisation ·
08 SA convergence · 09 runtime vs VMs · 10 scalability comparison.

## 8. Experiment configuration

Everything lives in `config/config.yaml`: topology type/size/latencies, server and VM
generation, traffic density, objective weights, greedy and SA parameters, and the experiment
sizes/repeats. Example - denser traffic and a latency-focused objective:

```yaml
communication: {relationships_per_vm: 10}
objective: {weights: {alpha: 0.3, beta: 0.5, gamma: 0.15, delta: 0.05}}
```

## 9. Performance evaluation methodology

* Each repeat builds one instance (seeded) and runs all three algorithms on it - same VMs,
  servers, topology, traffic and constraints. Repeats use different seeds; tables report mean/std.
* Metrics: weighted objective; communication cost `Σ f·hops`; average latency (frequency-weighted
  ms); total communication distance (Σ hops over communicating pairs); share of traffic kept on one
  server; CPU/RAM utilisation (mean/max/std, incl. background); reserved-bandwidth utilisation and
  actual egress network utilisation; load imbalance; servers used; migrations; runtime; convergence
  history; constraint violations (must be 0).
* Runtime: wall-clock; `runtime_incl_init_s` adds the Greedy time to SA's.
* Memory: process RSS change per algorithm and peak RSS (coarse), plus the analytic size of the
  distance matrices and sparse traffic (`ProblemInstance.footprint_mb`). The tests also measure
  Python-heap peak with `tracemalloc` for 50-500 VMs.
* **Read the results as a trade-off, not "SA wins everything".** SA optimises the *weighted*
  objective. With the default weights it spends some raw communication cost to gain a much better
  load balance; check the per-metric columns, not only the total.

## 10. CloudSim integration

`main.py` always writes `data/results/cloudsim/`: `hosts.csv`, `vms.csv`, `traffic.csv`,
`topology_edges.csv`, `host_latency_ms.csv`, `mapping_random|greedy|simulated_annealing.csv`,
`scenario.json`. Hosts are exported with background load subtracted; 1 core = 2500 MIPS.

Strategy:
1. Add CloudSim Plus (Maven `org.cloudsimplus:cloudsim-plus`) and compile `cloudsim/CloudSimValidator.java`.
2. `java ... CloudSimValidator data/results/cloudsim` - builds the hosts, forces each VM onto its mapped
   host through a custom `VmAllocationPolicy`, runs one CPU-bound cloudlet per VM and writes
   `cloudsim_summary.csv` per mapping (VMs placed/failed, hosts used, makespan, host CPU utilisation).
3. For network latency, extend it with `NetworkHost`/`NetworkVm`/`NetworkCloudlet` (send/receive
   tasks from `traffic.csv`, switch hierarchy from `topology_edges.csv`).
4. Compare the three mappings' `cloudsim_summary.csv` rows. **Keep these separate from the Python
   numbers**: Python results are model estimates, CloudSim results are simulated.

**Limitation: the Java file has not been compiled or run** (no CloudSim jar, no network here). It uses the
CloudSim Plus 7.x/8.x API from memory; expect to adjust imports/signatures. Plain CloudSim Plus hosts have
no topology-aware inter-VM latency, so step 3 is required for network measurements.

## 11. Testing

`python -m pytest -q` runs 40 tests: Dijkstra vs NetworkX and known fat-tree distances, communication
cost vs hand calculation/brute force, normalisation bounds, incremental vs full cost, CPU/RAM/BW
constraints, validity of all three algorithms, infeasible inputs, acceptance probability, cooling
schedules, convergence/termination, seed reproducibility, growth from 50 to 1000 VMs with time and
memory measurement, and CSV/CloudSim export.

## 12. Limitations and future work

* Synthetic data only; real traces (e.g. Azure/Google) would be more convincing.
* Only single-VM moves; adding swap moves and reheating would help at very high utilisation.
* Migration is a RAM-weighted count; it ignores migration distance, downtime and link contention.
* Link-level congestion is not modelled (distance-weighted traffic is a proxy); reported egress
  utilisation is not a constraint.
* Dense S×S distance matrices are O(S²) memory; beyond ~10k servers use float32 or on-demand
  per-rack distances. The SA loop is single-threaded pure Python + NumPy.
* Objective weights are not tuned; results depend on them.
* CloudSim validation is exported but not executed here.
