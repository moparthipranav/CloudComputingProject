# VM Placement Experiment Report

All values below were produced by actual algorithm execution in this run. They are **Python-model results**, not CloudSim simulation results.

* Total wall-clock time: 25.1 s
* Seed: 42, repeats per size: 3
* Objective weights: {'alpha': 0.7, 'beta': 0.2, 'gamma': 0.08, 'delta': 0.02}
* Migrations are counted against the Greedy placement (the SA starting point).

## Detailed comparison - 1000 VMs (seed 42)

| algorithm | total_objective | comm_cost | avg_latency_ms | total_distance_hops | load_imbalance | cpu_util_mean | cpu_util_max | ram_util_mean | ram_util_max | net_bw_util_mean | net_bw_util_max | migrations | runtime_s | constraint_violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.8986 | 1.643e+06 | 0.3138 | 2.835e+04 | 0.1847 | 0.6551 | 0.9939 | 0.6554 | 0.997 | 0.4303 | 0.7444 | 990 | 0.0183 | 0 |
| Greedy | 0.189 | 3.172e+05 | 0.03606 | 9912 | 0.2471 | 0.6197 | 0.9984 | 0.6126 | 0.9945 | 0.1672 | 0.5607 | 0 | 0.05466 | 0 |
| Simulated Annealing | 0.144 | 2.392e+05 | 0.02506 | 9238 | 0.1093 | 0.6336 | 0.9209 | 0.6302 | 0.9896 | 0.1292 | 0.41 | 771 | 2.799 | 0 |

### Change relative to Random / Greedy (percent, computed from the table above)

| algorithm | total_objective vs Greedy (%) | comm_cost vs Greedy (%) | avg_latency_ms vs Greedy (%) | load_imbalance vs Greedy (%) | total_objective vs Random (%) | comm_cost vs Random (%) | avg_latency_ms vs Random (%) | load_imbalance vs Random (%) |
|---|---|---|---|---|---|---|---|---|
| Random | 375.4 | 418.1 | 770.1 | -25.24 | nan | nan | nan | nan |
| Greedy | nan | nan | nan | nan | -78.96 | -80.7 | -88.51 | 33.77 |
| Simulated Annealing | -23.8 | -24.59 | -30.5 | -55.76 | -83.97 | -85.45 | -92.01 | -40.82 |

### Simulated Annealing run statistics

* Iterations: 180000, accepted moves: 15609 (uphill accepted: 7590)
* Initial temperature 0.000673, final 6.58e-08, stop reason: min_temperature
* Cost: start 0.18903 -> best 0.14404
* Largest incremental-vs-full cost drift observed: 9.16e-16

## Scalability (mean over 3 repeats)

| num_vms | algorithm | total_objective_mean | total_objective_std | comm_cost_mean | avg_latency_ms_mean | load_imbalance_mean | runtime_s_mean |
|---|---|---|---|---|---|---|---|
| 100 | Greedy | 0.299 | 0.01191 | 4.992e+04 | 0.08283 | 0.2745 | 0.004781 |
| 100 | Random | 0.8249 | 0.02303 | 1.462e+05 | 0.2808 | 0.2196 | 0.001805 |
| 100 | Simulated Annealing | 0.1659 | 0.01017 | 2.61e+04 | 0.03061 | 0.1418 | 0.65 |
| 250 | Greedy | 0.3205 | 0.01851 | 1.415e+05 | 0.07704 | 0.2937 | 0.01218 |
| 250 | Random | 0.8673 | 0.012 | 3.972e+05 | 0.2981 | 0.1913 | 0.005353 |
| 250 | Simulated Annealing | 0.2158 | 0.003708 | 9.184e+04 | 0.04067 | 0.1638 | 0.964 |
| 500 | Greedy | 0.3116 | 0.02234 | 2.82e+05 | 0.07181 | 0.2668 | 0.02595 |
| 500 | Random | 0.8954 | 0.006925 | 8.214e+05 | 0.3106 | 0.2004 | 0.009187 |
| 500 | Simulated Annealing | 0.2287 | 0.009788 | 2.028e+05 | 0.04408 | 0.1463 | 1.256 |
| 750 | Greedy | 0.2412 | 0.009155 | 3.145e+05 | 0.04764 | 0.2787 | 0.0343 |
| 750 | Random | 0.8947 | 0.005533 | 1.233e+06 | 0.3115 | 0.1892 | 0.01277 |
| 750 | Simulated Annealing | 0.1693 | 0.01083 | 2.153e+05 | 0.02883 | 0.1287 | 1.808 |
| 1000 | Greedy | 0.2099 | 0.02809 | 3.588e+05 | 0.04308 | 0.2575 | 0.04847 |
| 1000 | Random | 0.8975 | 0.001038 | 1.659e+06 | 0.3124 | 0.1905 | 0.01785 |
| 1000 | Simulated Annealing | 0.1481 | 0.003828 | 2.502e+05 | 0.02662 | 0.1083 | 2.366 |

Figures are in `results/graphs/`; raw tables in `results/reports/*.csv`.
