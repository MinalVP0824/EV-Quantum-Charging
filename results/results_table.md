| Method              | Valid   |   Peak load (kW) |   Energy cost (Rs) |   Avg wait (slots) |   Detour (Rs) |   Total objective (Rs) | Peak vs FCFS   | Cost vs FCFS   |
|:--------------------|:--------|-----------------:|-------------------:|-------------------:|--------------:|-----------------------:|:---------------|:---------------|
| FCFS                | yes     |            72.24 |              507   |               0.25 |            20 |                 751.76 | +0.0%          | +0.0%          |
| Classical optimum   | yes     |            59.51 |              419.5 |               1    |             0 |                 667.8  | -17.6%         | -17.3%         |
| Quantum (XY-QAOA)   | yes     |            59.51 |              419.5 |               1    |             0 |                 667.8  | -17.6%         | -17.3%         |
| Quantum (X-QAOA)    | yes     |            59.51 |              419.5 |               1    |             0 |                 667.8  | -17.6%         | -17.3%         |
| Simulated annealing | yes     |            59.51 |              419.5 |               1    |             0 |                 667.8  | -17.6%         | -17.3%         |

| Solver            | mixer   |   p |   cvar_alpha |   qubits |   transpiled_depth |   cx_gates |   cost_evaluations |   runtime_s |   shots |   prob_feasible |   prob_optimal | found_optimum   |   gap_pct |   fast_sim_vs_qiskit_fidelity |   random_valid_guess_prob_optimal |
|:------------------|:--------|----:|-------------:|---------:|-------------------:|-----------:|-------------------:|------------:|--------:|----------------:|---------------:|:----------------|----------:|------------------------------:|----------------------------------:|
| Quantum (XY-QAOA) | XY      |   5 |         0.25 |       15 |                613 |        523 |               1513 |       13.2  |    4096 |          0.588  |         0.1149 | True            |         0 |                             1 |                            0.0303 |
| Quantum (X-QAOA)  | X       |   5 |         0.25 |       15 |                348 |        510 |               1581 |       21.38 |    4096 |          0.0424 |         0.0013 | True            |         0 |                             1 |                            0.0303 |
