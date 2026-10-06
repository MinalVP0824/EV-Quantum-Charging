"""End-to-end pipeline: forecast -> QUBO -> QAOA -> baselines -> metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import brute_force, fcfs, simulated_annealing
from .data import generate_site_load
from .forecast import ForecastResult, slot_loads, train_and_forecast
from .problem import (Instance, QUBO, all_energies, bit_matrix, bits_to_assignment, build_qubo,
                      feasibility_mask, make_instance, schedule_metrics)
from .qaoa import QAOAResult, solve_qaoa

WINDOW_HOURS = {"toy": [18, 19], "demo": [18, 19, 20], "large": [18, 19, 20]}


@dataclass
class PipelineOutput:
    forecast: ForecastResult
    instance: Instance
    qubo: QUBO
    energies: np.ndarray
    feasible: np.ndarray
    assignments: dict
    metrics: dict
    qaoa: dict          # name -> QAOAResult
    table: pd.DataFrame
    quantum_table: pd.DataFrame


def run(instance="demo", p=5, alpha=0.25, shots=4096, seed=7, use_p90=False,
        gamma=0.02, compare_x_mixer=True, data: pd.DataFrame | None = None,
        callback=None) -> PipelineOutput:
    # 1) AI forecast
    df = data if data is not None else generate_site_load(seed=42)
    fc = train_and_forecast(df)
    hours = WINDOW_HOURS[instance]
    base = slot_loads(fc, hours, "p90" if use_p90 else "pred")

    # 2) problem + QUBO (forecast enters the Hamiltonian through base load)
    inst = make_instance(instance, base, gamma=gamma)
    q = build_qubo(inst)
    X = bit_matrix(q.n)
    E = all_energies(q.Q, q.offset, X)
    feas = feasibility_mask(q, inst, X)
    del X

    # 3) quantum solvers
    qres = {"Quantum (XY-QAOA)": solve_qaoa(q.Q_xy, q.offset_obj, E, feas, p=p, alpha=alpha,
                                            shots=shots, seed=seed, groups=q.groups, callback=callback)}
    if compare_x_mixer:
        qres["Quantum (X-QAOA)"] = solve_qaoa(q.Q, q.offset, E, feas, p=p, alpha=alpha,
                                              shots=shots, seed=seed)

    # 4) classical baselines
    bf_bits, _ = brute_force(E, feas)
    sa_bits, _ = simulated_annealing(q, seed=seed)
    assignments = {
        "FCFS": fcfs(inst),
        "Classical optimum": bits_to_assignment(bf_bits, q),
        "Quantum (XY-QAOA)": bits_to_assignment(qres["Quantum (XY-QAOA)"].best_bits, q),
    }
    if compare_x_mixer:
        assignments["Quantum (X-QAOA)"] = bits_to_assignment(qres["Quantum (X-QAOA)"].best_bits, q)
    assignments["Simulated annealing"] = bits_to_assignment(sa_bits, q)

    metrics = {k: schedule_metrics(a, inst) for k, a in assignments.items()}

    rows = []
    fc_ref = metrics["FCFS"]
    for name, m in metrics.items():
        rows.append({
            "Method": name,
            "Valid": "yes" if m["feasible"] else "no",
            "Peak load (kW)": m["peak_load_kw"],
            "Energy cost (Rs)": m["energy_cost_rs"],
            "Avg wait (slots)": m["avg_wait_slots"],
            "Detour (Rs)": m["detour_rs"],
            "Total objective (Rs)": m["objective_rs"],
            "Peak vs FCFS": f"{100 * (m['peak_load_kw'] / fc_ref['peak_load_kw'] - 1):+.1f}%",
            "Cost vs FCFS": f"{100 * (m['energy_cost_rs'] / fc_ref['energy_cost_rs'] - 1):+.1f}%",
        })
    table = pd.DataFrame(rows)

    n_feas = int(feas.sum())
    n_opt = int((np.isclose(E, E[feas].min()) & feas).sum())
    qrows = []
    for name, r in qres.items():
        s = r.summary()
        s.pop("params")
        s["random_valid_guess_prob_optimal"] = round(n_opt / n_feas, 4)
        qrows.append({"Solver": name, **s})
    qtable = pd.DataFrame(qrows)

    return PipelineOutput(fc, inst, q, E, feas, assignments, metrics, qres, table, qtable)
