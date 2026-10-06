"""Run the full pipeline from the command line and save figures + results.

    python run_pipeline.py                     # demo instance, 15 qubits
    python run_pipeline.py --instance toy      # 4 qubits, sanity check
    python run_pipeline.py --instance large    # 19 qubits, slower
    python run_pipeline.py --p 2 --p90         # deeper circuit, risk-averse forecast
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from evq import viz
from evq.pipeline import WINDOW_HOURS, run
from evq.problem import build_qubo, make_instance
from evq.qaoa import qaoa_circuit, qubo_to_ising


def main():
    ap = argparse.ArgumentParser(description="Hybrid quantum + AI EV charging scheduler")
    ap.add_argument("--instance", default="demo", choices=["toy", "demo", "large"])
    ap.add_argument("--p", type=int, default=5, help="QAOA depth")
    ap.add_argument("--alpha", type=float, default=0.25, help="CVaR alpha (1.0 = expectation)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gamma", type=float, default=0.02, help="peak-shaving weight (Rs per kW^2)")
    ap.add_argument("--p90", action="store_true", help="plan against the P90 forecast")
    ap.add_argument("--no-x-mixer", action="store_true", help="skip the standard X-mixer QAOA run")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    print(f"Running '{a.instance}' instance, QAOA p={a.p}, CVaR alpha={a.alpha} ...")
    r = run(a.instance, p=a.p, alpha=a.alpha, shots=a.shots, seed=a.seed, use_p90=a.p90,
            gamma=a.gamma, compare_x_mixer=not a.no_x_mixer)

    inst, q = r.instance, r.qubo
    print("\n== Forecast ==")
    for k, v in r.forecast.metrics.items():
        print(f"  {k}: {v:.2f}")
    print("  base load used (kW):", [round(x, 1) for x in inst.base_load])

    h, J, _ = qubo_to_ising(q.Q, q.offset)
    print("\n== Problem ==")
    print(f"  vehicles={len(inst.vehicles)} stations={len(inst.stations)} slots={inst.n_slots}")
    print(f"  qubits={q.n}  Ising terms: {int((abs(h) > 1e-12).sum())} Z + {len(J)} ZZ")
    print(f"  penalty weight={q.penalty:.1f}  valid schedules={int(r.feasible.sum())} of {2 ** q.n}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n== Schedules ==")
    print(r.table.to_string(index=False))
    print("\n== Quantum solver ==")
    print(r.quantum_table.drop(columns=["fast_sim_vs_qiskit_fidelity"]).to_string(index=False))

    # figures
    plot_names = ["FCFS", "Classical optimum", "Quantum (XY-QAOA)"]
    viz.forecast_plot(r.forecast, WINDOW_HOURS[a.instance]).savefig(out / "01_forecast.png", bbox_inches="tight")
    viz.load_plot(inst, {k: r.metrics[k] for k in ["FCFS", "Quantum (XY-QAOA)"]}).savefig(
        out / "02_load_per_slot.png", bbox_inches="tight")
    viz.schedule_grid(inst, {k: r.assignments[k] for k in plot_names}).savefig(
        out / "03_schedules.png", bbox_inches="tight")
    viz.convergence_plot(r.qaoa).savefig(out / "04_convergence.png", bbox_inches="tight")
    viz.distribution_plot(r.qaoa["Quantum (XY-QAOA)"], r.energies, r.feasible).savefig(
        out / "05_distribution.png", bbox_inches="tight")
    if len(r.qaoa) > 1:
        viz.mixer_comparison_plot(r.quantum_table.assign(
            mixer=r.quantum_table["mixer"], prob_optimal=r.quantum_table["prob_optimal"].replace(0, float("nan"))
        )).savefig(out / "06_mixer_comparison.png", bbox_inches="tight")
    # readable circuit picture: one XY-QAOA layer on the 4-qubit toy instance
    toy = make_instance("toy", inst.base_load[:2])
    tq = build_qubo(toy)
    h, J, _ = qubo_to_ising(tq.Q_xy, tq.offset_obj)
    small, _ = qaoa_circuit(h, J, tq.n, 1, tq.groups)
    small.draw("mpl", fold=-1, scale=0.9).savefig(out / "07_circuit_one_layer_toy.png", bbox_inches="tight")

    summary = {
        "instance": a.instance,
        "forecast": r.forecast.metrics,
        "base_load_kw": inst.base_load,
        "qubits": q.n,
        "penalty": q.penalty,
        "schedules": r.table.to_dict(orient="records"),
        "quantum": r.quantum_table.to_dict(orient="records"),
        "assignments": {k: {inst.vehicles[v].name: [inst.stations[s].name, inst.slot_labels[t]]
                            for v, [(s, t)] in a_.items()}
                        for k, a_ in r.assignments.items() if all(len(x) == 1 for x in a_.values())},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "results_table.md").write_text(r.table.to_markdown(index=False) + "\n\n"
                                          + r.quantum_table.to_markdown(index=False) + "\n")
    print(f"\nFigures and results saved to ./{out}/")


if __name__ == "__main__":
    main()
