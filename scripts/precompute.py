"""Precompute the results the app shows on first load, so every page opens instantly.

    python scripts/precompute.py            # everything (about 2 to 3 minutes)
    python scripts/precompute.py --city     # only the city digital twin
    python scripts/precompute.py --lab      # only the Quantum Lab demo
    python scripts/precompute.py --noisy    # only the noisy-simulator hardware run

Outputs
  results/precomputed/city_day.json   full-day Bengaluru simulation, all strategies
  results/precomputed/lab_demo.json   15-qubit demo: forecast, QUBO, QAOA, depth sweep, speed-ups
  results/hardware/noisy_FakeFez.json the hardware circuits on IBM's Heron noise model
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evq.city import simulate_day  # noqa: E402
from evq.pipeline import run  # noqa: E402
from evq.problem import all_energies, bit_matrix, feasibility_mask  # noqa: E402
from evq.qaoa import solve_qaoa, warm_amplitudes  # noqa: E402
from evq.serialize import pipeline_to_dict  # noqa: E402

OUT = ROOT / "results" / "precomputed"


def _save(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"  saved {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


def city():
    t = time.time()
    print("City digital twin (120 EVs, 6 hubs, 24 h) ...")
    out = simulate_day()
    _save(out, "city_day.json")
    q = out["strategies"]["quantum"]
    print(f"  {q['qaoa']['subproblems']} QAOA sub-problems, optimum found in {q['qaoa']['found_optimum_pct']}%,"
          f" peak rise FCFS {out['strategies']['fcfs']['kpis']['peak_rise_kw']} kW ->"
          f" quantum {q['kpis']['peak_rise_kw']} kW  ({time.time() - t:.0f}s)")


def lab():
    t = time.time()
    print("Quantum Lab demo (15 qubits) ...")
    r = run("demo", p=5, compare_x_mixer=True)
    d = pipeline_to_dict(r, "demo")
    q, inst = r.qubo, r.instance
    X = bit_matrix(q.n)
    E = all_energies(q.Q, q.offset, X)
    F = feasibility_mask(q, inst, X)
    del X

    sweep = []
    for p in range(1, 7):
        res = solve_qaoa(q.Q_xy, q.offset_obj, E, F, p=p, groups=q.groups, seed=7)
        sweep.append({"p": p, "p_optimal": res.p_optimal, "p_valid": res.p_feasible,
                      "cx_gates": res.cx_count, "found": res.sampled_optimal, "evaluations": res.n_evals})
    d["depth_sweep"] = sweep
    d["random_valid_p_optimal"] = float(1 / F.sum())

    # speed-ups at p = 2: cold start, warm start from each vehicle's cheapest option, parameter transfer
    cold = solve_qaoa(q.Q_xy, q.offset_obj, E, F, p=2, groups=q.groups, seed=7, backend="fast")
    greedy = [int(np.argmin(np.diag(q.Q_xy)[g])) for g in q.groups]
    warm = solve_qaoa(q.Q_xy, q.offset_obj, E, F, p=2, groups=q.groups, seed=7, backend="fast",
                      warm=[warm_amplitudes(len(g), i, eps=0.5) for g, i in zip(q.groups, greedy)])
    city_path = OUT / "city_day.json"
    transfer_params = None
    if city_path.exists():
        transfer_params = json.loads(city_path.read_text())["strategies"]["quantum"].get("qaoa_params")
    tr = solve_qaoa(q.Q_xy, q.offset_obj, E, F, p=2, groups=q.groups, seed=7, backend="fast",
                    init_params=transfer_params or list(cold.params))
    d["speedups"] = [
        {"method": "Cold start (grid search + COBYLA)", "evaluations": cold.n_evals,
         "p_optimal": cold.p_optimal, "found": cold.sampled_optimal},
        {"method": "Warm start (biased to each driver's cheapest slot)", "evaluations": warm.n_evals,
         "p_optimal": warm.p_optimal, "found": warm.sampled_optimal},
        {"method": "Parameter transfer (angles from the city run)", "evaluations": tr.n_evals,
         "p_optimal": tr.p_optimal, "found": tr.sampled_optimal},
    ]
    _save(d, "lab_demo.json")
    print(f"  done ({time.time() - t:.0f}s)")


def noisy():
    t = time.time()
    print("Hardware circuits on the FakeFez noise model ...")
    from qiskit_ibm_runtime.fake_provider import FakeFez

    from evq.hardware import run_on_backend, save
    res = run_on_backend(FakeFez(), label="Noisy simulator (IBM Heron noise model, FakeFez)")
    print(f"  saved {save(res, 'noisy_FakeFez.json').relative_to(ROOT)} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", action="store_true")
    ap.add_argument("--lab", action="store_true")
    ap.add_argument("--noisy", action="store_true")
    a = ap.parse_args()
    everything = not (a.city or a.lab or a.noisy)
    if a.city or everything:
        city()
    if a.lab or everything:
        lab()
    if a.noisy or everything:
        noisy()
