"""Checks for the city digital twin, warm start, parameter transfer and hardware tooling.

Run with:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit.quantum_info import Statevector  # noqa: E402

from evq import city as C  # noqa: E402
from evq.hardware import analyze, prepare  # noqa: E402
from evq.problem import all_energies, bit_matrix, build_qubo, feasibility_mask, make_instance  # noqa: E402
from evq.qaoa import (initial_state_circuit, initial_state_vector, solve_qaoa,  # noqa: E402
                      warm_amplitudes)

BASE = [41.2, 41.5, 29.1]
FORECAST = {"actual": [20 + 10 * np.sin(h / 24 * 2 * np.pi) for h in range(24)],
            "pred": [21 + 10 * np.sin(h / 24 * 2 * np.pi) for h in range(24)],
            "p10": [15.0] * 24, "p90": [30.0] * 24}


def _demo():
    inst = make_instance("demo", BASE)
    q = build_qubo(inst)
    X = bit_matrix(q.n)
    return inst, q, all_energies(q.Q, q.offset, X), feasibility_mask(q, inst, X)


def test_warm_start_state_matches_qiskit():
    _, q, _, _ = _demo()
    warm = [warm_amplitudes(len(g), 0, eps=0.4) for g in q.groups]
    a = initial_state_vector(q.n, q.groups, warm)
    b = Statevector(initial_state_circuit(q.n, q.groups, warm)).data
    assert abs(np.vdot(a, b)) ** 2 > 1 - 1e-9


def test_parameter_transfer_is_cheaper_and_still_optimal():
    _, q, E, F = _demo()
    cold = solve_qaoa(q.Q_xy, q.offset_obj, E, F, p=2, groups=q.groups, backend="fast")
    tr = solve_qaoa(q.Q_xy, q.offset_obj, E, F, p=2, groups=q.groups, backend="fast", init_params=cold.params)
    assert tr.transferred and tr.n_evals < cold.n_evals / 4
    assert tr.sampled_optimal


def test_city_subproblems_fit_16_qubits_and_strategies_are_valid():
    fleet = C.generate_fleet(36, seed=3)
    loads = C.feeder_loads(FORECAST)
    for strat in ["smart", "exact", "quantum"]:
        res = C.run_rolling(fleet, loads, strat, cold_compare=0)
        k = C.evaluate(res, fleet, loads)
        assert k["served"] + k["unserved"] == len(fleet)
        # no hub ever has more cars than ports, and nobody charges outside their window
        assert (res.state.ports <= np.array([h.ports for h in C.HUBS])).all()
        for ev in fleet:
            b = res.bookings.get(ev.id)
            if b:
                assert ev.arrival <= b[1] <= ev.departure
        if strat == "quantum":
            assert max(x.qubits for x in res.qlog) <= 16


def test_coordination_beats_charging_on_arrival():
    fleet = C.generate_fleet(80, seed=11)
    loads = C.feeder_loads(FORECAST)
    fcfs = C.evaluate(C.run_fcfs(fleet, loads), fleet, loads)
    exact = C.evaluate(C.run_rolling(fleet, loads, "exact"), fleet, loads)
    assert exact["system_cost_rs"] <= fcfs["system_cost_rs"]


def test_booking_returns_a_plan_inside_the_drivers_window():
    out = C.simulate_day(n_vehicles=30, fleet_seed=5, forecast_day=FORECAST, strategies=("quantum",), cold_compare=0)
    r = C.book_driver(out, energy=20, arrival=12, departure=16, pref=0)
    assert r["ok"]
    assert 12 <= r["plan"]["hour"] <= 16


def test_hardware_analysis_on_ideal_counts():
    prep = prepare("trio", "XY", p=1)
    probs = prep["ideal_probs"]
    n = prep["qubo"].n
    counts = {format(i, f"0{n}b"): int(round(4096 * p)) for i, p in enumerate(probs) if p > 1e-4}
    res = analyze(counts, prep)
    assert abs(res["p_valid"] - res["ideal_p_valid"]) < 0.02
    assert res["found_optimum"]
