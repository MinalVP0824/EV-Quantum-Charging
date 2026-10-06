"""Quick correctness checks.  Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evq.baselines import brute_force, fcfs  # noqa: E402
from evq.problem import (all_energies, assignment_to_bits, bit_matrix, build_qubo,  # noqa: E402
                         energy_of, feasibility_mask, make_instance, schedule_metrics)
from evq.qaoa import qubo_to_ising, solve_qaoa  # noqa: E402

BASE = [41.2, 41.5, 29.1]


def _setup(name="demo"):
    inst = make_instance(name, BASE)
    q = build_qubo(inst)
    X = bit_matrix(q.n)
    E = all_energies(q.Q, q.offset, X)
    feas = feasibility_mask(q, inst, X)
    return inst, q, X, E, feas


def test_ising_matches_qubo():
    _, q, X, E, _ = _setup()
    h, J, c = qubo_to_ising(q.Q, q.offset)
    Z = 1 - 2 * X
    Ei = c + Z @ h + sum(v * Z[:, i] * Z[:, j] for (i, j), v in J.items())
    assert np.allclose(E, Ei)


def test_ground_state_is_feasible():
    _, _, _, E, feas = _setup()
    assert feas[np.argmin(E)], "penalty too small: global minimum breaks a constraint"


def test_objective_matches_metrics():
    inst, q, _, E, feas = _setup()
    bits, best = brute_force(E, feas)
    from evq.problem import bits_to_assignment
    m = schedule_metrics(bits_to_assignment(bits, q), inst)
    assert np.isclose(m["objective_rs"], best, atol=0.01)


def test_fcfs_is_valid_and_not_better_than_optimum():
    inst, q, _, E, feas = _setup()
    a = fcfs(inst)
    assert schedule_metrics(a, inst)["feasible"]
    _, best = brute_force(E, feas)
    assert energy_of(q.Q, q.offset, assignment_to_bits(a, q)) >= best - 1e-9


def test_xy_qaoa_preserves_one_hot_and_finds_optimum():
    inst, q, _, E, feas = _setup("toy")
    r = solve_qaoa(q.Q_xy, q.offset_obj, E, feas, p=1, groups=q.groups, shots=1024)
    assert r.sim_fidelity > 0.999999
    assert r.sampled_optimal
    # every state with non-zero amplitude must pick exactly one option per vehicle
    X = bit_matrix(q.n)
    one_hot = np.all([X[:, g].sum(axis=1) == 1 for g in q.groups], axis=0)
    assert r.probs[~one_hot].sum() < 1e-9
