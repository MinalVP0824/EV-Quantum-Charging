"""Classical baselines: first-come-first-served, brute force and simulated annealing."""

import numpy as np

from .problem import Instance, QUBO


def fcfs(inst: Instance) -> dict:
    """Vehicles in arrival order grab the earliest free charger they can use.

    Ties between stations go to the one with the smaller detour, which mimics a
    driver heading to the nearest compatible charger as soon as possible.
    """
    taken = set()
    assign = {}
    order = sorted(range(len(inst.vehicles)), key=lambda v: (inst.vehicles[v].arrival, v))
    for v in order:
        veh = inst.vehicles[v]
        options = []
        for t in range(veh.arrival, min(veh.departure, inst.n_slots - 1) + 1):
            for s, st in enumerate(inst.stations):
                if veh.energy_kwh <= st.max_kw and (s, t) not in taken:
                    options.append((t, veh.detour.get(s, 0.0), s))
        if options:
            t, _, s = min(options)
            assign[v] = [(s, t)]
            taken.add((s, t))
    return assign


def brute_force(energies: np.ndarray, feasible: np.ndarray):
    """Exact optimum by enumerating every bitstring (fine up to ~22 variables)."""
    idx = np.flatnonzero(feasible)
    k = idx[np.argmin(energies[idx])]
    n = int(np.log2(len(energies)))
    return np.array([(k >> i) & 1 for i in range(n)]), float(energies[k])


def simulated_annealing(qubo: QUBO, sweeps=6000, restarts=30, seed=0):
    rng = np.random.default_rng(seed)
    Q = qubo.Q
    S = np.triu(Q, 1)
    S = S + S.T
    d = np.diag(Q)
    n = qubo.n
    best_x, best_E = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, n).astype(float)
        E = qubo.offset + d @ x + x @ np.triu(Q, 1) @ x
        for step in range(sweeps):
            T = 50.0 * (0.01 / 50.0) ** (step / sweeps)
            i = rng.integers(n)
            delta = (1 - 2 * x[i]) * (d[i] + S[i] @ x)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                x[i] = 1 - x[i]
                E += delta
                if E < best_E:
                    best_E, best_x = E, x.copy()
    return best_x.astype(int), float(best_E)
