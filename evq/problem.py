"""Charging-schedule problem, QUBO construction and schedule metrics.

Decision variable x[v,s,t] = 1 means vehicle v charges at station s in slot t.
Variables are only created where they make physical sense:
  * t lies between the vehicle's arrival and departure slot (deadline), and
  * the station can deliver the requested energy within one slot.

Objective (in rupees):
  energy cost    sum tariff_t * E_v * x
  grid peak      gamma * sum_t (base_t + sum_v E_v * x)^2     base_t = AI forecast
  waiting        wait_cost * (t - arrival_v) * x
  detour         detour_{v,s} * x
Penalties:
  each vehicle charges exactly once          A * (1 - sum x_v)^2
  one vehicle per charger per slot           B * sum_{pairs on same (s,t)} x_i x_j
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass
class Station:
    name: str
    max_kw: float


@dataclass
class Vehicle:
    name: str
    energy_kwh: float
    arrival: int
    departure: int
    detour: dict = field(default_factory=dict)  # station index -> rupee cost


@dataclass
class Instance:
    slot_labels: list
    tariff: list          # rupees per kWh per slot
    base_load: list       # forecast background load per slot (kW)
    stations: list
    vehicles: list
    gamma: float = 0.02   # rupees per kW^2 (peak-shaving weight)
    wait_cost: float = 15.0  # rupees per slot of waiting

    @property
    def n_slots(self):
        return len(self.slot_labels)


# ---------------------------------------------------------------- instances

def make_instance(name: str, base_load, tariff=None, gamma=0.02, wait_cost=15.0) -> Instance:
    if name == "toy":
        labels = ["18:00", "19:00"]
        stations = [Station("Hub A (22 kW)", 22)]
        vehicles = [
            Vehicle("V0 Fleet cab", 20, 0, 1),
            Vehicle("V1 Commuter", 10, 0, 1),
        ]
    else:
        labels = ["18:00", "19:00", "20:00"]
        stations = [Station("Hub A (22 kW)", 22), Station("Mall B (11 kW)", 11)]
        vehicles = [
            Vehicle("V0 Fleet cab", 20, 0, 2),
            Vehicle("V1 Commuter", 11, 0, 2, {1: 20}),
            Vehicle("V2 Delivery van", 18, 0, 1),
            Vehicle("V3 Resident", 7, 1, 2, {0: 15}),
        ]
        if name == "large":
            vehicles.append(Vehicle("V4 Night shift", 10, 1, 2, {1: 10}))
    base_load = list(base_load)[: len(labels)]
    if tariff is None:
        tariff = [9.5, 8.5, 6.0][: len(labels)]
    return Instance(labels, list(tariff), base_load, stations, vehicles, gamma, wait_cost)


# ---------------------------------------------------------------- variables

def build_variables(inst: Instance):
    var = []
    for v, veh in enumerate(inst.vehicles):
        for s, st in enumerate(inst.stations):
            if veh.energy_kwh > st.max_kw:
                continue
            for t in range(veh.arrival, min(veh.departure, inst.n_slots - 1) + 1):
                var.append((v, s, t))
    return var


@dataclass
class QUBO:
    Q: np.ndarray          # upper-triangular, diagonal = linear terms
    offset: float
    variables: list
    penalty: float
    Q_obj: np.ndarray      # objective only (no penalties)
    offset_obj: float
    Q_xy: np.ndarray       # objective + charger-conflict penalty (for the XY-mixer QAOA)
    groups: list           # variable indices per vehicle (one-hot blocks)

    @property
    def n(self):
        return len(self.variables)


def build_qubo(inst: Instance, penalty: float | None = None) -> QUBO:
    var = build_variables(inst)
    n = len(var)
    Q = np.zeros((n, n))
    g = inst.gamma

    # objective
    for i, (v, s, t) in enumerate(var):
        veh = inst.vehicles[v]
        E = veh.energy_kwh
        Q[i, i] += (
            inst.tariff[t] * E
            + g * (2 * inst.base_load[t] * E + E * E)
            + inst.wait_cost * (t - veh.arrival)
            + veh.detour.get(s, 0.0)
        )
    for i, j in combinations(range(n), 2):
        if var[i][2] == var[j][2]:
            Q[i, j] += 2 * g * inst.vehicles[var[i][0]].energy_kwh * inst.vehicles[var[j][0]].energy_kwh
    offset_obj = g * sum(b * b for b in inst.base_load)
    Q_obj = Q.copy()

    # automatic penalty: larger than the biggest saving any single violation can buy
    if penalty is None:
        marginal = np.diag(Q_obj) + (np.abs(Q_obj) + np.abs(Q_obj.T)).sum(axis=1) - 2 * np.abs(np.diag(Q_obj))
        penalty = 1.5 * float(marginal.max())

    groups = [[i for i, (vv, _, _) in enumerate(var) if vv == v] for v in range(len(inst.vehicles))]
    conflict = np.zeros_like(Q)
    for i, j in combinations(range(n), 2):
        if var[i][1:] == var[j][1:] and var[i][0] != var[j][0]:
            conflict[i, j] = penalty

    offset = offset_obj
    # exactly-once per vehicle: A(1 - sum x)^2 = A(1 - sum x + 2 sum_{i<j} x_i x_j)
    for v in range(len(inst.vehicles)):
        idx = [i for i, (vv, _, _) in enumerate(var) if vv == v]
        offset += penalty
        for i in idx:
            Q[i, i] -= penalty
        for i, j in combinations(idx, 2):
            Q[i, j] += 2 * penalty
    # one vehicle per charger per slot
    Q += conflict

    return QUBO(Q, offset, var, penalty, Q_obj, offset_obj, Q_obj + conflict, groups)


# ---------------------------------------------------------------- energies

def bit_matrix(n: int) -> np.ndarray:
    """Row k holds the bits of integer k, little-endian (column i = qubit i)."""
    k = np.arange(2 ** n, dtype=np.int64)
    return ((k[:, None] >> np.arange(n)) & 1).astype(np.float64)


def all_energies(Q: np.ndarray, offset: float, X: np.ndarray | None = None) -> np.ndarray:
    if X is None:
        X = bit_matrix(Q.shape[0])
    return offset + np.einsum("ki,ki->k", X @ Q, X)


def feasibility_mask(qubo: QUBO, inst: Instance, X: np.ndarray) -> np.ndarray:
    var = qubo.variables
    ok = np.ones(X.shape[0], dtype=bool)
    for v in range(len(inst.vehicles)):
        idx = [i for i, (vv, _, _) in enumerate(var) if vv == v]
        ok &= X[:, idx].sum(axis=1) == 1
    groups = {}
    for i, (_, s, t) in enumerate(var):
        groups.setdefault((s, t), []).append(i)
    for idx in groups.values():
        if len(idx) > 1:
            ok &= X[:, idx].sum(axis=1) <= 1
    return ok


def energy_of(Q, offset, x):
    x = np.asarray(x, dtype=float)
    return float(offset + x @ Q @ x)


# ---------------------------------------------------------------- schedules

def bits_to_assignment(bits, qubo: QUBO) -> dict:
    """Map a 0/1 vector to {vehicle: [(station, slot), ...]}."""
    out = {}
    for b, (v, s, t) in zip(bits, qubo.variables):
        if b:
            out.setdefault(v, []).append((s, t))
    return out


def assignment_to_bits(assign: dict, qubo: QUBO) -> np.ndarray:
    pos = {key: i for i, key in enumerate(qubo.variables)}
    x = np.zeros(qubo.n)
    for v, st in assign.items():
        for s, t in st:
            x[pos[(v, s, t)]] = 1
    return x


def is_feasible(assign: dict, inst: Instance) -> bool:
    used = set()
    for v in range(len(inst.vehicles)):
        if len(assign.get(v, [])) != 1:
            return False
        s, t = assign[v][0]
        if (s, t) in used:
            return False
        used.add((s, t))
    return True


def schedule_metrics(assign: dict, inst: Instance) -> dict:
    ev_load = [0.0] * inst.n_slots
    energy_cost = wait = detour = 0.0
    served = 0
    for v, veh in enumerate(inst.vehicles):
        slots = assign.get(v, [])
        if not slots:
            continue
        served += 1
        s, t = slots[0]
        ev_load[t] += veh.energy_kwh
        energy_cost += inst.tariff[t] * veh.energy_kwh
        wait += t - veh.arrival
        detour += veh.detour.get(s, 0.0)
    total_load = [b + e for b, e in zip(inst.base_load, ev_load)]
    peak_cost = inst.gamma * sum(x * x for x in total_load)
    return {
        "served": served,
        "unserved": len(inst.vehicles) - served,
        "feasible": is_feasible(assign, inst),
        "energy_cost_rs": round(energy_cost, 2),
        "peak_load_kw": round(max(total_load), 2),
        "ev_load_kw": [round(x, 2) for x in ev_load],
        "total_load_kw": [round(x, 2) for x in total_load],
        "avg_wait_slots": round(wait / max(served, 1), 3),
        "detour_rs": round(detour, 2),
        "objective_rs": round(energy_cost + peak_cost + inst.wait_cost * wait + detour, 2),
    }
