"""City-scale digital twin: a full day of EV charging across Bengaluru hubs.

How it works
  * A fleet of EVs (commuters, fleet cabs, delivery vans, residents) arrives through
    the day. Each vehicle needs a one-hour charging session before it leaves.
  * Every hour the planner re-forecasts the feeder load (AI day-ahead forecast plus an
    intraday correction from the latest measurement) and books the vehicles that just
    arrived. Bookings are final, like a real reservation system.
  * The hour's booking problem is decomposed into small sub-problems: vehicles whose
    options compete for the same charger or feeder are grouped, at most 4 vehicles
    with at most 4 options each, so every sub-problem fits in 16 qubits or fewer.
  * Each sub-problem is a QUBO solved with warm-started, constraint-preserving
    XY-QAOA. Angles are transferred from one sub-problem to the next, so training
    gets much cheaper after the first solve.

Strategies compared on the same day
  fcfs     drive to the preferred hub and charge at the first free port (today's behaviour)
  smart    uncoordinated smart charging: every driver's app picks the slot that is cheapest
           for that driver from the published prices and forecast, without seeing what the
           other drivers arriving in the same hour choose (this causes "herding")
  quantum  decomposed XY-QAOA (this project)
  exact    same decomposition solved by brute force (the classical reference for QAOA)

All locations are real Bengaluru neighbourhoods, but hub sizes, fleet and feeder loads
are illustrative. Tariffs follow the solar/non-solar time-of-day structure of India's
EV charging guidelines (0.7x during 09:00 to 16:00, 1.3x otherwise), on an illustrative
base rate. CO2 uses the CEA grid emission factor for FY 2023-24 (0.727 kg/kWh).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np

from .problem import all_energies, bit_matrix
from .qaoa import solve_qaoa, warm_amplitudes

# ------------------------------------------------------------------ constants

HOURS = 24
BASE_TARIFF = 7.0            # Rs/kWh, illustrative base rate
SOLAR_HOURS = range(9, 16)   # 09:00 to 15:59 counts as solar hours
TARIFF = [round(BASE_TARIFF * (0.7 if h in SOLAR_HOURS else 1.3), 2) for h in range(HOURS)]
GAMMA = 0.02                 # Rs per kW^2 per hour on each hub feeder (grid-stress price)
WAIT_COST = 12.0             # Rs per hour a driver waits beyond arrival
DETOUR_RS_PER_KM = 6.0       # Rs per extra km driven (charged both ways)
MAX_DETOUR_KM = 9.0
EMISSION_KG_PER_KWH = 0.727  # CEA CO2 baseline database, FY 2023-24 weighted average
MAX_GROUP = 4
MAX_OPTIONS = 4


@dataclass
class Hub:
    name: str
    lat: float
    lon: float
    ports: int
    kw: float
    feeder_scale: float      # multiplies the AI forecast profile to give this feeder's base load


HUBS = [
    Hub("Koramangala", 12.9352, 77.6245, 4, 60, 6.5),
    Hub("Indiranagar", 12.9784, 77.6408, 3, 30, 5.0),
    Hub("Whitefield", 12.9698, 77.7500, 4, 60, 7.0),
    Hub("Electronic City", 12.8452, 77.6602, 4, 60, 6.0),
    Hub("Hebbal", 13.0358, 77.5970, 3, 60, 5.5),
    Hub("Jayanagar", 12.9250, 77.5938, 3, 30, 5.0),
]

SEGMENTS = {
    # name: (share, arrival range, dwell range in hours, energy range kWh, preferred hubs)
    "Commuter":     (0.32, (8, 10), (7, 10), (14, 30), [0, 2, 3]),
    "Fleet cab":    (0.26, (7, 20), (1, 3), (14, 28), [0, 1, 2, 3, 4, 5]),
    "Delivery van": (0.14, (10, 15), (2, 4), (24, 40), [2, 3, 4]),
    "Resident":     (0.28, (17, 21), (3, 7), (10, 28), [0, 1, 4, 5]),
}


def haversine_km(a: Hub, b: Hub) -> float:
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


DIST = np.array([[haversine_km(a, b) for b in HUBS] for a in HUBS])


@dataclass
class EV:
    id: int
    segment: str
    energy: float
    arrival: int
    departure: int
    pref: int
    user: bool = False

    @property
    def label(self):
        return "You" if self.user else f"EV{self.id:03d}"


# ------------------------------------------------------------------ inputs

def generate_fleet(n: int = 120, seed: int = 11) -> list[EV]:
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s][0] for s in names])
    seg_idx = rng.choice(len(names), size=n, p=shares / shares.sum())
    fleet = []
    for i, si in enumerate(seg_idx):
        seg = names[si]
        _, (a0, a1), (d0, d1), (e0, e1), prefs = SEGMENTS[seg]
        arr = int(rng.integers(a0, a1 + 1))
        dep = int(min(HOURS - 1, arr + rng.integers(d0, d1 + 1)))
        energy = float(round(rng.uniform(e0, e1)))
        pref = int(rng.choice(prefs))
        if energy > HUBS[pref].kw:          # nudge to a hub that can deliver it
            pref = int(rng.choice([j for j in prefs if HUBS[j].kw >= energy] or [0]))
        fleet.append(EV(i, seg, energy, arr, dep, pref))
    fleet.sort(key=lambda e: (e.arrival, e.id))
    return fleet


def feeder_loads(forecast_day) -> dict:
    """Per-hub base feeder load (kW): actual, day-ahead forecast, P90 (24 x n_hubs)."""
    scale = np.array([h.feeder_scale for h in HUBS])
    act = np.outer(np.asarray(forecast_day["actual"], dtype=float), scale)
    pred = np.outer(np.asarray(forecast_day["pred"], dtype=float), scale)
    p90 = np.outer(np.asarray(forecast_day["p90"], dtype=float), scale)
    return {"actual": act, "pred": pred, "p90": p90}


def planning_forecast(loads: dict, hour: int, decay: float = 0.6, use_p90: bool = False) -> np.ndarray:
    """Day-ahead forecast corrected with the last measured error (persistence decay)."""
    base = (loads["p90"] if use_p90 else loads["pred"]).copy()
    if hour > 0:
        err = loads["actual"][hour - 1] - loads["pred"][hour - 1]
        for t in range(hour, HOURS):
            base[t] += err * decay ** (t - hour + 1)
    return base


# ------------------------------------------------------------------ state

@dataclass
class GridState:
    ev_kw: np.ndarray          # HOURS x hubs, EV load already booked
    ports: np.ndarray          # HOURS x hubs, ports already booked

    @classmethod
    def empty(cls):
        return cls(np.zeros((HOURS, len(HUBS))), np.zeros((HOURS, len(HUBS)), dtype=int))

    def copy(self):
        return GridState(self.ev_kw.copy(), self.ports.copy())

    def book(self, ev: EV, s: int, t: int):
        self.ev_kw[t, s] += ev.energy
        self.ports[t, s] += 1


def allowed_hubs(ev: EV) -> list[int]:
    return [s for s, h in enumerate(HUBS) if h.kw >= ev.energy and DIST[ev.pref, s] <= MAX_DETOUR_KM]


def option_cost(ev: EV, s: int, t: int, base: np.ndarray, state: GridState, grid: bool = True) -> float:
    """Cost of one booking. grid=False is what a driver's own app sees (no feeder stress)."""
    load = base[t, s] + state.ev_kw[t, s]
    return (TARIFF[t] * ev.energy
            + (GAMMA * ((load + ev.energy) ** 2 - load ** 2) if grid else 0.0)
            + WAIT_COST * (t - ev.arrival)
            + 2 * DETOUR_RS_PER_KM * DIST[ev.pref, s])


def candidate_options(ev: EV, base, state, k=MAX_OPTIONS, grid=True):
    opts = []
    for s in allowed_hubs(ev):
        for t in range(ev.arrival, ev.departure + 1):
            if state.ports[t, s] < HUBS[s].ports:
                opts.append((option_cost(ev, s, t, base, state, grid), s, t))
    opts.sort()
    return [(s, t) for _, s, t in opts[:k]]


# ------------------------------------------------------------------ sub-problem

@dataclass
class SubProblem:
    vehicles: list
    options: list               # per vehicle: list of (s, t)
    variables: list             # (vehicle index in group, s, t)
    groups: list                # qubit indices per vehicle
    Q: np.ndarray
    offset: float
    penalty: float


def build_subproblem(evs, base, state) -> SubProblem | None:
    options = [candidate_options(ev, base, state) for ev in evs]
    keep = [i for i, o in enumerate(options) if o]
    evs = [evs[i] for i in keep]
    options = [options[i] for i in keep]
    if not evs:
        return None
    # capacity pruning: an (s, t) slot may appear for at most `remaining ports` vehicles
    # (or for any number when only one port is left, which becomes a pairwise penalty)
    for s_t in {o for opts in options for o in opts}:
        s, t = s_t
        remaining = HUBS[s].ports - state.ports[t, s]
        holders = [i for i, opts in enumerate(options) if s_t in opts]
        if remaining >= 2 and len(holders) > remaining:
            ranked = sorted(holders, key=lambda i: options[i].index(s_t))
            for i in ranked[remaining:]:
                if len(options[i]) > 1:
                    options[i].remove(s_t)

    variables, groups = [], []
    for vi, opts in enumerate(options):
        groups.append(list(range(len(variables), len(variables) + len(opts))))
        variables += [(vi, s, t) for s, t in opts]
    n = len(variables)
    Q = np.zeros((n, n))
    for i, (vi, s, t) in enumerate(variables):
        Q[i, i] = option_cost(evs[vi], s, t, base, state)
    couplings = 0.0
    for i, j in combinations(range(n), 2):
        vi, si, ti = variables[i]
        vj, sj, tj = variables[j]
        if vi != vj and (si, ti) == (sj, tj):
            c = 2 * GAMMA * evs[vi].energy * evs[vj].energy     # exact feeder-peak coupling
            Q[i, j] += c
            couplings += c
    diag = np.diag(Q)
    penalty = 2 * (diag.max() - diag.min()) + 2 * couplings + 100.0
    for i, j in combinations(range(n), 2):
        vi, si, ti = variables[i]
        vj, sj, tj = variables[j]
        if vi != vj and (si, ti) == (sj, tj) and HUBS[si].ports - state.ports[ti, si] == 1:
            Q[i, j] += penalty                                   # only one port left
    return SubProblem(evs, options, variables, groups, Q, 0.0, penalty)


def subproblem_vectors(sp: SubProblem):
    n = len(sp.variables)
    X = bit_matrix(n)
    E = all_energies(sp.Q, sp.offset, X)
    feas = np.ones(len(E), dtype=bool)
    for g in sp.groups:
        feas &= X[:, g].sum(axis=1) == 1
    return X, E, feas


def decode(sp: SubProblem, bits) -> list:
    out = []
    for vi, g in enumerate(sp.groups):
        chosen = [k for k in g if bits[k]]
        _, s, t = sp.variables[chosen[0]]
        out.append((sp.vehicles[vi], s, t))
    return out


def exact_assignment(sp: SubProblem, X, E, feas):
    k = int(np.flatnonzero(feas)[np.argmin(E[feas])])
    return X[k].astype(int), float(E[k])


# ------------------------------------------------------------------ grouping

def make_groups(evs, base, state):
    """Group vehicles whose candidate options overlap (same hub and hour), <= MAX_GROUP."""
    cands = {ev.id: set(candidate_options(ev, base, state)) for ev in evs}
    remaining = list(evs)
    groups = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        pool = set(cands[seed.id])
        changed = True
        while changed and len(group) < MAX_GROUP:
            changed = False
            best = max(remaining, key=lambda e: len(pool & cands[e.id]), default=None)
            if best is not None and pool & cands[best.id]:
                group.append(best)
                remaining.remove(best)
                pool |= cands[best.id]
                changed = True
        groups.append(group)
    return groups


# ------------------------------------------------------------------ strategies

@dataclass
class QuantumLog:
    hour: int
    vehicles: int
    qubits: int
    evaluations: int
    transferred: bool
    p_optimal: float
    p_valid: float
    found_optimum: bool
    runtime_s: float
    gap_pct: float = 0.0
    cold_evaluations: int | None = None
    cold_p_optimal: float | None = None


@dataclass
class DayResult:
    strategy: str
    bookings: dict = field(default_factory=dict)       # ev id -> (s, t) or None
    state: GridState | None = None
    snapshots: list = field(default_factory=list)      # GridState before each hour's bookings
    qlog: list = field(default_factory=list)
    params: list | None = None                         # last trained QAOA angles
    runtime_s: float = 0.0


def run_fcfs(fleet, loads) -> DayResult:
    state = GridState.empty()
    res = DayResult("fcfs")
    t0 = time.time()
    for ev in fleet:
        hubs = sorted(allowed_hubs(ev), key=lambda s: DIST[ev.pref, s])
        choice = None
        for t in range(ev.arrival, ev.departure + 1):
            for s in hubs:
                if state.ports[t, s] < HUBS[s].ports:
                    choice = (s, t)
                    break
            if choice:
                break
        res.bookings[ev.id] = choice
        if choice:
            state.book(ev, *choice)
    res.state, res.runtime_s = state, time.time() - t0
    return res


def run_rolling(fleet, loads, strategy="quantum", p=2, alpha=0.25, shots=1024, seed=7,
                use_p90=False, cold_compare=0, progress=None) -> DayResult:
    """Hour-by-hour booking with 'smart', 'exact' or 'quantum' sub-problem solvers."""
    state = GridState.empty()
    res = DayResult(strategy)
    t0 = time.time()
    params = None
    by_hour = {}
    for ev in fleet:
        by_hour.setdefault(ev.arrival, []).append(ev)
    for hour in range(HOURS):
        res.snapshots.append(state.copy())
        arrivals = by_hour.get(hour, [])
        if not arrivals:
            continue
        base = planning_forecast(loads, hour, use_p90=use_p90)
        if strategy == "smart":
            seen = state.copy()                        # what every app sees at the top of the hour
            for ev in arrivals:
                choice = None
                for s, t in candidate_options(ev, base, seen, k=10 ** 6, grid=False):
                    if state.ports[t, s] < HUBS[s].ports:    # the charger itself must still be free
                        choice = (s, t)
                        break
                res.bookings[ev.id] = choice
                if choice:
                    state.book(ev, *choice)
            continue
        for group in make_groups(arrivals, base, state):
            sp = build_subproblem(group, base, state)
            for ev in group:
                res.bookings.setdefault(ev.id, None)
            if sp is None:
                continue
            X, E, feas = subproblem_vectors(sp)
            exact_bits, exact_E = exact_assignment(sp, X, E, feas)
            if strategy == "exact" or len(sp.vehicles) == 1:
                bits = exact_bits
            else:
                warm = [warm_amplitudes(len(g), 0, eps=0.5) for g in sp.groups]   # option 0 = cheapest alone
                r = solve_qaoa(sp.Q, sp.offset, E, feas, p=p, alpha=alpha, shots=shots, seed=seed,
                               groups=sp.groups, warm=warm, init_params=params, backend="fast",
                               grid=(16, 12), maxiter=150, transfer_maxiter=50)
                log = QuantumLog(hour, len(sp.vehicles), len(sp.variables), r.n_evals, r.transferred,
                                 r.p_optimal, r.p_feasible, bool(np.isclose(r.best_energy, exact_E)),
                                 r.runtime_s, round(100 * (r.best_energy - exact_E) / max(abs(exact_E), 1e-9), 3))
                if cold_compare and r.transferred and sum(q.cold_evaluations is not None for q in res.qlog) < cold_compare:
                    rc = solve_qaoa(sp.Q, sp.offset, E, feas, p=p, alpha=alpha, shots=shots, seed=seed,
                                    groups=sp.groups, warm=warm, backend="fast", grid=(16, 12), maxiter=150)
                    log.cold_evaluations, log.cold_p_optimal = rc.n_evals, rc.p_optimal
                res.qlog.append(log)
                params = r.params
                res.params = [float(x) for x in r.params]
                bits = r.best_bits if feas[int(sum(int(b) << i for i, b in enumerate(r.best_bits)))] else exact_bits
            for ev, s, t in decode(sp, bits):
                res.bookings[ev.id] = (s, t)
                state.book(ev, s, t)
            if progress:
                progress(hour, strategy)
    res.state, res.runtime_s = state, time.time() - t0
    return res


# ------------------------------------------------------------------ evaluation

def evaluate(res: DayResult, fleet, loads) -> dict:
    state = res.state
    actual = loads["actual"]
    total = actual + state.ev_kw
    limit = 1.25 * actual.max(axis=0)          # illustrative transformer rating per hub
    served = [ev for ev in fleet if res.bookings.get(ev.id)]
    energy = sum(ev.energy for ev in served)
    cost = sum(TARIFF[res.bookings[ev.id][1]] * ev.energy for ev in served)
    wait = [res.bookings[ev.id][1] - ev.arrival for ev in served]
    detour = sum(2 * DIST[ev.pref, res.bookings[ev.id][0]] for ev in served)
    solar = sum(ev.energy for ev in served if res.bookings[ev.id][1] in SOLAR_HOURS)
    city = total.sum(axis=1)
    grid_stress = GAMMA * float((total ** 2 - actual ** 2).sum())
    wait_rs = WAIT_COST * float(sum(wait))
    detour_rs = DETOUR_RS_PER_KM * detour
    ev_peak = float(state.ev_kw.sum(axis=1).max())
    return {
        "strategy": res.strategy,
        "served": len(served),
        "unserved": len(fleet) - len(served),
        "energy_kwh": round(energy, 1),
        "city_peak_kw": round(float(city.max()), 1),
        "city_peak_hour": int(city.argmax()),
        "worst_hub_peak_kw": round(float(total.max()), 1),
        "overload_hub_hours": int((total > limit).sum()),
        "energy_cost_rs": round(cost, 0),
        "grid_stress_rs": round(grid_stress, 0),
        "system_cost_rs": round(float(cost + grid_stress + wait_rs + detour_rs), 0),
        "ev_peak_kw": round(ev_peak, 1),
        "peak_rise_kw": round(float(city.max() - actual.sum(axis=1).max()), 1),
        "avg_wait_h": round(float(np.mean(wait)) if wait else 0.0, 2),
        "detour_km": round(float(detour), 1),
        "solar_share_pct": round(100 * solar / energy, 1) if energy else 0.0,
        "co2_t": round(energy * EMISSION_KG_PER_KWH / 1000, 2),
        "runtime_s": round(res.runtime_s, 1),
    }


def qlog_summary(res: DayResult) -> dict:
    q = res.qlog
    if not q:
        return {}
    cold = [x for x in q if x.cold_evaluations is not None]
    return {
        "subproblems": len(q),
        "max_qubits": max(x.qubits for x in q),
        "avg_qubits": round(float(np.mean([x.qubits for x in q])), 1),
        "found_optimum_pct": round(100 * float(np.mean([x.found_optimum for x in q])), 1),
        "found_optimum_count": int(sum(x.found_optimum for x in q)),
        "max_gap_pct": round(max(x.gap_pct for x in q), 2),
        "avg_p_optimal_pct": round(100 * float(np.mean([x.p_optimal for x in q])), 1),
        "avg_p_valid_pct": round(100 * float(np.mean([x.p_valid for x in q])), 1),
        "avg_evals_transfer": round(float(np.mean([x.evaluations for x in q if x.transferred])), 1)
        if any(x.transferred for x in q) else None,
        "avg_evals_cold_same_problems": round(float(np.mean([x.cold_evaluations for x in cold])), 1) if cold else None,
        "avg_evals_transfer_same_problems": round(float(np.mean([x.evaluations for x in cold])), 1) if cold else None,
        "avg_p_opt_cold_pct": round(100 * float(np.mean([x.cold_p_optimal for x in cold])), 1) if cold else None,
        "avg_p_opt_transfer_pct": round(100 * float(np.mean([x.p_optimal for x in cold])), 1) if cold else None,
    }


def simulate_day(n_vehicles=120, fleet_seed=11, forecast_day=None, p=2, use_p90=False,
                 strategies=("fcfs", "smart", "quantum", "exact"), cold_compare=12, progress=None) -> dict:
    """Run every strategy on the same day and return a JSON-ready result dictionary."""
    if forecast_day is None:
        from .data import generate_site_load
        from .forecast import train_and_forecast
        fc = train_and_forecast(generate_site_load(seed=42))
        forecast_day = {k: fc.target_day[k].tolist() for k in ["actual", "pred", "p10", "p90"]}
        forecast_day["mae_kw"] = fc.metrics["target_day_mae_kw"]
    fleet = generate_fleet(n_vehicles, fleet_seed)
    loads = feeder_loads(forecast_day)
    runs = {}
    for strat in strategies:
        if strat == "fcfs":
            runs[strat] = run_fcfs(fleet, loads)
        else:
            runs[strat] = run_rolling(fleet, loads, strat, p=p, use_p90=use_p90,
                                      cold_compare=cold_compare if strat == "quantum" else 0,
                                      progress=progress)
    return to_json(fleet, loads, forecast_day, runs, n_vehicles, fleet_seed, p)


def to_json(fleet, loads, forecast_day, runs, n_vehicles, fleet_seed, p) -> dict:
    out = {
        "meta": {
            "n_vehicles": n_vehicles, "fleet_seed": fleet_seed, "qaoa_depth": p,
            "tariff_rs_per_kwh": TARIFF, "solar_hours": [SOLAR_HOURS.start, SOLAR_HOURS.stop],
            "gamma": GAMMA, "wait_cost": WAIT_COST, "detour_rs_per_km": DETOUR_RS_PER_KM,
            "emission_kg_per_kwh": EMISSION_KG_PER_KWH, "forecast_mae_kw": forecast_day.get("mae_kw"),
        },
        "hubs": [asdict(h) for h in HUBS],
        "fleet": [asdict(ev) for ev in fleet],
        "forecast_profile": {k: list(map(float, forecast_day[k])) for k in ["actual", "pred", "p10", "p90"]},
        "base_actual": loads["actual"].round(2).tolist(),
        "base_pred": loads["pred"].round(2).tolist(),
        "strategies": {},
    }
    for name, res in runs.items():
        entry = {
            "kpis": evaluate(res, fleet, loads),
            "bookings": {str(k): (list(v) if v else None) for k, v in res.bookings.items()},
            "ev_kw": res.state.ev_kw.round(2).tolist(),
            "ports": res.state.ports.tolist(),
        }
        if res.qlog:
            entry["qaoa"] = qlog_summary(res)
            entry["qaoa_log"] = [asdict(x) for x in res.qlog]
        if res.params is not None:
            entry["qaoa_params"] = res.params
        if name == "quantum":
            entry["snapshots"] = {"ev_kw": [s.ev_kw.round(2).tolist() for s in res.snapshots],
                                  "ports": [s.ports.tolist() for s in res.snapshots]}
        out["strategies"][name] = entry
    return out


# ------------------------------------------------------------------ booking a single driver

def book_driver(city: dict, energy: float, arrival: int, departure: int, pref: int,
                p: int = 2, seed: int = 7, segment: str = "You") -> dict:
    """Plan one extra driver inside a precomputed city day (quantum strategy).

    The driver joins the vehicles arriving in the same hour. Their sub-problem is
    re-solved with XY-QAOA on top of the bookings already made earlier in the day.
    """
    snap = city["strategies"]["quantum"]["snapshots"]
    state = GridState(np.array(snap["ev_kw"][arrival]), np.array(snap["ports"][arrival]))
    loads = {"actual": np.array(city["base_actual"]), "pred": np.array(city["base_pred"])}
    loads["p90"] = loads["pred"]
    base = planning_forecast(loads, arrival)
    me = EV(-1, segment, float(energy), int(arrival), int(departure), int(pref), user=True)
    others = [EV(**{k: v for k, v in e.items()}) for e in city["fleet"] if e["arrival"] == arrival]

    # naive plan: drive to the preferred hub now and wait for the first free port
    naive = None
    for t in range(arrival, departure + 1):
        for s in sorted(allowed_hubs(me), key=lambda s: DIST[pref, s]):
            if state.ports[t, s] < HUBS[s].ports:
                naive = (s, t)
                break
        if naive:
            break

    groups = make_groups([me] + others, base, state)
    mine = next(g for g in groups if any(e.user for e in g))
    # book the other groups first only if they were scheduled before this one in the real run
    sp = build_subproblem(mine, base, state)
    if sp is None:
        return {"ok": False, "reason": "No free charger before your departure time at hubs within reach."}
    X, E, feas = subproblem_vectors(sp)
    exact_bits, exact_E = exact_assignment(sp, X, E, feas)
    if len(sp.vehicles) > 1:
        warm = [warm_amplitudes(len(g), 0, eps=0.5) for g in sp.groups]
        params = city["strategies"]["quantum"].get("qaoa_params")
        r = solve_qaoa(sp.Q, sp.offset, E, feas, p=p, shots=2048, seed=seed, groups=sp.groups,
                       warm=warm, backend="fast", grid=(16, 12), maxiter=150,
                       init_params=params if params and len(params) == 2 * p else None,
                       transfer_maxiter=60)
        bits = r.best_bits
        qinfo = {"qubits": len(sp.variables), "vehicles_in_subproblem": len(sp.vehicles),
                 "p_optimal_pct": round(100 * r.p_optimal, 1), "found_optimum": bool(np.isclose(r.best_energy, exact_E)),
                 "evaluations": r.n_evals}
    else:
        bits = exact_bits
        qinfo = {"qubits": len(sp.variables), "vehicles_in_subproblem": 1,
                 "p_optimal_pct": 100.0, "found_optimum": True, "evaluations": 0}
    plan = {ev.label: (s, t) for ev, s, t in decode(sp, bits)}
    if "You" not in plan:
        return {"ok": False, "reason": "No free charger before your departure time at hubs within reach."}
    s, t = plan["You"]

    def describe(s_t):
        if s_t is None:
            return None
        s_, t_ = s_t
        load = float(base[t_, s_] + state.ev_kw[t_, s_])
        return {"hub": HUBS[s_].name, "hub_index": s_, "hour": t_, "tariff": TARIFF[t_],
                "energy_cost_rs": round(TARIFF[t_] * energy, 0), "wait_h": t_ - arrival,
                "detour_km": round(float(2 * DIST[pref, s_]), 1),
                "feeder_load_kw": round(load, 1),
                "grid_stress_rs": round(GAMMA * ((load + energy) ** 2 - load ** 2), 0),
                "solar": t_ in SOLAR_HOURS}

    my_vi = next(i for i, ev in enumerate(sp.vehicles) if ev.user)
    alternatives = sorted(
        [{**describe((s_, t_)), "score_rs": round(option_cost(me, s_, t_, base, state), 0)}
         for s_, t_ in sp.options[my_vi]], key=lambda d: d["score_rs"])
    plan_d, naive_d = describe((s, t)), describe(naive)
    savings = None
    if naive_d:
        bill = naive_d["energy_cost_rs"] - plan_d["energy_cost_rs"]
        relief = naive_d["grid_stress_rs"] - plan_d["grid_stress_rs"]
        # proposed incentive: half of the grid cost a driver helps avoid is credited back
        savings = {"energy_bill_rs": round(bill, 0), "grid_relief_rs": round(relief, 0),
                   "credit_rs": round(max(relief, 0) * 0.5, 0),
                   "total_rs": round(bill + max(relief, 0) * 0.5, 0)}
    return {"ok": True, "plan": plan_d, "naive": naive_d, "alternatives": alternatives, "savings": savings,
            "quantum": qinfo, "co2_kg": round(energy * EMISSION_KG_PER_KWH, 1),
            "others_in_subproblem": [ev.label for ev in sp.vehicles if not ev.user]}
