"""Turn a PipelineOutput into a JSON-ready dict so the app can render instantly
from precomputed results and identically from a live run."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import WINDOW_HOURS, PipelineOutput
from .qaoa import qubo_to_ising


def _ts(series) -> list:
    return [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M") for x in series]


def qaoa_dict(r, energies, feasible, top=24) -> dict:
    idx = np.argsort(r.probs)[::-1][:top]
    opt = np.isclose(energies, r.optimal_energy) & feasible
    n = r.n_qubits
    summ = r.summary()
    return {
        "summary": summ,
        "history": [float(x) for x in r.history],
        "optimal_energy": float(r.optimal_energy),
        "distribution": [{"bits": format(int(k), f"0{n}b"), "prob": float(r.probs[k]),
                          "cost": float(energies[k]), "valid": bool(feasible[k]), "optimal": bool(opt[k])}
                         for k in idx],
    }


def pipeline_to_dict(r: PipelineOutput, instance_name: str) -> dict:
    inst, q = r.instance, r.qubo
    h, J, _ = qubo_to_ising(q.Q, q.offset)
    fc = r.forecast
    assign = {}
    for name, a in r.assignments.items():
        assign[name] = {inst.vehicles[v].name: [inst.stations[s].name, inst.slot_labels[t]]
                        for v, slots in a.items() for s, t in slots[:1]}
    return {
        "instance_name": instance_name,
        "forecast": {
            "metrics": fc.metrics,
            "window_hours": WINDOW_HOURS[instance_name],
            "history": {"timestamp": _ts(fc.history["timestamp"]), "demand_kw": fc.history["demand_kw"].round(3).tolist()},
            "target": {"timestamp": _ts(fc.target_day["timestamp"]),
                       **{k: fc.target_day[k].round(3).tolist() for k in ["actual", "pred", "p10", "p90"]}},
        },
        "instance": {
            "slots": inst.slot_labels, "tariff": inst.tariff, "base_load": [round(x, 2) for x in inst.base_load],
            "gamma": inst.gamma, "wait_cost": inst.wait_cost,
            "stations": [{"name": s.name, "max_kw": s.max_kw} for s in inst.stations],
            "vehicles": [{"name": v.name, "energy_kwh": v.energy_kwh, "arrival": inst.slot_labels[v.arrival],
                          "last_slot": inst.slot_labels[min(v.departure, inst.n_slots - 1)],
                          "detour": {inst.stations[s].name: c for s, c in v.detour.items()}}
                         for v in inst.vehicles],
        },
        "qubo": {"qubits": q.n, "penalty": round(q.penalty, 1), "z_terms": int((np.abs(h) > 1e-12).sum()),
                 "zz_terms": len(J), "valid_schedules": int(r.feasible.sum()), "bitstrings": 2 ** q.n},
        "metrics": r.metrics,
        "assignments": assign,
        "table": r.table.to_dict(orient="records"),
        "qaoa": {name: qaoa_dict(res, r.energies, r.feasible) for name, res in r.qaoa.items()},
    }
