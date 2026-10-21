"""Running the trained QAOA circuits on IBM quantum hardware (or a noisy simulator).

Workflow ("parameter transfer from simulator to hardware")
  1. Train the QAOA angles on the noiseless simulator.
  2. Bind them into the circuit, transpile for the target device and sample it.
  3. Score every measured bitstring classically: valid schedule? optimal schedule?

Experiments are kept small on purpose so that they survive today's hardware noise:
  toy   2 vehicles, 1 hub, 2 slots   ->  4 qubits
  trio  3 vehicles, 1 shared port    ->  6 qubits
each with the constraint-preserving XY mixer and the standard X mixer.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from .problem import (all_energies, bit_matrix, bits_to_assignment, build_qubo, feasibility_mask,
                      make_instance)
from .qaoa import qaoa_circuit, qubo_to_ising, solve_qaoa

BASE_LOAD = [41.2, 41.5, 29.1]          # forecast base load used across the demo
EXPERIMENTS = [("toy", "XY"), ("toy", "X"), ("trio", "XY"), ("trio", "X")]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "hardware"


def prepare(name: str, mixer: str, p: int = 1, seed: int = 7) -> dict:
    """Train angles on the ideal simulator and return the bound, measured circuit."""
    inst = make_instance(name, BASE_LOAD)
    q = build_qubo(inst)
    X = bit_matrix(q.n)
    E = all_energies(q.Q, q.offset, X)
    feas = feasibility_mask(q, inst, X)
    if mixer == "XY":
        Qc, off, groups = q.Q_xy, q.offset_obj, q.groups
    else:
        Qc, off, groups = q.Q, q.offset, None
    r = solve_qaoa(Qc, off, E, feas, p=p, groups=groups, seed=seed, backend="fast")
    h, J, _ = qubo_to_ising(Qc, off)
    qc, params = qaoa_circuit(h, J, q.n, p, groups)
    bound = qc.assign_parameters(dict(zip(params, r.params)))
    bound.measure_all()
    bound.name = f"{name}-{mixer}-p{p}"
    return {"name": name, "mixer": mixer, "p": p, "instance": inst, "qubo": q, "energies": E,
            "feasible": feas, "ideal_probs": r.probs, "params": [float(x) for x in r.params],
            "circuit": bound}


def analyze(counts: dict, prep: dict) -> dict:
    E, feas = prep["energies"], prep["feasible"]
    n = prep["qubo"].n
    shots = sum(counts.values())
    opt_E = float(E[feas].min())
    opt = np.isclose(E, opt_E) & feas
    p_valid = sum(c for b, c in counts.items() if feas[int(b, 2)]) / shots
    p_opt = sum(c for b, c in counts.items() if opt[int(b, 2)]) / shots
    best = min((b for b in counts if feas[int(b, 2)]), key=lambda b: E[int(b, 2)], default=None)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
    ideal = prep["ideal_probs"]
    # total-variation distance between hardware histogram and the ideal distribution
    emp = np.zeros(2 ** n)
    for b, c in counts.items():
        emp[int(b, 2)] = c / shots
    tvd = 0.5 * float(np.abs(emp - ideal).sum())
    schedule = None
    if best is not None:
        bits = [(int(best, 2) >> i) & 1 for i in range(n)]
        a = bits_to_assignment(bits, prep["qubo"])
        inst = prep["instance"]
        schedule = {inst.vehicles[v].name: inst.slot_labels[st[0][1]] for v, st in a.items()}
    return {
        "shots": shots,
        "p_valid": round(p_valid, 4), "p_optimal": round(p_opt, 4),
        "ideal_p_valid": round(float(ideal[feas].sum()), 4),
        "ideal_p_optimal": round(float(ideal[opt].sum()), 4),
        "random_p_valid": round(float(feas.mean()), 4),
        "random_p_optimal": round(float(opt.mean()), 4),
        "found_optimum": bool(best is not None and np.isclose(E[int(best, 2)], opt_E)),
        "best_schedule": schedule,
        "tvd_vs_ideal": round(tvd, 4),
        "top_counts": [{"bits": b, "count": c, "valid": bool(feas[int(b, 2)]), "optimal": bool(opt[int(b, 2)]),
                        "cost": round(float(E[int(b, 2)]), 1)} for b, c in top],
    }


def run_on_backend(backend, shots: int = 4096, p: int = 1, mitigation: bool = True,
                   experiments=EXPERIMENTS, label: str | None = None, submit_only: bool = False) -> dict:
    """Transpile and sample every experiment in one Sampler job on `backend`."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    preps = [prepare(n, m, p) for n, m in experiments]
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    isa = [pm.run(pr["circuit"]) for pr in preps]
    is_sim = bool(getattr(backend, "options", None) is not None and "Fake" in type(backend).__name__)
    sampler = SamplerV2(mode=backend)
    if mitigation and not is_sim:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
    job = sampler.run(isa, shots=shots)
    meta = {
        "backend": getattr(backend, "name", str(backend)),
        "kind": "noisy simulator" if is_sim else "IBM quantum hardware",
        "label": label or ("Noisy simulator" if is_sim else "IBM hardware"),
        "job_id": job.job_id() if callable(getattr(job, "job_id", None)) else None,
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "shots": shots, "p": p, "error_mitigation": bool(mitigation and not is_sim),
        "num_qubits_device": getattr(backend, "num_qubits", None),
    }
    if submit_only:
        return {"meta": meta, "pending": True,
                "experiments": [{"name": pr["name"], "mixer": pr["mixer"]} for pr in preps]}
    return collect(job.result(), preps, isa, meta)


def collect(result, preps, isa, meta) -> dict:
    out = {"meta": meta, "experiments": []}
    for pr, circ, pub in zip(preps, isa, result):
        counts = pub.data.meas.get_counts()
        ops = circ.count_ops()
        two_q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz"))
        out["experiments"].append({
            "name": pr["name"], "mixer": pr["mixer"], "p": pr["p"], "qubits": pr["qubo"].n,
            "params": pr["params"], "transpiled_depth": circ.depth(), "two_qubit_gates": int(two_q),
            "physical_qubits": sorted({circ.find_bit(q).index for inst in circ.data
                                       for q in inst.qubits if inst.operation.name != "barrier"})[:32],
            **analyze(counts, pr),
        })
    out["meta"]["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return out


def save(result: dict, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result, indent=2))
    return path


def load_all() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            d["file"] = f.name
            out.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    return out
