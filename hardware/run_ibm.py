"""Run the EV-charging QAOA circuits on a real IBM quantum computer.

One-time setup (free IBM Quantum Open Plan):
  1. Sign in at https://quantum.cloud.ibm.com and create an instance on the Open Plan.
  2. Copy your API key and the instance CRN from the dashboard.
  3. Save them once on your machine:
       python hardware/run_ibm.py --save-account --token YOUR_API_KEY --instance YOUR_CRN

Run (from the repository root):
  python hardware/run_ibm.py                       # least busy real device, waits for the result
  python hardware/run_ibm.py --backend ibm_fez     # a specific device
  python hardware/run_ibm.py --submit-only         # submit and exit, prints the job id
  python hardware/run_ibm.py --retrieve JOB_ID     # collect a submitted job later
  python hardware/run_ibm.py --fake FakeFez        # noisy local simulation, no account needed

Results go to results/hardware/*.json and appear on the Hardware page of the app.
The job uses a few seconds of QPU time (4 small circuits x 4096 shots).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evq.hardware import EXPERIMENTS, collect, prepare, run_on_backend, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run EV-charging QAOA on IBM quantum hardware")
    ap.add_argument("--save-account", action="store_true", help="store the API key and instance locally")
    ap.add_argument("--token", help="IBM Quantum API key")
    ap.add_argument("--instance", help="IBM Quantum instance CRN")
    ap.add_argument("--backend", help="device name, e.g. ibm_fez (default: least busy)")
    ap.add_argument("--fake", help="use a local fake backend instead, e.g. FakeFez or FakeTorino")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--no-mitigation", action="store_true", help="disable dynamical decoupling and twirling")
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--retrieve", metavar="JOB_ID")
    a = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    if a.save_account:
        if not a.token:
            sys.exit("--save-account needs --token (and ideally --instance)")
        QiskitRuntimeService.save_account(token=a.token, instance=a.instance, channel="ibm_quantum_platform",
                                          set_as_default=True, overwrite=True)
        print("Account saved. Now run: python hardware/run_ibm.py")
        return

    if a.fake:
        from qiskit_ibm_runtime import fake_provider
        backend = getattr(fake_provider, a.fake)()
        print(f"Running on local noisy simulator {a.fake} ({backend.num_qubits} qubits) ...")
        res = run_on_backend(backend, shots=a.shots, label=f"Noisy simulator ({a.fake})")
        path = save(res, f"noisy_{a.fake}.json")
        report(res, path)
        return

    service = QiskitRuntimeService(token=a.token, instance=a.instance) if a.token else QiskitRuntimeService()

    if a.retrieve:
        job = service.job(a.retrieve)
        print(f"Job {a.retrieve} status: {job.status()}")
        backend = job.backend()
        from qiskit.transpiler import generate_preset_pass_manager
        preps = [prepare(n, m, 1) for n, m in EXPERIMENTS]
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
        isa = [pm.run(pr["circuit"]) for pr in preps]
        meta = {"backend": backend.name, "kind": "IBM quantum hardware", "label": "IBM hardware",
                "job_id": a.retrieve, "shots": a.shots, "p": 1, "error_mitigation": not a.no_mitigation,
                "num_qubits_device": backend.num_qubits}
        res = collect(job.result(), preps, isa, meta)
        path = save(res, f"ibm_{backend.name}.json")
        report(res, path)
        return

    if a.backend:
        backend = service.backend(a.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=8)
    print(f"Using {backend.name} ({backend.num_qubits} qubits). Pending jobs: {backend.status().pending_jobs}")
    res = run_on_backend(backend, shots=a.shots, mitigation=not a.no_mitigation,
                         label=f"IBM hardware ({backend.name})", submit_only=a.submit_only)
    if a.submit_only:
        print(f"Submitted job {res['meta']['job_id']}. Collect it later with:")
        print(f"  python hardware/run_ibm.py --retrieve {res['meta']['job_id']}")
        return
    path = save(res, f"ibm_{backend.name}.json")
    report(res, path)


def report(res, path):
    print(f"\n{res['meta']['label']}  ->  saved to {path}")
    print(f"{'experiment':<12}{'qubits':>7}{'2q gates':>10}{'P(valid)':>10}{'ideal':>8}{'P(opt)':>9}{'ideal':>8}  found")
    for e in res["experiments"]:
        print(f"{e['name'] + ' ' + e['mixer']:<12}{e['qubits']:>7}{e['two_qubit_gates']:>10}"
              f"{100 * e['p_valid']:>9.1f}%{100 * e['ideal_p_valid']:>7.1f}%"
              f"{100 * e['p_optimal']:>8.1f}%{100 * e['ideal_p_optimal']:>7.1f}%  {e['found_optimum']}")
    print("\nCommit the JSON file and push; the app's Hardware page will show it.")


if __name__ == "__main__":
    main()
