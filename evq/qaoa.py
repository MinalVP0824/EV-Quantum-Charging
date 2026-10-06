"""QAOA solver built on Qiskit circuits.

Steps
  1. QUBO -> Ising Hamiltonian H = sum h_i Z_i + sum J_ij Z_i Z_j  (x = (1 - z) / 2)
  2. p-layer QAOA circuit, in one of two flavours:
       * "X"  : standard QAOA, |+>^n start, transverse-field mixer, constraints as penalties
       * "XY" : constraint-preserving QAOA (Hadfield et al., 2019). Each vehicle's
                variables form a one-hot block that starts in a W state and is mixed
                by an XY ring, so "charge exactly once" is never violated.
  3. Classical loop: COBYLA tunes (γ, β) to minimise the CVaR of the energy
     (Barkoutsos et al., 2020). alpha = 1 is plain expectation-value QAOA.
     Deeper circuits are warm-started layer by layer (INTERP, Zhou et al., 2020).
  4. The final Qiskit circuit is executed with the StatevectorSampler primitive and
     the best feasible measured bitstring becomes the schedule.

During the training loop a small vectorised simulator applies exactly the same
gates (diagonal cost phase, RX or XX+YY mixers) for speed. After training the
real Qiskit circuit is simulated once and its state is compared with the fast
simulator; the fidelity is reported so the shortcut is verifiable.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, XXPlusYYGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


# ------------------------------------------------------------------ Hamiltonian

def qubo_to_ising(Q: np.ndarray, offset: float):
    n = Q.shape[0]
    S = np.triu(Q, 1)
    S = S + S.T
    h = -np.diag(Q) / 2 - S.sum(axis=1) / 4
    J = {(i, j): Q[i, j] / 4 for i in range(n) for j in range(i + 1, n) if abs(Q[i, j]) > 1e-12}
    const = offset + np.diag(Q).sum() / 2 + np.triu(Q, 1).sum() / 4
    return h, J, const


def ising_operator(h, J, n) -> SparsePauliOp:
    terms = [("Z", [i], float(h[i])) for i in range(n) if abs(h[i]) > 1e-12]
    terms += [("ZZ", [i, j], float(c)) for (i, j), c in J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def _scale(h, J):
    return max(np.abs(h).max(), max((abs(c) for c in J.values()), default=0.0))


def _w_state(k: int) -> np.ndarray:
    vec = np.zeros(2 ** k)
    for i in range(k):
        vec[1 << i] = 1 / np.sqrt(k)
    return vec


def _xy_pairs(groups):
    pairs = []
    for g in groups:
        k = len(g)
        if k < 2:
            continue
        ring = [(g[a], g[a + 1]) for a in range(k - 1)]
        if k > 2:
            ring.append((g[-1], g[0]))
        pairs += ring[0::2] + ring[1::2]   # even pairs, then odd pairs
    return pairs


# ------------------------------------------------------------------ circuit

def initial_state_circuit(n: int, groups=None) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    if groups is None:
        qc.h(range(n))
    else:
        for g in groups:
            if len(g) == 1:
                qc.x(g[0])
            else:
                qc.append(StatePreparation(_w_state(len(g))), list(g))
    return qc


def qaoa_circuit(h, J, n: int, p: int, groups=None):
    """Parameterised QAOA circuit; parameters are ordered [γ1..γp, β1..βp]."""
    gam = ParameterVector("γ", p)
    bet = ParameterVector("β", p)
    scale = _scale(h, J)
    qc = QuantumCircuit(n, name=f"QAOA p={p}")
    qc.compose(initial_state_circuit(n, groups), inplace=True)
    pairs = _xy_pairs(groups) if groups is not None else None
    for layer in range(p):
        qc.barrier()
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gam[layer] * h[i] / scale, i)
        for (i, j), c in J.items():
            qc.rzz(2 * gam[layer] * c / scale, i, j)
        qc.barrier()
        if groups is None:
            qc.rx(2 * bet[layer], range(n))
        else:
            for a, b in pairs:
                qc.append(XXPlusYYGate(2 * bet[layer]), [a, b])
    return qc, list(gam) + list(bet)


# ------------------------------------------------------------------ fast simulator

class _FastQAOASim:
    """Applies the same gate sequence as qaoa_circuit with vectorised numpy."""

    def __init__(self, h, J, n, groups):
        self.n = n
        self.groups = groups
        k = np.arange(2 ** n, dtype=np.int64)
        Z = 1 - 2 * ((k[:, None] >> np.arange(n)) & 1)
        diag = Z @ h
        for (i, j), c in J.items():
            diag = diag + c * Z[:, i] * Z[:, j]
        self.diag = diag / _scale(h, J)
        self.psi0 = Statevector(initial_state_circuit(n, groups)).data.astype(complex)
        if groups is None:
            self.pairs = [(np.flatnonzero(((k >> i) & 1) == 0), np.flatnonzero(((k >> i) & 1) == 1))
                          for i in range(n)]
        else:
            self.pairs = []
            for a, b in _xy_pairs(groups):
                ba, bb = (k >> a) & 1, (k >> b) & 1
                self.pairs.append((np.flatnonzero((ba == 1) & (bb == 0)), np.flatnonzero((ba == 0) & (bb == 1))))

    def state(self, theta):
        p = len(theta) // 2
        psi = self.psi0.copy()
        for layer in range(p):
            g, b = theta[layer], theta[p + layer]
            psi *= np.exp(-1j * g * self.diag)
            c, s = np.cos(b), np.sin(b)
            for i0, i1 in self.pairs:
                a0, a1 = psi[i0], psi[i1]
                psi[i0] = c * a0 - 1j * s * a1
                psi[i1] = c * a1 - 1j * s * a0
        return psi

    def probs(self, theta):
        return np.abs(self.state(theta)) ** 2


def _cvar(probs, E_sorted, order, alpha):
    p = probs[order]
    c = np.cumsum(p)
    k = min(int(np.searchsorted(c, alpha)), len(p) - 1)
    head = float((p[:k] * E_sorted[:k]).sum())
    prev = float(c[k - 1]) if k > 0 else 0.0
    return (head + (alpha - prev) * E_sorted[k]) / alpha


# ------------------------------------------------------------------ result

@dataclass
class QAOAResult:
    p: int
    alpha: float
    mixer: str
    params: np.ndarray
    history: list = field(default_factory=list)
    probs: np.ndarray | None = None
    counts: dict | None = None
    best_bits: np.ndarray | None = None
    best_energy: float | None = None
    optimal_energy: float | None = None
    p_optimal: float = 0.0
    p_feasible: float = 0.0
    sampled_optimal: bool = False
    shots: int = 0
    n_qubits: int = 0
    depth: int = 0
    cx_count: int = 0
    n_evals: int = 0
    runtime_s: float = 0.0
    sim_fidelity: float = 0.0
    circuit: QuantumCircuit | None = None

    @property
    def gap_pct(self):
        return 100 * (self.best_energy - self.optimal_energy) / abs(self.optimal_energy)

    def summary(self) -> dict:
        return {
            "mixer": self.mixer, "p": self.p, "cvar_alpha": self.alpha,
            "qubits": self.n_qubits, "transpiled_depth": self.depth, "cx_gates": self.cx_count,
            "cost_evaluations": self.n_evals, "runtime_s": round(self.runtime_s, 2),
            "shots": self.shots,
            "prob_feasible": round(self.p_feasible, 4), "prob_optimal": round(self.p_optimal, 4),
            "found_optimum": self.sampled_optimal, "gap_pct": round(self.gap_pct, 3),
            "fast_sim_vs_qiskit_fidelity": round(self.sim_fidelity, 8),
            "params": [round(float(x), 4) for x in self.params],
        }


# ------------------------------------------------------------------ solver

def solve_qaoa(Q, offset, energies, feasible, p=1, alpha=0.25, shots=4096,
               seed=7, maxiter=300, grid=(24, 16), gamma_max=4 * np.pi, groups=None,
               callback=None) -> QAOAResult:
    """Q/offset define the cost Hamiltonian; energies/feasible are the full 2^n
    vectors of the penalised QUBO, used to score outcomes."""
    t0 = time.time()
    n = Q.shape[0]
    h, J, _ = qubo_to_ising(Q, offset)
    sim = _FastQAOASim(h, J, n, groups)
    order = np.argsort(energies)
    E_sorted = energies[order]
    history = []

    def f(theta):
        val = _cvar(sim.probs(theta), E_sorted, order, alpha)
        history.append(val)
        if callback:
            callback(len(history), val)
        return val

    # depth 1: coarse grid over (γ, β), then COBYLA.
    # γ multiplies the Hamiltonian normalised by its largest coefficient (the penalty),
    # so the objective differences need γ well above π to produce useful phases.
    beta_max = np.pi / 2 if groups is None else np.pi
    best_val, theta = np.inf, None
    for g in np.linspace(0.05, gamma_max, grid[0]):
        for b in np.linspace(0.05, beta_max, grid[1]):
            val = _cvar(sim.probs([g, b]), E_sorted, order, alpha)
            if val < best_val:
                best_val, theta = val, np.array([g, b])
    theta = minimize(f, theta, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2}).x

    # deeper circuits: INTERP warm start from the previous depth
    for depth in range(2, p + 1):
        g_prev, b_prev = theta[: depth - 1], theta[depth - 1:]
        xs_new = np.linspace(0, 1, depth)
        if depth == 2:
            g_new, b_new = np.repeat(g_prev, 2), np.repeat(b_prev, 2)
        else:
            xs_old = np.linspace(0, 1, depth - 1)
            g_new, b_new = np.interp(xs_new, xs_old, g_prev), np.interp(xs_new, xs_old, b_prev)
        theta = minimize(f, np.concatenate([g_new, b_new]), method="COBYLA",
                         options={"maxiter": maxiter, "rhobeg": 0.15}).x

    # build and run the real Qiskit circuit with the trained parameters
    qc, params = qaoa_circuit(h, J, n, p, groups)
    final = qc.assign_parameters(dict(zip(params, theta)))
    sv = Statevector(final)
    probs = sv.probabilities()
    fidelity = float(abs(np.vdot(sim.state(theta), sv.data)) ** 2)

    meas = final.copy()
    meas.measure_all()
    counts = StatevectorSampler(seed=seed).run([meas], shots=shots).result()[0].data.meas.get_counts()

    best_idx, best_E = None, np.inf
    for bitstr in counts:
        k = int(bitstr, 2)
        if feasible[k] and energies[k] < best_E:
            best_idx, best_E = k, float(energies[k])
    if best_idx is None:
        best_idx = int(max(counts, key=counts.get), 2)
        best_E = float(energies[best_idx])

    opt_E = float(energies[feasible].min())
    opt_mask = np.isclose(energies, opt_E) & feasible
    tq = transpile(final.decompose(reps=4) if groups is not None else final,
                   basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=seed)

    return QAOAResult(
        p=p, alpha=alpha, mixer="X" if groups is None else "XY", params=theta,
        history=history, probs=probs, counts=counts,
        best_bits=np.array([(best_idx >> i) & 1 for i in range(n)]),
        best_energy=best_E, optimal_energy=opt_E,
        p_optimal=float(probs[opt_mask].sum()), p_feasible=float(probs[feasible].sum()),
        sampled_optimal=bool(np.isclose(best_E, opt_E)), shots=shots,
        n_qubits=n, depth=tq.depth(), cx_count=int(tq.count_ops().get("cx", 0)),
        n_evals=len(history) + grid[0] * grid[1], runtime_s=time.time() - t0,
        sim_fidelity=fidelity, circuit=qc,
    )
