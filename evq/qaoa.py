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

Two speed-ups used by the city-scale simulation
  * Warm-start XY-QAOA: each vehicle block starts in a state biased towards a
    classical guess instead of the uniform W state (in the spirit of Egger et al., 2021).
  * Parameter transfer: angles trained on one sub-problem initialise the next one,
    so COBYLA starts close to a good solution and needs far fewer circuit evaluations
    (Brandao et al., 2018; Galda et al., 2021).

During training a small vectorised simulator applies exactly the same gates
(diagonal cost phase, RX or XX+YY mixers). With backend="qiskit" the real Qiskit
circuit is then simulated once and compared with the fast simulator; the fidelity is
reported so the shortcut is verifiable. backend="fast" skips that step and samples
from the fast simulator directly (used inside the city simulation for speed).
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
    if not terms:
        return SparsePauliOp("I" * n, coeffs=[0.0])
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def _scale(h, J):
    s = max(float(np.abs(h).max()) if len(h) else 0.0, max((abs(c) for c in J.values()), default=0.0))
    return s if s > 1e-12 else 1.0


def _w_state(k: int) -> np.ndarray:
    vec = np.zeros(2 ** k)
    for i in range(k):
        vec[1 << i] = 1 / np.sqrt(k)
    return vec


def warm_amplitudes(k: int, favoured: int, eps: float = 0.35) -> np.ndarray:
    """One-hot amplitudes biased towards option `favoured` (eps = weight left for the rest)."""
    if k == 1:
        return np.array([1.0])
    amp = np.full(k, np.sqrt(eps / (k - 1)))
    amp[favoured] = np.sqrt(1 - eps)
    return amp


def _block_state(amp: np.ndarray) -> np.ndarray:
    k = len(amp)
    vec = np.zeros(2 ** k)
    for i in range(k):
        vec[1 << i] = amp[i]
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

def initial_state_circuit(n: int, groups=None, warm=None) -> QuantumCircuit:
    """|+>^n (no groups), a W state per group, or a warm-start state per group.

    warm: optional list (one entry per group) of one-hot amplitude vectors.
    """
    qc = QuantumCircuit(n)
    if groups is None:
        qc.h(range(n))
        return qc
    for gi, g in enumerate(groups):
        if len(g) == 1:
            qc.x(g[0])
        elif warm is not None:
            qc.append(StatePreparation(_block_state(np.asarray(warm[gi], dtype=float))), list(g))
        else:
            qc.append(StatePreparation(_w_state(len(g))), list(g))
    return qc


def initial_state_vector(n: int, groups=None, warm=None) -> np.ndarray:
    """Same state as initial_state_circuit, built directly with numpy (little-endian)."""
    if groups is None:
        return np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
    k = np.arange(2 ** n, dtype=np.int64)
    psi = np.ones(2 ** n)
    for gi, g in enumerate(groups):
        amp = np.asarray(warm[gi], dtype=float) if warm is not None else np.full(len(g), 1 / np.sqrt(len(g)))
        bits = np.stack([(k >> q) & 1 for q in g], axis=1)
        one_hot = bits.sum(axis=1) == 1
        which = np.argmax(bits, axis=1)
        psi *= np.where(one_hot, amp[which], 0.0)
    return psi.astype(complex)


def qaoa_circuit(h, J, n: int, p: int, groups=None, warm=None):
    """Parameterised QAOA circuit; parameters are ordered [γ1..γp, β1..βp]."""
    gam = ParameterVector("γ", p)
    bet = ParameterVector("β", p)
    scale = _scale(h, J)
    qc = QuantumCircuit(n, name=f"QAOA p={p}")
    qc.compose(initial_state_circuit(n, groups, warm), inplace=True)
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

    def __init__(self, h, J, n, groups, warm=None):
        self.n = n
        self.groups = groups
        k = np.arange(2 ** n, dtype=np.int64)
        Z = 1 - 2 * ((k[:, None] >> np.arange(n)) & 1)
        diag = Z @ h
        for (i, j), c in J.items():
            diag = diag + c * Z[:, i] * Z[:, j]
        self.diag = diag / _scale(h, J)
        # built from the Qiskit circuit itself so the optimiser sees exactly the state the
        # hardware/circuit path prepares (COBYLA paths are sensitive to 1e-15 differences)
        self.psi0 = Statevector(initial_state_circuit(n, groups, warm)).data.astype(complex)
        if groups is None:
            self.pairs = [(np.flatnonzero(((k >> i) & 1) == 0), np.flatnonzero(((k >> i) & 1) == 1))
                          for i in range(n)]
        else:
            # restrict to basis states that are one-hot in every block (the only ones
            # with amplitude), which makes the XY mixer much cheaper for large n
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
    warm_start: bool = False
    transferred: bool = False
    circuit: QuantumCircuit | None = None

    @property
    def gap_pct(self):
        denom = abs(self.optimal_energy) if abs(self.optimal_energy) > 1e-9 else 1.0
        return 100 * (self.best_energy - self.optimal_energy) / denom

    def summary(self) -> dict:
        return {
            "mixer": self.mixer, "p": self.p, "cvar_alpha": self.alpha,
            "qubits": self.n_qubits, "transpiled_depth": self.depth, "cx_gates": self.cx_count,
            "cost_evaluations": self.n_evals, "runtime_s": round(self.runtime_s, 2),
            "shots": self.shots,
            "prob_feasible": round(self.p_feasible, 4), "prob_optimal": round(self.p_optimal, 4),
            "found_optimum": self.sampled_optimal, "gap_pct": round(self.gap_pct, 3),
            "fast_sim_vs_qiskit_fidelity": round(self.sim_fidelity, 8),
            "warm_start": self.warm_start, "parameter_transfer": self.transferred,
            "params": [round(float(x), 4) for x in self.params],
        }


# ------------------------------------------------------------------ solver

def _interp_next(theta, depth):
    g_prev, b_prev = theta[: depth - 1], theta[depth - 1:]
    if depth == 2:
        return np.concatenate([np.repeat(g_prev, 2), np.repeat(b_prev, 2)])
    xs_old, xs_new = np.linspace(0, 1, depth - 1), np.linspace(0, 1, depth)
    return np.concatenate([np.interp(xs_new, xs_old, g_prev), np.interp(xs_new, xs_old, b_prev)])


def solve_qaoa(Q, offset, energies, feasible, p=1, alpha=0.25, shots=4096,
               seed=7, maxiter=300, grid=(24, 16), gamma_max=4 * np.pi, groups=None,
               callback=None, warm=None, init_params=None, backend="qiskit",
               transfer_maxiter=60) -> QAOAResult:
    """Train and sample a QAOA circuit.

    Q/offset define the cost Hamiltonian; energies/feasible are the full 2^n vectors
    used to score outcomes (CVaR training, best feasible sample, optimality checks).

    warm         per-group one-hot amplitudes for a warm-started XY-QAOA (needs groups)
    init_params  2p angles from an earlier, similar problem (parameter transfer). The
                 grid search and depth ramp are skipped; COBYLA runs at most
                 `transfer_maxiter` evaluations from these angles.
    backend      "qiskit" builds, verifies and samples the real Qiskit circuit;
                 "fast" samples from the verified fast simulator (no transpile step).
    """
    t0 = time.time()
    n = Q.shape[0]
    h, J, _ = qubo_to_ising(Q, offset)
    sim = _FastQAOASim(h, J, n, groups, warm)
    order = np.argsort(energies)
    E_sorted = energies[order]
    history = []
    n_grid = 0

    def f(theta):
        val = _cvar(sim.probs(theta), E_sorted, order, alpha)
        history.append(val)
        if callback:
            callback(len(history), val)
        return val

    transferred = init_params is not None and len(init_params) == 2 * p
    if transferred:
        theta = minimize(f, np.asarray(init_params, dtype=float), method="COBYLA",
                         options={"maxiter": transfer_maxiter, "rhobeg": 0.1}).x
    else:
        # depth 1: coarse grid over (γ, β), then COBYLA.
        # γ multiplies the Hamiltonian normalised by its largest coefficient (often the
        # penalty), so objective differences need γ well above π to produce useful phases.
        beta_max = np.pi / 2 if groups is None else np.pi
        best_val, theta = np.inf, None
        for g in np.linspace(0.05, gamma_max, grid[0]):
            for b in np.linspace(0.05, beta_max, grid[1]):
                val = _cvar(sim.probs([g, b]), E_sorted, order, alpha)
                if val < best_val:
                    best_val, theta = val, np.array([g, b])
        n_grid = grid[0] * grid[1]
        theta = minimize(f, theta, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2}).x
        for depth in range(2, p + 1):
            theta = minimize(f, _interp_next(theta, depth), method="COBYLA",
                             options={"maxiter": maxiter, "rhobeg": 0.15}).x

    rng = np.random.default_rng(seed)
    qc = None
    depth = cx = 0
    fidelity = 1.0
    if backend == "qiskit":
        qc, params = qaoa_circuit(h, J, n, p, groups, warm)
        final = qc.assign_parameters(dict(zip(params, theta)))
        sv = Statevector(final)
        probs = sv.probabilities()
        fidelity = float(abs(np.vdot(sim.state(theta), sv.data)) ** 2)
        meas = final.copy()
        meas.measure_all()
        counts = StatevectorSampler(seed=seed).run([meas], shots=shots).result()[0].data.meas.get_counts()
        tq = transpile(final.decompose(reps=4) if groups is not None else final,
                       basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=seed)
        depth, cx = tq.depth(), int(tq.count_ops().get("cx", 0))
    else:
        probs = sim.probs(theta)
        probs = probs / probs.sum()
        draws = rng.choice(len(probs), size=shots, p=probs)
        idx, cnt = np.unique(draws, return_counts=True)
        counts = {format(int(i), f"0{n}b"): int(c) for i, c in zip(idx, cnt)}

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

    return QAOAResult(
        p=p, alpha=alpha, mixer="X" if groups is None else "XY", params=np.asarray(theta),
        history=history, probs=probs, counts=counts,
        best_bits=np.array([(best_idx >> i) & 1 for i in range(n)]),
        best_energy=best_E, optimal_energy=opt_E,
        p_optimal=float(probs[opt_mask].sum()), p_feasible=float(probs[feasible].sum()),
        sampled_optimal=bool(np.isclose(best_E, opt_E)), shots=shots,
        n_qubits=n, depth=depth, cx_count=cx,
        n_evals=len(history) + n_grid, runtime_s=time.time() - t0,
        sim_fidelity=fidelity, warm_start=warm is not None, transferred=transferred, circuit=qc,
    )
