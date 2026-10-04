"""The Ambainis quantum walk as a REAL Qiskit circuit (run on Aer).

Circuit = prepare psi0 -> K x [ oracle (flip lists that contain a twin) -> t x (D1 then D2) ] -> measure.
Basis: qwalk.build_lumped orbit basis (exact symmetry reduction, verified vs the full space).
D1, D2 are reflections 2P-I, loaded as UnitaryGate; oracle is a DiagonalGate of +-1.
"""
import math
import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, StatePreparation, UnitaryGate
from qiskit_aer import AerSimulator
from qwalk import build_lumped, run_rounds

SIM = AerSimulator(method="statevector")


def _pad(M, D):
    out = np.eye(D, dtype=complex); out[:M.shape[0], :M.shape[1]] = M
    return out


def walk_circuit(L, t, rounds, measure=False):
    """Circuit after `rounds` rounds. Returns (circuit, n_qubits)."""
    d = L["d"]; m = max(1, math.ceil(math.log2(d))); D = 1 << m
    psi = np.zeros(D, complex); psi[:d] = L["psi0"]
    sign = np.ones(D); sign[:d] = np.where(L["marked"], -1.0, 1.0)
    g1, g2 = UnitaryGate(_pad(L["D1"], D), label="D1"), UnitaryGate(_pad(L["D2"], D), label="D2")
    orc = DiagonalGate(list(sign.astype(complex)))
    qc = QuantumCircuit(m, m if measure else 0)
    qc.append(StatePreparation(psi), range(m))
    for _ in range(rounds):
        qc.append(orc, range(m))
        for _ in range(t):
            qc.append(g1, range(m)); qc.append(g2, range(m))
    if measure:
        qc.measure(range(m), range(m))
    return qc, m


def success_probs(L, t, K):
    """P(table has a collision) after k = 0..K rounds, from real Qiskit statevectors."""
    out, d = [], L["d"]
    for k in range(K + 1):
        qc, _ = walk_circuit(L, t, k); qc.save_statevector()
        sv = np.asarray(SIM.run(transpile(qc, SIM)).result().get_statevector())
        out.append(float((np.abs(sv[:d]) ** 2)[L["marked"]].sum()))
    return out


def measure_once(L, t, K, seed=None):
    """One shot: run the circuit, measure, return True if the measured table holds a twin."""
    qc, _ = walk_circuit(L, t, K, measure=True)
    key = next(iter(SIM.run(transpile(qc, SIM), shots=1, seed_simulator=seed).result().get_counts()))
    i = int(key, 2)
    return bool(i < L["d"] and L["marked"][i])


def full_space_qiskit(h, n, r, t, K):
    """No symmetry reduction: states |S,y> as basis indices, D1/D2 built from the real subset structure."""
    N = 1 << n; subs = list(combinations(range(N), r)); sid = {s: i for i, s in enumerate(subs)}
    states, tid = [], {}
    for s in subs:
        for y in range(N):
            if y not in s:
                states.append((sid[s], tid.setdefault(tuple(sorted(s + (y,))), len(tid))))
    dim = len(states); m = math.ceil(math.log2(dim)); D = 1 << m
    g1, g2 = np.array([s for s, _ in states]), np.array([T for _, T in states])
    def refl(g):
        P = np.zeros((D, D))
        for grp in np.unique(g):
            idx = np.flatnonzero(g == grp); P[np.ix_(idx, idx)] = 1 / len(idx)
        U = 2 * P - np.eye(D); U[dim:, dim:] = np.eye(D - dim); return U
    marked_S = np.array([len({h(x) for x in s}) < r for s in subs])
    mk = marked_S[g1]; sign = np.ones(D); sign[:dim] = np.where(mk, -1.0, 1.0)
    psi = np.zeros(D); psi[:dim] = 1 / math.sqrt(dim)
    u1, u2 = UnitaryGate(refl(g1)), UnitaryGate(refl(g2)); orc = DiagonalGate(list(sign.astype(complex)))
    out = []
    for k in range(1, K + 1):
        qc = QuantumCircuit(m); qc.append(StatePreparation(psi), range(m))
        for _ in range(k):
            qc.append(orc, range(m))
            for _ in range(t):
                qc.append(u1, range(m)); qc.append(u2, range(m))
        qc.save_statevector()
        sv = np.asarray(SIM.run(transpile(qc, SIM)).result().get_statevector())
        out.append(float((np.abs(sv[:dim]) ** 2)[mk].sum()))
    return out, m


if __name__ == "__main__":
    from bht_collision import make_hash
    for N, r, t, K in ((16, 3, 2, 4), (64, 5, 2, 4), (256, 8, 3, 4)):
        L = build_lumped(N, r)
        q = success_probs(L, t, K)[1:]; ref = run_rounds(L, t, K)
        print(f"N={N:>3} r={r} t={t}: qubits={max(1, math.ceil(math.log2(L['d'])))}  "
              f"max|Qiskit-NumPy|={max(abs(a-b) for a,b in zip(q, ref)):.1e}  p={['%.3f'%v for v in q]}")
    n, r, t, K = 3, 2, 1, 3
    fq, m = full_space_qiskit(make_hash(n), n, r, t, K)
    lq = success_probs(build_lumped(1 << n, r), t, K)[1:]
    print(f"FULL space N=8 r=2 ({m} qubits, no reduction) vs lumped Qiskit: max diff {max(abs(a-b) for a,b in zip(fq, lq)):.1e}")
    print("measure 400 shots (N=16,r=3,t=2,K=4): ", sum(measure_once(build_lumped(16,3), 2, 4, s) for s in range(400))/400)


def step_probs(L, t, K):
    """ONE circuit, state saved after every walk step. Returns {(k, s): P(list holds a twin)}."""
    d = L["d"]; m = max(1, math.ceil(math.log2(d))); D = 1 << m
    psi = np.zeros(D, complex); psi[:d] = L["psi0"]
    sign = np.ones(D); sign[:d] = np.where(L["marked"], -1.0, 1.0)
    g1, g2 = UnitaryGate(_pad(L["D1"], D)), UnitaryGate(_pad(L["D2"], D))
    orc = DiagonalGate(list(sign.astype(complex)))
    qc = QuantumCircuit(m); qc.append(StatePreparation(psi), range(m))
    for k in range(1, K + 1):
        qc.append(orc, range(m))
        for s in range(1, t + 1):
            qc.append(g1, range(m)); qc.append(g2, range(m)); qc.save_statevector(label=f"k{k}s{s}")
    data = SIM.run(transpile(qc, SIM)).result().data()
    return {(k, s): float((np.abs(np.asarray(data[f"k{k}s{s}"]))[:d] ** 2)[L["marked"]].sum())
            for k in range(1, K + 1) for s in range(1, t + 1)}
