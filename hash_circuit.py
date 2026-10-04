"""The scrambler (hash) built as a REAL quantum circuit, checked against the Python scrambler.

Python recipe:  v1 = x*A + C   ->   v2 = v1 xor (v1 >> s)   ->   v3 = v2*B   ->   fingerprint = v3 >> 1   (all mod 2^n)
Circuit recipe: registers  x (input, kept)   v (holds v1 then v2)   o (holds v3).   fingerprint = o[1:].
  multiply by a fixed number = add shifted copies (one quantum adder per 1-bit of the fixed number).
"""
import hashlib
import random

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder


def constants(pw, n):  # exactly the recipe in app.Hasher
    N = 1 << n
    rng = random.Random(int.from_bytes(hashlib.sha256(pw.encode()).digest(), "big") ^ n)
    return dict(A=rng.randrange(1, N, 2), B=rng.randrange(1, N, 2), C=rng.randrange(N), s=max(1, n // 2))


def hash_py(x, n, c):
    m = (1 << n) - 1
    v = (x * c["A"] + c["C"]) & m
    v ^= v >> c["s"]
    return ((v * c["B"]) & m) >> 1


def _mul_add(qc, src, dst, const, n, helper):
    """dst += const * src  (mod 2^n): one ripple-carry adder for every 1-bit of const (1 shared helper qubit)."""
    for k in range(n):
        if (const >> k) & 1:
            m = n - k
            qc.append(CDKMRippleCarryAdder(m, kind="fixed"), list(src[:m]) + list(dst[k:n]) + [helper])


def hash_circuit(n, c):
    qc = QuantumCircuit(3 * n + 1, name=f"scrambler{n}")
    x, v, o = list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n))
    h = 3 * n  # helper qubit, returned to 0 after every addition
    for k in range(n):  # start v at C
        if (c["C"] >> k) & 1:
            qc.x(v[k])
    _mul_add(qc, x, v, c["A"], n, h)                 # v = x*A + C
    for i in range(n - c["s"]):                   # v = v xor (v >> s), low bits first
        qc.cx(v[i + c["s"]], v[i])
    _mul_add(qc, v, o, c["B"], n, h)                 # o = v*B
    return qc, x, v, o


def toffoli_count(n, c):
    qc, *_ = hash_circuit(n, c)
    t = transpile(qc, basis_gates=["ccx", "cx", "x"], optimization_level=0)
    ops = t.count_ops()
    return dict(toffoli=int(ops.get("ccx", 0)), cx=int(ops.get("cx", 0)), qubits=t.num_qubits)


def verify(n, pw="hunter2"):
    """Run every input x through the circuit on Aer; compare with the Python scrambler."""
    from qiskit_aer import AerSimulator
    c, sim = constants(pw, n), AerSimulator()
    base, x, v, o = hash_circuit(n, c)
    bad = 0
    for val in range(1 << n):
        qc = QuantumCircuit(base.num_qubits, n - 1)
        for k in range(n):
            if (val >> k) & 1:
                qc.x(x[k])
        qc.compose(base, inplace=True)
        qc.measure(o[1:], range(n - 1))
        got = int(next(iter(sim.run(transpile(qc, sim, optimization_level=0), shots=1).result().get_counts())), 2)
        bad += got != hash_py(val, n, c)
    return dict(n=n, inputs=1 << n, wrong=bad)


if __name__ == "__main__":
    for n in (3, 4, 5):
        print(verify(n), toffoli_count(n, constants("hunter2", n)))
