"""Run the small quantum search on a real IBM chip, and compare with the perfect (ideal) answer.

  python ibm_run.py --fake                      # built-in noisy model of an IBM chip (no account needed)
  python ibm_run.py --real                      # real chip: needs a saved IBM account or env var QISKIT_IBM_TOKEN
  python ibm_run.py --real --backend ibm_xxx    # pick a chip by name (default: least busy)

Task: password -> number x. Find the ONE other number with the same fingerprint, using a Grover search
with 0,1,2,3.. search steps. More steps help at first, then noise (mistakes) wins. We plot ideal vs measured.
"""
import argparse, json, math, os
import numpy as np

from bht_collision import grover_circuit
from hash_circuit import constants, hash_py


def fold(pw, n):
    x = 0
    for b in pw.encode():
        x = (x * 31 + b) % (1 << n)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", default="hunter2"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4000); ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--real", action="store_true"); ap.add_argument("--fake", action="store_true"); ap.add_argument("--backend")
    a = ap.parse_args()
    n, c = a.n, constants(a.pw, a.n)
    x = fold(a.pw, n); fp = hash_py(x, n, c)
    twin = [y for y in range(1 << n) if y != x and hash_py(y, n, c) == fp]
    assert len(twin) == 1, "scrambler should be exactly 2-to-1"
    twin = twin[0]
    th = math.asin(1 / math.sqrt(1 << n))
    print(f"password {a.pw!r} -> number {x} -> fingerprint {fp}; its one twin is {twin}")
    circs = [grover_circuit(n, [twin], j) for j in range(a.max_steps + 1)]
    ideal = [math.sin((2 * j + 1) * th) ** 2 for j in range(a.max_steps + 1)]

    from qiskit import transpile
    if a.real:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService(token=os.environ.get("QISKIT_IBM_TOKEN")) if os.environ.get("QISKIT_IBM_TOKEN") else QiskitRuntimeService()
        be = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False, min_num_qubits=n)
        isa = [transpile(q, be, optimization_level=3) for q in circs]
        sam = SamplerV2(mode=be)
        sam.options.dynamical_decoupling.enable = True          # keeps idle qubits from drifting
        sam.options.twirling.enable_gates = True                # averages out gate mistakes
        res = sam.run(isa, shots=a.shots).result()
        counts = [r.data.c.get_counts() for r in res]
        label = be.name
    else:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        fake = FakeSherbrooke(); be = AerSimulator.from_backend(fake)
        isa = [transpile(q, be, optimization_level=3, seed_transpiler=1) for q in circs]
        counts = [be.run(q, shots=a.shots, seed_simulator=5 + i).result().get_counts() for i, q in enumerate(isa)]
        label = "model of IBM Sherbrooke (simulated noise, NOT a real run)"
    meas = [sum(v for k, v in cn.items() if int(k, 2) == twin) / a.shots for cn in counts]
    two_q = [sum(q.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for q in isa]
    print(f"{'steps':>5} {'ideal':>7} {'measured':>9} {'two-qubit gates':>16}   (guessing = {1 / (1 << n):.3f})")
    for j in range(len(circs)):
        print(f"{j:>5} {ideal[j]:>7.3f} {meas[j]:>9.3f} {two_q[j]:>16}")
    json.dump(dict(pw=a.pw, n=n, x=x, fingerprint=fp, twin=twin, backend=label, real=a.real, shots=a.shots,
                   steps=list(range(len(circs))), ideal=ideal, measured=meas, two_qubit_gates=two_q),
              open("ibm_results.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    w = 0.38; xs = np.arange(len(circs))
    plt.figure(figsize=(6.5, 4)); plt.bar(xs - w / 2, ideal, w, label="perfect machine", color="#d1d5db")
    plt.bar(xs + w / 2, meas, w, label=label[:40], color="#2563eb"); plt.axhline(1 / (1 << n), ls="--", c="k", lw=.8)
    plt.xlabel("search steps"); plt.ylabel("chance of finding the twin"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("ibm_results.png", dpi=160); print("saved ibm_results.json, ibm_results.png")


if __name__ == "__main__":
    main()
