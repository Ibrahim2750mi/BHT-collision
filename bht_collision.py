"""Toy hash collision finding: classical birthday vs quantum BHT (Qiskit + Aer).

Hash: n-bit input -> (n-1)-bit digest, built from a bijective mix then truncated,
so it is exactly 2-to-1 (every input has one collision partner).

Query accounting (all counted as calls to h):
  classical : one per hashed input
  BHT       : r table queries + 1 per Grover oracle call + 1 per measured-candidate check
Oracle truth table is built classically for simulation only and is NOT counted.
"""
import math
import random
import statistics
import sys

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def make_hash(n, seed=1, m=None):
    m = n - 1 if m is None else m
    rng = random.Random(seed)
    A, B = rng.randrange(1, 1 << n, 2), rng.randrange(1, 1 << n, 2)  # odd => bijective
    C = rng.randrange(1 << n)
    mask, s = (1 << n) - 1, max(1, n // 2)

    def h(x):
        y = (x * A + C) & mask
        y ^= y >> s
        y = (y * B) & mask
        return y >> (n - m)

    return h


def classical_birthday(h, n, rng):
    """Hash distinct random inputs until two digests match. Returns (pair, queries)."""
    seen = {}
    for q, x in enumerate(rng.sample(range(1 << n), 1 << n), start=1):
        d = h(x)
        if d in seen:
            return (seen[d], x), q
        seen[d] = x
    raise RuntimeError("no collision (hash is injective?)")


def _mcz(qc, n):
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)


def grover_circuit(n, marked, iters):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iters):
        for x in marked:  # phase oracle: -1 on each marked basis state
            zeros = [i for i in range(n) if not (x >> i) & 1]
            if zeros:
                qc.x(zeros)
            _mcz(qc, n)
            if zeros:
                qc.x(zeros)
        qc.h(range(n))  # diffusion
        qc.x(range(n))
        _mcz(qc, n)
        qc.x(range(n))
        qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def bht(h, n, rng, sim, r=None, lam=1.2, max_fail=30):
    """BHT with BBHT unknown-solution-count Grover. Returns (pair, stats)."""
    N = 1 << n
    r = r or max(2, round(N ** (1 / 3)))
    stats = {"queries": 0, "oracle_calls": 0, "circuits": 0, "restarts": 0, "r": r}
    while True:
        table = rng.sample(range(N), r)
        digests = {}
        for x in table:
            d = h(x)
            stats["queries"] += 1
            if d in digests:
                return (digests[d], x), stats
            digests[d] = x
        tset = set(table)
        marked = [x for x in range(N) if x not in tset and h(x) in digests]  # sim-only
        stats["marked"] = len(marked)
        if not marked:
            stats["restarts"] += 1
            continue
        m, fails = 1.0, 0
        while fails < max_fail:
            j = rng.randrange(max(1, int(m)))
            qc = grover_circuit(n, marked, j)
            counts = sim.run(qc, shots=1, seed_simulator=rng.randrange(1 << 30)).result().get_counts()
            x = int(next(iter(counts)), 2)
            stats["oracle_calls"] += j
            stats["queries"] += j + 1  # j oracle calls + classical check of candidate
            stats["circuits"] += 1
            if x not in tset and h(x) in digests:
                return (digests[h(x)], x), stats
            m = min(lam * m, math.sqrt(N))
            fails += 1
        stats["restarts"] += 1


def main(ns=(4, 6, 8), c_trials=500, q_trials=40, seed=0):
    rng = random.Random(seed)
    sim = AerSimulator()
    print(f"{'n':>2} {'N':>4} {'r':>2} | {'classical':>9} | {'BHT':>6} {'(oracle)':>8} | ok")
    for n in ns:
        h = make_hash(n)
        cq = [classical_birthday(h, n, rng)[1] for _ in range(c_trials)]
        bq, bo, ok = [], [], True
        for _ in range(q_trials):
            (a, b), st = bht(h, n, rng, sim)
            ok &= (a != b and h(a) == h(b))
            bq.append(st["queries"])
            bo.append(st["oracle_calls"])
        print(f"{n:>2} {1 << n:>4} {st['r']:>2} | {statistics.mean(cq):>9.1f} | "
              f"{statistics.mean(bq):>6.1f} {statistics.mean(bo):>8.1f} | {ok}")


if __name__ == "__main__":
    main()
