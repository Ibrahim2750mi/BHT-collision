"""Scaling study, n = 4..16 input bits (digest = n-1 bits, exactly 2-to-1).

Cost = hash queries needed to find a collision with success probability >= 2/3.
  classical : exact birthday formula (distinct draws, 2-to-1 hash)
  BHT       : table r + Grover iterations j + 1 check, k = r marked states (theory,
              validated against Qiskit Aer circuits below)
  walk      : exact simulation of the Ambainis walk (qwalk.py), best (r, t, K)
"""
import math
import random

import numpy as np

from bht_collision import grover_circuit, make_hash
from qwalk import best_walk_over_r

TARGET = 2 / 3


def classical_cost(N, target=TARGET):
    surv, mean, q, quant = 1.0, 0.0, 0, None
    while surv > 1e-12 and q < N // 2 + 1:
        mean += surv  # E[q] = sum_q P(no collision after q draws)
        q += 1
        surv *= (N - 2 * (q - 1)) / (N - (q - 1)) if N - 2 * (q - 1) > 0 else 0
        if quant is None and 1 - surv >= target:
            quant = q
    return quant, mean


def bht_cost(N, target=TARGET):
    best = None
    for r in range(2, min(N // 2, int(6 * N ** (1 / 3)) + 2) + 1):
        th = math.asin(math.sqrt(r / N))
        j = 0
        while j < 4 * N and math.sin((2 * j + 1) * th) ** 2 < target:
            j += 1
        if math.sin((2 * j + 1) * th) ** 2 < target:
            continue  # r/N too large: Grover cannot reach target (sin^2 stuck)
        c = r + j + 1
        if best is None or c < best[0]:
            best = (c, r, j)
    return best


def validate_grover_on_aer(n=8, r=6, shots=4000, seed=3):
    """Aer circuit success probability vs sin^2((2j+1)theta), k = r marked states."""
    from qiskit_aer import AerSimulator

    N, h, rng, sim = 1 << n, make_hash(n), random.Random(seed), AerSimulator()
    while True:
        table = rng.sample(range(N), r)
        if len({h(x) for x in table}) == r:  # no collision inside the table
            break
    tset, digests = set(table), {h(x) for x in table}
    marked = [x for x in range(N) if x not in tset and h(x) in digests]
    assert len(marked) == r
    th = math.asin(math.sqrt(len(marked) / N))
    print(f"Aer validation (n={n}, r={r}, {shots} shots):  j  circuit  theory")
    for j in (1, 2, 3, 4):
        counts = sim.run(grover_circuit(n, marked, j), shots=shots,
                         seed_simulator=seed + j).result().get_counts()
        p = sum(c for k, c in counts.items() if int(k, 2) in set(marked)) / shots
        print(f"{'':36}{j}  {p:.3f}    {math.sin((2 * j + 1) * th) ** 2:.3f}")


def main():
    validate_grover_on_aer()
    rows = []
    print(f"\n{'n':>2} {'N':>6} | {'classical':>9} {'(mean)':>7} | {'BHT':>5} {'r':>3} {'j':>3} |"
          f" {'walk':>5} {'r':>3} {'t':>3} {'K':>3} {'p':>5}")
    for n in range(4, 17, 2):
        N = 1 << n
        cq, cm = classical_cost(N)
        bc, br, bj = bht_cost(N)
        w = best_walk_over_r(N)
        rows.append((n, N, cq, bc, w["queries"]))
        print(f"{n:>2} {N:>6} | {cq:>9} {cm:>7.1f} | {bc:>5} {br:>3} {bj:>3} |"
              f" {w['queries']:>5} {w['r']:>3} {w['t']:>3} {w['K']:>3} {w['p']:.2f}")
    Ns = np.array([r[1] for r in rows[2:]], float)  # fit n >= 8
    print("\nlog-log slope (queries ~ N^s), n=8..16:")
    for name, col in (("classical", 2), ("BHT", 3), ("walk", 4)):
        s = np.polyfit(np.log(Ns), np.log([r[col] for r in rows[2:]]), 1)[0]
        print(f"  {name:>9}: s = {s:.3f}")
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for col, lab, st in ((2, "classical birthday", "o-"), (3, "BHT (Grover)", "s-"),
                         (4, "quantum walk", "^-")):
        ax.plot(N, [r[col] for r in rows], st, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("domain size N = 2^n")
    ax.set_ylabel("hash queries for P(success) >= 2/3")
    ax.set_title("Collision finding on a 2-to-1 toy hash")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("scaling.png", dpi=160)


if __name__ == "__main__":
    main()
