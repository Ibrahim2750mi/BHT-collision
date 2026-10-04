"""Real cost of each method, counted in 'hard steps' (Toffoli = a gate on 3 qubits, the expensive building block).

Scrambler cost H comes from the real circuit (hash_circuit.py), summed adder by adder.
Quantum oracle call = scrambler forward + scrambler backward (to clean up) + compare + flip.
A normal-computer try is charged H hard steps too, so both sides use one unit. That unit favours the quantum side: a quantum hard step is far slower than a normal-computer step (see speed_factor in the UI).
"""
import json
import math
from functools import lru_cache

from qiskit import transpile
from qiskit.circuit.library import CDKMRippleCarryAdder

from hash_circuit import constants, toffoli_count
from qwalk import best_walk_over_r
from scaling import classical_cost


@lru_cache(maxsize=None)
def adder_ccx(m):
    t = transpile(CDKMRippleCarryAdder(m, kind="fixed"), basis_gates=["ccx", "cx", "x"], optimization_level=0)
    return int(t.count_ops().get("ccx", 0))


def adder_fit():  # ccx(m) is linear in m: fit from m=1..12, check at 40
    a = adder_ccx(12) - adder_ccx(11); b = adder_ccx(12) - 12 * a
    assert adder_ccx(40) == a * 40 + b, "adder cost not linear"
    return a, b


def H_of(n, c):
    a, b = adder_fit()
    f = (lambda m: adder_ccx(m)) if n <= 40 else (lambda m: a * m + b)
    return sum(f(n - k) for const in (c["A"], c["B"]) for k in range(n) if (const >> k) & 1)


def costs(n, pw="hunter2", walk=True):
    N, c = 2.0 ** n, constants(pw, n)
    H = H_of(n, c)
    cmp_, diff = 2 * (n - 1) - 3, 2 * n - 3
    th1 = math.asin(1 / math.sqrt(N))
    j = max(0, round(math.pi / (4 * th1) - 0.5))
    g_total = j * (2 * H + cmp_ + diff)
    mean = classical_cost(1 << n)[1] if n <= 16 else math.sqrt(math.pi * N / 2)
    best = None
    for i in range(0, 400):  # list size r on a geometric grid, pick the cheapest in HARD STEPS
        r = max(2, round(2 ** (i * (n / 3 + 4) / 400 + 1)))
        if r >= N / 2:
            break
        th = math.asin(math.sqrt(r / N))
        jj = max(1, math.ceil((math.asin(math.sqrt(2 / 3)) / th - 1) / 2))
        tot = jj * (2 * H + r * cmp_ + diff)
        if best is None or tot < best[0]:
            best = (tot, r, jj)
    rq = max(2, round(N ** (1 / 3)))
    thq = math.asin(math.sqrt(rq / N)); jq = max(1, math.ceil((math.asin(math.sqrt(2 / 3)) / thq - 1) / 2))
    row = dict(n=n, N=N, H=H, qubits=3 * n + 1,
               crack=dict(classical_tries=N / 2, classical_units=N / 2 * H, grover_steps=j, quantum_units=g_total,
                          ratio=g_total / (N / 2 * H)),
               race=dict(classical_tries=mean, classical_units=mean * H, bht_r=best[1], bht_steps=best[2],
                         quantum_units=best[0], ratio=best[0] / (mean * H),
                         query_optimal=dict(r=rq, steps=jq, quantum_units=jq * (2 * H + rq * cmp_ + diff))))
    if walk and n <= 16:
        w = best_walk_over_r(1 << n)
        lb = w["r"] * H + (2 * w["t"] * w["K"] + 2) * 2 * H
        row["race"]["walk"] = dict(r=w["r"], t=w["t"], K=w["K"], quantum_units_lower_bound=lb, ratio=lb / (mean * H))
    return row


if __name__ == "__main__":
    for n in (3, 4, 5):  # the per-adder sum must equal the full circuit's count
        assert H_of(n, constants("hunter2", n)) == toffoli_count(n, constants("hunter2", n))["toffoli"], n
    print("adder-by-adder sum == full circuit count (n=3,4,5)")
    rows = [costs(n) for n in (4, 8, 12, 16, 24, 32, 48, 64)]
    cross = next((n for n in range(8, 4000, 4) if costs(n, walk=False)["race"]["ratio"] < 1), None)
    cross_c = next((n for n in range(4, 200, 2) if costs(n, walk=False)["crack"]["ratio"] < 1), None)
    out = dict(rows=rows, race_break_even_bits=cross, crack_break_even_bits=cross_c,
               verified=[__import__("hash_circuit").verify(n) for n in (3, 4, 5)], per_hash_note="H = hard steps for one scrambler run")
    json.dump(out, open("cost_table.json", "w"), indent=1)
    f = lambda v: f"{v:.3g}"
    print(f"\n{'n':>3} {'H':>6} | CRACK: classical  grover   x | RACE: classical   BHT(r)       x     walk(lb)  x")
    for r in rows:
        k, a = r["crack"], r["race"]; w = a.get("walk")
        print(f"{r['n']:>3} {r['H']:>6} | {f(k['classical_units']):>16} {f(k['quantum_units']):>8} {f(k['ratio']):>5} | "
              f"{f(a['classical_units']):>14} {f(a['quantum_units']):>8}({a['bht_r']}) {f(a['ratio']):>7} "
              f"{(f(w['quantum_units_lower_bound']) + ' ' + f(w['ratio'])) if w else '-':>16}")
    print("break-even bits: crack", cross_c, "| race (BHT)", cross)
