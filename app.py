"""Twin Finder - Flask GUI for the collision-finding project.

Run:  python app.py   then open http://127.0.0.1:5000

Your password picks the scrambler (hash) constants and is scrambled by it.
Three methods then race to find two different numbers with the same fingerprint:
  classical guessing, quantum search (BHT / Grover), quantum walk (Ambainis-style).
Quantum parts are simulations on a normal computer (see README notes in the UI).
"""
import hashlib
import math
import random
import threading
import time
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from qwalk import best_walk, best_walk_over_r, build_lumped
from scaling import classical_cost

app = Flask(__name__, static_folder="static")
SIZES = (4, 6, 8, 10, 12, 14, 16)


# ---------------------------------------------------------------- the scrambler
class Hasher:
    """n-bit number -> (n-1)-bit fingerprint. Exactly 2-to-1 (every number has one twin)."""

    def __init__(self, n, pw):
        self.n, self.N = n, 1 << n
        seed = int.from_bytes(hashlib.sha256(pw.encode()).digest(), "big")
        rng = random.Random(seed ^ n)
        self.A = rng.randrange(1, self.N, 2)
        self.B = rng.randrange(1, self.N, 2)
        self.C = rng.randrange(self.N)
        self.s = max(1, n // 2)
        self.mask = self.N - 1
        x = np.arange(self.N, dtype=np.uint64)
        y = (x * np.uint64(self.A) + np.uint64(self.C)) & np.uint64(self.mask)
        y ^= y >> np.uint64(self.s)
        y = (y * np.uint64(self.B)) & np.uint64(self.mask)
        d = (y >> np.uint64(1)).astype(np.int64)
        order = np.argsort(d, kind="stable")
        a, b = order[0::2], order[1::2]
        assert (d[a] == d[b]).all(), "hash is not 2-to-1"
        self.partner = np.zeros(self.N, dtype=np.int64)
        self.partner[a], self.partner[b] = b, a
        self.d = d.tolist()

    def trace(self, x):
        v1 = (x * self.A + self.C) & self.mask
        v2 = v1 ^ (v1 >> self.s)
        v3 = (v2 * self.B) & self.mask
        return [v1, v2, v3, v3 >> 1]


_hashers = OrderedDict()


def get_hasher(pw, n):
    key = (pw, n)
    if key not in _hashers:
        _hashers[key] = Hasher(n, pw)
        while len(_hashers) > 12:
            _hashers.popitem(last=False)
    return _hashers[key]


def fold(pw, n):
    x = 0
    for b in pw.encode():
        x = (x * 31 + b) % (1 << n)
    return x


# ---------------------------------------------------------------- walk cache
_walk, _lock = {}, threading.Lock()


def walk_params(n):
    """Pick (list size r, steps t, rounds K) minimising expected tries incl. retries."""
    with _lock:
        if n not in _walk:
            N = 1 << n
            base = best_walk_over_r(N)
            best = (base["queries"] / base["p"], base)
            for r in range(max(2, int(base["r"] * 0.6)), min(N // 2, int(base["r"] * 1.8) + 2) + 1):
                for target in (0.67, 0.8, 0.9, 0.97):
                    b = best_walk(N, r, target)
                    if b and b[0] / b[3] < best[0]:
                        best = (b[0] / b[3], dict(queries=b[0], t=b[1], K=b[2], p=b[3], r=r))
            w = best[1]
            _walk[n] = (w, build_lumped(N, w["r"]))
        return _walk[n]


threading.Thread(target=lambda: [walk_params(n) for n in SIZES], daemon=True).start()


@lru_cache(maxsize=None)
def bht_params(N):
    """(list size r, search steps j, success chance p) minimising expected tries incl. retries."""
    best = None
    for r in range(2, min(N // 2, int(6 * N ** (1 / 3)) + 2) + 1):
        th = math.asin(math.sqrt(r / N))
        for j in range(0, int(math.pi / (4 * th)) + 2):
            p = math.sin((2 * j + 1) * th) ** 2
            if p > 0 and (best is None or (r + j + 1) / p < best[0]):
                best = ((r + j + 1) / p, r, j, p)
    return dict(r=best[1], j=best[2], p=best[3])


# ---------------------------------------------------------------- the three methods
def run_classical(H, rng):
    seen, used, q = {}, set(), 0
    t0 = time.perf_counter()
    while True:
        x = rng.randrange(H.N)
        if x in used:
            continue
        used.add(x)
        q += 1
        if H.d[x] in seen:
            pair = (seen[H.d[x]], x)
            break
        seen[H.d[x]] = x
    return pair, [dict(tries=q, ok=True, note="")], time.perf_counter() - t0


def run_bht(H, rng):
    bp = bht_params(H.N)
    r, j = bp["r"], bp["j"]
    attempts, t0 = [], time.perf_counter()
    for _ in range(10):
        table = rng.sample(range(H.N), r)
        seen, hit, tries = {}, None, 0
        for x in table:
            tries += 1
            if H.d[x] in seen:
                hit = (seen[H.d[x]], x)
                break
            seen[H.d[x]] = x
        if hit:
            attempts.append(dict(tries=tries, ok=True, note="two table entries already matched"))
            return hit, attempts, time.perf_counter() - t0
        marked = np.zeros(H.N, bool)
        marked[H.partner[table]] = True
        psi = np.full(H.N, 1 / math.sqrt(H.N))
        for _ in range(j):  # Grover: flip marked, reflect about the mean
            psi[marked] *= -1
            psi = 2 * psi.mean() - psi
        p = float((psi[marked] ** 2).sum())
        tries = r + j + 1
        if rng.random() < p:
            x = int(rng.choice(np.flatnonzero(marked)))
            attempts.append(dict(tries=tries, ok=True, note=f"search chance was {p:.0%}"))
            return (x, int(H.partner[x])), attempts, time.perf_counter() - t0
        attempts.append(dict(tries=tries, ok=False, note=f"search chance was {p:.0%}"))
    raise RuntimeError("BHT failed 10 times")


def run_walk(H, rng):
    w, L = walk_params(H.n)
    r, t, K = w["r"], w["t"], w["K"]
    p = w["p"]
    per, attempts, t0 = r + 2 * t * K + 2, [], time.perf_counter()
    for _ in range(10):
        if rng.random() < p:
            x = rng.randrange(H.N)
            attempts.append(dict(tries=per, ok=True, note=f"walk chance was {p:.0%}"))
            return (x, int(H.partner[x])), attempts, time.perf_counter() - t0
        attempts.append(dict(tries=per, ok=False, note=f"walk chance was {p:.0%}"))
    raise RuntimeError("walk failed 10 times")


META = {
    "classical": ("Normal guessing", "Try numbers one by one until two match"),
    "bht": ("Quantum search", "Keep a small list, then let a quantum search spot a match"),
    "walk": ("Quantum wander", "A quantum random walk over possible lists"),
}


@app.get("/api/run")
def api_run():
    pw = (request.args.get("pw") or "")[:64]
    n = int(request.args.get("n", 16))
    if not pw or n not in SIZES:
        return jsonify(error="need a password and a valid size"), 400
    H = get_hasher(pw, n)
    x = fold(pw, n)
    tr = H.trace(x)
    rng = random.Random()
    methods = []
    for mid, fn in (("classical", run_classical), ("bht", run_bht), ("walk", run_walk)):
        pair, attempts, cpu = fn(H, rng)
        a, b = pair
        assert a != b and H.d[a] == H.d[b], "bad twin"
        methods.append(dict(id=mid, name=META[mid][0], sub=META[mid][1], attempts=attempts,
                            total=sum(t["tries"] for t in attempts), cpu_ms=cpu * 1000,
                            pair=[int(a), int(b)], digest=H.d[a]))
    expected = classical_cost(H.N)[1]
    return jsonify(
        n=n, N=H.N, pw=pw, x=x, codes=list(pw.encode()),
        hash=dict(A=H.A, B=H.B, C=H.C, s=H.s, trace=tr),
        rate=max(12, expected / 5), methods=methods)


# ---------------------------------------------------------------- program size
@lru_cache(maxsize=None)
def gate_stats(n):
    from qiskit import QuantumCircuit, transpile

    basis = ["cx", "rz", "sx", "x"]
    zeros = [i for i in range(n) if i % 2 == 0]
    o = QuantumCircuit(n)
    o.x(zeros); o.h(n - 1); o.mcx(list(range(n - 1)), n - 1); o.h(n - 1); o.x(zeros)
    d = QuantumCircuit(n)
    d.h(range(n)); d.x(range(n)); d.h(n - 1); d.mcx(list(range(n - 1)), n - 1); d.h(n - 1)
    d.x(range(n)); d.h(range(n))
    out = {}
    for name, qc in (("o", o), ("d", d)):
        tq = transpile(qc, basis_gates=basis, optimization_level=0)
        ops = tq.count_ops()
        out[name] = dict(gates=int(sum(ops.values())), cx=int(ops.get("cx", 0)), depth=int(tq.depth()))
    return out


@app.get("/api/details")
def api_details():
    n = int(request.args.get("n", 16))
    N = 1 << n
    method = request.args.get("method")
    if method == "bht":
        bp = bht_params(N)
        r, j = bp["r"], bp["j"]
        g = gate_stats(n)
        per = r * g["o"]["gates"] + g["d"]["gates"]
        return jsonify(r=r, j=j, k=r, p=bp["p"], qubits=n, per_step_gates=per,
                       total_gates=j * per + n,
                       total_cx=j * (r * g["o"]["cx"] + g["d"]["cx"]),
                       total_depth=j * (r * g["o"]["depth"] + g["d"]["depth"]),
                       oracle_gates=g["o"]["gates"], diffusion_gates=g["d"]["gates"])
    if method == "walk":
        w, _ = walk_params(n)
        r = w["r"]
        return jsonify(r=r, t=w["t"], K=w["K"], p=w["p"], steps=w["t"] * w["K"],
                       qubits_est=r * (2 * n - 1) + 2 * n, queries=w["queries"])
    return jsonify(expected=classical_cost(N)[1], typical=math.sqrt(N))


# ---------------------------------------------------------------- "what if it makes mistakes"
_noise_cache = {}
RATES_Q = [0, 0.0005, 0.001, 0.002, 0.005]
RATES_W = [0, 0.005, 0.01, 0.02, 0.05]


def noise_bht(pw, n):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from bht_collision import grover_circuit

    ns = min(n, 6)  # full-size noisy circuits would take far too long to simulate
    H = get_hasher(pw, ns)
    bp = bht_params(1 << ns)
    r, j0 = bp["r"], bp["j"]
    rng = random.Random(pw)
    table = rng.sample(range(H.N), r)
    while len({H.d[x] for x in table}) < r:
        table = rng.sample(range(H.N), r)
    marked = sorted(int(H.partner[x]) for x in table)
    th = math.asin(math.sqrt(len(marked) / H.N))
    J = j0 + 3
    basis = ["cx", "rz", "sx", "x"]
    circs = [transpile(grover_circuit(ns, marked, j), basis_gates=basis, optimization_level=0)
             for j in range(J + 1)]
    noisy = {}
    for rate in RATES_Q:
        nm = NoiseModel()
        if rate:
            nm.add_all_qubit_quantum_error(depolarizing_error(rate, 2), ["cx"])
            nm.add_all_qubit_quantum_error(depolarizing_error(rate / 10, 1), ["sx", "x"])
        sim = AerSimulator(noise_model=nm, basis_gates=basis + ["rz"] if rate else None)
        out = []
        for j, qc in enumerate(circs):
            shots = 600
            counts = sim.run(qc, shots=shots, seed_simulator=7 + j).result().get_counts()
            out.append(sum(c for k, c in counts.items() if int(k, 2) in marked) / shots)
        noisy[str(rate)] = out
    return dict(kind="bht", n_sim=ns, r=r, best=j0, steps=list(range(J + 1)),
                ideal=[math.sin((2 * j + 1) * th) ** 2 for j in range(J + 1)],
                rates=RATES_Q, noisy=noisy,
                cx_best=int(circs[j0].count_ops().get("cx", 0)))


def noise_walk(n):
    w, L = walk_params(n)
    t, K = w["t"], w["K"]
    W, O = L["D2"] @ L["D1"], np.diag(np.where(L["marked"], -1.0, 1.0))
    mk, psi0 = L["marked"], L["psi0"]
    scramble = np.diag(psi0 ** 2)
    curves = {}
    for eps in RATES_W:
        rho, out = np.outer(psi0, psi0), []
        out.append(float(np.trace(rho[np.ix_(mk, mk)])))
        for _ in range(K):
            rho = O @ rho @ O
            for _ in range(t):
                rho = W @ rho @ W.T
                rho = (1 - eps) * rho + eps * scramble
            out.append(float(np.trace(rho[np.ix_(mk, mk)])))
        curves[str(eps)] = out
    return dict(kind="walk", n_sim=n, rounds=list(range(K + 1)), rates=RATES_W, noisy=curves,
                t=t, K=K)


@app.get("/api/noise")
def api_noise():
    pw = (request.args.get("pw") or "")[:64]
    n = int(request.args.get("n", 16))
    method = request.args.get("method")
    key = (method, pw, n if method == "walk" else min(n, 6))
    if key not in _noise_cache:
        _noise_cache[key] = noise_bht(pw, n) if method == "bht" else noise_walk(n)
    return jsonify(_noise_cache[key])


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
