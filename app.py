"""Twin Finder - Flask GUI for the collision-finding project.

Run:  python app.py   then open http://127.0.0.1:5000

Your password picks the scrambler (hash) constants and is scrambled by it.
Three methods then race to find two different numbers with the same fingerprint:
  classical guessing, quantum search (BHT / Grover), quantum walk (Ambainis-style).
Quantum parts are simulations on a normal computer (see README notes in the UI).
"""
import hashlib
import json
import math
import queue
import random
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

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


def prep(pw, n, kind):
    """Turn the typed password into the number x that gets scrambled."""
    if not pw or n not in SIZES:
        return None, "type a password and pick a valid size"
    if kind == "number":
        if not pw.isdigit() or int(pw) >= (1 << n):
            return None, f"Number mode: digits only, and below {1 << n:,} for {n} bits"
        return int(pw), None
    return fold(pw, n), None


def codes_for(pw, kind):
    return [int(c) for c in pw] if kind == "number" else list(pw.encode())


def find_text_twin(t, n, pw, rng):
    """A readable string (a-z, 0-9) that folds to number t, so it scrambles like the password."""
    alpha = [ord(c) for c in "abcdefghijklmnopqrstuvwxyz0123456789"]
    M = 1 << n
    for _ in range(300000):
        pre = [rng.choice(alpha) for _ in range(5)]
        v = 0
        for b in pre:
            v = (v * 31 + b) % M
        ok = [c for c in alpha if (v * 31 + c) % M == t]
        if ok:
            s = "".join(map(chr, pre + [rng.choice(ok)]))
            if s != pw:
                return s
    return None


@app.get("/api/run")
def api_run():
    pw = (request.args.get("pw") or "")[:64]
    n = int(request.args.get("n", 16))
    kind = request.args.get("kind", "text")
    x, err = prep(pw, n, kind)
    if err:
        return jsonify(error=err), 400
    H = get_hasher(pw, n)
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
        mode="race", kind=kind, n=n, N=H.N, pw=pw, x=x, codes=codes_for(pw, kind),
        hash=dict(A=H.A, B=H.B, C=H.C, s=H.s, trace=tr),
        rate=max(12, expected / 5), methods=methods)


# ---------------------------------------------------------------- crack mode (targeted twin)
def crack_classical(H, x, rng):
    target, order, t0 = H.d[x], list(range(H.N)), time.perf_counter()
    rng.shuffle(order)
    tries = 0
    for y in order:
        if y == x:
            continue
        tries += 1
        if H.d[y] == target:
            return y, [dict(tries=tries, ok=True, note="")], time.perf_counter() - t0
    raise RuntimeError("no twin (hash is not 2-to-1?)")


def crack_grover(H, x, rng):
    N = H.N
    th = math.asin(1 / math.sqrt(N))
    j = max(0, round(math.pi / (4 * th) - 0.5))
    marked = np.zeros(N, bool)
    marked[H.partner[x]] = True
    t0, attempts = time.perf_counter(), []
    for _ in range(6):
        psi = np.full(N, 1 / math.sqrt(N))
        for _ in range(j):
            psi[marked] *= -1
            psi = 2 * psi.mean() - psi
        p = float((psi[marked] ** 2).sum())
        ok = rng.random() < p
        attempts.append(dict(tries=j + 1, ok=ok, note=f"search chance was {p:.1%}"))
        if ok:
            return int(H.partner[x]), attempts, time.perf_counter() - t0
    raise RuntimeError("quantum search failed 6 times")


@app.get("/api/crack")
def api_crack():
    pw = (request.args.get("pw") or "")[:64]
    n = int(request.args.get("n", 16))
    kind = request.args.get("kind", "text")
    x, err = prep(pw, n, kind)
    if err:
        return jsonify(error=err), 400
    H = get_hasher(pw, n)
    rng = random.Random()
    methods = []
    for mid, fn, nm, sub in (
            ("classical", crack_classical, "Normal guessing", "Try numbers one by one until one has your fingerprint"),
            ("grover", crack_grover, "Quantum search", "Let a quantum search home in on the one number that matches")):
        t, attempts, cpu = fn(H, x, rng)
        assert t != x and H.d[t] == H.d[x], "bad twin"
        methods.append(dict(id=mid, name=nm, sub=sub, attempts=attempts, total=sum(a["tries"] for a in attempts),
                            cpu_ms=cpu * 1000, twin=int(t), digest=H.d[x]))
    twin = methods[0]["twin"]
    text_twin = find_text_twin(twin, n, pw, rng) if kind == "text" else None
    return jsonify(mode="crack", kind=kind, n=n, N=H.N, pw=pw, x=x, codes=codes_for(pw, kind),
                   hash=dict(A=H.A, B=H.B, C=H.C, s=H.s, trace=H.trace(x)),
                   rate=max(12, (H.N / 2) / 5), methods=methods, twin=twin, text_twin=text_twin)


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
    if method == "grover":
        th = math.asin(1 / math.sqrt(N))
        j = max(0, round(math.pi / (4 * th) - 0.5))
        g = gate_stats(n)
        per = g["o"]["gates"] + g["d"]["gates"]
        return jsonify(j=j, p=math.sin((2 * j + 1) * th) ** 2, qubits=n, per_step_gates=per,
                       total_gates=j * per + n, total_cx=j * (g["o"]["cx"] + g["d"]["cx"]),
                       classical=N / 2)
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


def _noisy_grover(ns, marked, j0, J):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from bht_collision import grover_circuit

    th = math.asin(math.sqrt(len(marked) / (1 << ns)))
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
    return dict(kind="bht", n_sim=ns, best=j0, steps=list(range(J + 1)),
                ideal=[math.sin((2 * j + 1) * th) ** 2 for j in range(J + 1)],
                rates=RATES_Q, noisy=noisy, cx_best=int(circs[j0].count_ops().get("cx", 0)))


def noise_bht(pw, n):
    ns = min(n, 6)  # full-size noisy circuits would take far too long to simulate
    H = get_hasher(pw, ns)
    bp = bht_params(1 << ns)
    r, j0 = bp["r"], bp["j"]
    rng = random.Random(pw)
    table = rng.sample(range(H.N), r)
    while len({H.d[x] for x in table}) < r:
        table = rng.sample(range(H.N), r)
    marked = sorted(int(H.partner[x]) for x in table)
    out = _noisy_grover(ns, marked, j0, j0 + 3)
    out["r"] = r
    return out


def noise_crack(pw, n, kind):
    ns = min(n, 6)
    H = get_hasher(pw, ns)
    x = int(pw) % (1 << ns) if kind == "number" else fold(pw, ns)
    th = math.asin(1 / math.sqrt(1 << ns))
    j0 = max(0, round(math.pi / (4 * th) - 0.5))
    return _noisy_grover(ns, [int(H.partner[x])], j0, j0 + 3)


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
    kind = request.args.get("kind", "text")
    key = (method, pw, kind, n if method == "walk" else min(n, 6))
    if key not in _noise_cache:
        _noise_cache[key] = (noise_bht(pw, n) if method == "bht" else
                             noise_crack(pw, n, kind) if method == "grover" else noise_walk(n))
    return jsonify(_noise_cache[key])



# ================================================================ LIVE RACE (server-sent events)
# Each method runs in its own thread and really does its work: it computes fingerprints,
# runs search steps, and reports its try count as it goes. Every hash call / search step is
# held to the same tries-per-second rate so a human can watch (the work itself is real).
class Abort(Exception):
    pass


class Pacer:
    def __init__(self, mid, rate, emit, stop):
        self.mid, self.rate, self.emit, self.stop = mid, rate, emit, stop
        self.t0 = time.perf_counter()
        self.tries, self.slept, self.last = 0, 0.0, 0.0
        self.attempt, self.att0, self.planned = 1, 0, 1
        self.phase, self.cur, self.p, self.log = "", None, None, []

    def new_attempt(self, planned, phase):
        self.att0, self.planned, self.phase, self.cur, self.p = self.tries, planned, phase, None, None
        self.push(force=True)

    def end_attempt(self, ok, note=""):
        self.log.append(dict(tries=self.tries - self.att0, ok=ok, note=note))
        self.push(force=True)
        self.emit(dict(type="attempt_end", id=self.mid, ok=ok, att=self.attempt, tries=self.tries))
        self.attempt += 1

    def tick(self, k=1):
        if self.stop.is_set():
            raise Abort()
        self.tries += k
        ahead = self.t0 + self.tries / self.rate - time.perf_counter()
        if ahead > 0.002:
            time.sleep(ahead)
            self.slept += ahead
        self.push()

    def push(self, force=False):
        now = time.perf_counter()
        if force or now - self.last > 0.035:
            self.last = now
            self.emit(dict(type="progress", id=self.mid, tries=self.tries, att=self.attempt,
                           att_tries=self.tries - self.att0, planned=self.planned,
                           phase=self.phase, cur=self.cur, p=self.p))

    def elapsed(self):
        return time.perf_counter() - self.t0

    def cpu_ms(self):
        return (self.elapsed() - self.slept) * 1000


class AerGrover:
    """Real Qiskit circuit, simulated on Aer: state after i search steps."""

    def __init__(self, n, marked):
        import qiskit_aer  # noqa: F401  (registers save_statevector)
        from qiskit_aer import AerSimulator
        self.n, self.marked, self.sim = n, marked, AerSimulator(method="statevector")

    def probs(self, iters):
        from bht_collision import grover_circuit
        qc = grover_circuit(self.n, self.marked, iters, measure=False)
        qc.save_statevector()
        sv = self.sim.run(qc).result().get_statevector()
        return np.abs(np.asarray(sv)) ** 2


def grover_live(H, marked, j, pc, rng, engine):
    """j live search steps (each one oracle call), then a measurement. Returns found x or None."""
    N, mk = H.N, marked
    psi = np.full(N, 1 / math.sqrt(N))
    aer = AerGrover(H.n, [int(m) for m in np.flatnonzero(mk)]) if engine == "qiskit" else None
    probs = psi ** 2
    for i in range(1, j + 1):
        if aer:
            probs = aer.probs(i)
        else:
            psi[mk] *= -1
            psi = 2 * psi.mean() - psi
            probs = psi ** 2
        pc.p, pc.cur = float(probs[mk].sum()), None
        pc.phase = f"Quantum search step {i} of {j}"
        pc.tick()
    pc.phase, pc.cur = "Looking at the result", None
    pc.tick()
    probs = probs / probs.sum()
    x = int(np.random.default_rng(rng.randrange(1 << 32)).choice(N, p=probs))
    return x if mk[x] else None


def live_classical_race(H, rng, pc, expected):
    pc.new_attempt(expected, "Guessing numbers and writing down fingerprints")
    seen, used = {}, set()
    while True:
        y = rng.randrange(H.N)
        if y in used:
            continue
        used.add(y)
        d = H.d[y]
        pc.cur = [y, d]
        pc.tick()
        if d in seen:
            pc.end_attempt(True)
            return (seen[d], y)
        seen[d] = y


def live_bht(H, rng, pc, engine):
    bp = bht_params(H.N)
    r, j = bp["r"], bp["j"]
    for _ in range(10):
        pc.new_attempt(r + j + 1, "Writing down a short list")
        table, seen, hit = rng.sample(range(H.N), r), {}, None
        for x in table:
            pc.cur = [x, H.d[x]]
            pc.tick()
            if H.d[x] in seen:
                hit = (seen[H.d[x]], x)
                break
            seen[H.d[x]] = x
        if hit:
            pc.end_attempt(True, "two list entries already matched")
            return hit
        marked = np.zeros(H.N, bool)
        marked[H.partner[table]] = True
        x = grover_live(H, marked, j, pc, rng, engine)
        pc.end_attempt(x is not None)
        if x is not None:
            return (x, int(H.partner[x]))
    raise RuntimeError("quantum search failed 10 times")


def live_walk(H, rng, pc):
    w, L = walk_params(H.n)
    r, t, K = w["r"], w["t"], w["K"]
    D1, D2, mk, psi0 = L["D1"], L["D2"], L["marked"], L["psi0"]
    O = np.where(mk, -1.0, 1.0)
    for _ in range(10):
        pc.new_attempt(r + 2 * t * K + 2, "Building a random list")
        for _ in range(r):
            x = rng.randrange(H.N)
            pc.cur = [x, H.d[x]]
            pc.tick()
        v = psi0.copy()
        pc.cur, pc.p = None, float((v[mk] ** 2).sum())
        for k in range(1, K + 1):
            v = v * O  # flag lists that contain a twin
            for s in range(1, t + 1):
                v = D2 @ (D1 @ v)  # one wandering step (swap one number in the list)
                pc.p = float((v[mk] ** 2).sum())
                pc.phase = f"Wandering: round {k} of {K}, step {s} of {t}"
                pc.tick(2)
        pc.phase = "Looking at the list"
        pc.tick(2)
        ok = rng.random() < float((v[mk] ** 2).sum())
        pc.end_attempt(ok)
        if ok:
            x = rng.randrange(H.N)
            return (x, int(H.partner[x]))
    raise RuntimeError("walk failed 10 times")


def live_crack_classical(H, x, rng, pc, expected):
    pc.new_attempt(expected, "Trying numbers, checking each fingerprint")
    order = list(range(H.N))
    rng.shuffle(order)
    for y in order:
        if y == x:
            continue
        pc.cur = [y, H.d[y]]
        pc.tick()
        if H.d[y] == H.d[x]:
            pc.end_attempt(True)
            return y
    raise RuntimeError("no twin found")


def live_crack_grover(H, x, rng, pc, engine):
    th = math.asin(1 / math.sqrt(H.N))
    j = max(0, round(math.pi / (4 * th) - 0.5))
    marked = np.zeros(H.N, bool)
    marked[H.partner[x]] = True
    for _ in range(10):
        pc.new_attempt(j + 1, "Quantum search")
        found = grover_live(H, marked, j, pc, rng, engine)
        pc.end_attempt(found is not None)
        if found is not None:
            return found
    raise RuntimeError("quantum search failed 10 times")


def _sse(ev):
    return f"data: {json.dumps(ev)}\n\n"


def _meta(mode):
    if mode == "crack":
        return [dict(id="classical", name="Normal guessing", sub="Try numbers one by one until one has your fingerprint"),
                dict(id="grover", name="Quantum search", sub="Let a quantum search home in on the one number that matches")]
    return [dict(id=k, name=META[k][0], sub=META[k][1]) for k in ("classical", "bht", "walk")]


def _rate(N, mode, speed=1.0):
    return max(12, (N / 2 if mode == "crack" else classical_cost(N)[1]) / 5) * speed


@app.get("/api/prepare")
def api_prepare():
    pw = (request.args.get("pw") or "")[:64]
    n = int(request.args.get("n", 16))
    kind, mode = request.args.get("kind", "text"), request.args.get("mode", "race")
    speed = min(1.0, max(0.02, float(request.args.get("speed", 1))))
    x, err = prep(pw, n, kind)
    if err:
        return jsonify(error=err), 400
    H = get_hasher(pw, n)
    if mode == "race":
        walk_params(n)  # warm the cache so the live race starts instantly
    return jsonify(mode=mode, kind=kind, n=n, N=H.N, pw=pw, x=x, codes=codes_for(pw, kind),
                   hash=dict(A=H.A, B=H.B, C=H.C, s=H.s, trace=H.trace(x)),
                   rate=_rate(H.N, mode, speed), methods=_meta(mode))


@app.get("/api/stream")
def api_stream():
    pw = (request.args.get("pw") or "")[:64]
    n = int(request.args.get("n", 16))
    kind, mode = request.args.get("kind", "text"), request.args.get("mode", "race")
    x, err = prep(pw, n, kind)
    if err:
        return jsonify(error=err), 400
    H = get_hasher(pw, n)
    engine = "qiskit" if request.args.get("engine") == "qiskit" and n <= 8 else "numpy"
    speed = min(1.0, max(0.02, float(request.args.get("speed", 1))))
    rate, stop, q = _rate(H.N, mode, speed), threading.Event(), queue.Queue()
    expected = H.N / 2 if mode == "crack" else classical_cost(H.N)[1]
    twin = int(H.partner[x]) if mode == "crack" else None
    text_twin = find_text_twin(twin, n, pw, random.Random()) if mode == "crack" and kind == "text" else None
    if mode == "race":
        walk_params(n)

    def work(mid):
        pc, rng = Pacer(mid, rate, q.put, stop), random.Random()
        try:
            if mode == "race":
                a, b = {"classical": lambda: live_classical_race(H, rng, pc, expected),
                        "bht": lambda: live_bht(H, rng, pc, engine),
                        "walk": lambda: live_walk(H, rng, pc)}[mid]()
                assert a != b and H.d[a] == H.d[b], "bad twin"
                res = dict(pair=[int(a), int(b)], digest=H.d[a])
            else:
                t = live_crack_classical(H, x, rng, pc, expected) if mid == "classical" \
                    else live_crack_grover(H, x, rng, pc, engine)
                assert t != x and H.d[t] == H.d[x], "bad twin"
                res = dict(twin=int(t), digest=H.d[x], text_twin=text_twin)
            q.put(dict(type="done", id=mid, total=pc.tries, attempts=pc.log, elapsed=pc.elapsed(),
                       cpu_ms=pc.cpu_ms(), **res))
        except Abort:
            pass
        except Exception as e:  # report instead of hanging the page
            q.put(dict(type="error", id=mid, message=str(e)))

    def gen():
        threads = [threading.Thread(target=work, args=(m["id"],), daemon=True) for m in _meta(mode)]
        try:
            yield _sse(dict(type="start", engine=engine, rate=rate))
            for t in threads:
                t.start()
            while True:
                try:
                    yield _sse(q.get(timeout=0.25))
                except queue.Empty:
                    if not any(t.is_alive() for t in threads):
                        break
            while not q.empty():
                yield _sse(q.get_nowait())
            yield _sse(dict(type="end"))
        finally:
            stop.set()

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    # works whether index.html sits in static/ or directly next to app.py
    base = Path(__file__).resolve().parent
    for d in (base / "static", base):
        if (d / "index.html").exists():
            return send_from_directory(d, "index.html")
    return "index.html not found. Put it next to app.py or in a static/ folder.", 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
