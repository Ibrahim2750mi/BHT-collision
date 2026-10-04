"""Quantum-walk collision finder (Ambainis-style walk on the Johnson graph).

State space: |S, y>, S = r-subset of inputs (the "table"), y not in S.
One walk step W = D2 . D1:
  D1 = reflection about uniform-over-y (y not in S), for each S
  D2 = reflection about uniform-over-x (x in T = S + y), for each T
  (then S' = T - x is the new table) -- this is Szegedy/Ambainis on J(N, r).
Search round = phase-flip all |S, y> with S containing a collision, then t steps of W.

The toy hash is exactly 2-to-1, so collision pairs form a perfect matching on the
N inputs. That symmetry (permute pairs / swap inside a pair) lets us simulate the walk
exactly on a tiny "orbit" subspace of dimension ~r instead of C(N, r)*(N-r).
`brute_force_check` verifies the reduction against the full Hilbert space (N=16).

Query accounting: setup r, each walk step 2 (add h(y), erase h(x)), final read-out 2.
"""
import math
from itertools import combinations

import numpy as np


def _lc(n, k):  # log binomial
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def build_lumped(N, r):
    """Return dict with D1, D2, oracle sign, psi0, marked mask on the orbit basis."""
    P = N // 2  # number of collision pairs
    orbits, idx = [], {}
    for a in range(r // 2 + 1):
        b = r - 2 * a
        if a + b > P:
            continue
        for t, cnt in (("c", b), ("f", N - r - b)):
            if cnt > 0:
                idx[(a, t)] = len(orbits)
                orbits.append((a, t, cnt))
    d = len(orbits)
    Pi1 = np.zeros((d, d))
    for a in {o[0] for o in orbits}:
        v = np.zeros(d)
        for (aa, t), i in idx.items():
            if aa == a:
                v[i] = math.sqrt(orbits[i][2] / (N - r))
        Pi1 += np.outer(v, v)
    Pi2 = np.zeros((d, d))
    for aT in range((r + 1) // 2 + 1):
        bT = r + 1 - 2 * aT
        v = np.zeros(d)
        if aT >= 1 and (aT - 1, "c") in idx:
            v[idx[(aT - 1, "c")]] = math.sqrt(2 * aT / (r + 1))
        if bT > 0 and (aT, "f") in idx:
            v[idx[(aT, "f")]] = math.sqrt(bT / (r + 1))
        if v.any():
            assert abs(v @ v - 1) < 1e-9, "Pi2 vector not normalised"
            Pi2 += np.outer(v, v)
    I = np.eye(d)
    D1, D2 = 2 * Pi1 - I, 2 * Pi2 - I
    marked = np.array([o[0] >= 1 for o in orbits])
    logp = np.array([
        _lc(P, a) + _lc(P - a, r - 2 * a) + (r - 2 * a) * math.log(2) + math.log(cnt)
        - _lc(N, r) - math.log(N - r)
        for a, t, cnt in orbits
    ])
    psi0 = np.sqrt(np.exp(logp))
    assert abs(psi0 @ psi0 - 1) < 1e-8, "initial state not normalised"
    return dict(D1=D1, D2=D2, marked=marked, psi0=psi0, d=d)


def run_rounds(L, t, K):
    """Success probability after k=1..K rounds of (oracle, W^t). Returns list."""
    W = L["D2"] @ L["D1"]
    Wt = np.linalg.matrix_power(W, t)
    O = np.where(L["marked"], -1.0, 1.0)
    M = Wt * O[None, :]
    v, out = L["psi0"].copy(), []
    for _ in range(K):
        v = M @ v
        out.append(float((v[L["marked"]] ** 2).sum()))
    return out


def best_walk(N, r, target=2 / 3):
    """Cheapest (t, K) reaching success >= target. Returns (queries, t, K, p) or None."""
    L = build_lumped(N, r)
    W = L["D2"] @ L["D1"]
    O = np.where(L["marked"], -1.0, 1.0)
    tmax = int(4 * math.sqrt(r)) + 2
    Kcap = int(4 * math.sqrt(2 * N) / r) + 10
    best, Wt = None, np.eye(L["d"])
    for t in range(1, tmax + 1):
        Wt = Wt @ W
        M = Wt * O[None, :]
        v = L["psi0"].copy()
        for K in range(1, Kcap + 1):
            v = M @ v
            cost = r + 2 * t * K + 2
            if best and cost >= best[0]:
                break
            p = float((v[L["marked"]] ** 2).sum())
            if p >= target:
                best = (cost, t, K, p)
                break
    return best


def best_walk_over_r(N, target=2 / 3):
    rs = range(2, min(N // 2, int(5 * N ** (1 / 3)) + 2) + 1)
    res = [(best_walk(N, r, target), r) for r in rs]
    res = [(b, r) for b, r in res if b]
    b, r = min(res, key=lambda z: z[0][0])
    return dict(queries=b[0], t=b[1], K=b[2], p=b[3], r=r)


def brute_force_check(h, n, r, t=2, K=4):
    """Full-Hilbert-space walk vs lumped walk. Returns max abs difference."""
    N = 1 << n
    subs = list(combinations(range(N), r))
    sid = {s: i for i, s in enumerate(subs)}
    tid, states = {}, []
    for s in subs:
        for y in range(N):
            if y not in s:
                T = tuple(sorted(s + (y,)))
                states.append((sid[s], tid.setdefault(T, len(tid))))
    g1 = np.array([s for s, _ in states])
    g2 = np.array([T for _, T in states])
    marked_S = np.array([len({h(x) for x in s}) < r for s in subs])
    marked = marked_S[g1]

    def refl(v, g, ng):
        mean = np.bincount(g, v, ng) / np.bincount(g, minlength=ng)
        return 2 * mean[g] - v

    v = np.ones(len(states)) / math.sqrt(len(states))
    brute = []
    for _ in range(K):
        v = np.where(marked, -v, v)
        for _ in range(t):
            v = refl(refl(v, g1, len(subs)), g2, len(tid))
        brute.append(float((v[marked] ** 2).sum()))
    lumped = run_rounds(build_lumped(N, r), t, K)
    return max(abs(a - b) for a, b in zip(brute, lumped)), brute, lumped


if __name__ == "__main__":
    from bht_collision import make_hash

    for n, r, t in ((4, 3, 1), (4, 3, 2), (4, 4, 2), (3, 2, 1)):
        h = make_hash(n)
        diff, bf, lm = brute_force_check(h, n, r, t=t)
        print(f"N={1 << n:>2} r={r} t={t}: max|brute-lumped| = {diff:.2e}  "
              f"(p after 4 rounds: {bf[-1]:.4f} vs {lm[-1]:.4f})")
