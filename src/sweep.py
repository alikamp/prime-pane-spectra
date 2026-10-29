#!/usr/bin/env python3
"""
sweep.py -- one segmented pass, all checkpoints.

Measures C_s(N) = #{ p <= N : p and p+s both prime } for every even shift
s <= SMAX, at ~30 log-spaced checkpoints N, in a SINGLE pass to NMAX.

Convention: p <= N, and p+s may exceed N. Normalisation is then exactly
Li2(N) with no truncation correction.

The question: D(N) = <log( C_s(N) / (S(s) Li2(N)) )> - log(2 C2)
is robustly negative and shrinking. Is it a deterministic correction to the
leading Hardy-Littlewood form, or the fluctuating HL error term? Three
checkpoints could not tell. Thirty can.

Self-contained -- runs here or on Colab. Memory is O(segment), not O(N),
so NMAX is limited by time, not RAM.

    python3 sweep.py --nmax 1e9
    python3 sweep.py --nmax 1e10 --seg 5e7      # Colab
"""

import argparse, json, time
import numpy as np
from scipy.integrate import quad

C2 = 0.66016181584686957393
LOG2C2 = np.log(2 * C2)
SMAX = 1200


def base_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return np.flatnonzero(s)


def smallest_factor(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i:: i] = np.where(spf[i:: i] == 0, i, spf[i:: i])
    return spf


def odd_factors(s, spf):
    out = []
    while s > 1:
        p = int(spf[s])
        if p > 2:
            out.append(p)
        while s % p == 0:
            s //= p
    return out


def seg_sieve(lo, hi, bp):
    """Primality of [lo, hi) as a bool array."""
    seg = np.ones(hi - lo, dtype=bool)
    if lo < 2:
        seg[: max(0, min(2, hi) - lo)] = False
    for p in bp:
        if p * p >= hi:
            break
        start = max(p * p, ((lo + p - 1) // p) * p)
        if start >= hi:
            continue
        seg[start - lo:: p] = False
    return seg


def li2(x):
    return quad(lambda t: 1.0 / np.log(t) ** 2, 2, x, limit=500)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=float, default=1e9)
    ap.add_argument("--seg", type=float, default=2e7)
    ap.add_argument("--out", default="sweep_results.json")
    a = ap.parse_args()
    NMAX, M = int(a.nmax), int(a.seg)

    shifts = np.arange(2, SMAX + 1, 2)
    spf = smallest_factor(SMAX)
    hl = np.array([sum(np.log((p - 1) / (p - 2))
                       for p in odd_factors(int(s), spf)) for s in shifts])

    checks = sorted(set(int(x) for x in
                        np.logspace(6, np.log10(NMAX), 30)))
    bounds = sorted(set(checks) | set(range(0, NMAX + 1, M)) | {NMAX})
    bounds = [b for b in bounds if b <= NMAX]
    if bounds[0] != 0:
        bounds = [0] + bounds

    print(f"NMAX = {NMAX:,}   segment = {M:,}   "
          f"{len(shifts)} shifts   {len(checks)} checkpoints")
    bp = base_primes(int(NMAX ** 0.5) + 1)
    print(f"base primes to {int(NMAX**0.5)+1:,}: {len(bp):,}\n")

    running = np.zeros(len(shifts), dtype=np.int64)
    results, t0 = [], time.time()

    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        seg = seg_sieve(lo, min(hi + SMAX, NMAX + SMAX + 2), bp)
        L = hi - lo
        for j, s in enumerate(shifts):
            running[j] += np.count_nonzero(seg[:L] & seg[s:s + L])
        del seg
        if hi in checks:
            Lk = li2(hi)
            d = np.log(running / Lk) - hl
            w = running / running.sum()
            direct = float((w * d).sum())
            ratio = float(np.mean(running / (np.exp(hl) * Lk)))
            sd = float(np.sqrt((w * (d - direct) ** 2).sum() / len(d)))
            results.append(dict(N=hi, pi2=int(running[0]), li2=Lk,
                                direct=direct, D=direct - LOG2C2,
                                ratio=ratio, rel=ratio / (2 * C2) - 1,
                                scatter=sd))
            print(f"  N={hi:>13,}  pi_2={running[0]:>12,}  "
                  f"D={direct-LOG2C2:+.6f}  rel={100*(ratio/(2*C2)-1):+.4f}%  "
                  f"[{time.time()-t0:6.0f}s]")

    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}  ({time.time()-t0:.0f}s total)")

    # ---------------- analysis ----------------
    N = np.array([r["N"] for r in results], float)
    D = np.array([r["D"] for r in results])
    rel = np.array([r["rel"] for r in results])
    Ln = np.log(N)

    print(f"\n=== is D(N) deterministic? ===")
    print(f"sign changes in D: {int(np.sum(np.diff(np.sign(D)) != 0))}"
          f"   monotone: {bool(np.all(np.diff(D) > 0) or np.all(np.diff(D) < 0))}")

    mask = D < 0
    if mask.sum() > 4:
        sl, ic = np.polyfit(Ln[mask], np.log(-D[mask]), 1)
        res = np.log(-D[mask]) - (sl * Ln[mask] + ic)
        print(f"power law  D ~ -N^alpha :  alpha = {sl:+.3f}"
              f"   resid sd = {res.std():.3f}")

    for name, b in (("1/log N", 1 / Ln), ("1/log^2 N", 1 / Ln ** 2),
                    ("log^2N/sqrt(N)", Ln ** 2 / np.sqrt(N)),
                    ("1/sqrt(N)", 1 / np.sqrt(N))):
        c = float(np.linalg.lstsq(b[:, None], D, rcond=None)[0][0])
        r = D - c * b
        print(f"  {name:<16} a={c:+.5f}   resid rms={r.std():.6f}"
              f"   (|D| rms={np.abs(D).std():.6f})")

    print("\nif D scatters without converging -> error term, form is right")
    print("if D follows a clean law        -> a correction term to name")


if __name__ == "__main__":
    main()
