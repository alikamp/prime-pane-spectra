#!/usr/bin/env python3
"""
resolving_angles.py -- inverse design of the lift angle.

Instead of sweeping the departure angle and seeing what appears, CHOOSE the
angle to resolve a target prime r.

Pane basis: integer n at (n mod q, n // q).
Lift delta at departure angle phi  ->  shift  s = a + q*b,  (a,b) = delta*(cos,sin).
The angle resolves r  <=>  r | s  <=>  (a,b) in the index-r sublattice

        Lambda_r = { (a,b) : a + q b = 0  (mod r) }

Shortest vector of a reduced basis of Lambda_r  =  cheapest resolving lift.

Measured signature: excess of prime coincidences over a sieve-to-P null is
        E(s) = prod_{ r | s, r > P } (r-1)/(r-2)
i.e. exactly the prime factors of s that the null's sieve cannot see.
"""

import numpy as np
from math import gcd, atan2, degrees, hypot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 50_000_000
Q = 210                 # 2*3*5*7  -> these primes are STRUCTURAL
SIEVE_P = 11            # null sieves to 11 -> primes > 11 are RESOLVABLE


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return s


def factors(n):
    out, d, n = [], 2, int(n)
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def predicted_excess(s, P=SIEVE_P):
    """Excess of primes over a sieve-to-P null at shift s."""
    e = 1.0
    for r in factors(abs(s)):
        if r > P:
            e *= (r - 1) / (r - 2)
    return e


def reduced_resolving_vectors(r, q=Q, nvec=3, box=60):
    """Shortest primitive vectors of  Lambda_r  INTERSECT  Lambda_2.

    Lambda_r = {(a,b): a+qb = 0 mod r}  resolves r.
    Lambda_2 = {s even}: since q is even, s = a (mod 2), so a must be even,
    otherwise the parity obstruction kills the count outright. The usable
    resolving angles are the intersection, not Lambda_r alone.
    """
    qr = q % r
    cands = []
    for b in range(-box, box + 1):
        for a in range(-box, box + 1):
            if a == 0 and b == 0:
                continue
            if (a + qr * b) % r:
                continue
            if a % 2:                          # parity obstruction
                continue
            if gcd(abs(a), abs(b)) != 1:      # primitive directions only
                continue
            cands.append((hypot(a, b), a, b))
    cands.sort()
    seen, out = set(), []
    for n, a, b in cands:
        key = (a // gcd(abs(a) or 1, abs(b) or 1), b)
        ang = round(degrees(atan2(b, a)), 6)
        if ang in seen:
            continue
        seen.add(ang)
        out.append((a, b))
        if len(out) >= nvec:
            break
    return out


def pair_count(S, s):
    s = abs(int(s))
    if s <= 0 or s >= len(S):
        return 0
    return int(np.count_nonzero(S[: len(S) - s] & S[s:]))


def main():
    print(f"sieving to {N:,} ...")
    S = sieve(N)
    npr = int(S.sum())

    rough = np.ones(N + 1, dtype=bool)
    rough[:2] = False
    for p in (2, 3, 5, 7, 11):
        rough[p * 2:: p] = False
    rng = np.random.default_rng(7)
    keep = rng.choice(np.flatnonzero(rough), size=npr, replace=False)
    R = np.zeros(N + 1, dtype=bool)
    R[keep] = True
    print(f"{npr:,} primes;  null sieved to {SIEVE_P}, matched density\n")

    print(f"pane basis q = {Q} = {'*'.join(map(str, factors(Q)))}")
    print(f"  structural primes (divide q, unresolvable by angle): {factors(Q)}")
    print(f"  resolvable primes: coprime to q and > {SIEVE_P}\n")

    # ---- baseline FIRST: shifts with nothing above the sieve to resolve ---
    # The null's overall pair density is not exactly the primes', so the
    # excess must be measured against this baseline, not against 1.
    base = []
    for b in range(1, 14):
        for a in range(-13, 14):
            s = a + Q * b
            if s <= 0 or s % 2:
                continue
            if any(x > SIEVE_P for x in factors(s)):
                continue
            cp, cn = pair_count(S, s), pair_count(R, s)
            if cp and cn:
                base.append(cp / cn)
    base = np.array(base)
    B = base.mean()
    print(f"=== baseline (shifts with all factors <= {SIEVE_P}) ===")
    print(f"{len(base)} control shifts;  ratio = {B:.4f} +/- {base.std():.4f}")
    print("  nothing above the sieve to resolve -> this is the zero point\n")

    targets = [13, 17, 19, 23, 29, 31, 37, 43, 53, 61, 73, 89, 101, 127, 163]
    print(f"{'r':>5}{'shortest (a,b)':>16}{'lift':>7}{'phi(deg)':>10}"
          f"{'shift s':>9}{'meas E':>9}{'pred E':>9}{'sigma':>7}")

    rows = []
    for r in targets:
        vecs = reduced_resolving_vectors(r)
        if not vecs:
            continue
        a, b = vecs[0]
        s = a + Q * b
        if abs(s) >= N // 2 or s == 0:
            continue
        cp, cn = pair_count(S, s), pair_count(R, s)
        if cp == 0 or cn == 0:
            continue
        meas = (cp / cn) / B                    # baseline-corrected
        pred = predicted_excess(s)
        err = meas * np.sqrt(1 / cp + 1 / cn)
        sig = (meas - 1) / err
        rows.append((r, meas, pred, sig, err, s))
        print(f"{r:>5}{str((a,b)):>16}{hypot(a,b):>7.2f}"
              f"{degrees(atan2(b,a)):>10.2f}{s:>9}"
              f"{meas:>9.4f}{pred:>9.4f}{sig:>7.1f}")

    ok = [x for x in rows if x[3] >= 3]
    print(f"\n  resolved at >=3 sigma: {[x[0] for x in ok]}")
    if rows:
        dev = np.array([x[1] - x[2] for x in rows])
        er = np.array([x[4] for x in rows])
        print(f"  measured vs predicted: mean deviation "
              f"{dev.mean():+.4f}  ({(dev/er).mean():+.2f} sigma)")

    # ---- aperture limit --------------------------------------------------
    print("\n=== aperture / resolution limit ===")
    for r in (13, 31, 101, 127, 199):
        e = (r - 1) / (r - 2)
        need = (3 / (e - 1)) ** 2 * 2
        print(f"  r={r:>4}  resolving power {e:.4f}  "
              f"pairs needed for 3-sigma: {need:,.0f}")

    # ---- plot -------------------------------------------------------------
    rr = np.array([x[0] for x in rows], float)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax[0].errorbar(rr, [x[1] for x in rows], yerr=[x[4] for x in rows],
                   fmt="o", ms=5, color="C3", label="measured", capsize=3)
    g = np.linspace(rr.min(), rr.max(), 400)
    ax[0].plot(g, (g - 1) / (g - 2), "k--", lw=1,
               label=r"predicted $(r-1)/(r-2)$")
    ax[0].axhline(1, color="C7", lw=.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("target prime $r$")
    ax[0].set_ylabel("excess over sieve null")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Resolving a chosen prime via departure angle", fontsize=9)

    ax[1].plot(rr, [x[3] for x in rows], "o", ms=5, color="C0")
    ax[1].axhline(3, color="C3", ls="--", lw=1, label=r"3$\sigma$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("target prime $r$")
    ax[1].set_ylabel("detection significance")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"Aperture limit at N={N:,}", fontsize=9)
    fig.tight_layout(); fig.savefig("F_resolving_angles.png", dpi=130)
    print("\n-> F_resolving_angles.png")


if __name__ == "__main__":
    main()
