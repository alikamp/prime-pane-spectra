#!/usr/bin/env python3
"""
lift_resonance.py -- the "lift the top pane" geometry.

Pane: integer n sits at (n mod q, n // q).  Two such panes, parallel.
A line crosses both with departure angle (theta, phi).  Lifting the top
pane by delta translates its puncture pattern by delta*(cos phi, sin phi),
which in arithmetic terms is a SHIFT of

        s = round(delta cos phi) + q * round(delta sin phi)

Coincidences between the panes = #{p <= N : p and p+s both prime}.
So the departure angle selects which prime difference you are counting.

Prediction (Hardy-Littlewood k-tuple conjecture):
        #{p<=N : p+s prime}  ~  S(s) * Li_2(N),     s even
        S(s) = 2 C2 * prod_{p | s, p odd} (p-1)/(p-2)
        C2   = 0.66016181584686957393...
Odd s gives essentially zero (parity obstruction).
"""

import numpy as np
from math import pi, cos, sin, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C2 = 0.66016181584686957393
N = 10_000_000
Q = 210                      # 2*3*5*7 : one pane column per residue class


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return s


def odd_prime_factors(s):
    out, d = [], 3
    s = int(s)
    while s % 2 == 0:
        s //= 2
    while d * d <= s:
        if s % d == 0:
            out.append(d)
            while s % d == 0:
                s //= d
        d += 2
    if s > 1:
        out.append(s)
    return out


def singular_series(s):
    if s % 2:
        return 0.0
    v = 2 * C2
    for p in odd_prime_factors(s):
        v *= (p - 1) / (p - 2)
    return v


def li2(n):
    x = np.arange(2, n + 1, dtype=np.float64)
    return float(np.sum(1.0 / np.log(x) ** 2))


def pair_count(S, s):
    if s <= 0 or s >= len(S):
        return 0
    return int(np.count_nonzero(S[: len(S) - s] & S[s:]))


def main():
    print(f"sieving to {N:,} ...")
    S = sieve(N)
    npr = int(S.sum())
    L2 = li2(N)
    print(f"{npr:,} primes;  Li_2(N) = {L2:,.0f}")

    # --- structure-matched null: coprime to 2,3,5,7,11, same density -------
    rough = np.ones(N + 1, dtype=bool)
    rough[:2] = False
    for p in (2, 3, 5, 7, 11):
        rough[p * 2:: p] = False
    idx = np.flatnonzero(rough)
    rng = np.random.default_rng(7)
    keep = rng.choice(idx, size=npr, replace=False)
    R = np.zeros(N + 1, dtype=bool)
    R[keep] = True
    print(f"null set: {int(R.sum()):,} points, coprime to 2,3,5,7,11\n")

    # ---------------------------------------------------------------- test 1
    # Shift spectrum: does the arithmetic of s drive the coincidence count?
    print("=== shift spectrum, s = 2..60 ===")
    print(f"{'s':>4}{'primes':>12}{'HL pred':>12}{'ratio':>8}"
          f"{'null':>12}{'p/null':>8}")
    rows = []
    for s in range(2, 61):
        cp = pair_count(S, s)
        cn = pair_count(R, s)
        hl = singular_series(s) * L2
        rows.append((s, cp, hl, cn))
        if s % 2 == 0 and s <= 30:
            r = cp / hl if hl else float("nan")
            print(f"{s:>4}{cp:>12,}{hl:>12,.0f}{r:>8.3f}"
                  f"{cn:>12,}{cp/max(cn,1):>8.3f}")

    ev = [(s, cp, hl, cn) for s, cp, hl, cn in rows if s % 2 == 0]
    od = [(s, cp, hl, cn) for s, cp, hl, cn in rows if s % 2 == 1]
    print(f"\nmean even-s count: {np.mean([r[1] for r in ev]):,.0f}")
    print(f"mean  odd-s count: {np.mean([r[1] for r in od]):,.0f}"
          f"   (parity obstruction)")
    ratios = np.array([r[1] / r[2] for r in ev])
    print(f"measured / Hardy-Littlewood over even s:"
          f"  mean {ratios.mean():.4f}  sd {ratios.std():.4f}")

    # s=2 vs s=6 : the sharpest single contrast (3 | 6 doubles the density)
    c2_, c6 = pair_count(S, 2), pair_count(S, 6)
    n2_, n6 = pair_count(R, 2), pair_count(R, 6)
    print(f"\ns=6 / s=2   primes {c6/c2_:.4f}   null {n6/max(n2_,1):.4f}"
          f"   HL predicts {singular_series(6)/singular_series(2):.4f}")

    # ---------------------------------------------------------------- test 2
    # Departure-angle sweep at fixed lift.
    print("\n=== departure-angle sweep, lift delta = 60, q = 210 ===")
    delta = 60
    phis = np.linspace(0, 2 * pi, 1441)
    shifts, cnts, nulls = [], [], []
    for ph in phis:
        a = int(round(delta * cos(ph)))
        b = int(round(delta * sin(ph)))
        s = a + Q * b
        shifts.append(s)
        cnts.append(pair_count(S, abs(s)))
        nulls.append(pair_count(R, abs(s)))
    cnts = np.array(cnts, float)
    nulls = np.array(nulls, float)
    shifts = np.array(shifts)

    good = cnts > 0
    print(f"angles sampled: {len(phis)};  nonzero coincidence: {good.sum()}")
    print(f"peak count {cnts.max():,.0f} at phi="
          f"{np.degrees(phis[cnts.argmax()]):.2f} deg, shift s={shifts[cnts.argmax()]}")
    print(f"dynamic range over departure angle: "
          f"{cnts[good].max()/cnts[good].min():.1f}x")

    # how much of the angular structure survives the null?
    m = good & (nulls > 0)
    exc = cnts[m] / nulls[m]
    print(f"primes/null across angles: mean {exc.mean():.3f}"
          f"  min {exc.min():.3f}  max {exc.max():.3f}")

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    sev = [r[0] for r in ev]
    ax[0].plot(sev, [r[1] for r in ev], "o-", ms=4, color="C3", label="primes")
    ax[0].plot(sev, [r[2] for r in ev], "--", color="k", lw=1,
               label="Hardy-Littlewood")
    ax[0].plot([r[0] for r in od], [r[1] for r in od], "s", ms=3, color="C7",
               label="odd s (parity)")
    ax[0].set_xlabel("shift s"); ax[0].set_ylabel("coincidences")
    ax[0].legend(fontsize=8); ax[0].set_title("Shift spectrum", fontsize=9)

    ax[1].plot(sev, ratios, "o", ms=4, color="C3")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set_ylim(0.9, 1.1)
    ax[1].set_xlabel("shift s"); ax[1].set_ylabel("measured / HL")
    ax[1].set_title("Agreement with singular series", fontsize=9)

    axp = fig.add_subplot(1, 3, 3, projection="polar")
    ax[2].remove()
    axp.plot(phis, cnts, lw=0.8, color="C3")
    axp.set_title(f"Coincidences vs departure angle\n(lift={delta}, q={Q})",
                  fontsize=9)
    fig.tight_layout(); fig.savefig("E_lift_resonance.png", dpi=130)
    print("\n-> E_lift_resonance.png")


if __name__ == "__main__":
    main()
