#!/usr/bin/env python3
"""
interval_test.py -- the decisive test: deterministic correction or fluctuation?

The checkpoints in a sweep are CUMULATIVE: [0,N] and [0,2N] share every prime
below N, so consecutive D values are strongly correlated. A slowly drifting
fluctuation produces no sign changes either, so "D is always negative" over
cumulative checkpoints proves much less than it looks like.

Differencing the saved per-shift counts gives the deficit on DISJOINT
intervals [N_k, N_{k+1}]:

    C_s(N_{k+1}) - C_s(N_k)   counted against   Li2(N_{k+1}) - Li2(N_k)

Those intervals share no primes, so they are near-independent samples.

    deterministic correction -> interval deficits stay negative
    fluctuation              -> interval deficits scatter in sign about 0

Run:  python3 interval_test.py sweep_1e10.json
"""

import json, sys
import numpy as np
from scipy.integrate import quad

C2 = 0.66016181584686957393
LOG2C2 = np.log(2 * C2)


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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sweep_1e10.json"
    res = json.load(open(path))
    res = [r for r in res if "counts" in r]
    if len(res) < 3:
        print(f"{path}: only {len(res)} checkpoints carry per-shift counts.")
        print("Re-run with the patched sweep_fast.py (it saves 'counts').")
        return

    nshift = len(res[0]["counts"])
    smax = 2 * nshift
    shifts = np.arange(2, smax + 1, 2)
    spf = smallest_factor(smax)
    hl = np.array([sum(np.log((p - 1) / (p - 2))
                       for p in odd_factors(int(s), spf)) for s in shifts])
    S = np.exp(hl)

    N = np.array([r["N"] for r in res], float)
    C = np.array([r["counts"] for r in res], float)
    L = np.array([r["li2"] for r in res], float)

    print(f"{path}: {len(res)} checkpoints, {nshift} shifts, "
          f"N up to {N[-1]:,.0f}\n")

    print("=== cumulative [0,N] (correlated) ===")
    cum = np.array([float(np.mean(C[k] / (S * L[k]))) / (2 * C2) - 1
                    for k in range(len(res))])
    print(f"  all negative: {bool(np.all(cum < 0))}   "
          f"sign changes: {int(np.sum(np.diff(np.sign(cum)) != 0))}")

    print("\n=== disjoint intervals [N_k, N_k+1] (near-independent) ===")
    dC = np.diff(C, axis=0)
    dL = np.diff(L)
    rel = np.array([float(np.mean(dC[k] / (S * dL[k]))) / (2 * C2) - 1
                    for k in range(len(dC))])
    # Poisson-scale expectation for each interval
    npair = dC.sum(axis=1)
    pois = 1.0 / np.sqrt(npair / len(shifts)) / np.sqrt(len(shifts))

    print(f"{'interval':>34}{'rel deficit':>14}{'Poisson':>11}{'ratio':>8}")
    for k in range(len(rel)):
        print(f"  [{N[k]:>13,.0f}, {N[k+1]:>13,.0f}]"
              f"{100*rel[k]:>13.4f}%{100*pois[k]:>10.4f}%"
              f"{rel[k]/pois[k]:>8.1f}")

    neg = int(np.sum(rel < 0))
    print(f"\n  negative: {neg}/{len(rel)}")
    print(f"  mean {100*rel.mean():+.5f}%   sd {100*rel.std():.5f}%")
    tstat = rel.mean() / (rel.std() / np.sqrt(len(rel)))
    print(f"  t = {tstat:+.2f} on {len(rel)-1} dof")
    # sign test
    from math import comb
    p_sign = sum(comb(len(rel), i) for i in range(neg, len(rel) + 1)) / 2 ** len(rel)
    print(f"  sign test p = {p_sign:.2e}")

    print("\n  trend in interval deficit with N:")
    m = rel < 0
    if m.sum() > 3:
        sl = np.polyfit(np.log(N[1:][m]), np.log(-rel[m]), 1)[0]
        print(f"    interval deficit ~ N^{sl:+.3f}")
        print("    ~ -0.5 means it tracks sample size -> fluctuation")
        print("    ~  0.0 means a fixed relative offset -> correction term")

    print("\n  VERDICT")
    if neg == len(rel) and abs(tstat) > 4:
        print("   every disjoint interval is deficient -> deterministic")
    elif neg <= len(rel) * 0.65:
        print("   intervals scatter in sign -> fluctuation; leading form stands")
    else:
        print("   mixed; needs more decades")


if __name__ == "__main__":
    main()
