#!/usr/bin/env python3
"""
aperture.py -- extending the lens by reading the whole resolving lattice.

Single-angle method: one shift per target prime r, excess (r-1)/(r-2),
noise ~ 1/sqrt(pairs).  Aperture dies near r ~ 100.

Lattice method: every even shift s is a departure angle that resolves ALL
prime factors of s at once.  Over many shifts each prime r appears in many
of them, so fit every prime simultaneously:

    log( C_s / Li2(N) )  =  b0  +  SUM_{p | s, p odd}  b_p * log((p-1)/(p-2))

Hardy-Littlewood  <=>  b_p = 1 for every p, and b0 = log(2 C2).
The b_p are what the lens actually measures, now with the full lattice behind
each one instead of a single angle.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C2 = 0.66016181584686957393
N = 50_000_000
SMAX = 3000                 # shifts 2,4,...,SMAX  = departure angles sampled
MIN_OCC = 4                 # fit a prime only if it divides >= this many shifts


def sieve_bool(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return s


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
    print(f"sieving primes to {N:,} ...")
    S = sieve_bool(N)
    npr = int(S.sum())
    x = np.arange(2, N + 1, dtype=np.float64)
    L2 = float(np.sum(1.0 / np.log(x) ** 2))
    del x
    print(f"{npr:,} primes;  Li_2(N) = {L2:,.0f}\n")

    spf = smallest_factor(SMAX)
    shifts = np.arange(2, SMAX + 1, 2)

    print(f"measuring {len(shifts)} departure angles (even shifts <= {SMAX}) ...")
    counts = np.empty(len(shifts), dtype=np.float64)
    for i, s in enumerate(shifts):
        counts[i] = np.count_nonzero(S[: N + 1 - s] & S[s:])
    print(f"pair counts: min {counts.min():,.0f}  max {counts.max():,.0f}\n")

    # ---- build the design matrix ----------------------------------------
    facs = [odd_factors(int(s), spf) for s in shifts]
    occ = {}
    for f in facs:
        for p in f:
            occ[p] = occ.get(p, 0) + 1
    fit_p = sorted(p for p, c in occ.items() if c >= MIN_OCC)
    pidx = {p: j for j, p in enumerate(fit_p)}
    print(f"{len(fit_p)} primes divide >= {MIN_OCC} shifts "
          f"(range {fit_p[0]}..{fit_p[-1]}) -> fitted individually")

    n, m = len(shifts), len(fit_p) + 1
    X = np.zeros((n, m))
    X[:, 0] = 1.0                                   # b0
    offset = np.zeros(n)                            # primes held at b_p = 1
    for i, f in enumerate(facs):
        for p in f:
            w = np.log((p - 1) / (p - 2))
            if p in pidx:
                X[i, 1 + pidx[p]] = w
            else:
                offset[i] += w

    y = np.log(counts / L2) - offset
    wt = np.sqrt(counts)                            # var(log C) ~ 1/C
    Xw, yw = X * wt[:, None], y * wt

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    dof = n - m
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(Xw.T @ Xw)
    err = np.sqrt(np.diag(cov))

    print(f"\nfit: {n} angles, {m} parameters, dof {dof}, "
          f"reduced chi2 {s2:.3f}\n")
    print(f"  b0 = {beta[0]:+.5f} +/- {err[0]:.5f}   "
          f"log(2*C2) = {np.log(2*C2):+.5f}  "
          f"({(beta[0]-np.log(2*C2))/err[0]:+.1f} sigma)")

    print(f"\n{'r':>6}{'occ':>6}{'b_r':>10}{'err':>9}{'sigma(b=1)':>12}"
          f"{'single-angle sigma':>20}")
    rows = []
    for p in fit_p:
        j = 1 + pidx[p]
        b, e = beta[j], err[j]
        dev = (b - 1) / e
        # what one angle would have given: signal/noise for a single shift
        sig1 = (1 / (p - 2)) / np.sqrt(2 / counts.mean())
        rows.append((p, occ[p], b, e, dev, 1 / e, sig1))
        if p in (3, 5, 7, 11, 13, 17, 31, 61, 101, 151, 199, 251, 349) \
           or p == fit_p[-1]:
            print(f"{p:>6}{occ[p]:>6}{b:>10.4f}{e:>9.4f}{dev:>12.1f}"
                  f"{sig1:>20.1f}")

    b = np.array([r[2] for r in rows])
    e = np.array([r[3] for r in rows])
    pull = (b - 1) / e
    print(f"\n  all {len(rows)} fitted primes:  mean b_p = {b.mean():.4f}")
    print(f"  pull (b_p - 1)/err :  mean {pull.mean():+.2f}  sd {pull.std():.2f}"
          f"   (expect 0 +/- 1 if HL holds)")
    nout = int(np.sum(np.abs(pull) > 3))
    print(f"  |pull| > 3 : {nout} / {len(rows)}")

    # ---- aperture comparison --------------------------------------------
    print("\n=== aperture: single angle vs full lattice ===")
    print(f"{'r':>6}{'single-angle sigma':>20}{'lattice sigma':>16}{'gain':>8}")
    for p, o, bb, ee, dv, lat, s1 in rows:
        if p in (13, 31, 101, 199, 349, 499):
            print(f"{p:>6}{s1:>20.1f}{lat:>16.1f}{lat/max(s1,1e-9):>8.1f}x")
    reach = [r[0] for r in rows if 1 / r[3] >= 3]
    print(f"\n  largest prime resolved at >=3 sigma: {max(reach) if reach else None}")
    print(f"  (single-angle limit at this N was ~101)")

    # ---- plots ------------------------------------------------------------
    pr = np.array([r[0] for r in rows], float)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].errorbar(pr, b, yerr=e, fmt="o", ms=3, lw=.8, color="C3", capsize=2)
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("prime $r$")
    ax[0].set_ylabel(r"fitted $\beta_r$")
    ax[0].set_ylim(0, 2)
    ax[0].set_title(r"Per-prime singular-series exponent ($\beta=1$ is HL)",
                    fontsize=9)

    ax[1].plot(pr, [r[6] for r in rows], "o", ms=3, color="C7",
               label="single angle")
    ax[1].plot(pr, 1 / e, "o", ms=3, color="C0", label="full lattice")
    ax[1].axhline(3, color="C3", ls="--", lw=1, label=r"3$\sigma$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("prime $r$"); ax[1].set_ylabel("detection significance")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Aperture extension", fontsize=9)
    fig.tight_layout(); fig.savefig("G_aperture.png", dpi=130)
    print("\n-> G_aperture.png")


if __name__ == "__main__":
    main()
