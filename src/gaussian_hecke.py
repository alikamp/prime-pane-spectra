#!/usr/bin/env python3
"""
gaussian_hecke.py -- the genuinely 2D case.

In Z[i] the ANGLE of a prime carries arithmetic. Hecke Grossencharacters

    xi_k(pi) = (pi/|pi|)^{4k}

are well defined on ideals (the exponent 4 kills the unit group {+-1,+-i}),
and each has its own L-function with its own zeros. Nothing in the rational
1D picture reaches them.

Spectrum, exactly as in the rational case but weighted by the character:

    F_k(t) = sum over prime ideals p, powers m, N(p^m) <= X
             log N(p) * N(p^m)^{-1/2} * Re[ xi_k(p)^m * exp(-i t log N(p^m)) ]

Peaks of -F_k(t) sit at the ordinates of the zeros of L(s, xi_k).

VALIDATION, k = 0:  xi_0 = 1, so L(s, xi_0) = zeta_K(s) = zeta(s) L(s, chi_4).
The k=0 spectrum MUST therefore show the Riemann zeros AND the zeros of the
Dirichlet L-function mod 4, interleaved. That is a sharp falsifiable check
that the Gaussian machinery is right. Reference chi_4 zeros are computed
here from scratch via Hurwitz zeta, not taken from memory.

PREDICTION, k >= 1: a different discrete spectrum, not reducible to zeta.
Those ordinates are output for checking against LMFDB.

Prime ideals of Z[i]:
  p = 2             ramified, (1+i),  N = 2,   angle pi/4
  p = 1 mod 4       split, p = a^2+b^2, two ideals, N = p, angles +-atan(b/a)
  p = 3 mod 4       inert, (p),        N = p^2, angle 0
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZETA_GAMMA = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
              52.970321, 56.446248, 59.347044]


def primes_up_to(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return np.flatnonzero(s)


def two_squares(p):
    """p = a^2 + b^2 for p = 1 mod 4, by trial on a."""
    for a in range(1, int(p ** 0.5) + 1):
        r = p - a * a
        b = int(round(r ** 0.5))
        if b * b == r:
            return a, b
    raise ValueError(p)


def prime_ideals(X):
    """(norm, angle) for every prime ideal of Z[i] with norm <= X."""
    out = []
    for p in primes_up_to(X):
        p = int(p)
        if p == 2:
            out.append((2, np.pi / 4))                 # (1+i), ramified
        elif p % 4 == 1:
            a, b = two_squares(p)
            th = np.arctan2(b, a)
            out.append((p, th))                        # split: two ideals,
            out.append((p, -th))                       # conjugate pair
        else:
            if p * p <= X:
                out.append((p * p, 0.0))               # inert, norm p^2
    return out


def spectrum(ideals, k, X, tmax, dt):
    """F_k(t) over prime ideal powers with norm <= X."""
    logs, wts, phases = [], [], []
    for Np, th in ideals:
        lp = np.log(Np)
        q, m = Np, 1
        while q <= X:
            logs.append(m * lp)
            wts.append(lp / np.sqrt(q))
            phases.append(4.0 * k * th * m)            # arg of xi_k(p)^m
            q *= Np
            m += 1
    logs = np.array(logs); wts = np.array(wts); phases = np.array(phases)

    t = np.arange(0.0, tmax, dt)
    L = logs.max()
    taper = 0.5 * (1 + np.cos(np.pi * logs / L))
    w = wts * taper
    F = np.zeros_like(t)
    CH = 2000
    for i in range(0, len(logs), CH):
        ang = phases[i:i+CH, None] - np.outer(logs[i:i+CH], t)
        F += w[i:i+CH] @ np.cos(ang)
    S = -F
    win = int(8.0 / dt)
    env = np.sqrt(np.convolve(S ** 2, np.ones(win) / win, mode="same")) + 1e-12
    return t, S / env, len(logs)


def chi4_zeros(tmax):
    """Ordinates of L(s, chi_4) on the critical line, computed from scratch."""
    from mpmath import mp, zeta, mpf, mpc, findroot
    mp.dps = 25

    def Lc(t):
        s = mpc(0.5, t)
        return (mpf(4) ** (-s)) * (zeta(s, mpf(1) / 4) - zeta(s, mpf(3) / 4))

    # Z-like real function: rotate off the known phase by tracking sign flips
    # of the real and imaginary parts jointly via |L| minima.
    ts = np.arange(1.0, tmax, 0.05)
    vals = np.array([abs(complex(Lc(float(x)))) for x in ts])
    zeros = []
    for i in range(1, len(ts) - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1] and vals[i] < 0.35:
            try:
                r = findroot(lambda z: Lc(z), mpf(float(ts[i])))
                rr = float(r.real if hasattr(r, "real") else r)
                if 0 < rr < tmax and all(abs(rr - z) > 0.1 for z in zeros):
                    zeros.append(rr)
            except Exception:
                pass
    return sorted(zeros)


def peaks(t, S, thresh=1.5):
    """Local maxima of the envelope-normalised spectrum. S is in units of the
    local RMS, so a fixed cut works across the whole range; a fraction-of-max
    cut silently drops real peaks wherever the spectrum is weaker."""
    p = np.flatnonzero((S[1:-1] > S[:-2]) & (S[1:-1] > S[2:])) + 1
    p = p[S[p] > thresh]
    return np.sort(t[p])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", type=float, default=2e6)
    ap.add_argument("--tmax", type=float, default=45.0)
    ap.add_argument("--dt", type=float, default=0.004)
    a = ap.parse_args()
    X, tmax, dt = int(a.X), a.tmax, a.dt

    print(f"enumerating prime ideals of Z[i] with norm <= {X:,} ...")
    ide = prime_ideals(X)
    nsplit = sum(1 for n, th in ide if th not in (0.0,) and n != 2)
    print(f"  {len(ide):,} prime ideals "
          f"({nsplit:,} split, {sum(1 for n,th in ide if th==0.0):,} inert, 1 ramified)")

    print("\ncomputing reference zeros of L(s, chi_4) from Hurwitz zeta ...")
    c4 = chi4_zeros(tmax)
    print(f"  {len(c4)} found: " + ", ".join(f"{z:.4f}" for z in c4[:8]))

    zg = [g for g in ZETA_GAMMA if g < tmax]
    expected0 = sorted(zg + c4)

    results = {}
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for row, k in enumerate((0, 1, 2)):
        t, S, nterm = spectrum(ide, k, X, tmax, dt)
        pk = peaks(t, S)
        results[k] = pk
        ax[row].plot(t, S, lw=0.8, color=f"C{row}")
        ax[row].set_ylabel(f"$k={k}$")
        if k == 0:
            for g in zg:
                ax[row].axvline(g, color="C3", ls="--", lw=.8, alpha=.75)
            for g in c4:
                ax[row].axvline(g, color="C2", ls=":", lw=1.1, alpha=.85)
        ax[row].set_title(f"$L(s,\\xi_{k})$ spectrum, {nterm:,} terms",
                          fontsize=9, loc="left")
    ax[-1].set_xlabel("$t$")
    fig.suptitle("Hecke Grossencharacter spectra over $\\mathbb{Z}[i]$  "
                 "(k=0: dashed = zeta zeros, dotted = $L(\\chi_4)$ zeros)")
    fig.tight_layout(); fig.savefig("I_gaussian_hecke.png", dpi=130)

    res = 2 * np.pi / np.log(X)
    print(f"\nspectral resolution 2pi/log X = {res:.3f}")

    print(f"\n=== k=0 VALIDATION: must show zeta zeros AND L(chi_4) zeros ===")
    print(f"{'expected':>12}{'source':>10}{'nearest peak':>14}{'delta':>10}")
    hit = 0
    for g in expected0:
        src = "zeta" if g in zg else "chi_4"
        j = np.argmin(np.abs(results[0] - g))
        d = results[0][j] - g
        ok = abs(d) < res
        hit += ok
        print(f"{g:>12.4f}{src:>10}{results[0][j]:>14.4f}{d:>+10.4f}"
              f"{'  <-- match' if ok else ''}")
    print(f"\n  {hit}/{len(expected0)} recovered within one resolution element")
    print(f"  peaks found: {len(results[0])}  (expected {len(expected0)})")

    print(f"\n=== k=1, k=2: spectra no 1D construction reaches ===")
    for k in (1, 2):
        near = [g for g in expected0
                if min(abs(results[k] - g)) < res] if len(results[k]) else []
        print(f"  k={k}: {len(results[k])} peaks, first 8: "
              + ", ".join(f"{x:.3f}" for x in results[k][:8]))
        print(f"       coinciding with a k=0 ordinate: {len(near)}"
              f"/{len(expected0)}")
    print("\n  -> k>=1 ordinates are predictions; check against LMFDB")
    print("-> I_gaussian_hecke.png")


if __name__ == "__main__":
    main()
