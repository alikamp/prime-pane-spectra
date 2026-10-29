#!/usr/bin/env python3
"""
facets.py -- the same structure seen from two angular symmetries.

Z[i]   units {+-1, +-i}        -> 4-fold  -> character (pi/|pi|)^{4k}
Z[w]   units {+-1,+-w,+-w^2}   -> 6-fold  -> character (pi/|pi|)^{6k}
       w = exp(2 pi i/3),  N(a+bw) = a^2 - ab + b^2

Prediction, and the point of the script:

    zeta_K(s) = zeta(s) * L(s, chi_K)   for every quadratic K

so the k=0 spectrum of EACH ring must show the Riemann zeros -- the facet
common to every viewing angle -- plus its own quadratic character's zeros:
chi_4 for Z[i], chi_3 for Z[w]. The shared part is shared; the rest is not.

For k >= 1 the two rings carry unrelated Hecke L-functions, so their
ordinates should look like independent sequences. That is tested against a
matched-density null, not asserted.

Reference chi_3 and chi_4 zeros are computed here from Hurwitz zeta.
"""

import numpy as np
from mpmath import mp, mpf, mpc, zeta as mpzeta, findroot

mp.dps = 25

ZETA_GAMMA = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832]


def primes_up_to(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return np.flatnonzero(s)


# ---------------------------------------------------------------- Z[i]
def gaussian_ideals(X):
    out = []
    for p in primes_up_to(X):
        p = int(p)
        if p == 2:
            out.append((2, np.pi / 4))
        elif p % 4 == 1:
            for a in range(1, int(p ** 0.5) + 1):
                r = p - a * a
                b = int(round(r ** 0.5))
                if b * b == r:
                    th = np.arctan2(b, a)
                    out.append((p, th))
                    out.append((p, -th))
                    break
        else:
            if p * p <= X:
                out.append((p * p, 0.0))
    return out


# ---------------------------------------------------------------- Z[w]
def eisenstein_ideals(X):
    """N(a + b w) = a^2 - ab + b^2,  w = exp(2 pi i / 3)."""
    out = []
    for p in primes_up_to(X):
        p = int(p)
        if p == 3:
            out.append((3, np.pi / 6))              # ramified (1 - w)
        elif p % 3 == 1:
            found = None
            for a in range(1, 2 * int(p ** 0.5) + 2):
                disc = 4 * p - 3 * a * a            # b = (a +- sqrt(disc))/2
                if disc < 0:
                    break
                r = int(round(disc ** 0.5))
                if r * r == disc and (a + r) % 2 == 0:
                    found = (a, (a + r) // 2)
                    break
            if found:
                a, b = found
                x = a - b / 2.0
                y = b * np.sqrt(3) / 2.0
                th = np.arctan2(y, x)
                out.append((p, th))
                out.append((p, -th))
        else:
            if p * p <= X:
                out.append((p * p, 0.0))
    return out


def spectrum(ideals, k, X, tmax, dt, fold):
    logs, wts, ph = [], [], []
    for Np, th in ideals:
        lp = np.log(Np)
        q, m = Np, 1
        while q <= X:
            logs.append(m * lp)
            wts.append(lp / np.sqrt(q))
            ph.append(fold * k * th * m)
            q *= Np
            m += 1
    logs = np.array(logs); wts = np.array(wts); ph = np.array(ph)
    t = np.arange(0.0, tmax, dt)
    L = logs.max()
    w = wts * 0.5 * (1 + np.cos(np.pi * logs / L))
    F = np.zeros_like(t)
    for i in range(0, len(logs), 2000):
        F += w[i:i+2000] @ np.cos(ph[i:i+2000, None]
                                  - np.outer(logs[i:i+2000], t))
    S = -F
    win = int(8.0 / dt)
    env = np.sqrt(np.convolve(S ** 2, np.ones(win) / win, mode="same")) + 1e-12
    return t, S / env


def peaks(t, S, thresh=1.5):
    p = np.flatnonzero((S[1:-1] > S[:-2]) & (S[1:-1] > S[2:])) + 1
    return np.sort(t[p[S[p] > thresh]])


def dirichlet_zeros(q, tmax):
    """Odd quadratic character mod q (q = 3 or 4), via Hurwitz zeta."""
    if q == 4:
        res = [(1, 1), (3, -1)]
    else:
        res = [(1, 1), (2, -1)]

    def Lc(t):
        s = mpc(0.5, t)
        return sum(c * mpzeta(s, mpf(a) / q) for a, c in res) * mpf(q) ** (-s)

    ts = np.arange(1.0, tmax, 0.05)
    v = np.array([abs(complex(Lc(float(x)))) for x in ts])
    out = []
    for i in range(1, len(ts) - 1):
        if v[i] < v[i-1] and v[i] < v[i+1] and v[i] < 0.4:
            try:
                r = float(findroot(Lc, mpf(float(ts[i]))).real)
                if 0 < r < tmax and all(abs(r - z) > 0.1 for z in out):
                    out.append(r)
            except Exception:
                pass
    return sorted(out)


def match(found, expected, res):
    return [(g, found[np.argmin(np.abs(found - g))]) for g in expected
            if len(found) and abs(found[np.argmin(np.abs(found - g))] - g) < res]


if __name__ == "__main__":
    X, TMAX, DT = 3_000_000, 26.0, 0.004
    res = 2 * np.pi / np.log(X)
    print(f"X = {X:,}   resolution 2pi/log X = {res:.3f}\n")

    gi, ei = gaussian_ideals(X), eisenstein_ideals(X)
    print(f"Z[i]: {len(gi):,} prime ideals   Z[w]: {len(ei):,} prime ideals")

    print("\ncomputing reference Dirichlet zeros from Hurwitz zeta ...")
    c4 = [z for z in dirichlet_zeros(4, TMAX)]
    c3 = [z for z in dirichlet_zeros(3, TMAX)]
    print(f"  L(chi_4): " + ", ".join(f"{z:.4f}" for z in c4[:6]))
    print(f"  L(chi_3): " + ", ".join(f"{z:.4f}" for z in c3[:6]))
    zg = [g for g in ZETA_GAMMA if g < TMAX]

    specs = {}
    for name, ide, fold in (("Z[i]", gi, 4), ("Z[w]", ei, 6)):
        for k in (0, 1, 2):
            t, S = spectrum(ide, k, X, TMAX, DT, fold)
            specs[(name, k)] = peaks(t, S)

    print("\n=== k=0: the shared facet ===")
    for name, own, lab in (("Z[i]", c4, "chi_4"), ("Z[w]", c3, "chi_3")):
        pk = specs[(name, 0)]
        mz = match(pk, zg, res)
        mo = match(pk, own, res)
        print(f"\n  {name}  ({len(pk)} peaks; expect "
              f"{len(zg)} zeta + {len([z for z in own if z < TMAX])} {lab})")
        print(f"    zeta zeros found : {len(mz)}/{len(zg)}   "
              + ", ".join(f"{a:.3f}->{b:.3f}" for a, b in mz[:4]))
        print(f"    {lab} zeros found: {len(mo)}/"
              f"{len([z for z in own if z < TMAX])}   "
              + ", ".join(f"{a:.3f}->{b:.3f}" for a, b in mo[:4]))

    print("\n=== is the shared facet actually shared? ===")
    a0, b0 = specs[("Z[i]", 0)], specs[("Z[w]", 0)]
    shared = [g for g in zg
              if len(a0) and len(b0)
              and abs(a0[np.argmin(np.abs(a0 - g))] - g) < res
              and abs(b0[np.argmin(np.abs(b0 - g))] - g) < res]
    print(f"  zeta ordinates present in BOTH rings' k=0 spectra: "
          f"{len(shared)}/{len(zg)}")
    print("  " + ", ".join(f"{g:.4f}" for g in shared))
    only4 = [z for z in c4 if z < TMAX
             and len(b0) and abs(b0[np.argmin(np.abs(b0 - z))] - z) > res]
    only3 = [z for z in c3 if z < TMAX
             and len(a0) and abs(a0[np.argmin(np.abs(a0 - z))] - z) > res]
    print(f"  chi_4 ordinates ABSENT from Z[w]: {len(only4)}/"
          f"{len([z for z in c4 if z < TMAX])}")
    print(f"  chi_3 ordinates ABSENT from Z[i]: {len(only3)}/"
          f"{len([z for z in c3 if z < TMAX])}")

    print("\n=== k>=1: independent facets? ===")
    rng = np.random.default_rng(0)
    for k in (1, 2):
        A, B = specs[("Z[i]", k)], specs[("Z[w]", k)]
        if not len(A) or not len(B):
            continue
        hit = sum(1 for g in A if abs(B[np.argmin(np.abs(B - g))] - g) < res)
        null = []
        for _ in range(400):
            R = np.sort(rng.uniform(0, TMAX, len(B)))
            null.append(sum(1 for g in A
                            if abs(R[np.argmin(np.abs(R - g))] - g) < res))
        mu, sd = np.mean(null), np.std(null)
        print(f"  k={k}: Z[i] {len(A)} peaks, Z[w] {len(B)} peaks, "
              f"coincidences {hit}  (null {mu:.1f} +- {sd:.1f}, "
              f"z = {(hit-mu)/max(sd,1e-9):+.1f})")
    print("\n  z ~ 0 means the two views carry independent information.")
