#!/usr/bin/env python3
"""
moire_primes.py -- Geometric / optical resonance analysis of the primes.

Four experiments:

  A. Embeddings            : Ulam, Sacks, polar, log-spiral maps N -> R^2
  B. Angular resonance     : rigid-rotation coincidence count N(theta, eps),
                             measured against two null models.
  C. Coincidence lattices  : enumerate exact "magic angles" of Z^2 (Pythagorean
                             rotations, index Sigma = product of primes = 1 mod 4)
                             and test whether PRIMES resonate there beyond chance.
  D. Mellin resonance      : log-radial power spectrum of the prime measure.
                             This is the C*-group version of the overlay idea;
                             peaks should land on the Riemann zeta zeros.

Usage:
    python3 moire_primes.py --all
    python3 moire_primes.py --section B --nmax 20000
"""

import argparse
import numpy as np
from math import gcd, cos, sin, pi, log, atan2, sqrt
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261028)

# --------------------------------------------------------------------------
# 0. Number-theoretic primitives
# --------------------------------------------------------------------------

def primes_up_to(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return np.flatnonzero(s)


def coprime_to_primorial(n, k=4):
    """Residue-class null: integers coprime to the first k primes.
    Same small-moduli structure as the primes, none of the deeper arithmetic."""
    small = primes_up_to(100)[:k]
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in small:
        s[p * 2:: p] = False
    return np.flatnonzero(s)


# --------------------------------------------------------------------------
# A. Embeddings  Phi : N -> R^2
# --------------------------------------------------------------------------

def ulam_coords(nmax):
    """Square (Ulam) spiral. coords[n] = position of integer n, n >= 1."""
    coords = np.zeros((nmax + 1, 2))
    x = y = 0
    n = 1
    coords[1] = (0.0, 0.0)
    step = 1
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    d = 0
    while n < nmax:
        for _ in range(2):
            dx, dy = dirs[d % 4]
            for _ in range(step):
                if n >= nmax:
                    break
                x += dx
                y += dy
                n += 1
                coords[n] = (x, y)
            d += 1
            if n >= nmax:
                break
        step += 1
    return coords


def embed(ns, kind="sacks", alpha=1.0):
    ns = np.asarray(ns, dtype=float)
    if kind == "sacks":
        r = np.sqrt(ns)
        th = 2 * np.pi * np.sqrt(ns)
    elif kind == "polar":
        r = ns
        th = ns
    elif kind == "logspiral":
        r = ns
        th = alpha * np.log(ns)
    else:
        raise ValueError(kind)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


def embed_ulam(ns, nmax):
    c = ulam_coords(nmax)
    return c[np.asarray(ns, dtype=int)]


# --------------------------------------------------------------------------
# B. Angular resonance  N(theta, eps)  with nulls
# --------------------------------------------------------------------------

def coincidence_count(P, theta, eps, tree=None):
    c, s = cos(theta), sin(theta)
    R = np.array([[c, -s], [s, c]])
    Q = P @ R.T
    t1 = tree if tree is not None else cKDTree(P)
    return t1.count_neighbors(cKDTree(Q), eps)


def angular_scan(P, thetas, eps):
    tree = cKDTree(P)
    return np.array([coincidence_count(P, t, eps, tree) for t in thetas],
                    dtype=float)


def null_sets(nmax, k, rng):
    """Three nulls of matched cardinality k, in increasing strength.
       random : uniform integers                 (Cramer-like, no residue structure)
       odd    : uniform ODD integers             (parity structure only)
       rough  : coprime to 2,3,5,7,11            (full small-moduli structure)
    All randomise WHICH elements are 'prime' while preserving the stated structure."""
    allint = np.arange(2, nmax + 1)
    odd = allint[allint % 2 == 1]
    rough = coprime_to_primorial(nmax, k=5)
    out = {}
    for name, pool in (("random", allint), ("odd", odd), ("rough", rough)):
        kk = min(k, len(pool))
        out[name] = np.sort(rng.choice(pool, size=kk, replace=False))
    return out


def section_B(nmax=20000, eps=0.02, dtheta_deg=0.005):
    """Fine angular scan on the ULAM lattice.

    NOTE: coincidence peaks have angular width ~ eps/R and are delta-like.
    A scan coarser than that steps straight over every peak -- this is the
    single easiest way to conclude 'no signal' when the signal is enormous.
    """
    print(f"\n[B] Fine angular scan on Ulam lattice (nmax={nmax}, eps={eps})")
    pr = primes_up_to(nmax)
    coords = ulam_coords(nmax)
    P = coords[pr]
    R = np.abs(P).max()
    nulls = null_sets(nmax, len(pr), RNG)
    Rn = coords[nulls["rough"]]

    thetas = np.deg2rad(np.arange(0.0, 90.0, dtheta_deg))
    print(f"    {len(pr)} primes, max radius {R:.0f}, {len(thetas)} angles")
    print(f"    peak width ~ {np.rad2deg(eps/R):.4f} deg   "
          f"sampling {dtheta_deg} deg  -> {'RESOLVED' if dtheta_deg < np.rad2deg(eps/R) else 'ALIASED'}")

    sp = angular_scan(P, thetas, eps)
    sr = angular_scan(Rn, thetas, eps)

    ang = csl_angles(400)
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for a, (sig, lab, col) in zip(ax, [(sp, "primes", "C3"),
                                       (sr, "null: coprime to 2,3,5,7,11", "C0")]):
        a.plot(np.rad2deg(thetas), sig, lw=0.7, color=col)
        for th, S, *_ in ang:
            d = np.rad2deg(th)
            if d < 90:
                a.axvline(d, color="k", ls=":", lw=0.5, alpha=.45)
        a.set_ylabel("coincidence count")
        a.set_title(f"{lab}   (dotted = exact Pythagorean / CSL angles)",
                    fontsize=9, loc="left")
    ax[-1].set_xlabel("rotation angle (degrees)")
    fig.suptitle("Rigid-rotation coincidence comb, Ulam embedding")
    fig.tight_layout(); fig.savefig("B_angular_resonance.png", dpi=130)
    print("    -> B_angular_resonance.png")
    print(f"    primes: median={np.median(sp):.1f}  max={sp.max():.0f} "
          f"(={sp.max()/max(np.median(sp),1):.0f}x median)")
    print(f"    null  : median={np.median(sr):.1f}  max={sr.max():.0f} "
          f"(={sr.max()/max(np.median(sr),1):.0f}x median)")
    return thetas, sp, sr


def _kurt(x):
    return float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3)


# --------------------------------------------------------------------------
# C. Coincidence-site lattices: exact magic angles of Z^2
# --------------------------------------------------------------------------

def csl_angles(sigma_max=200):
    """Rotations mapping Z^2 onto a sublattice of index Sigma.
    theta = 2*arg(a+bi), Sigma = a^2+b^2 odd, gcd(a,b)=1."""
    out = []
    amax = int(sqrt(sigma_max)) + 1
    for a in range(1, amax + 1):
        for b in range(1, a):
            if gcd(a, b) != 1:
                continue
            S = a * a + b * b
            if S > sigma_max or S % 2 == 0:
                continue
            out.append((2 * atan2(b, a), S, a, b))
    out.sort()
    # dedupe by angle
    ded, seen = [], set()
    for th, S, a, b in out:
        key = round(th, 12)
        if key not in seen:
            seen.add(key)
            ded.append((th, S, a, b))
    return ded


def section_C(nmax=200000, eps=0.01, sigma_max=200, n_real=5):
    """Do primes resonate at the CSL magic angles MORE than a structure-matched
    null? eps must be small: these are exact coincidences, not broad peaks."""
    print(f"\n[C] Magic-angle (CSL) test, Ulam embedding, nmax={nmax}, eps={eps}")
    ang = csl_angles(sigma_max)
    print(f"    {len(ang)} exact coincidence rotations with Sigma <= {sigma_max}")
    print(f"    sample Sigma: {sorted({S for _,S,_,_ in ang})[:12]}")
    print(f"    all Sigma odd: {all(S % 2 for _, S, _, _ in ang)}; "
          f"all prime factors = 1 mod 4: {all(_only_1mod4(S) for _,S,_,_ in ang)}")

    coords = ulam_coords(nmax)
    pr = primes_up_to(nmax)
    P = coords[pr]
    delta = len(pr) / nmax
    print(f"    {len(pr)} primes, density delta={delta:.4f}, "
          f"max radius {np.abs(P).max():.0f}")

    def magic_counts(pts):
        t = cKDTree(pts)
        return np.array([coincidence_count(pts, th, eps, t) for th, *_ in ang], float)

    mp = magic_counts(P)

    # full-lattice exact coincidence totals, per angle -> the geometric baseline
    full = magic_counts(coords[1:nmax + 1])
    # Cramer expectation: each exact lattice coincidence survives with prob delta^2
    cramer = full * delta ** 2

    results = {"primes": mp}
    for name in ("random", "odd", "rough"):
        acc = []
        for r in range(n_real):
            rng = np.random.default_rng(1000 + r)
            ns = null_sets(nmax, len(pr), rng)[name]
            acc.append(magic_counts(coords[ns]))
        results[name] = np.mean(acc, axis=0)
        results[name + "_sd"] = np.std([a.sum() for a in acc])

    print(f"\n    {'set':<22}{'total coincidences':>20}{'ratio to primes':>18}")
    tot_p = mp.sum()
    print(f"    {'PRIMES':<22}{tot_p:>20.0f}{1.0:>18.2f}")
    print(f"    {'Cramer (delta^2)':<22}{cramer.sum():>20.0f}"
          f"{tot_p/cramer.sum():>18.2f}")
    for name, lab in (("random", "null: random ints"),
                      ("odd", "null: random odds"),
                      ("rough", "null: coprime 2..11")):
        s = results[name].sum()
        print(f"    {lab:<22}{s:>20.0f}{tot_p/s:>18.2f}"
              f"   (sd over reals {results[name+'_sd']:.0f})")

    print("\n    Interpretation: ratio ~1 against a null means the resonance is")
    print("    a property of that null's structure, not of primality.")

    sig = np.array([S for _, S, _, _ in ang], float)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].scatter(sig, mp, s=26, color="C3", label="primes", zorder=3)
    ax[0].scatter(sig, results["rough"], s=26, marker="x", color="C0",
                  label="null: coprime 2..11")
    ax[0].scatter(sig, results["random"], s=18, marker="^", color="C7",
                  label="null: random ints")
    ax[0].set_xlabel(r"coincidence index $\Sigma$")
    ax[0].set_ylabel("exact coincidences")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[0].set_title(r"Coincidences vs CSL index (expect $\propto 1/\Sigma$)",
                    fontsize=9)

    ratio = mp / np.maximum(results["rough"], 1e-9)
    ax[1].scatter(sig, ratio, s=26, color="C3")
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\Sigma$")
    ax[1].set_ylabel("primes / structure-matched null")
    ax[1].set_title("Excess over small-moduli structure", fontsize=9)
    fig.tight_layout(); fig.savefig("C_magic_angles.png", dpi=130)
    print("    -> C_magic_angles.png")
    return results


def _only_1mod4(S):
    d = 3
    while d * d <= S:
        while S % d == 0:
            if d % 4 != 1:
                return False
            S //= d
        d += 2
    return S == 1 or S % 4 == 1


# --------------------------------------------------------------------------
# D. Mellin / log-radial resonance  ->  Riemann zeros
# --------------------------------------------------------------------------

KNOWN_GAMMA = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
               37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
               52.970321, 56.446248, 59.347044]


def section_D(X=1_000_000, tmax=60.0, dt=0.005):
    print(f"\n[D] Mellin resonance of the prime measure, X={X}")
    pr = primes_up_to(X)
    # von Mangoldt support: prime powers p^k <= X
    logs, wts = [], []
    for p in pr:
        lp = log(p)
        q, k = p, 1
        while q <= X:
            logs.append(k * lp)
            wts.append(lp / sqrt(q))        # Lambda(n) / sqrt(n)
            q *= p
            k += 1
    logs = np.array(logs); wts = np.array(wts)
    print(f"    {len(logs)} prime-power terms, log-range [0, {logs.max():.2f}]")

    # Taper in log-space: suppresses the sidelobe ringing from the hard cutoff
    # at log X, which otherwise dominates and smears every peak.
    L = logs.max()
    taper = 0.5 * (1 + np.cos(pi * logs / L))        # Hann in log n
    wts = wts * taper

    t = np.arange(0.0, tmax, dt)
    F = np.zeros_like(t)
    CH = 2000
    for i in range(0, len(logs), CH):
        F += wts[i:i+CH] @ np.cos(np.outer(logs[i:i+CH], t))
    S = -F

    # Normalise by a local RMS envelope so peaks at large t are comparable
    # to peaks at small t (the raw spectrum falls off steeply).
    w = int(8.0 / dt)
    env = np.sqrt(np.convolve(S ** 2, np.ones(w) / w, mode="same")) + 1e-12
    Sn = S / env

    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(t, S, lw=0.8, color="C7"); ax[0].set_ylabel("raw  $-F(t)$")
    ax[1].plot(t, Sn, lw=0.9, color="C0"); ax[1].set_ylabel("envelope-normalised")
    for a in ax:
        for g in KNOWN_GAMMA:
            if g < tmax:
                a.axvline(g, color="C3", ls="--", lw=0.9, alpha=.75)
    ax[1].set_xlabel(r"$t$  (log-radial frequency)")
    ax[0].set_title("Mellin spectrum of the prime measure; dashed = known zeta zeros",
                    fontsize=10)
    fig.tight_layout(); fig.savefig("D_mellin_spectrum.png", dpi=130)
    print("    -> D_mellin_spectrum.png")

    pk = np.flatnonzero((Sn[1:-1] > Sn[:-2]) & (Sn[1:-1] > Sn[2:])) + 1
    pk = pk[Sn[pk] > 0.5 * Sn.max()]
    found = np.sort(t[pk])
    print(f"    {len(found)} significant peaks found")
    print(f"    spectral resolution  2*pi/log(X) = {2*pi/L:.3f}")
    print("\n      known gamma   nearest peak      delta")
    errs = []
    for g in KNOWN_GAMMA:
        if g >= tmax or len(found) == 0:
            break
        j = np.argmin(np.abs(found - g))
        d = found[j] - g
        errs.append(abs(d))
        flag = "  <-- match" if abs(d) < 2 * pi / L else ""
        print(f"      {g:9.6f}   {found[j]:12.4f}   {d:+8.4f}{flag}")
    if errs:
        hit = sum(e < 2 * pi / L for e in errs)
        print(f"\n    {hit}/{len(errs)} zeros recovered within one resolution element")
    return t, Sn


# --------------------------------------------------------------------------

def section_A(nmax=20000):
    print(f"\n[A] Embedding gallery, nmax={nmax}")
    pr = primes_up_to(nmax)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, kind in zip(ax, ["sacks", "polar", "logspiral"]):
        P = embed(pr, kind, alpha=2.0)
        a.scatter(P[:, 0], P[:, 1], s=0.7, color="k")
        a.set_aspect("equal"); a.set_title(kind, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Prime embeddings")
    fig.tight_layout(); fig.savefig("A_embeddings.png", dpi=130)
    print("    -> A_embeddings.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all")
    ap.add_argument("--nmax", type=int, default=20000)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    s = "all" if a.all else a.section
    if s in ("all", "A"): section_A(a.nmax)
    if s in ("all", "B"): section_B(a.nmax)
    if s in ("all", "C"): section_C()
    if s in ("all", "D"): section_D()


if __name__ == "__main__":
    main()
