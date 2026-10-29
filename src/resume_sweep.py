#!/usr/bin/env python3
"""
resume_sweep.py -- continue an interrupted sweep from its last checkpoint.

A sweep JSON written by the patched sweep_fast.py stores the full per-shift
cumulative counts at every checkpoint. The last one is a complete restart
state: seed `running` from it and sieve only [N_last, NMAX].

    python3 resume_sweep.py --json sweep_1e11.json --nmax 1e11 --seg 1e8

Checkpoints are recomputed for --nmax, so pass the SAME --nmax the original
run used or the grid will not line up. Output is appended to the same file
(a .bak copy is written first).

Validate with --selftest: runs to 5e8, resumes to 1e9, and compares against
a direct run to 1e9. All 600 shift counts must match exactly.
"""

import argparse, json, os, shutil, time
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


def pack64(seg):
    b = np.packbits(seg, bitorder="little")
    pad = (-len(b)) % 8
    # + 4 whole zero words of headroom: the shifted operand is built from
    # W[q:-1] | W[q+1:], which is one word shorter than W, so without slack
    # the last word of the left operand falls outside the overlap and up to
    # 64 values per segment go uncounted at the largest shifts.
    b = np.concatenate([b, np.zeros(pad + 32, np.uint8)])
    return b.view(np.uint64)


def shifted_and_count(W, Wleft, s):
    q, r = divmod(s, 64)
    if q >= len(W):
        return 0
    if r == 0:
        sh = W[q:]
    else:
        sh = (W[q:-1] >> np.uint64(r)) | (W[q + 1:] << np.uint64(64 - r))
    n = min(len(Wleft), len(sh))
    return int(np.bitwise_count(Wleft[:n] & sh[:n]).sum())


def clear_from(W, L):
    out = W.copy()
    q, r = divmod(L, 64)
    if q < len(out):
        if r:
            out[q] &= np.uint64((1 << r) - 1)
            out[q + 1:] = 0
        else:
            out[q:] = 0
    return out


def li2(x):
    return quad(lambda t: 1.0 / np.log(t) ** 2, 2, x, limit=500)[0]


def run(nmax, seg, out, start_N=0, start_counts=None, prior=None, quiet=False):
    """Sweep (start_N, nmax], seeded from start_counts. Returns results list."""
    shifts = np.arange(2, SMAX + 1, 2)
    spf = smallest_factor(SMAX)
    hl = np.array([sum(np.log((p - 1) / (p - 2))
                       for p in odd_factors(int(s), spf)) for s in shifts])

    checks = sorted(set(int(x) for x in np.logspace(6, np.log10(nmax), 30)))
    checks = [c for c in checks if c > start_N]
    bounds = sorted(set(checks) | set(range(0, nmax + 1, seg)) | {nmax})
    bounds = [b for b in bounds if start_N <= b <= nmax]
    if not bounds or bounds[0] != start_N:
        bounds = [start_N] + bounds

    running = (np.array(start_counts, np.int64) if start_counts is not None
               else np.zeros(len(shifts), np.int64))
    bp = base_primes(int(nmax ** 0.5) + 1)
    results = list(prior) if prior else []
    t0 = time.time()
    if not quiet:
        print(f"resuming from N={start_N:,} to {nmax:,}; "
              f"{len(checks)} checkpoints remain", flush=True)

    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        s_ = seg_sieve(lo, hi + SMAX + 2, bp)
        L = hi - lo
        W = pack64(s_)
        del s_
        WL = clear_from(W, L)
        for j, sft in enumerate(shifts):
            running[j] += shifted_and_count(W, WL, int(sft))
        del W, WL
        if hi in checks:
            Lk = li2(hi)
            d = np.log(running / Lk) - hl
            w = running / running.sum()
            direct = float((w * d).sum())
            ratio = float(np.mean(running / (np.exp(hl) * Lk)))
            sd = float(np.sqrt((w * (d - direct) ** 2).sum()))
            results.append(dict(N=hi, pi2=int(running[0]), li2=Lk,
                                direct=direct, D=direct - LOG2C2,
                                ratio=ratio, rel=ratio / (2 * C2) - 1,
                                sd=sd, se=sd / np.sqrt(len(d)),
                                counts=running.tolist()))
            if not quiet:
                print(f"  N={hi:>14,}  pi_2={running[0]:>13,}  "
                      f"D={direct-LOG2C2:+.6f}  rel={100*(ratio/(2*C2)-1):+.4f}%"
                      f"  se={sd/np.sqrt(len(d)):.2e}  [{time.time()-t0:7.0f}s]",
                      flush=True)
            if out:
                json.dump(results, open(out, "w"), indent=1)
    return results


def selftest():
    print("validating resume against a direct run ...")
    a = run(500_000_000, 50_000_000, None, quiet=True)
    b = run(1_000_000_000, 50_000_000, None, start_N=a[-1]["N"],
            start_counts=a[-1]["counts"], prior=a, quiet=True)
    c = run(1_000_000_000, 50_000_000, None, quiet=True)
    ok = b[-1]["counts"] == c[-1]["counts"]
    print(f"  resumed final N = {b[-1]['N']:,}   direct final N = {c[-1]['N']:,}")
    print(f"  pi_2 resumed={b[-1]['pi2']:,}  direct={c[-1]['pi2']:,}")
    print(f"  all 600 shift counts identical: {ok}")
    if not ok:
        bad = sum(1 for x, y in zip(b[-1]["counts"], c[-1]["counts"]) if x != y)
        print(f"  {bad} shifts differ")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="sweep_1e11.json")
    ap.add_argument("--nmax", type=float, default=1e11)
    ap.add_argument("--seg", type=float, default=1e8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    prior = json.load(open(a.json))
    prior = [r for r in prior if "counts" in r]
    if not prior:
        print("No checkpoint in that file carries per-shift counts.")
        print("It was written by an older sweep_fast.py -- must start over.")
        return
    last = prior[-1]
    print(f"{a.json}: {len(prior)} usable checkpoints, "
          f"last N = {last['N']:,} ({100*last['N']/a.nmax:.1f}% of target)")
    shutil.copy(a.json, a.json + ".bak")
    run(int(a.nmax), int(a.seg), a.json,
        start_N=last["N"], start_counts=last["counts"], prior=prior)
    print(f"\ndone -> {a.json}")


if __name__ == "__main__":
    main()
