#!/usr/bin/env python3
"""
sweep_fast.py -- bit-packed version of sweep.py. Same output, ~8x faster.

The bottleneck in sweep.py is one byte-wise AND over the whole segment per
shift: 600 passes x 2 bytes/element. Packing the segment to bits and using
uint64 words cuts memory traffic 8x, and np.bitwise_count does the counting
in one vectorised pass.

Counting p <= N with p, p+s prime becomes, in packed form:
    popcount( W_left  &  (W >> s bits) )
where W_left is W with every bit at index >= L cleared, so the tail of the
lookahead never leaks into the count.

Identical interface and JSON output to sweep.py.
    python3 sweep_fast.py --nmax 1e11 --seg 1e8
    python3 sweep_fast.py --nmax 1e8 --check      # verify vs byte method
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
    """bool array -> uint64 words, bit i of the array = bit i%64 of word i//64."""
    b = np.packbits(seg, bitorder="little")
    pad = (-len(b)) % 8
    # + 4 whole zero words of headroom: the shifted operand is built from
    # W[q:-1] | W[q+1:], which is one word shorter than W, so without slack
    # the last word of the left operand falls outside the overlap and up to
    # 64 values per segment go uncounted at the largest shifts.
    b = np.concatenate([b, np.zeros(pad + 32, np.uint8)])
    return b.view(np.uint64)


def shifted_and_count(W, Wleft, s):
    """popcount( Wleft & (W >> s bits) )."""
    q, r = divmod(s, 64)
    if q >= len(W):
        return 0
    if r == 0:
        sh = W[q:]
    else:
        hi = W[q:-1] >> np.uint64(r)
        lo = W[q + 1:] << np.uint64(64 - r)
        sh = hi | lo
    n = min(len(Wleft), len(sh))
    return int(np.bitwise_count(Wleft[:n] & sh[:n]).sum())


def clear_from(W, L):
    """copy of W with every bit at index >= L cleared"""
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


def check(nmax=20_000_000, seg=5_000_000):
    """byte method vs bit method, must agree exactly"""
    shifts = np.arange(2, SMAX + 1, 2)
    bp = base_primes(int(nmax ** 0.5) + 1)
    a = np.zeros(len(shifts), np.int64)
    b = np.zeros(len(shifts), np.int64)
    for lo in range(0, nmax, seg):
        hi = min(lo + seg, nmax)
        s_ = seg_sieve(lo, hi + SMAX + 2, bp)
        L = hi - lo
        for j, sft in enumerate(shifts):
            a[j] += np.count_nonzero(s_[:L] & s_[sft:sft + L])
        W = pack64(s_)
        WL = clear_from(W, L)
        for j, sft in enumerate(shifts):
            b[j] += shifted_and_count(W, WL, int(sft))
    ok = np.array_equal(a, b)
    print(f"byte vs bit over N={nmax:,}: identical = {ok}")
    if not ok:
        d = np.flatnonzero(a != b)
        print(f"  {len(d)} mismatches, first at s={shifts[d[0]]}: "
              f"{a[d[0]]} vs {b[d[0]]}")
    print(f"  pi_2 = {a[0]:,}")
    return ok


def main():
    global SMAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=float, default=1e9)
    ap.add_argument("--seg", type=float, default=1e8)
    ap.add_argument("--out", default="sweep_fast.json")
    ap.add_argument("--shifts-out", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smax", type=int, default=SMAX,
                    help="max shift; halving it roughly halves runtime "
                         "and raises the standard error by sqrt(2)")
    a = ap.parse_args()
    if a.check:
        check()
        return
    NMAX, M = int(a.nmax), int(a.seg)
    SMAX = a.smax

    shifts = np.arange(2, SMAX + 1, 2)
    spf = smallest_factor(SMAX)
    hl = np.array([sum(np.log((p - 1) / (p - 2))
                       for p in odd_factors(int(s), spf)) for s in shifts])

    checks = sorted(set(int(x) for x in np.logspace(6, np.log10(NMAX), 30)))
    bounds = sorted(set(checks) | set(range(0, NMAX + 1, M)) | {NMAX})
    bounds = [b for b in bounds if b <= NMAX]
    if bounds[0] != 0:
        bounds = [0] + bounds

    print(f"NMAX = {NMAX:,}   segment = {M:,}   {len(shifts)} shifts   "
          f"{len(checks)} checkpoints   [bit-packed]")
    bp = base_primes(int(NMAX ** 0.5) + 1)
    print(f"base primes to {int(NMAX**0.5)+1:,}: {len(bp):,}\n")

    running = np.zeros(len(shifts), np.int64)
    results, t0 = [], time.time()

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
            sd = float(np.sqrt((w * (d - direct) ** 2).sum()))      # per-shift sd
            se = sd / np.sqrt(len(d))                                # std error
            results.append(dict(N=hi, pi2=int(running[0]), li2=Lk,
                                direct=direct, D=direct - LOG2C2,
                                ratio=ratio, rel=ratio / (2 * C2) - 1,
                                sd=sd, se=se,
                                counts=running.tolist()))
            print(f"  N={hi:>14,}  pi_2={running[0]:>13,}  "
                  f"D={direct-LOG2C2:+.6f}  rel={100*(ratio/(2*C2)-1):+.4f}%  "
                  f"se={se:.2e}  [{time.time()-t0:7.0f}s]", flush=True)
            json.dump(results, open(a.out, "w"), indent=1)

    print(f"\nwrote {a.out}  ({time.time()-t0:.0f}s)")

    N = np.array([r["N"] for r in results], float)
    D = np.array([r["D"] for r in results])
    Ln = np.log(N)
    print(f"\nsign changes in D: {int(np.sum(np.diff(np.sign(D)) != 0))}")
    m = D < 0
    if m.sum() > 4:
        sl, ic = np.polyfit(Ln[m], np.log(-D[m]), 1)
        r = np.log(-D[m]) - (sl * Ln[m] + ic)
        print(f"power law  D ~ -N^alpha :  alpha = {sl:+.4f}  resid sd {r.std():.3f}")
        # alpha on the top two decades only, where fluctuation matters least
        hi_ = m & (N > N.max() / 100)
        if hi_.sum() > 4:
            sl2 = np.polyfit(Ln[hi_], np.log(-D[hi_]), 1)[0]
            print(f"  top 2 decades only    :  alpha = {sl2:+.4f}")
    for name, b in (("1/log N", 1 / Ln), ("1/log^2 N", 1 / Ln ** 2),
                    ("log^2N/sqrt(N)", Ln ** 2 / np.sqrt(N)),
                    ("1/sqrt(N)", 1 / np.sqrt(N)),
                    ("logN/sqrt(N)", Ln / np.sqrt(N))):
        c = float(np.linalg.lstsq(b[:, None], D, rcond=None)[0][0])
        print(f"  {name:<16} a={c:+.6f}   resid rms={(D-c*b).std():.3e}")


if __name__ == "__main__":
    main()
