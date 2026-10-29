#!/usr/bin/env python3
"""
hecke_zeros.py -- independent reference zeros for L(s, xi_k) over Z[i].

The measured k>=1 spectra need a check that does not come from the same
code that produced them. LMFDB identifies the k=1 object as newform
4.5.b.a (weight 5, level 4, CM by Q(i)) and its coefficients match the
character exactly (a_2 = (1+i)^4 = -4, a_5 = (2+i)^4+(2-i)^4 = -14).

So compute that L-function's zeros from scratch:

  theta series   a_n = (1/4) sum_{a^2+b^2=n} (a+bi)^{4k}
                 (each ideal has 4 generators and (i*alpha)^{4k} = alpha^{4k})

  completed L    Lambda(s) = sum_n a_n [ X^{-s} Gamma(s, X)
                                       + eps X^{s-w} Gamma(w-s, X) ],
                 X = 2*pi*n/sqrt(N),  w = 4k+1 the classical weight

  zeros          Lambda(w/2 + it) = 0  ->  ordinate t, which is exactly the
                 analytic normalisation the measured spectrum works in.

Converges exponentially in n; ~60 terms is plenty.
"""

import numpy as np
from mpmath import mp, mpf, mpc, gammainc, gamma, findroot, pi as mppi

mp.dps = 30


def theta_coeffs(k, nmax):
    """a_n = (1/4) sum_{a^2+b^2=n} (a+bi)^{4k}, n = 1..nmax"""
    e = 4 * k
    a = [mpc(0)] * (nmax + 1)
    R = int(nmax ** 0.5) + 1
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            n = x * x + y * y
            if 1 <= n <= nmax:
                a[n] += mpc(x, y) ** e
    return [z / 4 for z in a]


def make_lambda(k, N=4, nterms=80, eps=1):
    w = 4 * k + 1                      # classical weight
    a = theta_coeffs(k, nterms)
    sq = mpf(N) ** mpf(0.5)

    def Lam(s):
        tot = mpc(0)
        for n in range(1, nterms + 1):
            if a[n] == 0:
                continue
            X = 2 * mppi * n / sq
            tot += a[n] * (X ** (-s) * gammainc(s, X)
                           + eps * X ** (s - w) * gammainc(w - s, X))
        return tot
    return Lam, w


def find_eps(k):
    """Lambda is real on the critical line for eps=+1, imaginary for eps=-1."""
    best = None
    for e in (1, -1):
        Lam, w = make_lambda(k, eps=e)
        v = Lam(mpc(mpf(w) / 2, mpf("3.7")))
        score = abs(v.imag) / (abs(v) + mpf("1e-40"))
        if best is None or score < best[0]:
            best = (score, e)
    return best[1]


def zeros(k, tmax, step=0.05):
    eps = find_eps(k)
    Lam, w = make_lambda(k, eps=eps)
    half = mpf(w) / 2

    def Z(t):
        return Lam(mpc(half, mpf(t))).real

    ts = np.arange(0.2, tmax, step)
    vals = [float(Z(float(t))) for t in ts]
    out = []
    for i in range(len(ts) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            try:
                r = float(findroot(Z, (float(ts[i]), float(ts[i + 1])),
                                   solver="bisect", tol=1e-18))
                if all(abs(r - z) > 1e-4 for z in out):
                    out.append(r)
            except Exception:
                pass
    return eps, sorted(out)


MEASURED = {
    1: [7.536, 9.948, 12.104, 15.196, 16.760, 18.556, 19.800, 21.524],
    2: [4.116, 7.392, 11.252, 12.784, 15.068, 16.128, 19.096, 20.432],
}

if __name__ == "__main__":
    for k in (1, 2):
        w = 4 * k + 1
        print(f"\n=== k={k}: weight {w}, level 4, CM by Q(i) "
              f"{'(newform 4.5.b.a)' if k==1 else ''} ===")
        a = theta_coeffs(k, 10)
        print("  theta coefficients a_1..a_9: "
              + ", ".join(str(int(round(float(z.real)))) for z in a[1:10]))
        eps, zs = zeros(k, 24.0)
        print(f"  root number eps = {eps:+d}")
        print(f"  computed zeros: " + ", ".join(f"{z:.4f}" for z in zs[:10]))
        m = MEASURED[k]
        print(f"\n  {'computed':>12}{'measured':>12}{'delta':>10}")
        for z in zs[:8]:
            j = int(np.argmin([abs(z - x) for x in m]))
            print(f"  {z:>12.4f}{m[j]:>12.4f}{m[j]-z:>+10.4f}")
