"""Figure 1 of the FinFM paper: cascade size error grows like 1/(1 - lambda),
breach-shock error does not.

Random liquidation books, not data. Each trial draws a smooth book mu whose
cascade map has slope at most lambda, perturbs it into a nearby book nu
(density noise plus small atoms), and computes both cascades exactly.
Price impact is linear, phi(v) = L v, so least fixed points are exact.

Run: python scripts/make_figure.py  (writes paper/figures/cascade_error.pdf)
Needs numpy and matplotlib. Seeded, so the figure is reproducible.
"""

import pathlib
import sys

import numpy as np

rng = np.random.default_rng(20261119)
OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures" / "cascade_error.pdf"


class Book:
    """Liquidation map: point masses plus piecewise-constant densities."""

    def __init__(self, ax=(), am=(), edges=(), dens=()):
        self.ax = np.asarray(ax, float)
        self.am = np.asarray(am, float)
        self.edges = np.asarray(edges, float)
        self.dens = np.asarray(dens, float)

    def V(self, D, left=False):
        D = np.atleast_1d(np.asarray(D, float))
        hit = self.ax[None, :] < D[:, None] if left else self.ax[None, :] <= D[:, None]
        out = (hit * self.am[None, :]).sum(1)
        if len(self.dens):
            lo, hi = self.edges[:-1], self.edges[1:]
            cover = np.clip(D[:, None] - lo[None, :], 0, (hi - lo)[None, :])
            out = out + (cover * self.dens[None, :]).sum(1)
        return out

    def breakpoints(self):
        return np.unique(np.concatenate([self.ax, self.edges]))

    def density_at(self, a, b):
        if np.any((self.ax >= a) & (self.ax <= b) & (self.am > 0)):
            return np.inf
        if not len(self.dens):
            return 0.0
        lo, hi = self.edges[:-1], self.edges[1:]
        mask = (hi > a) & (lo <= b)
        return float(self.dens[mask].max()) if mask.any() else 0.0


def lfp(book, delta, L):
    """Least fixed point of D -> delta + L V(D): inf{D >= delta : D - L V(D) >= delta}."""
    bps = book.breakpoints()
    pts = np.concatenate([[delta], bps[bps > delta], [np.inf]])
    for a, b in zip(pts[:-1], pts[1:]):
        Ga = a - L * book.V(a)[0]
        if Ga >= delta - 1e-15:
            return a
        mid = a + 1.0 if np.isinf(b) else 0.5 * (a + b)
        slope = 1 - L * book.density_at(mid, mid)
        if slope > 0:
            D = a + (delta - Ga) / slope
            if D < b:
                return D
    raise RuntimeError("no fixed point found")


def sup_gap(b1, b2):
    pts = np.unique(np.concatenate([b1.breakpoints(), b2.breakpoints()]))
    right = np.abs(b1.V(pts) - b2.V(pts))
    left = np.abs(b1.V(pts, left=True) - b2.V(pts, left=True))
    return float(max(right.max(), left.max()))


def breach_shock(book, w, L, lo=-5.0, hi=5.0, iters=60):
    """inf{delta : D*(delta) >= w}, by bisection (D* is nondecreasing in delta)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if lfp(book, mid, L) >= w:
            hi = mid
        else:
            lo = mid
    return hi


def trial(lam):
    L = rng.uniform(0.5, 3.0)
    K, top = 16, 40.0
    edges = np.sort(np.concatenate([[0.0, top], rng.uniform(0, top, K - 1)]))
    dens = lam * rng.uniform(0.85, 1.0, K) / L
    dens[np.argmax(edges[1:] - edges[:-1])] = lam / L  # slope bound attained somewhere
    mu = Book(edges=edges, dens=dens)
    delta = rng.uniform(0.05, 0.4)
    # Model error near the current price: misplaced selling in [0, 2 delta].
    # Extra atoms add selling; a density cut on the first bin removes some.
    ax = rng.uniform(0, 2 * delta, 6)
    am = rng.exponential(0.01, 6)
    cut = np.ones(K)
    cut[0] = 1 - rng.uniform(0, 0.05)
    nu = Book(ax, am, edges, dens * cut)  # nu never steeper than mu
    eta = sup_gap(mu, nu)
    size_err = abs(lfp(mu, delta, L) - lfp(nu, delta, L)) / (L * eta)
    w = lfp(mu, delta, L)  # level the true cascade reaches
    br_err = abs(breach_shock(mu, w, L) - breach_shock(nu, w, L)) / (L * eta)
    return size_err, br_err


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = np.repeat([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98], 40)
    rows = np.array([trial(l) for l in lams])
    bound = 1 / (1 - lams)
    assert np.all(rows[:, 0] <= bound + 1e-9), "Theorem 1 violated"
    assert np.all(rows[:, 1] <= 1 + 1e-6), "Proposition 5 violated"

    # TrueType fonts in the PDF: AAAI rejects Type 3 fonts.
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    jit = rng.uniform(-0.006, 0.006, len(lams))
    grid = np.linspace(0.05, 0.985, 200)
    ax.plot(grid, 1 / (1 - grid), color="#1f4e79", lw=1.2, label=r"bound $1/(1-\bar\lambda)$ (Thm. 1)")
    ax.scatter(lams + jit, rows[:, 0], s=6, color="#1f4e79", alpha=0.55, lw=0, label="cascade size error")
    ax.axhline(1.0, color="#b5462e", lw=1.0, ls="--", label="bound 1 (breach shocks)")
    ax.scatter(lams + jit, rows[:, 1], s=6, color="#b5462e", alpha=0.55, lw=0, marker="^", label="breach shock error")
    ax.set_yscale("log")
    ax.set_xlabel(r"slope bound $\bar\lambda$ of the cascade map")
    ax.set_ylabel(r"error $/\ L_\varphi \eta$")
    ax.set_xlim(0, 1)
    lo = max(rows.min() * 0.7, 1e-3)
    ax.set_ylim(lo, 400)
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    by = {l: rows[lams == l] for l in np.unique(lams)}
    for l, r in by.items():
        print(f"lambda={l:.2f}  size/bound max={np.max(r[:, 0] * (1 - l)):.3f}  breach max={r[:, 1].max():.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
