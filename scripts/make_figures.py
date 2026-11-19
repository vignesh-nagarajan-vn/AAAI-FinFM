"""Figures of the FinFM paper.

Figure 1 (cascade_geometry.pdf): how map error becomes cascade error, drawn for
  exact two-line books with linear impact. Panels: a smooth map, a wall within
  reach, and the cascade as a function of the shock.
Figure 2 (cascade_error.pdf): cascade size error grows like 1/(1 - lambda),
  breach-shock error does not. Random liquidation books, not data. Each trial
  draws a smooth book mu whose cascade map has slope at most lambda, perturbs it
  into a nearby book nu (density noise plus small atoms), and computes both
  cascades exactly.
Figure 3 (impact_transmission.pdf): the price impact a liquidator's sale on a
  constant-product pool transmits to the reference exchange through arbitrage,
  simulated round by round, against the first-order formula and the bound L_phi v.

Price impact in Figures 1 and 2 is linear, phi(v) = L v, so least fixed points are
exact. All text in the figures is 9 point and set by LaTeX in the paper's fonts, so
the PDFs embed only Type 1 fonts, as the AAAI-27 kit requires. Needs a LaTeX install.

Run: python scripts/make_figures.py  (writes paper/figures/*.pdf)
Needs numpy and matplotlib. Seeded, so the figures are reproducible.
"""

import math
import pathlib
import sys

import numpy as np

rng = np.random.default_rng(20261119)
FIG = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures"

BLUE = "#1f4e79"   # true book; contrast 8.7:1 on white
RED = "#b5462e"    # model book; contrast 5.4:1 on white
GRAY = "#4d4d4d"
BLACK = "#000000"
COL, TEXT = 3.3125, 7.0  # AAAI-27 column and text widths in inches; figures are included at scale 1


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


# ------------------------------------------------------------ Figure 2 (random books)

def figure_error(plt):
    lams = np.repeat([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98], 40)
    rows = np.array([trial(l) for l in lams])
    bound = 1 / (1 - lams)
    assert np.all(rows[:, 0] <= bound + 1e-9), "Theorem 1 violated"
    assert np.all(rows[:, 1] <= 1 + 1e-6), "Proposition 4 violated"

    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

    fig, ax = plt.subplots(figsize=(COL, 2.45))
    jit = rng.uniform(-0.006, 0.006, len(lams))
    grid = np.linspace(0.05, 0.985, 200)
    h1, = ax.plot(grid, 1 / (1 - grid), color=BLUE, lw=1.2)
    h2 = ax.scatter(lams + jit, rows[:, 0], s=7, color=BLUE, lw=0, marker="o")
    h3 = ax.axhline(1.0, color=RED, lw=1.0, ls="--")
    h4 = ax.scatter(lams + jit, rows[:, 1], s=9, color=RED, lw=0, marker="^")
    ax.set_yscale("log")
    ax.set_xlabel(r"slope bound $\bar\lambda$")
    ax.set_ylabel(r"error $/\,L_\varphi\eta$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.005, 200)
    ticks = [0.01, 0.1, 1, 10, 100]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "1", "10", "100"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend([h1, h2, h3, h4],
              [r"size bound $1/(1\mm\bar\lambda)$", "size error", "breach bound 1", "breach error"],
              ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              handlelength=1.8, columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "cascade_error.pdf")
    plt.close(fig)
    for l in np.unique(lams):
        r = rows[lams == l]
        print(f"lambda={l:.2f}  size/bound max={np.max(r[:, 0] * (1 - l)):.3f}  breach max={r[:, 1].max():.3f}")


# ------------------------------------------------------------ Figure 1 (geometry)

def figure_geometry(plt):
    """Two-line books with L = 1 and mu's density lam on [0, inf).

    (a) lam = 0.8, delta = 0.2, model error L eta = 0.1 placed at drop 0.
    (b) lam = 0.5, delta = 0.5, L eta = 0.2, a wall of mass L m = 0.8 at 1.2,
        which is D*_mu + r / 2 with r = 0.4, as in Proposition 2.
    (c) the cascade D*(delta) for the books of (b).
    Placing the error at drop 0 makes T_nu = T_mu + L eta exactly, so every
    fixed point below is closed form and checked against lfp().
    """
    # (a)
    lam_a, d_a, e_a = 0.8, 0.2, 0.1
    mu_a = Book(edges=[0.0, 50.0], dens=[lam_a])
    nu_a = Book([0.0], [e_a], [0.0, 50.0], [lam_a])
    Dm_a, Dn_a = lfp(mu_a, d_a, 1.0), lfp(nu_a, d_a, 1.0)
    r_a = e_a / (1 - lam_a)
    assert abs(Dm_a - 1.0) < 1e-12 and abs(Dn_a - Dm_a - r_a) < 1e-12

    # (b) and (c)
    lam_b, d_b, e_b, wall, m = 0.5, 0.5, 0.2, 1.2, 0.8
    mu_b = Book([wall], [m], [0.0, 50.0], [lam_b])
    nu_b = Book([0.0, wall], [e_b, m], [0.0, 50.0], [lam_b])
    Dm_b, Dn_b = lfp(mu_b, d_b, 1.0), lfp(nu_b, d_b, 1.0)
    r_b = e_b / (1 - lam_b)
    assert abs(Dm_b - 1.0) < 1e-12 and abs(Dn_b - 3.0) < 1e-12
    assert abs(wall - (Dm_b + r_b / 2)) < 1e-12 and Dn_b - Dm_b >= m

    def T(book, delta, D):
        return delta + book.V(D)

    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 2.2))

    # Panel (a): smooth map.
    ax = axs[0]
    D = np.linspace(0, 2.0, 200)
    ax.plot(D, D, color=GRAY, lw=0.8, ls=":")
    ax.plot(D, T(mu_a, d_a, D), color=BLUE, lw=1.3, label=r"true $T_\mu$")
    ax.plot(D, T(nu_a, d_a, D), color=RED, lw=1.3, ls="--", label=r"model $T_\nu$")
    ax.plot([Dm_a], [Dm_a], "o", color=BLUE, ms=5)
    ax.plot([Dn_a], [Dn_a], "s", color=RED, ms=5, mfc="white", mew=1.2)
    x0 = 0.6
    ax.annotate("", xy=(x0, T(nu_a, d_a, x0)[0]), xytext=(x0, T(mu_a, d_a, x0)[0]),
                arrowprops=dict(arrowstyle="-", lw=1.6, color="black", shrinkA=0, shrinkB=0))
    ax.annotate(r"$L_\varphi\eta$", xy=(x0, T(nu_a, d_a, x0)[0] - 0.05), xytext=(0.3, 1.05),
                ha="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY, shrinkA=1, shrinkB=1))
    ax.annotate("", xy=(Dn_a, 0.45), xytext=(Dm_a, 0.45),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="black", shrinkA=0, shrinkB=0))
    ax.plot([Dm_a, Dm_a], [0.45, Dm_a], color=GRAY, lw=0.6, ls=":")
    ax.plot([Dn_a, Dn_a], [0.45, Dn_a], color=GRAY, lw=0.6, ls=":")
    ax.text(0.5 * (Dm_a + Dn_a), 0.18, r"$r=L_\varphi\eta/(1\mm\bar\lambda)$", ha="center")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"drop $D$")
    ax.set_ylabel(r"next drop $T(D)$")
    ax.set_title("(a) smooth map", fontsize=9)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, borderaxespad=0.2)

    # Panel (b): a wall within reach.
    ax = axs[1]
    lo = np.linspace(0, wall, 100, endpoint=False)
    hi = np.linspace(wall, 3.4, 100)
    ax.plot([0, 3.4], [0, 3.4], color=GRAY, lw=0.8, ls=":")
    for seg in (lo, hi):
        ax.plot(seg, T(mu_b, d_b, seg), color=BLUE, lw=1.3)
        ax.plot(seg, T(nu_b, d_b, seg), color=RED, lw=1.3, ls="--")
    ax.axvline(wall, color=GRAY, lw=0.6)
    ax.text(wall + 0.06, 0.15, r"wall, mass $m$")
    ax.plot([Dm_b], [Dm_b], "o", color=BLUE, ms=5)
    ax.plot([Dn_b], [Dn_b], "s", color=RED, ms=5, mfc="white", mew=1.2)
    ax.annotate("", xy=(Dn_b, 0.75), xytext=(Dm_b, 0.75),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="black", shrinkA=0, shrinkB=0))
    ax.plot([Dm_b, Dm_b], [0.75, Dm_b], color=GRAY, lw=0.6, ls=":")
    ax.plot([Dn_b, Dn_b], [0.75, Dn_b], color=GRAY, lw=0.6, ls=":")
    ax.text(2.05, 0.92, r"$\geq L_\varphi m$", ha="center")
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel(r"drop $D$")
    ax.set_title("(b) wall within reach", fontsize=9)

    # Panel (c): the cascade against the shock, for the books of (b).
    ax = axs[2]
    dg = np.linspace(0, 1.0, 1001)
    Dmu = np.array([lfp(mu_b, d, 1.0) for d in dg])
    Dnu = np.array([lfp(nu_b, d, 1.0) for d in dg])
    assert np.allclose(Dnu[:-200], Dmu[200:], atol=1e-9)   # nu's curve = mu's shifted left by L eta
    for curve, kw in ((Dmu, dict(color=BLUE, lw=1.3)), (Dnu, dict(color=RED, lw=1.3, ls="--"))):
        jump = np.where(np.diff(curve) > 0.5)[0]
        cut = jump[0] + 1
        ax.plot(dg[:cut], curve[:cut], **kw)
        ax.plot(dg[cut:], curve[cut:], **kw)
    level = 2.0
    b_mu, b_nu = breach_shock(mu_b, level, 1.0, 0, 1), breach_shock(nu_b, level, 1.0, 0, 1)
    assert abs((b_mu - b_nu) - e_b) < 1e-9
    ax.axhline(level, color=GRAY, lw=0.6, ls=":")
    ax.annotate("", xy=(b_mu, level), xytext=(b_nu, level),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="black", shrinkA=0, shrinkB=0))
    ax.text(0.5 * (b_mu + b_nu), level + 0.12, r"breach: $L_\varphi\eta$", ha="center", va="bottom")
    ax.annotate("", xy=(d_b, Dn_b), xytext=(d_b, Dm_b),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="black", shrinkA=0, shrinkB=0))
    ax.text(d_b - 0.04, 1.45, "size", ha="right")
    ax.plot([d_b], [Dm_b], "o", color=BLUE, ms=5)
    ax.plot([d_b], [Dn_b], "s", color=RED, ms=5, mfc="white", mew=1.2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel(r"shock $\delta$")
    ax.set_ylabel(r"cascade $D^\ast(\delta)$")
    ax.set_title("(c) cascade against shock", fontsize=9)

    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(FIG / "cascade_geometry.pdf")
    plt.close(fig)
    print(f"geometry: (a) D*={Dm_a:.3f}, {Dn_a:.3f}, r={r_a:.3f}; (b) D*={Dm_b:.3f}, {Dn_b:.3f}, r={r_b:.3f}; "
          f"(c) breach shocks {b_nu:.3f}, {b_mu:.3f}")


# ------------------------------------------------------------ Figure 3 (impact constant)

class Pool:
    """Constant-product pool with fee f on the input; reserves x (collateral), y (numeraire)."""

    def __init__(self, x, y, f):
        self.x, self.y, self.f = float(x), float(y), float(f)

    @property
    def price(self):
        return self.y / self.x

    def sell(self, v):
        vin = v * (1 - self.f)
        out = self.y * vin / (self.x + vin)
        self.x += v
        self.y -= out
        return out

    def buy_to_price(self, target):
        if target <= self.price:
            return 0.0
        a, b, c = 1 - self.f, self.y * (2 - self.f), self.y ** 2 - self.x * self.y * target
        q = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        qin = q * (1 - self.f)
        out = self.x * qin / (self.y + qin)
        self.y += q
        self.x -= out
        return out

    def sell_to_price(self, target):
        if target >= self.price:
            return 0.0
        a, b, c = 1 - self.f, self.x * (2 - self.f), self.x ** 2 - self.x * self.y / target
        v = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        self.sell(v)
        return v


def phi_dex(X, f, v):
    return math.log(X + v) + math.log(X + (1 - f) * v) - 2 * math.log(X)


def resolve(X, X0, p, f, psi, a, tau, v, steps=2000):
    """Same simulation as math-derivations/model_layer_checks.py in the source project:
    the liquidator sells v on the pool, then an arbitrageur closes a fraction a of the
    log gap each round while it exceeds tau, and the exchange moves by psi * (collateral
    traded there) / X0. Returns the exchange's resolved log drop."""
    pool = Pool(X, X * p, f)
    cex = math.log(p)
    pool.sell(v)
    for _ in range(steps):
        gap = math.log(pool.price) - cex
        if abs(gap) <= tau or abs(gap) < 1e-15:
            break
        target = pool.price * math.exp(-a * gap)
        if gap < 0:
            cex -= psi * pool.buy_to_price(target) / X0
        else:
            cex += psi * pool.sell_to_price(target) / X0
    return math.log(p) - cex


def figure_impact(plt):
    X = X0 = 1e4
    p, f, psi, tau, a = 2000.0, 0.003, 0.5, 0.01, 0.5
    k = psi * (1 - f) * X / ((2 - f) * X0)
    L = psi * (1 - f) / (X0 * (1 + k))
    vs = np.linspace(0, 0.06, 13) * X
    full = np.array([resolve(X, X0, p, f, psi, a, 0.0, v) for v in vs])
    dead = np.array([resolve(X, X0, p, f, psi, a, tau, v) for v in vs])
    first = np.array([k / (1 + k) * phi_dex(X, f, v) for v in vs])
    bound = L * vs
    assert np.all(full <= bound + 1e-12), "transmitted impact exceeds L_phi v"
    kpost = psi * (1 - f) * (X + vs) / ((2 - f) * X0)
    assert np.all(full - dead <= kpost * tau / (1 + kpost) * (1 + 1e-2) + 1e-12), "dead zone bound violated"

    vv = np.linspace(0, 0.06, 200) * X
    x, xx = vs / X * 100, vv / X * 100
    fo = np.array([k / (1 + k) * phi_dex(X, f, v) for v in vv]) * 100
    allow = k * tau / (1 + k) * 100
    fig, ax = plt.subplots(figsize=(COL, 2.3), layout="constrained")
    g0 = ax.fill_between(xx, np.maximum(fo - allow, 0), fo, color="#d9d9d9", lw=0)
    g1, = ax.plot(xx, L * vv * 100, color=BLACK, lw=0.9, ls="--")
    g2, = ax.plot(xx, fo, color=BLUE, lw=1.2)
    g3, = ax.plot(x, full * 100, "o", color=BLUE, ms=4)
    g4, = ax.plot(x, dead * 100, "^", color=RED, ms=4.5, mfc="white", mew=1.0)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 2.6)
    ax.set_xlabel(r"sale $v$, \% of pool reserve $X$")
    ax.set_ylabel(r"exchange drop, \%")
    fig.legend([g1, g2, g0, g3, g4],
               [r"bound $L_\varphi v$", "first order", r"$k\tau/(1+k)$ below it",
                "simulated", r"simulated, dead zone $\tau$"],
               ncol=2, frameon=False, loc="outside upper center",
               handlelength=1.8, columnspacing=1.0)
    fig.savefig(FIG / "impact_transmission.pdf")
    plt.close(fig)
    gap = np.max(np.abs(full[1:] - first[1:]) / first[1:])
    print(f"impact: k={k:.4f}, L*X={L * X:.4f}, max rel gap simulated vs first order {gap:.4f}, "
          f"max dead-zone shortfall {np.max(full - dead):.5f} vs k tau/(1+k) {k * tau / (1 + k):.5f}")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 9 pt text set by LaTeX in the paper's own fonts (newtxtext, Computer Modern
    # math), so the PDFs embed Type 1 fonts: AAAI rejects Type 3 and CID fonts.
    plt.rcParams.update({
        "text.usetex": True,
        # Matplotlib's Type 1 subsetting drops glyph slot 0, where TeX math fonts
        # keep the minus sign, so figures write minus as \mm, from the text font.
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage{textcomp}"
                               r"\newcommand{\mm}{\mathbin{\mbox{\textminus}}}",
        "font.size": 9, "font.family": "serif", "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    })
    FIG.mkdir(parents=True, exist_ok=True)
    figure_error(plt)      # first, so the seeded draws match the earlier version
    figure_geometry(plt)
    figure_impact(plt)
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    sys.exit(main())
