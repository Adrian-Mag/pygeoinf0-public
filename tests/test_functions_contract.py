"""The contract every function of the bank passes (plan 021; experiments/EXP-037-function-bank/DESIGN.md 4): for every
registered family, every domain kind it declares, the standard domains of that kind and every example:
vectorisation, finite values, the support, closed forms against an independent quadrature, interior support at a
distance from the boundary, the gradient against finite differences, membership, and the claims' statuses."""
import itertools
import math

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0.functions as F
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.functions import geometry as geo
from pygeoinf0.spaces.membership import check

FLAT = ("interval", "box", "disk", "ball", "shell", "cylinder")


def cases():
    out = []
    for name, e in sorted(F.entries().items()):
        for k in e.kinds:
            for i, D in enumerate(geo.standard(k)):
                for j, params in enumerate(e.example_params(D)):
                    out.append(pytest.param(name, D, params, id=f"{name}-{k}{i}-{j}"))
    return out


def build(name, D, params):
    return F.entries()[name].builder(D, **params)


def _gl_box(lo, hi, n=48):
    t, w = np.polynomial.legendre.leggauss(n)
    axes = [(0.5 * (b - a) * (t + 1) + a, 0.5 * (b - a) * w) for a, b in zip(lo, hi)]
    G = np.meshgrid(*[a for a, _ in axes], indexing="ij")
    W = np.ones_like(G[0])
    for i, (_, wi) in enumerate(axes):
        shape = [1] * len(axes)
        shape[i] = -1
        W = W * wi.reshape(shape)
    return [g.ravel() for g in G], W.ravel()


def _polar(f, c, r, breaks, D=None):
    """int f over the geodesic ball (c, r) in polar coordinates about c: Gauss-Legendre in the distance with panels
    split at the support's breaks, Gauss-Legendre in cos(theta) and the periodic trapezoid rule in the azimuth. On a
    sphere (radius R) geodesic polar coordinates, x = cos(rho/R) c + R sin(rho/R) e(phi), area R sin(rho/R) drho dphi;
    on a circle x = c rotated by +-rho/R; on a torus (flat) the Euclidean ones, wrapped by the minimum image."""
    k = None if D is None else geo.kind(D)
    d = geo.dimension(D) if k in ("sphere", "circle") else len(c)
    edges = sorted({0.0, r, *[b for b in breaks if 0 < b < r], *[r * (1 - 2.0 ** -j) for j in range(1, 7)]})
    t, w = np.polynomial.legendre.leggauss({1: 80, 2: 64, 3: 40}[d])
    rho = np.concatenate([0.5 * (b - a) * (t + 1) + a for a, b in itertools.pairwise(edges)])
    wr = np.concatenate([0.5 * (b - a) * w for a, b in itertools.pairwise(edges)])
    if d == 1:
        dirs, wd = np.array([[1.0], [-1.0]]), np.ones(2)
    elif d == 2:
        phi = 2 * np.pi * np.arange(128) / 128
        dirs, wd = np.c_[np.cos(phi), np.sin(phi)], np.full(128, 2 * np.pi / 128)
    else:
        u, wu = np.polynomial.legendre.leggauss(40)
        phi = 2 * np.pi * np.arange(64) / 64
        U, P = np.meshgrid(u, phi, indexing="ij")
        s = np.sqrt(1 - U ** 2)
        dirs = np.c_[(s * np.cos(P)).ravel(), (s * np.sin(P)).ravel(), U.ravel()]
        wd = np.repeat(wu, 64) * 2 * np.pi / 64
    if k == "sphere":
        R = float(D.radius)
        n = c / np.linalg.norm(c)
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        E = dirs[:, :1] * e1 + dirs[:, 1:2] * np.cross(n, e1)              # unit tangents at c
        a = rho / R
        X = R * (np.cos(a)[:, None, None] * n + np.sin(a)[:, None, None] * E[None, :, :])
        W = (wr * R * np.sin(a))[:, None] * wd[None, :]
    elif k == "circle":
        R, phi0 = float(D.radius), np.arctan2(c[1] - D.centre[1], c[0] - D.centre[0])
        ang = phi0 + np.outer(rho / R, dirs[:, 0])
        X = np.stack([D.centre[0] + R * np.cos(ang), D.centre[1] + R * np.sin(ang)], axis=-1)
        W = wr[:, None] * wd[None, :]
    else:
        X = c + rho[:, None, None] * dirs[None, :, :]
        W = (wr * rho ** (d - 1))[:, None] * wd[None, :]
    return float(np.sum(W * f(*np.moveaxis(X, -1, 0))))


def reference_integral(f, D):
    """An integral independent of the family's closed form: for radial functions with an interior support (breaks
    declared), geodesic polar coordinates about the centre split at the breaks; else Gauss-Legendre on the
    support's bounding box for compact supports on flat domains (two panels per axis); else a fine domain
    quadrature."""
    s, k = f.support, geo.kind(D)
    if s.kind == "interior" and s.centre is not None and s.breaks is not None:
        return _polar(f, np.asarray(s.centre, float), float(s.radius), s.breaks, D), 1e-9
    if f.family == "plateau":                                   # the box and its kinks: panels between them
        axes = []
        t, w = np.polynomial.legendre.leggauss(32)
        for b in f.breaks:
            pieces = [(0.5 * (q - p) * (t + 1) + p, 0.5 * (q - p) * w) for p, q in zip(b[:-1], b[1:])]
            axes.append((np.concatenate([n for n, _ in pieces]), np.concatenate([x for _, x in pieces])))
        grid = np.meshgrid(*[a[0] for a in axes], indexing="ij")
        weights = np.ones_like(grid[0])
        for i, a in enumerate(axes):
            weights = weights * a[1].reshape([-1 if j == i else 1 for j in range(len(axes))])
        return float(np.sum(weights * f(*grid))), 1e-10
    if s.kind != "full" and s.centre is not None and k in FLAT:
        c, r = np.asarray(s.centre, float), float(s.radius)
        total = 0.0
        for lo_hi in np.ndindex(*([2] * len(c))):
            lo = [c[i] - r + r * lo_hi[i] for i in range(len(c))]
            X, w = _gl_box(lo, [lo[i] + r for i in range(len(c))], 40 if len(c) == 3 else 64)
            total += w @ f(*X)
        return total, 1e-7
    X, w = geo.quadrature(D, {1: 2000, 2: 160}.get(geo.dimension(D), 48))
    return w @ f(*X.T), (1e-4 if k in ("ball", "shell", "cylinder") else 1e-5)     # this rule's own error on a curved boundary


@pytest.mark.parametrize("name, D, params", cases())
def test_contract(name, D, params):
    f = build(name, D, params)
    X, _ = geo.quadrature(D, 12)
    vals = f(*X.T)
    # 1-2: vectorisation and finite values
    assert vals.shape == (X.shape[0],) and np.all(np.isfinite(vals))
    for p in X[:: max(1, len(X) // 5)]:
        assert np.isclose(f(*p), f(*[np.atleast_1d(v) for v in p])[0])
    # 3: support
    s = f.support
    if s.kind != "full":
        if s.contains is not None:
            outside = ~np.asarray(s.contains(*X.T), bool)
        elif s.centre is not None:
            outside = geo.distance(D, s.centre, *X.T) > s.radius * (1 + 1e-9)
        else:
            outside = np.zeros(len(X), bool)
        assert np.all(np.abs(vals[outside]) <= 1e-12 * max(1.0, np.max(np.abs(vals))))
    # 4: closed forms against an independent quadrature
    if "integral" in f.closed:
        v = f.closed["integral"][0]
        q, tol = reference_integral(f, D)
        assert abs(q - v) <= tol * max(abs(v), 1e-300) + 1e-14, (q, v)
    # 5: interior support stays away from the boundary
    if s.kind == "interior" and s.centre is not None:
        assert s.radius < geo.room(D, s.centre) and s.radius < geo.injectivity(D)
    # 6: gradient against central differences
    if f.gradient is not None:
        P = X[:: max(1, len(X) // 7)]
        h = 1e-6 * geo.volume(D) ** (1.0 / geo.dimension(D))
        G = f.gradient(*P.T)
        scale = max(1e-12, np.max(np.abs(G)))
        for i in range(P.shape[1]):
            E = np.zeros(P.shape[1])
            E[i] = h
            fd = (f(*(P + E).T) - f(*(P - E).T)) / (2 * h)
            assert np.max(np.abs(fd - G[:, i])) <= 1e-5 * scale + 1e-9
    # 7: membership
    assert check(f, pg.L2(D))[0]
    try:
        H1 = pg.Sobolev(D, 1, length_scale=0.2)
        H2 = pg.Sobolev(D, 2, length_scale=0.2)
    except (Unsupported, NotImplementedError, TypeError, ValueError):
        H1 = H2 = None
    if H1 is not None:
        smax, strict = f.regularity.sobolev_bound()
        assert check(f, H1)[0] == (1 < smax if strict else 1 <= smax)
        if f.regularity.order == math.inf:
            assert check(f, H2)[0]
    # 8: claims
    for c in f.claims:
        assert c.status in ("theorem", "derived", "checked")
        if c.status == "theorem":
            assert c.reason


def test_bank_is_optional_and_lists_itself():
    import subprocess
    import sys
    code = "import sys, pygeoinf0; print('pygeoinf0.functions' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"
    names = [n for n, _, _ in F.catalogue()]
    assert "bump" in names and "gaussian" in names
    B = pg.Ball(1.0)
    assert F.on(B).bump(radius=0.3).family == "bump" and F.ball.bump(B, radius=0.3).family == "bump"
    with pytest.raises(pg.Refused, match="gaussian is not defined on a torus"):
        F.ball.gaussian(pg.Torus((0.0, 1.0), (0.0, 1.0)))


def test_refusals_and_membership_reasons():
    B = pg.Ball(1.0)
    with pytest.raises(pg.Refused, match="leaves the domain"):
        F.on(B).bump(radius=1.5)
    with pytest.raises(pg.Refused, match="must be positive"):
        F.on(B).gaussian(width=-1.0)
    f = F.on(B).gaussian()
    assert f.element(pg.L2(B)) is not None
    with pytest.raises(pg.Refused, match="integral 0"):
        F.on(B).bump(radius=0.3).scaled(0.0).normalised()


def test_framework_fixes_from_p1():
    """Fixes the P1 implementers asked for (workflow wf_2099323b-aa5): centres checked against the domain, graded
    radial panels for profiles steep at the edge, the band integral, and the multiplier key of scaled()."""
    from scipy.integrate import quad
    S = pg.Sphere(1.0)
    with pytest.raises(pg.Refused, match="is not on"):
        F.on(S).gaussian(centre=(0.0, 0.0, 2.0))
    with pytest.raises(pg.Refused, match="a centre is a point"):
        F.on(S).bump(centre=("a", "b", "c"))
    with pytest.raises(pg.Refused, match="is outside"):
        F.on(pg.Ball(1.0)).bump(centre=(2.0, 0.0, 0.0), radius=0.1)
    I = pg.Interval("[0, 1]")
    for k in (1e-4, 1.0):
        v = F.on(I).bump(radius=0.4, k=k).integral()[0]
        ref = 2 * quad(lambda r, k=k: np.exp(-k / (1 - (r / 0.4) ** 2)), 0, 0.4, epsabs=0, epsrel=1e-13, limit=500)[0]
        assert abs(v - ref) < 1e-13 * ref
    B = pg.Ball(1.0)
    band = geo.radial_shell_mass(B, lambda r: np.ones_like(r), 0.2, 0.5)
    assert np.isclose(band, 4 * np.pi / 3 * (0.5 ** 3 - 0.2 ** 3), rtol=1e-14)
    f = F.on(B).bump(radius=0.3).scaled(2.0)
    assert f.params["multiplier"] == 2.0 and "scale" not in f.params


def test_measures_targets_and_catalogue():
    """Plan 021 P2: f.measure() lives on the support (accurate SOLA functionals for narrow kernels), normalisation
    against a weighted measure, plain parameters, and FUNCTIONS.md is the generated catalogue."""
    from pathlib import Path

    from scipy.integrate import quad
    I = pg.Interval("[0, 1]")
    t = F.on(I).bump(centre=0.5, radius=0.1).normalised()
    assert t.params["centre"] == (0.5,) and type(t.params["centre"][0]) is float
    S = t.support_set()
    assert (S.a, S.b) == (0.4, 0.6)
    exact = quad(lambda x: t(np.array([x]))[0] * (np.sin(2 * np.pi * x) + x), 0.4, 0.6, epsabs=0, epsrel=1e-13)[0]
    for M in (pg.L2(I), pg.Sobolev(I, 1, length_scale=0.1)):
        v = pg.Functionals(M.riesz([t.measure(M.measure)]))(M(lambda x: np.sin(2 * np.pi * x) + x)).values[0]
        assert abs(v - exact) < 1e-10
    B = pg.Ball(1.0)
    tb = F.on(B).bump(centre=(0, 0, 0.4), radius=0.3).normalised()
    assert isinstance(tb.support_set(), pg.Ball) and tb.support_set().radius == 0.3
    assert F.on(B).gaussian().support_set() is B
    one = pg.Functionals(pg.L2(B).riesz([tb.measure()]))(pg.L2(B)(lambda x, y, z: 1 + 0 * x)).values[0]
    assert abs(one - 1) < 1e-6
    w = pg.LebesgueMeasure(I).with_density(lambda r: r ** 2, name="r^2")
    tw = F.on(I).bump(centre=0.6, radius=0.15).normalised(against=w)
    assert tw.closed["mass"][:2] == (1.0, "checked")
    M = pg.L2(w)
    assert abs(pg.Functionals(M.riesz([tw.measure(w)]))(M(lambda r: 1 + 0 * r)).values[0] - 1) < 1e-10
    assert abs(tw.scaled(2.0).closed["mass"][0] - 2.0) < 1e-15
    with pytest.raises(pg.Refused, match="needs the Lebesgue measure"):
        tw.measure(pg.LebesgueMeasure(pg.Interval(0, 2)))
    table = F.catalogue_table()
    assert f"{len(F.entries())} families" in table and "| `bump` |" in table
    assert (Path(__file__).parents[1] / "FUNCTIONS.md").read_text() == table + "\n", \
        "FUNCTIONS.md is stale: python -m pygeoinf0.functions > FUNCTIONS.md"
