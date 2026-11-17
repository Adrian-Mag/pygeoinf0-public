"""Separable functionals k (x) Y_lj on harmonic spaces, their structured Grams, and density kernels (plan 009 G2, G13;
EXP-030 DESIGN G2, G13).

    G = pg.Functionals.separable(M, radial, harmonics, against=..., taper=..., into=D)

harmonics: one label (l, j) per functional (the package's real harmonics, spaces/harmonic.py; convention= reads
them in another basis, e.g. harmonics.THESIS). radial: one radial kernel per functional (a pg.Tabulated, a callable of r,
or a number on a sphere), or on a direct sum a dict {summand: kernels} (None: the functional does not touch that
summand). Functional i is

    m -> sum_k  int k_i^(k)(r) m^(k)_(l_i j_i)(r) dnu_k(r)       (ball or shell summands: nu_k = against, default
                                                                  Lebesgue r^2 dr, whatever the space's density)
          + sum_k  c_i^(k) R^2 m^(k)_(l_i j_i)                    (sphere summands: the coefficient against the
                                                                    sphere's own measure R^2 dOmega)

The functional integrates against the domain's own measure lambda (or against=); the space's measure mu = w lambda (a
constant c or a radial rho, plan 064) and the summand's direct-sum weight enter only through the representer, which is
k dnu/dmu (x) Y_(l_i j_i) divided by the weight (plan 066: on L2(c dOmega) the sphere coefficients are divided by c,
on L2(rho dx) the radial factor by c rho; so G m does not depend on the density). The radial factor is
converted to the space's measure (Radon-Nikodym, measures/densities.py) and refused (pg.Refused; the decision is
spaces/dual.py) when it is not square integrable, unless a taper (a declared modification, ASSERTED) makes it so;
it is exactly 0 where every kernel vanishes identically (Tabulated.zero_end). Equal kernel objects are realised once.

Structure (Theorem G2.1, NOTES.md section 3): <k (x) Y_lj, S k' (x) Y_l'j'> = delta_ll' delta_jj' <k, S_l k'> for every
S that commutes with rotations, so G.structured_gram(S) assembles the Gram matrix harmonic by harmonic and returns a
pg.BlockDiagonal with one block per (l, j), blocks shared where the same kernels appear (the same objects, in the same
order: in orthonormal units the CMB data repeat across orders t, NOTES.md section 3), and G.structured_cross(T, S) the
pairings with another separable family (the factors of a property family, properties/families.py). Both are what the
data-space decomposition asks for (solvers/dataspace/decomposition.py); claims DERIVED (orthonormality) and THEOREM (Schur, for S).

Density kernels on interval spaces: pg.Functionals(M, kernels, against=nu, taper=tau): k dnu/dmu (times tau) on L2(mu).

quadrature=pg.realisations.Trapezoid(n) computes the radial integrals of the structured Grams by that rule instead
(the thesis's realisation, S1), with the measured difference from the exact Gram in the claims; on a direct sum a dict
{summand: rule} chooses per summand (a summand absent from it is integrated exactly), e.g. one parent rule
Trapezoid(n, over=(a, b)) shared by the layers that cut [a, b].
"""
import math
import warnings

import numpy as np

from ..core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, Claim
from ..spaces.elements import Family
from ..core.errors import Unsupported
from ..quadrature import gauss
from ..realisations.series import harmonic_degrees as H
from ..realisations.series import legendre as LG
from ..realisations.registry import provides
from ..spaces.base import Euclidean
from .functionals import Functionals, FunctionalsAdjoint
from ..spaces.identity import identical


# ------------------------------------------------------------------------------------------------ radial kernels
def _breaks_of(item):
    b = getattr(item, "breaks", None)
    return [] if b is None else list(np.asarray(b, float))


def _values(item, x):
    if provides(item, "piecewise_legendre"):
        return LG.evaluate(item, x)[..., 0]
    if callable(item):
        return np.broadcast_to(np.asarray(item(x), float), x.shape)
    return np.full(x.shape, float(item))


def radial_family(a, b, items, scale, mu, against=None, taper=None, tol=1e-14, label="functional"):
    """(legendre.PiecewiseLegendre with one column per item, claims): the radial factors k_i (dnu/dmu) tau * scale_i on [a, b]."""
    from ..spaces.dual import l2_density
    claims = []
    rho = None
    if against is not None:
        rho, claims = l2_density(against, mu, items, taper, label)
    if taper is not None:
        claims.append(Claim(f"the {label}s are modified by the declared taper {getattr(taper, '__name__', 'tau')}", ASSERTED,
                            "a modification of the relation stated by the user (G13)"))
    brk = sorted({x for it in items for x in _breaks_of(it) if a < x < b} |
                 ({x for x in (rho.breaks if rho else ()) if a < x < b}))
    lo, hi, cut = a, b, None
    for x0, (order, margin, s) in ({} if rho is None else getattr(rho, "representer", {})).items():
        inward = 1.0 if x0 <= a else -1.0
        if order < 0:            # singular but square integrable: zero on [x0, x0 + delta], tail energy ~ delta^margin
            delta = max((b - a) * 1e-26 ** (1.0 / margin), 1e-300)
            cut = (x0, delta, margin, s)
            lo, hi = (a + delta, b) if inward > 0 else (a, b - delta)
        if order == math.inf and None not in (z := [getattr(it, "zero_end", lambda i: None)(inward) for it in items]):
            lo, hi = (max(lo, min(z)), hi) if inward > 0 else (lo, min(hi, max(z)))   # every kernel exactly 0 next to
            continue                                                                    # x0: representers 0 (LG.pad)
        reach = math.log2((b - a) / (delta if order < 0 else 1e-12 * (b - a)))
        brk += [x0 + inward * (b - a) * 2.0 ** -k for k in range(1, int(reach) + 1)]   # geometric breaks: the fit
    brk = sorted(set(x for x in brk if lo < x < hi))                                     # resolves the end
    scale = np.asarray(scale, float)

    def exact(it):
        p = it if provides(it, "piecewise_legendre") else getattr(it, "pl", None)
        return p is not None and abs(p.breaks[0] - a) <= 1e-12 * abs(b) and abs(p.breaks[-1] - b) <= 1e-12 * abs(b)
    if rho is None and taper is None and all(exact(it) for it in items):
        pls = [it if provides(it, "piecewise_legendre") else it.pl for it in items]
        return LG.combine(LG.hstack(pls), np.diag(scale)), claims

    def factor(x):
        x = np.asarray(x, float)
        f = np.ones_like(x) if rho is None else rho(x)
        if taper is not None:
            f = f * _values(taper, x)
        return f

    def covers(it):          # a pg.Tabulated (exactly piecewise linear, any rule) defined on all of [a, b]
        br = getattr(it, "breaks", None)
        return hasattr(it, "_v0") and br is not None and br[0] <= a + 1e-12 * abs(b) and br[-1] >= b - 1e-12 * abs(b)
    if cut is None and all(covers(it) for it in items):
        pl, e2, npieces = _times_factor(items, factor, scale, [lo] + brk + [hi], tol)
        claims.append(Claim(f"representers realised as the kernels (exactly piecewise polynomial) times the common "
                            f"factor (dnu/dmu) tau, fitted once on {npieces} pieces; products exact; largest estimated "
                            f"error {float(np.sqrt(e2).max()):.2g} in L2(r^2 dr)", CHECKED,
                            "the factor's fit checked at other Gauss points per piece, times the kernels' sup per piece"))
        return LG.pad(LG.PiecewiseLegendre(pl.breaks, pl.coefs, np.sqrt(e2) / max(abs(b), 1e-300)), a, b), claims

    def fun(x):
        return np.stack([_values(it, x) for it in items], axis=-1) * (factor(x)[:, None] * scale[None, :])
    if rho is not None and rho.orders or taper is not None:
        pl, unresolved = _fit_locally(fun, [lo] + brk + [hi], len(items), tol)
    else:
        pl, unresolved = LG.fit(fun, lo, hi, N=len(items), breaks=brk, tol=tol)
    if unresolved:
        warnings.warn(f"{unresolved} radial piece(s) of the {label} representers did not converge")
    e2 = _r2_error(fun, pl)
    if cut is not None:                                    # the excluded end: int_0^delta g^2 r^s dr by its power law
        x0, delta, margin, s = cut
        edge = lo if x0 <= a else hi
        g = fun(np.array([edge]))[0]
        e2 = e2 + g * g * delta ** (s + 1) / margin
        claims.append(Claim(f"representers set to zero within {delta:.2g} of the singular point {x0:g} (they are square "
                            "integrable there; the neglected part is in the stated error)", CHECKED,
                            "the local power law of k dnu/dmu (measures/densities.py)"))
    pl = LG.pad(LG.PiecewiseLegendre(pl.breaks, pl.coefs, np.sqrt(e2) / max(abs(b), 1e-300)), a, b)   # sr_error multiplies by b
    claims.append(Claim(f"representers realised piecewise (Legendre, {pl.npieces} pieces), largest estimated error "
                        f"{float(np.sqrt(e2).max()):.2g} in L2(r^2 dr)", CHECKED,
                        "adaptive fit with the kernels' breakpoints, checked at other Gauss points per piece"))
    return pl, claims


def _times_factor(items, factor, scale, edges, tol):
    """(PiecewiseLegendre, squared errors in L2(r^2 dr), pieces): columns k_i * f * scale_i, with every k_i a pg.Tabulated (exactly
    piecewise linear) and f a smooth factor common to all of them (here (dnu/dmu) tau). Only f is fitted (one column,
    adaptively, each interval at its own scale); the kernels are restricted exactly to the fit's pieces and the products
    are exact on the polynomials. The error of column i is bounded per piece by sup |k_i| times the factor's error
    there, the latter estimated at other Gauss points (as _r2_error does for a full fit)."""
    f, unresolved = _fit_locally(lambda x: factor(x)[:, None], edges, 1, tol)
    if unresolved:
        warnings.warn(f"{unresolved} radial piece(s) of the common factor did not converge")
    nb = f.breaks
    lo, hi, mids = nb[:-1], nb[1:], 0.5 * (nb[:-1] + nb[1:])
    c0 = np.empty((len(mids), len(items)))
    c1 = np.zeros_like(c0)
    for i, it in enumerate(items):
        x0, x1, v0, v1 = it._x0, it._x1, it._v0, it._v1
        k = np.clip(np.searchsorted(x0, mids, side="right") - 1, 0, len(x0) - 1)   # a Tabulated: linear pieces,
        slope = (v1[k] - v0[k]) / (x1[k] - x0[k])                                   # restricted from its end values
        ylo, yhi = v0[k] + slope * (lo - x0[k]), v0[k] + slope * (hi - x0[k])
        c0[:, i], c1[:, i] = 0.5 * (ylo + yhi), 0.5 * (yhi - ylo)
    coefs = [np.vstack([c0[j], c1[j]]) for j in range(len(mids))]
    K = LG.combine(LG.PiecewiseLegendre(nb, coefs, np.zeros(len(items))), np.diag(scale))
    prod = LG.multiply(K, f)
    e2 = np.zeros(len(items))
    for j, (C, Ck) in enumerate(zip(f.coefs, K.coefs)):                             # per piece: the factor's error in
        t, w = gauss(C.shape[0] + 9)                      # L2(r^2 dr), times sup |k_i|^2
        x = mids[j] + 0.5 * (hi[j] - lo[j]) * t
        d = factor(x) - LG.polyval(t, C)[:, 0]
        e2 += float((0.5 * (hi[j] - lo[j]) * w * x * x) @ (d * d)) * np.abs(Ck).sum(axis=0) ** 2
    return prod, (np.full_like(e2, np.inf) if unresolved else e2), len(mids)      # unresolved: no bound


def _fit_locally(fun, edges, N, tol):
    """LG.fit interval by interval: each with its own scale of values (a representer that grows like a power towards a
    singular end, or peaks at a taper's scale, would otherwise set one scale for the whole interval)."""
    breaks, coefs, e2, unresolved = [edges[0]], [], np.zeros(N), 0
    for x0, x1 in zip(edges[:-1], edges[1:]):
        pl, u = LG.fit(fun, x0, x1, N=N, tol=tol)
        breaks += list(pl.breaks[1:])
        coefs += list(pl.coefs)
        e2 += pl.err ** 2
        unresolved += u
    return LG.PiecewiseLegendre(breaks, coefs, np.sqrt(e2)), unresolved


def _r2_error(fun, pl):
    """int (g - fit)^2 r^2 dr per column, by Gauss points other than the fit's (an estimate of the realisation error)."""
    out = np.zeros(pl.N)
    for i, C in enumerate(pl.coefs):
        x0, x1 = pl.breaks[i], pl.breaks[i + 1]
        t, w = gauss(C.shape[0] + 9)
        x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * t
        d = fun(x) - LG.polyval(t, C)
        out += (0.5 * (x1 - x0) * w * x * x) @ (d * d)
    return out


def density_family(space, kernels, against=None, taper=None):
    """(Family of representers, claims) of f -> int k f dnu on a function space on an interval (module docstring)."""
    from ..spaces.dual import l2_density
    B = space.backend
    if getattr(B, "realisation_name", None) != "legendre" or getattr(B, "dim", None) != 1:
        raise Unsupported("density functionals with against= are formed on spaces on an interval here; on a ball or "
                          "shell use pg.Functionals.separable")
    items = list(kernels)
    groups = {}
    for i, k in enumerate(items):
        nu = against if against is not None else getattr(k, "against", None)
        groups.setdefault(id(nu), (nu, []))[1].append(i)
    B, claims, parts, order = space.backend, [], [], []
    for nu, idx in groups.values():
        ks = [items[i] for i in idx]
        rho = None
        if nu is not None:
            rho, cl = l2_density(nu, space.measure, ks, taper)
            claims += cl
            if any(o < 0 for o, _, _ in getattr(rho, "representer", {}).values()):
                raise Unsupported("these representers are square integrable but singular at an end; on a space of an "
                                  "interval with a density the realisation loses relative accuracy there. Use the radial "
                                  "factor of a ball or shell (pg.Functionals.separable), or a taper")
        brk = sorted({x for it in ks for x in _breaks_of(it)} | set(rho.breaks if rho else ()))

        def fun(x, ks=ks, rho=rho):
            x = np.asarray(x, float)
            f = np.ones_like(x) if rho is None else rho(x)
            if taper is not None:
                f = f * _values(taper, x)
            return np.stack([_values(k, x) for k in ks], axis=-1) * f[..., None]
        if (rho is not None and rho.orders or taper is not None) and B.realisation_name == "legendre":
            a, b = space.domain.a, space.domain.b                   # local scales (radial_family)
            for x0 in (x for x, o in (rho.orders.items() if rho is not None else ()) if o < 0):
                brk = brk + [x0 + (1.0 if x0 <= a else -1.0) * (b - a) * 2.0 ** -k for k in range(1, 41)]
            R, u = _fit_locally(fun, [a] + sorted({x for x in brk if a < x < b}) + [b], len(ks), space.tol)
        else:
            R, u = B.fit(fun, len(ks), breaks=brk)
        if u:
            warnings.warn(f"{u} piece(s) of the density representers did not converge")
        parts.append(R)
        order += idx
    R = parts[0] if len(parts) == 1 else B.hstack(parts)
    inv = np.argsort(order)
    if not np.array_equal(inv, np.arange(len(order))):
        R = B.take(R, inv)
    if taper is not None:
        claims.append(Claim("the functionals are modified by the declared taper", ASSERTED, "stated by the user (G13)"))
    return Family(space, R, name="k"), claims


# ------------------------------------------------------------------------------------------------ construction
def _per_summand(value, k, names):
    if isinstance(value, dict):
        return value.get(names[k], value.get(k)) if names is not None else value.get(0)
    return value


def separable(space, radial, harmonics, against=None, taper=None, into=None, name="G", quadrature=None,
              convention=None):
    """pg.Functionals.separable (module docstring)."""
    from ..spaces.harmonic import index
    lab = np.asarray(harmonics, int).reshape(-1, 2)
    lm, N = index(lab[:, 0], lab[:, 1]), len(lab)
    sign = np.ones(N) if convention is None else convention.factors(lab[:, 0], lab[:, 1])
    L = int(lab[:, 0].max()) if N else 0
    names = getattr(space, "names", None)
    spaces = list(space.summands) if names is not None else [space]
    weights = list(space.weights) if names is not None else [1.0]
    if names is None:
        radial = {0: radial}
    keyed = {(space.index(key) if names is not None else 0): items for key, items in radial.items()}
    parts, layout, claims = [None] * len(spaces), {}, []
    for k, items in keyed.items():
        items = list(items)
        if len(items) != N:
            raise ValueError(f"{len(items)} radial kernels for {N} harmonic labels")
        Hk, w = spaces[k], weights[k]
        kind = H.require(Hk, "separable functionals")
        present = np.array([i for i, it in enumerate(items) if it is not None], int)
        if not len(present):
            continue
        if kind == "sphere":
            if _per_summand(against, k, names) is not None:
                raise Unsupported("on a sphere summand the kernels are the coefficients of the representer (no against=)")
            vals = np.array([float(items[i]) for i in present]) * sign[present] / (w * Hk.backend.c)   # dlambda/dmu (066)
            R = H.sphere_separable(Hk.backend, vals, lm[present], L)
            layout[k] = ("sphere", R, present, vals)
        else:
            keys, uniq, col = {}, [], []
            for i in present:
                key = (id(items[i]), float(sign[i]))
                if key not in keys:
                    keys[key] = len(uniq)
                    uniq.append((items[i], float(sign[i])))
                col.append(keys[key])
            Bk = Hk.backend
            nu = _per_summand(against, k, names)            # the functional's measure: default Lebesgue r^2 dr (066)
            if nu is None and Bk.rho is not None:
                nu = Bk.radial_lebesgue()
            div = w * (Bk.c if nu is None else 1.0)         # no against=: the representer is k / (w c) exactly
            pl, cl = radial_family(Bk.a, Bk.b, [u[0] for u in uniq], [u[1] / div for u in uniq], Bk.radial_measure(),
                                   nu, _per_summand(taper, k, names), Hk.tol,
                                   label=f"functional ({names[k]!r})" if names is not None else "functional")
            claims += cl
            R = Bk.separable(pl, lm[present], np.asarray(col, int), L)
            layout[k] = ("radial", R, present, np.asarray(col, int))
        parts[k] = (R, present)
    if names is None:
        if parts[0] is None or len(parts[0][1]) != N:
            raise ValueError("every functional needs a kernel on a space that is not a direct sum")
        fam = Family(space, parts[0][0], name="k")
    else:
        from ..realisations.series.direct_sum import DirectSumRealisation
        fam = Family(space, DirectSumRealisation(parts, N), name="k")
    claims.append(Claim(f"separable: functional i is <k_i (x) Y_(l_i j_i), m> ({N} functionals, degrees <= {L})", DERIVED,
                        "built from radial kernels and declared harmonics"))
    return SeparableFunctionals(fam, into, name, lab, lm, layout, weights, quadrature, claims)


def _weighted_radial(space):
    """True when a ball or shell summand of space has a measure other than Lebesgue (a density, plan 064)."""
    spaces = list(space.summands) if getattr(space, "names", None) is not None else [space]
    return any(getattr(getattr(H, "backend", None), "harmonic", None) == "radial" and
               (H.backend.c != 1.0 or H.backend.rho is not None) for H in spaces)


class SeparableFunctionals(Functionals):
    """Functionals k_i (x) Y_(l_i j_i) with structured Grams (module docstring)."""

    def __init__(self, family, into, name, labels, lm, layout, weights, quadrature, claims):
        super().__init__(family, into=into, name=name)
        self.labels, self.lm, self.layout, self.weights = labels, np.asarray(lm, int), layout, list(weights)
        self.quadrature = quadrature
        self._claims += claims
        self._groups = None
        self._radial_ops = False

    def _compose(self, other):
        if isinstance(other, FunctionalsAdjoint):             # G G*: its blocks at every size, on any diagonal metric (plan 045)
            from .plain_gram import plain_normal
            out = plain_normal(self, other.G)
            if out is not None:
                return out
        return super()._compose(other)

    def groups(self):
        """[(lm, rows)] per harmonic, in the order of first appearance; rows ascending (the data's own order)."""
        if self._groups is None:
            _, first = np.unique(self.lm, return_index=True)
            self._groups = [(int(self.lm[f]), np.flatnonzero(self.lm == self.lm[f])) for f in np.sort(first)]
        return self._groups

    def _where(self, k, rows):
        """(positions of rows among summand k's members, the rows that are present there)."""
        present = self.layout[k][2]
        where = np.full(len(self.lm), -1)
        where[present] = np.arange(len(present))
        p = where[rows]
        return p[p >= 0], np.flatnonzero(p >= 0)

    def _block(self, rows, other, rows_o, f, l, stats, summands=None):
        """<g_rows, S g'_rows_o> for one harmonic: sum over summands of w_k f_k(l) <parts>."""
        out = np.zeros((len(rows), len(rows_o)))
        groups = {}                                       # radial parts coupling several summands (pieces of a domain)
        for k, lay in self.layout.items():
            lay_o = other.layout.get(k)
            if lay_o is None or (summands is not None and k not in summands):
                continue
            pa, ia = self._where(k, rows)
            pb, ib = other._where(k, rows_o)
            if not len(pa) or not len(pb):
                continue
            act = f[k](l) if self._radial_ops else ("scalar", float(np.asarray(f[k](np.array([l])), float).ravel()[0]))
            if act is None or (act[0] == "scalar" and act[1] == 0):
                continue
            c = self.weights[k] * (act[1] if act[0] == "scalar" else 1.0)
            Hk = self.domain.summands[k] if hasattr(self.domain, "summands") else self.domain
            if lay[0] == "sphere":
                B = Hk.backend
                Gk = np.asarray(B.gram(B.take(lay[1], pa), B.take(lay_o[1], pb)), float)
            else:
                Ra, Rb = lay[1], lay_o[1]
                if act[0] == "radial" and len(act) > 3 or (act[0] == "radial" and self._rule(k) is not None):
                    radial_part(self, act, k, lay, lay_o, pa, pb, ia, ib, c, out, groups, len(rows), len(rows_o))
                    continue
                if act[0] == "radial":                       # <k_a, (sum_n lam_n phi_n (x) phi_n) k_b>, exact pieces
                    from ..realisations.series.radial_harmonic import rgram
                    Aa = rgram(Ra.pl.take(np.asarray(lay[3][pa], int)), act[1])
                    Ab = Aa if (Rb is Ra and np.array_equal(lay[3][pa], lay_o[3][pb])) else \
                        rgram(Rb.pl.take(np.asarray(lay_o[3][pb], int)), act[1])
                    out[np.ix_(ia, ib)] += c * (Aa * act[2]) @ Ab.T
                    continue
                Bk = Hk.backend                              # c rho r^2 dr: the summand's measure (plan 064)
                Gk = Bk.radial_gram(Ra, lay[3][pa], Rb, lay_o[3][pb])
                rule = self._rule(k)
                if rule is not None:
                    x, w = rule.nodes(Ra.pl.breaks[0], Ra.pl.breaks[-1])
                    Va = LG.evaluate(Ra.pl, x)[:, lay[3][pa]]
                    Vb = LG.evaluate(Rb.pl, x)[:, lay_o[3][pb]]
                    dens = Bk.c * (1.0 if Bk.rho is None else LG.evaluate(Bk.rho, x)[:, 0])
                    Gq = Va.T @ ((w * x * x * dens)[:, None] * Vb)
                    stats.append(float(np.abs(Gq - Gk).max() / max(float(np.abs(Gk).max()), 1e-300)))
                    Gk = Gq
            out[np.ix_(ia, ib)] += c * Gk
        for c, lam, SA, SB in groups.values():
            out += c * (SA * lam) @ SB.T
        return out

    def _rule(self, k):
        """The quadrature rule of summand k (None: exact)."""
        q = self.quadrature
        if not isinstance(q, dict):
            return q
        names = getattr(self.domain, "names", None)
        return q.get(names[k], q.get(k)) if names is not None else q.get(0)

    def _scalars(self, S):
        from .equivariance import degree_operators, degree_scalars
        self._radial_ops = False
        f = degree_scalars(S, self.domain)
        if f is None and S is not None:
            if _weighted_radial(self.domain):          # radial operators pair in r^2 dr: not with a density (plan 064);
                return None                            # None: the caller's general route (inner products)
            f = degree_operators(S, self.domain)       # plan 009 G5: a radial operator per degree (known part)
            self._radial_ops = f is not None
        return f

    def _claims_for(self, S, stats, what):
        from .equivariance import claim
        cl = [Claim(f"{what}: block diagonal over the harmonics (l, j)", DERIVED,
                    "<k (x) Y_lj, S k' (x) Y_l'j'> = 0 unless (l, j) = (l', j') (orthonormal harmonics, S by degree)",
                    key="block diagonal")]
        if S is not None and self._radial_ops:
            cl.append(Claim("S commutes with rotations and acts by a radial operator per degree and summand", THEOREM,
                            "a function of the Laplacian with rotation-invariant conditions (Schur; EXP-030 Theorem "
                            "G2.1)", key="rotation invariant"))
            cl += [c for c in S.claims if c.key == "truncation"]
        elif S is not None:
            cl.append(claim(S))
        if self.quadrature is not None:
            q = self.quadrature
            what = q.name if not isinstance(q, dict) else "rules " + "; ".join(sorted({r.name for r in q.values()}))
            cl.append(Claim(f"radial integrals by the {what} (a realisation option), not exactly: largest "
                            f"relative difference from the exact blocks {max(stats, default=0.0):.3g}", CHECKED,
                            "measured against Gauss-Legendre per piece"))
        return cl

    def structured_gram(self, S=None, summands=None):
        """The Gram matrix <g_i, S g_j> as a pg.BlockDiagonal (one block per harmonic, shared where the kernels
        repeat), or None when S is not derived to commute with rotations (module docstring). summands: the positions
        of the summands whose parts are counted (default all: the per-summand Grams add up to the Gram)."""
        D = self.codomain
        f = self._scalars(S)
        if f is None or not (D.is_standard or D.weights is not None):       # a diagonal metric only multiplies (plan 045)
            return None
        keys, mats, shared, order, stats = {}, [], [], [], []
        for u, rows in self.groups():
            l = math.isqrt(u)
            key = [l]
            for k, lay in sorted(self.layout.items()):
                if summands is not None and k not in summands:
                    continue
                pa, ia = self._where(k, rows)
                key.append((k, tuple(ia), tuple(lay[3][pa]) if lay[0] == "radial" else tuple(np.round(lay[3][pa], 17))))
            key = tuple(key)
            if key not in keys:
                keys[key] = len(mats)
                mats.append(self._block(rows, self, rows, f, l, stats, summands))
            shared.append(keys[key])
            order.append(rows)
        from .blocks import BlockDiagonal
        return BlockDiagonal(D if D.is_standard else Euclidean(D.dim), mats, name=f"{self.name}S{self.name}*", shared=shared,
                             order=np.concatenate(order),
                             claims=self._claims_for(S, stats, f"the Gram matrix of {self.name}: {len(shared)} harmonics, "
                                                                f"{len(mats)} distinct blocks"))

    def select_rows(self, idx, into):
        """The functionals idx only (into the selected data space), with the same radial factors and harmonics: the
        structure (and the structured Grams) kept (relations/selection.py, R.select(rows))."""
        from ..realisations.series.direct_sum import DirectSumRealisation
        idx = np.asarray(idx, int)
        pos = np.full(len(self.lm), -1)
        pos[idx] = np.arange(len(idx))
        lm = self.lm[idx]
        names = getattr(self.domain, "names", None)
        spaces = list(self.domain.summands) if names is not None else [self.domain]
        L = int(np.floor(np.sqrt(lm.max()))) if len(lm) else 0
        layout, parts = {}, [None] * len(spaces)
        for k, (kind, R, present, aux) in self.layout.items():
            keep = pos[present] >= 0
            if not keep.any():
                continue
            pr, ax = pos[present[keep]], np.asarray(aux)[keep]
            Rn = H.sphere_separable(spaces[k].backend, ax, lm[pr], L) if kind == "sphere" else \
                spaces[k].backend.separable(R.pl, lm[pr], ax.astype(int), L)
            layout[k], parts[k] = (kind, Rn, pr, ax), (Rn, pr)
        fam = Family(self.domain, DirectSumRealisation(parts, len(idx)) if names is not None else parts[0][0], name="k")
        lab = np.asarray(self.labels).reshape(-1, 2)[idx]
        return SeparableFunctionals(fam, into, f"{self.name}_J", lab, lm, layout, self.weights, self.quadrature,
                                    [c for c in self._claims])

    def structured_cross(self, T, S=None, summands=None):
        """(<g_i, S t_b>, <t_b, S t_c>) with a separable family T on the same space, harmonic by harmonic (the parts in
        the given summands only, when given); None otherwise."""
        if not isinstance(T, SeparableFunctionals) or not identical(T.domain, self.domain):
            return None
        f = self._scalars(S)
        if f is None:
            return None
        cross, stats = np.zeros((len(self.lm), len(T.lm))), []
        mine = dict(self.groups())
        for u, rows_t in T.groups():
            if u in mine:
                cross[np.ix_(mine[u], rows_t)] = self._block(mine[u], T, rows_t, f, math.isqrt(u), stats, summands)
        tt = T.structured_gram(S, summands)
        if tt is None:
            return None
        from ..settings import settings
        return cross, (tt.matrix() if T.codomain.dim <= settings.dense_max else tt)


# ------------------------------------------------------------------------------------------------ radial parts
def radial_part(G, act, k, lay, lay_o, pa, pb, ia, ib, c, out, groups, na, nb):
    """A radial operator per degree with a multiplier m, a scalar d per summand and a group of summands that are
    pieces of one domain (module docstring), for SeparableFunctionals._block: <k_a, d m phi_n> on this summand's interval, by the summand's
    rule when it has one, else exactly; within a group the pieces' pairings add before the product (out of
    probability/degrees.py in plan 070 P7; the integrals: realisations/series/radial_integrals.py)."""
    from ..realisations.series.radial_integrals import coefficients
    ext = act[3] if len(act) > 3 else None
    ext = dict(m=None, d=1.0, group=None) if ext is None else (ext if isinstance(ext, dict) else
                                                              dict(m=ext, d=1.0, group=None))
    Ra, Rb = lay[1], lay_o[1]
    a, b = float(Ra.pl.breaks[0]), float(Ra.pl.breaks[-1])
    rule = G._rule(k)
    ca, cb = np.asarray(lay[3][pa], int), np.asarray(lay_o[3][pb], int)
    Aa = ext["d"] * coefficients(Ra.pl.take(ca), act[1], ext["m"], a, b, rule)
    Ab = Aa if (Rb is Ra and np.array_equal(ca, cb)) else \
        ext["d"] * coefficients(Rb.pl.take(cb), act[1], ext["m"], a, b, rule)
    if ext["group"] is None:
        out[np.ix_(ia, ib)] += c * (Aa * act[2]) @ Ab.T
        return
    g = groups.get(ext["group"])
    if g is None:
        g = groups[ext["group"]] = [c, act[2], np.zeros((na, len(act[2]))), np.zeros((nb, len(act[2])))]
    elif abs(g[0] - c) > 1e-14 * abs(c):
        raise ValueError("the pieces of one radial part carry different weights")
    g[2][ia] += Aa
    g[3][ib] += Ab
