"""Riesz representers in L2(lambda) of measures of full dimension, kept symbolic (EXP-031, DESIGN 3.4, question 6).

For mu = w sigma_S with S a set of dimension d inside the closure of the domain D (a subinterval, a sub-box, a disk in
a box, the domain itself) and lambda = rho dx the space's measure, the functional u -> int u dmu is bounded on
L2(lambda) iff d mu / d lambda = w 1_S / rho is in L2(lambda) (spaces/dual.py), and that is its representer. It is
kept as what it is, the measure, not fitted: values are w(x) 1_S(x) / rho(x), and pairings are

    <k_mu, f>      = int_S w f dsigma             (f an element given by a formula: Gauss quadrature on S, its pieces;
                                                   w = 1 on a coordinate box of a box or torus: exact for f's tensor
                                                   realisation, realisations/rkhs/box_form.py)
    <k_mu, k_nu>   = int_{S cap T} w_mu w_nu / rho dx    (exact where S cap T is representable: equal sets, one of them
                                                   the whole domain, intervals, boxes, sub-paths of a path,
                                                   rectangles of a flat patch, arcs of a circle, convex polytopes
                                                   (tubes), disjoint sets: sets/intersections.py; on a polytope with
                                                   w = rho = 1 its volume;
                                                   refused otherwise)

Combinations (a k_mu + b k_nu, G* y) stay symbolic (the representer of the signed measure). Operations that need a
realisation in the space's backend (adding a formula, plotting, pg.Inclusion) fit w 1_S / rho with S's faces as breaks,
which is exact for exact weights on intervals and coordinate boxes; on other sets they are refused (Refused), since a
fit across the boundary of S would be an undisclosed substitution.
"""
import numpy as np

from ..core.claims import DERIVED, Claim, Claims
from .elements import Family, Function, _coords
from ..core.errors import Refused, SupportOutsideDomain
from ..core.logic import Truth
from ..sets.intersections import intersection
from ..measures import (DensityMeasure, LebesgueMeasure, SignedMeasure, as_measures, constant_density, terms_of,
                       total_mass)
from ..sets import Interval

FORMULA_NODES = {1: 64, 2: 48, 3: 24}      # Gauss points per axis (per piece on intervals) for int_S w f dsigma; the
                                           # pairings are also computed at 3/4 of these, the difference is the estimate
PAIR_WARN = 1e-8                           # relative estimate of a representer pairing above which a warning is issued
MOMENT_AGREE = 1e-10                       # relative difference of two orders under which a moment's fixed rule stands
PAIR_MAX = 96                              # Gauss points per axis at which a pairing stops refining (plan 044)


def weight_values(m, X):
    return m.weight.values(X) if isinstance(m, DensityMeasure) else np.ones(len(X))


def _rho(space, X):
    lam = space.measure
    return lam.weight.values(X) if isinstance(lam, DensityMeasure) else np.ones(len(X))


def members_at(S, X):
    if isinstance(S, Interval):
        return np.atleast_1d(S.closure.contains(X[:, :1]))
    return np.atleast_1d(S.contains(X))


def _density(space, mu):
    """x -> (d mu / d lambda)(x), a function of the Cartesian coordinates."""
    terms = terms_of(mu)

    def f(*X):
        shape = np.broadcast(*X).shape
        P = np.stack([np.broadcast_to(np.asarray(a, float), shape).ravel() for a in X], axis=1)
        v = np.zeros(len(P))
        for c, m in terms:
            inside = members_at(m.set, P)
            if inside.any():
                v[inside] += c * weight_values(m, P[inside])
        return (v / _rho(space, P)).reshape(shape)
    return f


def _breaks(space, mu):
    """(breaks per natural axis, representable?): the faces of the sets in the domain's natural coordinates."""
    D = space.domain
    if isinstance(D, Interval):
        pts = set()
        for _, m in terms_of(mu):
            pts |= {p for p in (m.set.a, m.set.b) if D.a < p < D.b}
            if isinstance(m, DensityMeasure):
                pts |= {float(p[0]) for p in m.weight.breaks if D.a < p[0] < D.b}
        return sorted(pts) or None, True
    from ..domains import Box
    axes, degree = [set() for _ in range(D.dim)], 0
    for _, m in terms_of(mu):
        S = m.set
        if S == D or S.same_up_to_null_set(D):
            f = _bank(getattr(m, "weight", None))            # a bank weight: its support's natural breaks (plan 035)
            nb = f.fit_breaks(D) if f is not None else None
            for k, b in enumerate(nb or ()):
                axes[k] |= set(b or ())
            degree = max(degree, getattr(nb, "degree", None) or 0)
            continue
        if isinstance(S, Box) and isinstance(D, Box):
            for k, s in enumerate(S.sides):
                axes[k] |= {p for p in (s.a, s.b) if D.sides[k].a < p < D.sides[k].b}
            continue
        return None, False
    out = tuple(sorted(a) or None for a in axes)
    if degree:
        from ..realisations.backends import NaturalBreaks
        out = NaturalBreaks(out, degree)
    return out, True


def _bank(wt):
    """The bank function behind a weight (functions/base.BankFunction: it knows its support), or None."""
    while wt is not None and not hasattr(wt, "f"):
        wt = getattr(wt, "base", None)
    f = getattr(wt, "f", None)
    return f if callable(getattr(f, "fit_breaks", None)) and hasattr(f, "nonzero_on") else None


def _nonzero(mu):
    """Why the representer of mu is not zero (for the fit's check, spaces/elements.Function.realisation), or None."""
    terms = terms_of(mu)
    if len(terms) != 1 or not isinstance(terms[0][1], DensityMeasure):
        return None
    m = terms[0][1]
    f = _bank(m.weight)
    why = f.nonzero_on(m.set) if f is not None else None
    if why:
        return why
    Z, err = m.mass()
    return f"its measure has mass {Z:.3g} (estimated error {err:.1g})" if abs(Z) > 10 * err and Z != 0 else None


class DensityRepresenter(Function):
    """The representer d mu / d lambda of a measure of full dimension in L2(lambda), symbolic (module docstring)."""

    __slots__ = ("measure", "_representable")

    def __init__(self, space, mu, name=None):
        breaks, ok = _breaks(space, mu)
        super().__init__(space, source=_density(space, mu), name=name or f"k[{mu}]", breaks=breaks,
                         expect=lambda: _nonzero(mu))
        self.measure, self._representable = mu, ok

    @property
    def realisation(self):
        if self._pl is None and not self._representable:
            raise Refused(f"{self.name} is kept symbolic in {self.space}: realising w 1_S / rho in the space's backend "
                          "would fit it across the boundary of S (an undisclosed substitution). Its pairings with "
                          "formulas and with other such representers are exact: use M.inner, pg.Functional, Functionals",
                          needs=("S a coordinate box of the domain, or the domain itself",))
        return Function.realisation.fget(self)

    @property
    def error(self):
        return 0.0 if self._pl is None else Function.error.fget(self)

    def __call__(self, *args):
        v = self._src(*_coords(self.space, args))
        return float(v) if np.ndim(v) == 0 else v

    def _lin(self, a, other, b, name):
        if isinstance(other, DensityRepresenter) and other.space == self.space:
            return DensityRepresenter(self.space, SignedMeasure([(a, self.measure), (b, other.measure)]), name)
        return Function._lin(self, a, other, b, name)

    def __neg__(self):
        return self * -1.0

    def __mul__(self, c):
        if np.ndim(c) == 0 and isinstance(c, (int, float, np.number)):
            return DensityRepresenter(self.space, SignedMeasure([(float(c), self.measure)]), f"{float(c):g}·{self.name}")
        return Function.__mul__(self, c)

    __rmul__ = __mul__

    def integral(self, over=None):
        if over is None:                                  # int (d mu / d lambda) d lambda = mu(S)
            return total_mass(self.measure)
        return Function.integral(self, over)

    @property
    def claims(self):
        return Claims([Claim(f"{self.name} is in {self.space}", DERIVED, "the Radon-Nikodym derivative d mu / d lambda, "
                             "square integrable by the boundedness decision (spaces/dual.py)"),
                       Claim("kept symbolic", DERIVED, "pairings: Gauss quadrature on S (formulas), exact intersections "
                                                       "(representers)")])

    def __repr__(self):
        return f"{self.name} ∈ {self.space}  [symbolic: d({self.measure})/dλ]"


class DensityRepresenters(Family):
    """A family of symbolic representers (one per measure)."""

    __slots__ = ("measures",)

    def __init__(self, space, measures, name="k"):
        super().__init__(space, None, name=name)
        self.measures = list(measures)

    @property
    def realisation(self):
        if self._pl is None:
            self._pl = self.space.backend.hstack([self[i].realisation for i in range(len(self))])
        return self._pl

    def __len__(self):
        return len(self.measures)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            i = int(i) % len(self)
            return DensityRepresenter(self.space, self.measures[i], name=f"{self.name}[{i}]")
        return DensityRepresenters(self.space, [self.measures[j] for j in np.arange(len(self))[i]], self.name)

    def errors(self):
        return np.zeros(len(self))

    def combine(self, C):
        """The members sum_i C[i, j] k_i (j = 0..M-1): symbolic, the representers of the combined measures."""
        C = np.asarray(C, float).reshape(len(self), -1)
        out = [SignedMeasure([(C[i, j], m) for i, m in enumerate(self.measures) if C[i, j] != 0] or
                             [(0.0, self.measures[0])]) for j in range(C.shape[1])]
        return DensityRepresenters(self.space, out, name=f"{self.name}·C")

    def sum(self):
        return self.combine(np.ones((len(self), 1)))[0]

    def __mul__(self, c):
        if np.ndim(c) == 0:
            return DensityRepresenters(self.space, [SignedMeasure([(float(c), m)]) for m in self.measures], self.name)
        return NotImplemented

    __rmul__ = __mul__

    def info(self):
        return f"Family {self.name}: {len(self)} symbolic representers in {self.space} (spaces/representers.py)"

    def __repr__(self):
        return f"Family {self.name} of {len(self)} symbolic representers in {self.space}"


def symbolic(u):
    return isinstance(u, (DensityRepresenter, DensityRepresenters))


def representable(u):
    """Whether the symbolic representer(s) u can be realised in the space's backend (their sets are the domain,
    coordinate boxes or intervals: spaces/representers._breaks)."""
    if u._pl is not None:
        return True
    return all(_breaks(u.space, m)[1] for m in _measures(u))


def _measures(u):
    return [u.measure] if isinstance(u, DensityRepresenter) else u.measures


def _shape(f, g, G):
    if isinstance(f, Function) and isinstance(g, Function):
        return float(G[0, 0])
    if isinstance(f, Function):
        return G[0]
    if isinstance(g, Function):
        return G[:, 0]
    return G


def pairing_space(space, e):
    """The space whose backend reads the element e in <X.riesz(mu), e> = int e dmu: e's own (equal to space, possibly
    realised otherwise; plan 071 P3). A symbolic representer is a measure, not a realisation, so the pairing is the one
    of e's space bit for bit, whatever realises the space it was called on."""
    return e.space


def inner(space, f, g, fallback):
    """<f, g> in L2(lambda) when f or g is symbolic; fallback(f, g) otherwise. With one side an element, the moments are
    read in that element's realisation (pairing_space); the two-order estimate is kept on space.backend, where the
    caller reads it."""
    sf, sg = symbolic(f), symbolic(g)
    if not (sf or sg):
        return fallback(f, g)
    for u in (f, g):
        if u.space != space:
            from ..core.errors import SpaceMismatch
            raise SpaceMismatch(f"{u.name} is in {u.space}, not in {space}")
    if sf and sg and _measures(f) is _measures(g) and len(_measures(f)) > 1:     # a Gram: one triangle, mirrored
        ms = _measures(f)
        G = np.zeros((len(ms), len(ms)))
        for i, m in enumerate(ms):
            for j in range(i, len(ms)):
                G[i, j] = G[j, i] = pair(space, m, ms[j])
    elif sf and sg:
        G = np.array([[pair(space, m, n) for n in _measures(g)] for m in _measures(f)])
    elif sf:
        G = moments(pairing_space(space, g), _measures(f), g)
    else:
        G = moments(pairing_space(space, f), _measures(g), f).T
    if not (sf and sg):                                    # two orders: the estimate, kept on the backend (O4)
        e_ = g if sf else f
        lo = moments(pairing_space(space, e_), _measures(f if sf else g), e_, low=True)
        e = np.abs((G if sf else G.T) - lo)
        space.backend.last_pairing_error = e if sf else None
    return _shape(f, g, G)


def norm(space, f, fallback):
    if not symbolic(f):
        return fallback(f)
    if f.space != space:
        from ..core.errors import SpaceMismatch
        raise SpaceMismatch(f"{f.name} is in {f.space}, not in {space}")
    sq = np.array([pair(space, m, m) for m in _measures(f)])        # the diagonal only (EXP-032 finding 4)
    r = np.sqrt(np.maximum(sq, 0.0))
    return float(r[0]) if isinstance(f, Function) else r


def _points_of(m):
    wt = getattr(m, "weight", None)
    return () if wt is None or wt.singular_at is None else [float(p[0]) for p in wt.singular_at]


def moments(space, measures, g, low=False):
    """(len(measures), N): int g dmu_i for an element or family g with a realisation (a formula); low: at 3/4 of
    the order (the second order of the estimate)."""
    B, R = space.backend, g.realisation
    brk = B.breaks(R) if isinstance(space.domain, Interval) else None
    out = []
    for mu in measures:
        row = 0.0
        for c, m in terms_of(mu):
            S = m.set
            exact = None if brk is not None or not isinstance(m, LebesgueMeasure) else \
                getattr(B, "box_integral", lambda R, S: None)(R, S)        # coordinate boxes: exact (EXP-032 f. 8)
            if exact is not None:
                row = row + c * exact
                continue
            from ..realisations.series.symbolic_harmonics import realised_moment    # radial x harmonic: on the coefficients (plan 044)
            exact = realised_moment(space, m, R)
            if exact is not None:
                row = row + c * exact
                continue
            n = FORMULA_NODES.get(int(S.dim), 24)
            n = int(0.75 * n) if low else n
            br = () if brk is None or not isinstance(S, Interval) else np.ravel(brk)
            if isinstance(m, DensityMeasure):
                br = tuple(br) + tuple(float(p[0]) for p in m.weight.breaks) if isinstance(S, Interval) else br
            from ..core.adaptive import adapted_nodes_on
            from ..measures.weights import nodes_on
            if isinstance(m, DensityMeasure) and m.weight.kind != "constant":    # the weight decides (plan 022 P2)
                X, W, _ = adapted_nodes_on(S, n, m.weight.values, _points_of(m), br, exponent=m.weight.exponent)
            else:
                X, W = nodes_on(S, n, _points_of(m), br)
            v = B.evaluate(R, *[X[:, k] for k in range(X.shape[1])]).reshape(len(X), -1)
            term = (W * weight_values(m, X)) @ v
            if int(S.dim) > 1:                                 # two orders apart: the realisation's kinks (plan 044)
                term = _refined(space, m, R, S, term, n, low, nodes_on, v)
            row = row + c * term
        out.append(np.broadcast_to(row, (len(g) if isinstance(g, Family) else 1,)))
    return np.array(out, float)


def _refined(space, m, R, S, term, n, low, nodes_on, v):
    """term, or the rule of realisations/quadrature.py where term and a lower order disagree; the low call of the same pairing
    takes the same answer (kept on the backend), so the estimate in inner stays the rule's."""
    B = space.backend
    key = (id(m), id(R))
    kept = B.__dict__.get("_last_moment")
    if low:
        return kept[1] if kept is not None and kept[0] == key else term
    X, W = nodes_on(S, int(0.75 * n), _points_of(m), ())
    other = (W * weight_values(m, X)) @ B.evaluate(R, *[X[:, k] for k in range(X.shape[1])]).reshape(len(X), -1)
    if np.all(np.abs(term - other) <= MOMENT_AGREE * max(float(np.abs(term).max()), 1e-300)):
        B.__dict__["_last_moment"] = None
        return term
    from ..realisations.quadrature import moment
    better = moment(space, m, R)
    B.__dict__["_last_moment"] = None if better is None else (key, better)
    return term if better is None else better


def pair(space, mu, nu):
    """<k_mu, k_nu> = int (d mu/d lambda)(d nu/d lambda) d lambda: exact intersections, Gauss quadrature on them."""
    from ..measures.weights import integrate_on
    total, err = 0.0, 0.0
    for c, m in terms_of(mu):
        for e, n in terms_of(nu):
            J = intersection(m.set, n.set, space.domain)
            if J is None:
                continue
            if hasattr(J, "cells") and all(isinstance(x, LebesgueMeasure) for x in (m, n, space.measure)):
                total += c * e * J.volume                  # a polytope (intersections rule 5), integrand 1: exact
                continue
            pts =list(_points_of(m)) + list(_points_of(n)) + list(_points_of(space.measure))

            def f(X, m=m, n=n):
                return weight_values(m, X) * weight_values(n, X) / _rho(space, X)
            k = FORMULA_NODES.get(int(J.dim), 24)
            kw = dict(points=pts if isinstance(J, Interval) else (),
                      breaks=_interval_breaks(m, n, space) if isinstance(J, Interval) else ())
            hi, lo = integrate_on(J, f, k, **kw), integrate_on(J, f, int(0.75 * k), **kw)    # two orders (O4)
            while int(J.dim) > 1 and abs(hi - lo) > PAIR_WARN * abs(hi) and k < PAIR_MAX:   # a weight on a piece: refine first
                k *= 2
                hi, lo = integrate_on(J, f, k, **kw), hi
            total += c * e * hi
            err += abs(c * e) * abs(hi - lo)
    if err > PAIR_WARN * max(abs(total), 1e-300):
        import warnings
        from ..core.errors import RealisationWarning
        rel = err / abs(total) if total else float("inf")          # total = 0: disjoint supports seen by the rule
        warnings.warn(f"the L2 pairing of {mu} and {nu} carries an estimated relative error {rel:.1e} "
                      "(two quadrature orders on the intersection)", RealisationWarning, stacklevel=4)
    return total


def _interval_breaks(m, n, space):
    out = []
    for x in (m, n, space.measure):
        dens = getattr(x, "density", None)
        if dens is not None:
            out += list(dens.breaks[1:-1])
    return out


def _check_support(space, mu):
    D = space.domain
    for _, m in terms_of(mu):
        S = m.set
        if isinstance(D, Interval):
            if S.a < D.a or S.b > D.b:
                raise SupportOutsideDomain(f"{S} is not contained in {D}")
            continue
        if S == D or S.same_up_to_null_set(D):
            continue
        from ..domains import inside_closure
        exact = S.inside(D) if callable(getattr(S, "inside", None)) else Truth.UNKNOWN   # a restricted set S cap D
        exact = inside_closure(S, D) if exact is Truth.UNKNOWN else exact
        if exact is not Truth.UNKNOWN:
            if exact is Truth.FALSE:
                raise SupportOutsideDomain(f"the support of {m} leaves {D} (decided exactly)")
            continue
        from ..domains.base import require_in_closure
        from ..quadrature import rule
        X, _ = rule(S, 8)
        box = np.array(S.bounding_box, float)
        corners = np.array(np.meshgrid(*box, indexing="ij")).reshape(len(box), -1).T
        require_in_closure(D, np.vstack([X, corners]) if hasattr(S, "sides") else X, m)


def riesz_l2(space, mu):
    """M.riesz for L2(lambda) and L^p(lambda) (see the module docstring and _LebesgueBase.riesz)."""
    from .dual import require_bounded
    items, many = as_measures(mu, space.domain.cartesian_dim)
    for m in items:
        require_bounded(m, space)
    lebesgue = isinstance(space.measure, LebesgueMeasure)
    c = constant_density(space.measure)                 # L2(c dx), plan 064: d mu / d lambda = (d mu / dx) / c
    if not many:
        m = items[0]
        if c is not None and isinstance(m, LebesgueMeasure) and m.set == space.domain:
            return space.constant(1.0 / c)
        if lebesgue and isinstance(m, LebesgueMeasure) and isinstance(m.set, Interval):
            _check_support(space, m)
            return space.indicator(m.set)
        _check_support(space, m)
        return DensityRepresenter(space, m)
    for m in items:
        _check_support(space, m)
    return DensityRepresenters(space, items, name="riesz")


def average(space, S, weight=None):
    """The bounded functional u -> int_S u w dsigma / int_S w dsigma (w = 1 by default): the Functional of the
    normalised measure (question 11; S a set inside the closure of the domain, or a measure), a Functional."""
    from .dual import Functional
    from ..measures import Measure, measure_of
    if isinstance(S, str):
        S = Interval(S)
    mu = S if isinstance(S, Measure) and weight is None else measure_of(S)
    lam = getattr(space, "measure", None)
    if weight is None and isinstance(lam, DensityMeasure) and isinstance(mu, LebesgueMeasure) and \
            lam.weight.kind != "constant":    # a constant density normalises away (plan 064)
        if isinstance(mu.set, Interval):
            def weight(x, w=lam.weight):
                return w.values(np.asarray(x, float).reshape(-1, 1))
        else:
            def weight(*X, w=lam.weight):
                shape = np.broadcast(*X).shape
                P = np.stack([np.broadcast_to(np.asarray(a, float), shape).ravel() for a in X], axis=1)
                return w.values(P).reshape(shape)
    if weight is not None:                    # on L2(rho dx) the default is the average for the space's measure
        mu = mu.with_density(weight)
    T = mu.normalised()
    k = space.riesz(T)
    k.name = f"average over {mu.set}" + ("" if weight is None else " (weighted)")
    return Functional.from_representer(k)
