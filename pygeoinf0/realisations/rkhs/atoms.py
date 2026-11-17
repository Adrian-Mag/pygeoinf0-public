"""Measures as atoms for the Sobolev backends: points, segments, flat patches and spheres in flat coordinates, and the
check that every support lies in the closure of the domain.

An atom is a piece of a measure with constant density; its geometry is stored once, in explicit columns:

    kind      P                E1                  E2                  radius
    POINT     the point        0                   0                   0
    SEGMENT   start            end - start         0                   0
    PATCH     corner           first edge          second edge         0        (P + a E1 + b E2, a, b in [0, 1])
    SPHERE    centre           0                   0                   R
    CIRCLE    centre           0                   0                   R        (in R^2)
    WEIGHTED  centre           0                   0                   bounding radius   (EXP-031: a piece of a weighted
              measure, its geometry and weight in the piece column: weighted.py, weighted_volume.py)

The energies and values of atoms are one table indexed by kinds (pairs.py); a new kind of atom needs a branch in
atoms_from, its support in Atoms.support, and its row of the table.
"""
import enum

import numpy as np

from ...core.errors import SpaceMismatch, SupportOutsideDomain, Unsupported
from ...core.logic import Truth
from ...domains.base import require_in_closure, round_inside
from ...quadrature import gauss


class Kind(enum.IntEnum):
    POINT = 0
    SEGMENT = 1
    PATCH = 2
    SPHERE = 3
    CIRCLE = 4
    WEIGHTED = 5


class Rows:
    """The geometry of selected atoms (arrays with one row per atom), at least two-dimensional: 1-D flat coordinates
    are embedded in R^2 (segments on a line are parallel pieces in kernels.py)."""

    __slots__ = ("P", "E1", "E2", "radius", "piece")

    def __init__(self, P, E1, E2, radius, piece=None):
        if P.shape[1] == 1:
            P, E1, E2 = (np.concatenate([X, np.zeros((len(X), 1))], axis=1) for X in (P, E1, E2))
        self.P, self.E1, self.E2, self.radius, self.piece = P, E1, E2, radius, piece

    @property
    def Q(self):
        return self.P + self.E1


class Atoms:
    """Measures decomposed into atoms (see the module docstring): columns P, E1, E2 (m, dim), radius (m,), kind (m,),
    weight (m,), owner (m,) = the index of the measure each atom belongs to; n measures."""

    __slots__ = ("P", "E1", "E2", "radius", "kind", "weight", "owner", "n", "self_energy", "hmatrix", "spectral",
                 "piece")

    def __init__(self, P, E1, E2, radius, kind, weight, owner, n, piece=None):
        self.P, self.E1, self.E2 = (np.asarray(X, float) for X in (P, E1, E2))
        self.radius, self.kind = np.asarray(radius, float), np.asarray(kind, int)
        self.weight, self.owner, self.n = np.asarray(weight, float), np.asarray(owner, int), int(n)
        self.piece = np.full(len(self.kind), None, object) if piece is None else np.asarray(piece, object)
        self.self_energy = None                      # (energies, error estimates) of the measures, once computed
        self.hmatrix = None                          # the compressed atom energies (hmatrix.py), once built
        self.spectral = None                         # the plan of the spectral engine (spectral.py), once built

    @property
    def Q(self):
        return self.P + self.E1

    @staticmethod
    def concat(items):
        offs = np.cumsum([0] + [a.n for a in items])

        def cat(name):
            return np.concatenate([getattr(a, name) for a in items])
        return Atoms(cat("P"), cat("E1"), cat("E2"), cat("radius"), cat("kind"), cat("weight"),
                     np.concatenate([a.owner + o for a, o in zip(items, offs)]), int(offs[-1]), cat("piece"))

    def subset(self, idx):
        """The atoms of the measures idx (distinct), renumbered in that order."""
        idx = np.atleast_1d(np.asarray(idx, int))
        if len(np.unique(idx)) != len(idx):
            raise ValueError("repeated members")
        pos = np.full(self.n, -1)
        pos[idx] = np.arange(len(idx))
        k = pos[self.owner] >= 0
        return Atoms(self.P[k], self.E1[k], self.E2[k], self.radius[k], self.kind[k], self.weight[k],
                     pos[self.owner[k]], len(idx), self.piece[k])

    def rows(self, idx, shift=None):
        """The geometry of atoms idx, translated by shift, or mapped by an image y -> R y + t (green.Image)."""
        pc = self.piece[idx]
        if shift is not None and np.any(self.kind[idx] == Kind.WEIGHTED):       # images of weighted pieces (O3)
            from .weighted_contacts import transform
            R, t = (shift.R, shift.t) if hasattr(shift, "R") else (1.0, shift)
            pc = np.array([None if p is None else transform(p, R, t) for p in pc], object)
        if hasattr(shift, "R"):
            R = shift.R
            return Rows(self.P[idx] * R + shift.t, self.E1[idx] * R, self.E2[idx] * R, self.radius[idx], pc)
        P = self.P[idx] if shift is None else self.P[idx] + shift
        return Rows(P, self.E1[idx], self.E2[idx], self.radius[idx], pc)

    def extent(self):
        """Centres and radii (m, dim), (m,) of balls containing the atoms."""
        sph = (self.kind == Kind.SPHERE) | (self.kind == Kind.CIRCLE) | (self.kind == Kind.WEIGHTED)
        M = np.where(sph[:, None], self.P, self.P + 0.5 * (self.E1 + self.E2))
        r = 0.5 * np.maximum(np.linalg.norm(self.E1 + self.E2, axis=1), np.linalg.norm(self.E1 - self.E2, axis=1))
        return M, np.where(sph, self.radius, r)

    def support(self, q):
        """The support of atom q as a set with its natural measure (None for a point), in flat coordinates."""
        from ...sets import Interval
        from ...domains import Circle, Flat, Sphere
        k, P = self.kind[q], self.P[q]
        if k == Kind.POINT:
            return None
        if k == Kind.SEGMENT and len(P) == 1:
            a, b = sorted((float(P[0]), float(P[0] + self.E1[q][0])))
            return Interval(a, b)
        if k == Kind.SEGMENT:
            return Flat(P, [self.E1[q]])
        if k == Kind.PATCH:
            return Flat(P, [self.E1[q], self.E2[q]])
        if k == Kind.CIRCLE:
            return Circle(float(self.radius[q]), tuple(P))
        return Sphere(float(self.radius[q]), tuple(P))


def atoms_from(measures, domain, flat, periods=None, origin=None, unflat=None):
    """Atoms (flat coordinates) of a list of measures: Dirac measures (points), arc length on segments and polylines,
    area on flat patches and spheres (in 3-D domains), arc length on circles (in 2-D domains), subintervals of an interval domain. Every support must lie in
    the closure of the domain (checked at sample points: ends and Gauss nodes of every piece, 26 points of a sphere):
    a ray that leaves the domain must be clipped, since its integral depends on values outside. A signed measure
    sum_t c_t mu_t gives the atoms of every mu_t, all owned by its index, with their weights times c_t."""
    from ...measures import DensityMeasure, DiracMeasure, LebesgueMeasure, PieceDensity, terms_of
    from ...sets import Interval
    from ...domains import Circle, Flat, Polyline, Sphere
    cdim = domain.cartesian_dim
    cols, kind, owner, weight, objs = [], [], [], [], []
    from .segment_partition import adapts, common_segment_edges
    common = common_segment_edges([(e, mu.weight.values) for nu in measures for _, mu in terms_of(nu)
                                   if adapts(mu) and (e := _segment_ends(mu.set)) is not None])          # shared piece edges of weighted segments (plan 022 P2)
    for i, coef, mu in ((i, a, m) for i, nu in enumerate(measures) for a, m in terms_of(nu)):
        S, w = mu.set, None
        simple = None
        if isinstance(mu, DensityMeasure) and not isinstance(mu, PieceDensity):    # weighted measures (EXP-031)
            simple = _constant_weight(mu, coef, domain)
            if simple is not None:
                mu, coef, w = simple
        if (isinstance(mu, DensityMeasure) and not isinstance(mu, PieceDensity) and simple is None) or \
                (isinstance(mu, LebesgueMeasure) and _needs_piece(S, domain)):
            for piece in _pieces_of(mu, domain, flat, periods, origin, unflat, common):
                cols.append((piece.centre, 0 * piece.centre, 0 * piece.centre, piece.radius))
                kind.append(Kind.WEIGHTED)
                owner.append(i)
                weight.append(coef)
                objs.append(piece)
            continue
        if isinstance(mu, DiracMeasure):
            x = np.atleast_1d(np.asarray(mu.x0, float))
            pieces, k = [(x, 0 * x, 0 * x, 0.0)], Kind.POINT
        elif isinstance(mu, LebesgueMeasure) and isinstance(S, Flat) and S.dim == 1:
            p, q = S.endpoints
            pieces, k = [(p, q - p, 0 * p, 0.0)], Kind.SEGMENT
        elif isinstance(mu, (LebesgueMeasure, PieceDensity)) and isinstance(S, Polyline):
            a, b = S.segments
            pieces, k = [(p, q - p, 0 * p, 0.0) for p, q in zip(a, b)], Kind.SEGMENT
            w = getattr(mu, "weights", w)
        elif isinstance(mu, LebesgueMeasure) and isinstance(S, (Flat, Sphere)) and S.dim == 2:
            if cdim != 3:
                raise SpaceMismatch(f"{mu} lives in R^3; {domain} has {cdim} Cartesian coordinates")
            if domain.dim != 3:
                raise Unsupported(f"area measures as functionals on the surface {domain} are not implemented")
            if isinstance(S, Flat):
                o, (e1, e2) = np.array(S.origin), np.array(S.edges)
                pieces, k = [(o, e1, e2, 0.0)], Kind.PATCH
            else:
                c = np.array(S.centre, float)
                pieces, k = [(c, 0 * c, 0 * c, float(S.radius))], Kind.SPHERE
        elif isinstance(mu, LebesgueMeasure) and isinstance(S, Circle):
            if cdim != 2 or domain.dim != 2:
                raise Unsupported(f"a circle as a functional needs a 2-D domain, not {domain}")
            c = np.array(S.centre, float)
            pieces, k = [(c, 0 * c, 0 * c, float(S.radius))], Kind.CIRCLE
        elif isinstance(mu, LebesgueMeasure) and isinstance(S, Interval) and isinstance(domain, Interval):
            pieces, k = [(np.array([S.a]), np.array([S.b - S.a]), np.zeros(1), 0.0)], Kind.SEGMENT
        else:
            raise Unsupported(f"Riesz representers of {mu} in a space on {domain} are not implemented (points, "
                              "segments, polylines, patches and spheres in 3-D domains, circles in 2-D domains, and "
                              "subintervals of an interval are)")
        if k == Kind.SEGMENT and len(pieces) > 1 and cdim > 1 and periods is None:    # polylines: vectorised
            ends = np.array([p for p, *_ in pieces] + [pieces[-1][0] + pieces[-1][1]])
            if ends.shape[1] != cdim:
                raise SpaceMismatch(f"{mu} lives in R^{ends.shape[1]}; {domain} has {cdim} Cartesian coordinates")
            t = 0.5 * (gauss(16)[0] + 1)
            require_in_closure(domain, np.vstack([ends] + [p + np.outer(t, e) for p, e, *_ in pieces]), mu,
                               segment=k == Kind.SEGMENT)
            fp = flat(ends)
            cols += [(fp[j], fp[j + 1] - fp[j], 0 * fp[j], 0.0) for j in range(len(pieces))]
            kind += [k] * len(pieces)
            owner += [i] * len(pieces)
            objs += [None] * len(pieces)
            weight += list(coef * (np.ones(len(pieces)) if w is None else np.asarray(w, float)))
            continue
        for q, (p, e1, e2, r) in enumerate(pieces):
            if p.size != cdim:
                raise SpaceMismatch(f"{mu} lives in R^{p.size}; {domain} has {cdim} Cartesian coordinates")
            inside = round_inside(domain, p, r) if k in (Kind.SPHERE, Kind.CIRCLE) else None
            if inside is False:
                raise SupportOutsideDomain(f"{mu} leaves {domain} (a sphere or circle must lie in its closure)")
            if inside is None:
                require_in_closure(domain, _samples(k, p, e1, e2, r), mu, segment=k == Kind.SEGMENT)
            fp = flat(np.atleast_2d(p))[0]
            fe1, fe2 = (flat(np.atleast_2d(p + e))[0] - fp for e in (e1, e2))    # flat maps are affine on pieces
            if periods is not None:                       # translate the piece into the fundamental cell
                fp = fp - np.floor((fp - origin) / periods) * periods
            cols.append((fp, fe1, fe2, r))
            kind.append(k)
            owner.append(i)
            objs.append(None)
            weight.append(coef * (1.0 if w is None else float(w[q])))     # a polyline's density (one piece or periodic)
    P, E1, E2, R = (np.array(c) for c in zip(*cols))
    pc = np.empty(len(objs), object)
    pc[:] = objs
    return Atoms(P, E1, E2, R, kind, np.array(weight), owner, len(measures), pc)


def _constant_weight(mu, coef, domain):
    """(measure, coefficient, piece weights) for a DensityMeasure that the constant-weight atoms represent exactly: a
    constant weight c on a set those atoms take (c times the set's own measure), or one value per segment of a
    polyline; None otherwise (then weighted pieces, EXP-031 W3)."""
    from ...measures import LebesgueMeasure
    wt = mu.weight
    if _needs_piece(mu.set, domain):
        return None
    if wt.kind == "constant":
        return LebesgueMeasure(mu.set), coef * wt.c, None
    if wt.kind == "pieces" and hasattr(mu.set, "_L"):
        return LebesgueMeasure(mu.set), coef, wt.c
    return None


def _needs_piece(S, domain):
    """Sets that only weighted pieces represent: sub-domains of full dimension, curves given by callables, arcs,
    simplicial sets."""
    from ...domains import Arc, Ball, Box, Curve, Disk, Flat, Shell, Simplicial
    return isinstance(S, (Box, Disk, Ball, Shell, Curve, Arc, Simplicial)) or \
        (isinstance(S, Flat) and S.dim == domain.cartesian_dim == S.cartesian_dim and S.dim > 1)


class FlatWeight:
    """A weight as a function of flat (here Cartesian) points (m, d) -> (m,), with .constant for constant weights."""

    def __init__(self, f=None, constant=None):
        self.f, self.constant = f, None if constant is None else float(constant)

    def __call__(self, X):
        X = np.atleast_2d(X)
        return np.full(len(X), self.constant) if self.constant is not None else np.asarray(self.f(X), float)


def _singular_pieces(mu, wt, domain):
    """Segments and subintervals with a weight declared singular at points (|w| ~ dist^-a): split at the points on
    them into SingularSegmentPieces (rules graded towards the point). Only a < 1/2 (w in L2 of the piece, so its
    autocorrelation is finite at 0, which the same-piece route needs) and only 1-D pieces; others are refused."""
    from ...sets import Interval
    from ...domains import Flat, Polyline
    from .weighted import SegmentPiece, SingularSegmentPiece
    S = mu.set
    if not (isinstance(S, Interval) or (isinstance(S, Flat) and S.dim == 1) or isinstance(S, Polyline)):
        raise Unsupported(f"Riesz representers of {mu}: declared singular weights are realised on segments, polylines "
                          "and subintervals only (graded routes towards points of surfaces and volumes are not "
                          "implemented)")
    if not wt.exponent < 0.5:
        raise Unsupported(f"Riesz representers of {mu}: |w| ~ dist^-{wt.exponent:g} is not in L2 of the piece "
                          "(a >= 1/2): the same-piece route would need the expansion of its singular autocorrelation, "
                          "which is not implemented (bounded functionals with 1/2 <= a < 1 - max(0, d/2 - s) exist)")
    w = FlatWeight(wt.values)
    segs = [(np.array([S.a]), np.array([S.b]))] if isinstance(S, Interval) else \
        [S.endpoints] if isinstance(S, Flat) else list(zip(*S.segments))
    out = []
    for p, q in segs:
        L = float(np.linalg.norm(q - p))
        u = (q - p) / L
        cuts = sorted({float(np.clip((x - p) @ u, 0, L)) for x in wt.singular_at
                       if np.linalg.norm((x - p) - ((x - p) @ u) * u) <= 1e-12 * L and -1e-12 * L <= (x - p) @ u <= L * (1 + 1e-12)})
        if not cuts:
            out.append(SegmentPiece(p, q, w))
            continue
        edges = sorted({0.0, L} | set(cuts))
        for a, b in zip(edges[:-1], edges[1:]):
            A, B, M = p + a * u, p + b * u, p + 0.5 * (a + b) * u
            if a in cuts and b in cuts:                            # singular at both ends: two halves
                out += [SingularSegmentPiece(A, M, w), SingularSegmentPiece(B, M, w)]
            elif a in cuts:
                out.append(SingularSegmentPiece(A, B, w))
            elif b in cuts:
                out.append(SingularSegmentPiece(B, A, w))          # the singular point first
            else:
                out.append(SegmentPiece(A, B, w))
                continue
            for piece in out[-2 if a in cuts and b in cuts else -1:]:
                piece.exponent = wt.exponent
    return out


def _segment_ends(S):
    """(p, q) of a segment or subinterval, else None (the type test stays with the measure parser)."""
    from ...sets import Interval
    from ...domains import Flat
    if isinstance(S, Interval):
        return np.array([S.a], float), np.array([S.b], float)
    if isinstance(S, Flat) and S.dim == 1:
        return tuple(np.atleast_1d(np.asarray(e, float)) for e in S.endpoints)
    return None


def _pieces_of(mu, domain, flat, periods=None, origin=None, unflat=None, common=None):
    """The weighted pieces of w dsigma (or of a set's own measure, w = 1) in the flat coordinates of the domain."""
    from ...measures import DensityMeasure
    from ...sets import Interval
    from ...domains import (Arc, Ball, Box, Circle, Curve, Disk, Flat, Polyline, Shell, Simplicial, Sphere,
                           inside_closure)
    from .weighted import CurvePiece, SegmentPiece, SpherePiece
    from .segment_partition import segment_pieces
    from .weighted_volume import ParallelogramPiece, RoundPiece
    S, cdim = mu.set, domain.cartesian_dim
    wt = mu.weight if isinstance(mu, DensityMeasure) else None
    sing = wt is not None and wt.singular
    w = FlatWeight(constant=1.0) if wt is None else FlatWeight(constant=wt.c) if wt.kind == "constant" else \
        FlatWeight(wt.values)
    test = np.vstack([np.zeros(cdim), np.eye(cdim)]) * 0.5
    if not np.allclose(flat(test), test):
        out = _on_path(mu, wt, domain, flat, unflat, periods, origin)
        if out is not None:
            return out
        raise Unsupported(f"weighted measures (EXP-031) need Cartesian flat coordinates; {domain!r} has others")

    def per_piece(j):
        return w if wt is None or wt.kind != "pieces" else FlatWeight(constant=wt.c[j])
    if sing:
        out = _singular_pieces(mu, wt, domain)
    elif isinstance(S, Interval):
        out = segment_pieces([S.a], [S.b], w, wt, common)
    elif isinstance(S, Flat) and S.dim == 1:
        p, q = S.endpoints
        out = segment_pieces(p, q, w, wt, common)
    elif isinstance(S, Polyline):
        out = [SegmentPiece(p, q, per_piece(j)) for j, (p, q) in enumerate(zip(*S.segments))]
    elif isinstance(S, Simplicial) and S.dim == 1 and S.chart is None:
        out = [SegmentPiece(S.vertices[c[0]], S.vertices[c[1]], per_piece(j)) for j, c in enumerate(S.cells)]
    elif isinstance(S, Simplicial) and S.dim == 3 and S.chart is None and cdim == 3:
        from .weighted_solids import TetraPiece
        out = [TetraPiece(*S.vertices[c], per_piece(j)) for j, c in enumerate(S.cells)]
    elif isinstance(S, Simplicial) and S.dim == 2 and S.chart is None and (cdim == 2 or domain.dim == 3):
        from .weighted_volume import TrianglePiece
        out = [TrianglePiece(*S.vertices[c], per_piece(j)) for j, c in enumerate(S.cells)]
    elif isinstance(S, (Curve, Arc, Circle)):
        if isinstance(S, Curve):
            C = S
        else:
            par = S.parametrisation()
            C = Curve(par.gamma, par.t0, par.t1, derivative=par.derivative, breaks=par.breaks, closed=par.closed)
        out = [CurvePiece(C, w)]
    elif isinstance(S, Sphere) and cdim == 3:
        out = [SpherePiece(S.centre, S.radius, w)]
    elif isinstance(S, Box) and S.dim == 2 == cdim:
        out = [ParallelogramPiece([s.a for s in S.sides], np.diag([s.length for s in S.sides]), w)]
    elif (isinstance(S, Box) or (isinstance(S, Flat) and S.dim == 3)) and S.dim == 3 == cdim:
        from .weighted_solids import ParallelepipedPiece
        o, E = ([s.a for s in S.sides], np.diag([s.length for s in S.sides])) if isinstance(S, Box) else \
            (S.origin, np.array(S.edges))
        out = [ParallelepipedPiece(o, E, w)]
    elif isinstance(S, Flat) and S.dim == 2 and S.cartesian_dim == cdim:     # a parallelogram (2-D) or a patch (3-D)
        if cdim == 3 and domain.dim != 3:
            raise Unsupported(f"area measures as functionals on the surface {domain} are not implemented")
        out = [ParallelogramPiece(S.origin, np.array(S.edges), w)]
    elif (isinstance(S, Disk) and cdim == 2) or (isinstance(S, Ball) and cdim == 3):
        out = [RoundPiece(S.centre, S.radius, w)]
    elif isinstance(S, Shell) and cdim == 3:
        from .weighted_solids import ShellPiece
        out = [ShellPiece(S.centre, S.inner, S.outer, w)]
    elif isinstance(S, Simplicial) and S.chart is not None:
        R = S.chart.radius
        if abs(S.volume - 4 * np.pi * R * R) > 1e-10 * 4 * np.pi * R * R or (wt is not None and wt.kind == "pieces"):
            raise Unsupported(f"Riesz representers of {mu}: cells of a sphere chart are realised only when they cover "
                              "the sphere (the set is then the sphere) with a weight given on it; per-cell values or a "
                              "partial cover would need geodesic polar routes with breaks at great-circle edges")
        out = [SpherePiece(S.chart.centre, R, w)]
    else:
        raise Unsupported(f"Riesz representers of {mu} in Sobolev spaces on {domain} are not implemented (weighted "
                          "segments, polylines, curves, arcs, circles, spheres; boxes, parallelograms and disks in 2-D; "
                          "balls in 3-D)")
    if periods is not None:                          # a torus: every set is in it modulo the lattice; one cell at most
        from .weighted_contacts import transform
        P = np.asarray(periods, float)
        if any(2 * p.radius >= float(np.min(P)) for p in out):
            raise Unsupported(f"{mu} is as large as a period of {domain}: its images would overlap it")
        return [transform(p, 1.0, -np.floor((p.centre - np.asarray(origin, float)) / P) * P) for p in out]
    inside = inside_closure(S, domain) if not isinstance(S, Interval) else Truth.of(domain.a <= S.a and S.b <= domain.b)
    if inside is Truth.FALSE:
        raise SupportOutsideDomain(f"the support of {mu} leaves {domain} (decided exactly)")
    if inside is Truth.UNKNOWN:
        require_in_closure(domain, np.vstack([p.formula_nodes()[0] for p in out]), mu)
    return out


def _on_path(mu, wt, domain, flat, unflat, periods, origin):
    """w dsigma on a sub-path S of a one-dimensional domain whose flat coordinate is not Cartesian (an arc of a sphere,
    a circle, a curve; EXP-032 finding 6): one SegmentPiece [s0, s1] in the domain's own coordinate, the weight read
    through unflat. s0, s1 come from S's parametrisation (unwrapped by the period on circles); the piece is accepted
    only when s is monotone along S and s1 - s0 is S's length, which is what makes ds of the coordinate the arc length
    of S (then int u w dsigma over S is the integral over [s0, s1] of the coordinate). None when this does not apply."""
    from .weighted import SegmentPiece
    from .weighted_contacts import transform
    S = mu.set
    if domain.dim != 1 or unflat is None or getattr(S, "dim", None) != 1 or not hasattr(S, "parametrisation"):
        return None
    par = S.parametrisation()
    e = np.unique(np.r_[par.t0, [b for b in par.breaks if par.t0 < b < par.t1], par.t1])
    t = np.unique(np.r_[e, (e[:-1, None] + np.diff(e)[:, None] * (0.5 * (gauss(16)[0] + 1))).ravel()])
    X = np.asarray(par.gamma(t), float).reshape(len(t), -1)
    if X.shape[1] != domain.cartesian_dim:
        raise SpaceMismatch(f"{mu} lives in R^{X.shape[1]}; {domain} has {domain.cartesian_dim} Cartesian coordinates")
    s = np.asarray(flat(X), float)[:, 0]
    if not np.all(np.isfinite(s)):
        raise SupportOutsideDomain(f"the support of {mu} does not lie on {domain} (sampled along its parametrisation)")
    if periods is not None:
        s = np.unwrap(s, period=float(periods[0]))
    d, length = np.diff(s), float(S.volume)
    if not (np.all(d > 0) or np.all(d < 0)) or abs(abs(s[-1] - s[0]) - length) > 1e-8 * length:
        raise Unsupported(f"Riesz representers of {mu} on {domain}: the set is not a sub-path traversed once in the "
                          f"domain's coordinate, or that coordinate is not its arc length (range {abs(s[-1] - s[0]):.12g}"
                          f", length {length:.12g})")
    if wt is not None and wt.singular:
        raise Unsupported(f"Riesz representers of {mu}: declared singular weights on sub-paths of {domain} are not "
                          "implemented (only in Cartesian flat coordinates)")
    w = FlatWeight(constant=1.0 if wt is None else wt.c) if wt is None or wt.kind == "constant" else \
        FlatWeight(lambda F: wt.values(np.atleast_2d(unflat(np.atleast_2d(F)))))
    piece = SegmentPiece([min(s[0], s[-1])], [max(s[0], s[-1])], w)
    if periods is None:
        return [piece]
    P = np.asarray(periods, float)
    if 2 * piece.radius >= float(np.min(P)):
        raise Unsupported(f"{mu} is as long as the period of {domain}: its images would overlap it")
    return [transform(piece, 1.0, -np.floor((piece.centre - np.asarray(origin, float)) / P) * P)]


def _samples(k, p, e1, e2, r):
    """Points of a piece at which its support is checked against the domain."""
    t = 0.5 * (gauss(16)[0] + 1)
    if k == Kind.CIRCLE:                             # 2000 points of the circle (other domains than round_inside)
        a = 2 * np.pi * np.arange(2000) / 2000
        return p + r * np.stack([np.cos(a), np.sin(a)], axis=1)
    if k == Kind.SPHERE:                             # 2000 points of a Fibonacci lattice on the sphere
        i = np.arange(2000) + 0.5
        z, a = 1 - 2 * i / 2000, np.pi * (1 + 5 ** 0.5) * i
        return p + r * np.stack([np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z], axis=1)
    s = np.r_[0.0, t, 1.0]
    X = p + np.outer(s, e1)
    if k == Kind.PATCH:
        X = (X[:, None, :] + np.outer(s, e2)[None]).reshape(-1, len(p))
    return X


def atoms_inside(A, D):
    """Whether every atom of A (flat = Cartesian coordinates, or the coordinate of an interval) lies in the closed set D
    (plan 035 P4: restriction to a piece keeps such a realisation; moved from domains/pieces.py by plan 069 P2: the
    atoms are a realisation's), a Truth (plan 067 P1; was True, False or None): TRUE
    or FALSE, decided exactly for points, segments and spheres in intervals, boxes, balls and shells and for patches in
    convex ones; UNKNOWN when it is not decided (weighted pieces, circles, patches in a shell)."""
    from ...domains import Box
    from ...domains.partition import TOL
    from ...domains.pieces import annuli
    from ...sets import Interval
    rd = annuli(D)
    rd = rd[0] if rd is not None and len(rd) == 1 else None          # a ball or shell: (centre, inner, outer)
    if isinstance(D, Interval):
        tol = TOL * max(1.0, abs(D.a), abs(D.b))
        inside = lambda X: bool(np.all((X[:, 0] >= D.a - tol) & (X[:, 0] <= D.b + tol)))      # noqa: E731
    elif isinstance(D, Box):
        lo, hi = np.array([s.a for s in D.sides]), np.array([s.b for s in D.sides])
        tol = TOL * float(np.max(hi - lo))
        inside = lambda X: bool(np.all((X >= lo - tol) & (X <= hi + tol)))                     # noqa: E731
    elif rd is not None:
        C, a, b = rd
        tol = TOL * b
        inside = lambda X: bool(np.all(np.linalg.norm(X - C, axis=1) <= b + tol))              # noqa: E731
    else:
        return Truth.UNKNOWN
    for q in range(len(A.P)):
        k, P, E1, E2 = A.kind[q], A.P[q], A.E1[q], A.E2[q]
        if k == Kind.POINT:
            ok = inside(P[None]) and (rd is None or rd[1] == 0 or np.linalg.norm(P - rd[0]) >= rd[1] - tol)
        elif k == Kind.SEGMENT:
            ok = inside(np.stack([P, P + E1]))                   # convex sets: the ends suffice
            if ok and rd is not None and rd[1] > 0:              # a shell: the nearest point to the centre
                t = float(np.clip(-(P - rd[0]) @ E1 / max(E1 @ E1, 1e-300), 0.0, 1.0))
                ok = np.linalg.norm(P + t * E1 - rd[0]) >= rd[1] - tol
        elif k == Kind.SPHERE:
            R = float(A.radius[q])
            if isinstance(D, Box):
                ok = inside(np.stack([P - R, P + R]))
            else:
                d = float(np.linalg.norm(P - rd[0]))
                ok = d + R <= rd[2] + tol and (rd[1] == 0 or abs(d - R) >= rd[1] - tol)
        elif k == Kind.PATCH and (rd is None or rd[1] == 0):
            ok = inside(np.stack([P, P + E1, P + E2, P + E1 + E2]))
        else:
            return Truth.UNKNOWN
        if not ok:
            return Truth.FALSE
    return Truth.TRUE
