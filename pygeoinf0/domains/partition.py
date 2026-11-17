"""Partitions of a domain into pieces (plan 035 P3, design D1).

    P = pg.Partition(pg.Ball(1.0), {"core": pg.Ball(0.55), "mantle": pg.Shell(0.55, 1.0)})
    P = pg.Partition("[0, 1]", ["[0, 0.4]", "[0.4, 1]"], names=["left", "right"])
    P = pg.LayeredBall([0.55, 0.8, 1.0], names=["a", "b", "c"]).partition         the radial partition of a layered ball

A Partition checks that its pieces tile the domain up to a null set, from their descriptions and only where that is
decided exactly (to TOL relative to the domain's size):

    Interval      coordinate intervals: sorted, the first starts and the last ends at the domain's ends, each ends where
                  the next starts
    Ball, Shell   concentric balls and shells (one centre): the same test on their radial intervals
    Box           boxes of its dimension inside it, pairwise disjoint interiors, volumes adding up to its volume
    Sphere, band  latitude bands and polar caps of the one sphere (domains/bands.py): the same test on their
                  colatitude intervals (caps about other axes and other shapes are refused: Unsupported)

and refuses anything else (Unsupported: it cannot decide), a gap, an overlap or a piece outside (NotATiling, a Refused
naming the pieces). It owns the seams: P.seams, one Seam per interface between two pieces with its two sides (below:
nearer the centre, the lower coordinate or the smaller colatitude; above: the other).

P.union is the union of the pieces, a Region whose measure is the domain's (plan 067 P2: a Partition is a family of
regions, not a region itself). Refinement is a tree: a piece may itself be a Partition (of that piece), P.refine(name, pieces) returns a new partition
with one piece cut again (nothing is mutated), and P.flat() is the partition of the same domain into the leaves (their
names must be distinct). Spaces live on leaves (pg.PiecewiseSpace, spaces/piecewise.py).

    P.split(mu, side=None)   {name: mu restricted to the closed piece} (mu.restricted, plan 035 P1) for the pieces mu
                             charges, with a claim that they sum to mu: the seams are null sets for measures of full
                             dimension and for paths; a point or sphere mass on a seam needs side= (never halved), one on
                             the outer boundary belongs to the one piece that touches it
    X.restrictions(P), P.glue(X) the operators u -> (u|Omega_k)_k and its inverse (operators/pieces.py)
    P.functionals(C, mu)     on C(K): the functionals int . d(mu|K_k) on the closed pieces (C.split_functional)
"""
import collections

import numpy as np

from .solids import Ball
from .solids import Box
from .layered import Shell
from .manifolds import Sphere

TOL = 1e-12          # relative to the domain's size: endpoints, radii and centres closer than this are equal

Seam = collections.namedtuple("Seam", "name below above at set")
Seam.__doc__ = """An interface between two pieces: its name ("below/above"), the pieces on its two sides (below: nearer the
centre, the lower coordinate or the smaller colatitude), where it is (a point of an interval, a radius, a colatitude, or
(axis, value) for boxes) and the set (a Sphere for balls and shells, None otherwise)."""


def _as_set(S):
    from ..sets import Interval
    return Interval(S) if isinstance(S, str) else S


def _round(S):
    """(centre, inner, outer) of a Ball or Shell, else None."""
    if isinstance(S, Shell):
        return np.array(S.centre, float), S.inner, S.outer
    if isinstance(S, Ball):
        return np.array(S.centre, float), 0.0, S.radius
    return None


def _kind(S):
    from ..sets import Interval
    if isinstance(S, Interval):
        return "interval"
    if isinstance(S, Box):
        return "box"
    if _round(S) is not None:
        return "round"
    from .bands import colatitudes
    if colatitudes(S) is not None:
        return "band"
    return None


def _tile_line(lo, hi, items, tol, what):
    """Names of items (name, a, b) sorted along [lo, hi], refused unless they tile it (NotATiling)."""
    from ..core.errors import NotATiling
    items = sorted(items, key=lambda t: (t[1], t[2]))
    for n, a, b in items:
        if b - a <= tol:
            raise NotATiling(f"the piece {n!r} has no {what} extent ({a:g} to {b:g})", needs=("pieces of positive size",))
        if a < lo - tol or b > hi + tol:
            raise NotATiling(f"the piece {n!r} ({a:g} to {b:g}) leaves the domain ({lo:g} to {hi:g})",
                             needs=("pieces inside the domain",))
    cover = ("pieces that cover the domain",)
    if items[0][1] > lo + tol:
        raise NotATiling(f"a gap from {lo:g} to {items[0][1]:g} (before {items[0][0]!r})", needs=cover)
    if items[-1][2] < hi - tol:
        raise NotATiling(f"a gap from {items[-1][2]:g} to {hi:g} (after {items[-1][0]!r})", needs=cover)
    for (n, a, b), (m, c, d) in zip(items[:-1], items[1:]):
        if c > b + tol:
            raise NotATiling(f"a gap from {b:g} to {c:g} between {n!r} and {m!r}", needs=cover)
        if c < b - tol:
            raise NotATiling(f"{n!r} and {m!r} overlap from {c:g} to {min(b, d):g}",
                             needs=("pieces that meet only on their boundaries",))
    return [t[0] for t in items]


class Partition:
    """The domain cut into pieces that tile it up to a null set (module docstring). pieces: a dict {name: set or
    Partition of that set}, or a list (names= or "piece 0", "piece 1", ...)."""

    def __init__(self, domain, pieces, names=None):
        from ..core.errors import Unsupported
        domain = _as_set(domain)
        if isinstance(pieces, dict):
            if names is not None:
                raise ValueError("give names= only with a list of pieces")
            items = list(pieces.items())
        else:
            pieces = list(pieces)
            names = [f"piece {k}" for k in range(len(pieces))] if names is None else list(names)
            if len(names) != len(pieces):
                raise ValueError(f"{len(pieces)} pieces need {len(pieces)} names")
            items = list(zip(names, pieces))
        if not items:
            raise ValueError("a partition needs at least one piece")
        if len({n for n, _ in items}) != len(items):
            raise ValueError("the pieces need distinct names")
        self.domain, self.names = domain, tuple(n for n, _ in items)
        self.children = {n: (p if isinstance(p, Partition) else None) for n, p in items}
        self.pieces = {n: (p.domain if isinstance(p, Partition) else _as_set(p)) for n, p in items}
        kind = _kind(domain)
        if kind is None or any(_kind(S) != kind for S in self.pieces.values()):
            raise Unsupported(f"whether these pieces tile {domain!r} is not decided here: radial intervals of a ball or "
                              "shell (concentric balls and shells), coordinate intervals of an interval, boxes in a "
                              f"box and colatitude intervals of a sphere (bands and polar caps) are (pieces: {', '.join(repr(S) for S in self.pieces.values())})")
        self.kind = kind
        getattr(self, f"_check_{kind}")()

    # ---------------------------------------------------------------- the tiling checks
    def _check_interval(self):
        I = self.domain
        tol = TOL * max(1.0, abs(I.a), abs(I.b))
        self.order = tuple(_tile_line(I.a, I.b, [(n, S.a, S.b) for n, S in self.pieces.items()], tol, "length"))
        self.seams = tuple(Seam(f"{a}/{b}", a, b, self.pieces[b].a, None) for a, b in zip(self.order[:-1], self.order[1:]))
        self._how = "the endpoints, sorted: each piece ends where the next starts"

    def _check_round(self):
        from ..core.errors import Unsupported
        C, a, b = _round(self.domain)
        tol = TOL * b
        rings = {}
        for n, S in self.pieces.items():
            c, lo, hi = _round(S)
            if np.linalg.norm(c - C) > tol:
                raise Unsupported(f"the piece {n!r} = {S!r} is not concentric with {self.domain!r}: only radial "
                                  "partitions of balls and shells are decided here")
            rings[n] = (lo, hi)
        self.order = tuple(_tile_line(a, b, [(n, lo, hi) for n, (lo, hi) in rings.items()], tol, "radial"))
        self.seams = tuple(Seam(f"{x}/{y}", x, y, rings[y][0], Sphere(rings[y][0], tuple(C)))
                           for x, y in zip(self.order[:-1], self.order[1:]))
        self._how = "concentric balls and shells whose radial intervals tile the domain's (sorted)"

    def _check_box(self):
        from ..core.errors import NotATiling
        B = self.domain
        lo, hi = np.array([s.a for s in B.sides]), np.array([s.b for s in B.sides])
        tol = TOL * float(np.max(hi - lo))
        box = {}
        for n, S in self.pieces.items():
            if S.dim != B.dim:
                raise NotATiling(f"the piece {n!r} = {S!r} is a box of another dimension than {B!r}")
            a, b = np.array([s.a for s in S.sides]), np.array([s.b for s in S.sides])
            if np.any(a < lo - tol) or np.any(b > hi + tol):
                raise NotATiling(f"the piece {n!r} = {S!r} leaves {B!r}", needs=("pieces inside the domain",))
            box[n] = (a, b)
        names = list(box)
        seams = []
        for i, n in enumerate(names):
            for m in names[i + 1:]:
                (a, b), (c, d) = box[n], box[m]
                ov = np.minimum(b, d) - np.maximum(a, c)
                if np.all(ov > tol):
                    raise NotATiling(f"{n!r} and {m!r} overlap", needs=("pieces that meet only on their boundaries",))
                touch = np.flatnonzero(np.abs(ov) <= tol)
                if len(touch) == 1 and np.all(np.delete(ov, touch) > tol):
                    k = int(touch[0])
                    x, y = (n, m) if a[k] < c[k] else (m, n)
                    seams.append(Seam(f"{x}/{y}", x, y, (k, float(max(a[k], c[k]))), None))
        vol = sum(float(np.prod(b - a)) for a, b in box.values())
        whole = float(np.prod(hi - lo))
        if vol < whole * (1 - 1e-12):
            raise NotATiling(f"the pieces cover a volume {vol:.12g} of {whole:.12g}: a gap",
                             needs=("pieces that cover the domain",))
        self.order, self.seams = tuple(self.names), tuple(seams)
        self._how = "boxes inside the domain with pairwise disjoint interiors whose volumes add up to its volume"

    def _check_band(self):
        from ..core.errors import Unsupported
        from .bands import colatitudes, same_sphere
        S, a, b = colatitudes(self.domain)
        rings = {}
        for n, T in self.pieces.items():
            U, lo, hi = colatitudes(T)
            if not same_sphere(S, U):
                raise Unsupported(f"the piece {n!r} = {T!r} lies on another sphere than {self.domain!r}")
            rings[n] = (lo, hi)
        self.order = tuple(_tile_line(a, b, [(n, lo, hi) for n, (lo, hi) in rings.items()], TOL, "colatitude"))
        self.seams = tuple(Seam(f"{x}/{y}", x, y, rings[y][0], None) for x, y in zip(self.order[:-1], self.order[1:]))
        self._how = "bands and polar caps of one sphere about its axis whose colatitude intervals tile the domain's"

    # ---------------------------------------------------------------- structure
    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, name):
        """The set of a piece, by name (a leaf of a refined piece too)."""
        if name in self.pieces:
            return self.pieces[name]
        for child in self.children.values():
            if child is not None:
                try:
                    return child[name]
                except KeyError:
                    pass
        raise KeyError(f"{name!r} is not a piece of {self} (pieces: {', '.join(map(repr, self.names))})")

    def seam(self, below, above):
        """The seam between two pieces (in either order)."""
        for s in self.seams:
            if {s.below, s.above} == {below, above}:
                return s
        raise KeyError(f"{below!r} and {above!r} do not share a seam in {self}")

    @property
    def is_flat(self):
        return all(c is None for c in self.children.values())

    def refine(self, name, pieces, names=None):
        """A new partition with the piece name cut into pieces (a Partition of that piece, checked as such)."""
        if name not in self.pieces:
            raise KeyError(f"{name!r} is not a piece of {self}")
        child = pieces if isinstance(pieces, Partition) else Partition(self.pieces[name], pieces, names)
        if child.domain != self.pieces[name]:
            raise ValueError(f"the refinement partitions {child.domain!r}, not the piece {name!r} = {self.pieces[name]!r}")
        return Partition(self.domain, {n: (child if n == name else (self.children[n] or self.pieces[n]))
                                       for n in self.names})

    def leaves(self):
        """{leaf name: set}, depth first in the order of the pieces."""
        out = {}
        for n in self.names:
            sub = self.children[n].leaves() if self.children[n] is not None else {n: self.pieces[n]}
            clash = set(sub) & set(out)
            if clash:
                raise ValueError(f"the leaves {sorted(clash)} appear twice: name the pieces of a refinement apart")
            out.update(sub)
        return out

    def flat(self):
        """The partition of the same domain into the leaves of the tree (itself when it is flat)."""
        if self.is_flat:
            return self
        if getattr(self, "_flat", None) is None:
            self._flat = Partition(self.domain, self.leaves())
        return self._flat

    # ---------------------------------------------------------------- measures
    def split(self, mu, side=None):
        """{name: mu restricted to the closed piece} for the pieces mu charges (module docstring); .claims says they sum
        to mu."""
        from ..measures import SignedMeasure, terms_of
        parts = {}
        for c, m in terms_of(mu):
            got, ambiguous = {}, []
            for n in self.names:
                try:
                    r = m.restricted(self.pieces[n], side)
                except ValueError as e:              # on the boundary of the piece and no side given
                    if side is not None or "side=" not in str(e):
                        raise
                    rb, ra = m.restricted(self.pieces[n], "below"), m.restricted(self.pieces[n], "above")
                    r = rb if rb is not None else ra
                    if r is not None:
                        ambiguous.append(n)
                if r is not None:
                    got[n] = r
            if len(ambiguous) > 1 or (ambiguous and len(got) > 1):
                raise ValueError(f"{m} lies on a seam of {self} (between {', '.join(map(repr, got))}), where a field on "
                                 "pieces may have two values: say side='below' or side='above'")
            for n, r in got.items():
                parts.setdefault(n, []).append((c, r))
        out = Split({n: (t[0][1] if len(t) == 1 and t[0][0] == 1.0 else SignedMeasure(t))
                     for n, t in sorted(parts.items(), key=lambda kv: self.names.index(kv[0]))})
        out.measure, out.partition = mu, self
        return out

    # ---------------------------------------------------------------- spaces and operators (operators/pieces.py)
    def functionals(self, X, mu, side=None):
        """On C(K): {piece: f -> int f d(mu|K_k) on C(K_k)}, a seam mass once (side=); called on u, sum_k l_k(R_k u) =
        int u dmu (C.split_functional)."""
        from ..core.errors import Refused
        if not hasattr(X, "split_functional"):
            raise Refused(f"P.functionals is for C(K) (functionals are measures); on {X} a functional is a Riesz "
                          "representer", needs=("pg.PiecewiseSpace(P, fields).functional(field, mu) and "
                                                ".piece_functionals",))
        return X.split_functional(self, mu, side)

    def glue(self, X):
        """G: (+)_k X(Omega_k) -> X(Omega), the inverse (and for L2 the adjoint) of X.restrictions(P)."""
        return X.restrictions(self).glue

    @property
    def union(self):
        """The union of the pieces, a Region (plan 067 P2): membership is the pieces' (exact); its dimension and its
        Hausdorff measure are the domain's, since the pieces tile it up to a null set (the tiling claim)."""
        from ..sets import RegionUnion
        return RegionUnion(list(self.pieces.values()), dim=self.domain.dim, volume=self.domain.volume,
                           why=f"the pieces tile {self.domain!r} up to a null set (the claim 'tiling')")

    @property
    def claims(self):
        from ..core.claims import DERIVED, Claim, Claims
        return Claims([Claim(f"the {len(self)} pieces tile {self.domain!r} up to a null set", DERIVED,
                             f"decided exactly from their descriptions: {self._how} (to {TOL:g} relative)", key="tiling"),
                       Claim(f"{len(self.seams)} seams, each between two pieces", DERIVED,
                             "the shared boundaries found by the tiling check", key="seams")])

    def __eq__(self, other):
        return isinstance(other, Partition) and other.domain == self.domain and other.names == self.names and \
            all(other.pieces[n] == self.pieces[n] and other.children[n] == self.children[n] for n in self.names)

    def __hash__(self):
        return hash(("Partition", self.names))

    def __repr__(self):
        parts = ", ".join(f"{n}: {self.children[n]!r}" if self.children[n] is not None else f"{n}: {self.pieces[n]!r}"
                          for n in self.names)
        return f"Partition({self.domain!r} = {parts})"


class Split(dict):
    """{piece name: restricted measure}, from Partition.split; .claims says that the parts sum to the measure."""

    @property
    def claims(self):
        from ..core.claims import DERIVED, Claim, Claims
        return Claims([Claim(f"the {len(self)} parts sum to {self.measure}", DERIVED,
                             "the pieces tile the domain up to a null set (the partition's check) and each part is the "
                             "restriction to a closed piece; a point or sphere mass on a seam went to the side given, "
                             "once, and one on the outer boundary to the one piece touching it", key="split")])


def breaks_for(domain, sets):
    """The boundaries of sets inside domain as breaks per natural axis of domain (what a realisation takes), when they
    are coordinate surfaces of domain: interior endpoints on an interval, box faces on a box, concentric radii on a
    ball or shell; None when they are not (a fit across them would not be told where they are)."""
    kind = _kind(domain)
    if kind is None or any(_kind(S) != kind for S in sets):
        return None
    if kind == "interval":
        return sorted({p for S in sets for p in (S.a, S.b) if domain.a < p < domain.b}) or None
    if kind == "box":
        if any(S.dim != domain.dim for S in sets):
            return None
        return tuple(sorted({p for S in sets for p in (S.sides[k].a, S.sides[k].b)
                             if domain.sides[k].a < p < domain.sides[k].b}) or None for k in range(domain.dim))
    if kind == "band":
        from .bands import colatitudes
        lo, hi = colatitudes(domain)[1:]
        return (tuple(sorted({t for S in sets for t in colatitudes(S)[1:] if lo < t < hi})) or None, None)
    C, a, b = _round(domain)
    radii = set()
    for S in sets:
        c, lo, hi = _round(S)
        if np.linalg.norm(c - C) > TOL * b:
            return None
        radii |= {r for r in (lo, hi) if a < r < b}
    return (tuple(sorted(radii)) or None, None, None)


def carry_breaks(domain, D, br):
    """Breaks br of a function on domain (per natural axis, or points on an interval) for a fit on the piece D: the
    interior ones on an interval; on a box or a concentric ball or shell the same per-axis breaks (the same natural
    coordinates); None otherwise, and for a bank function's natural breaks (recomputed for D: core/points.hints)."""
    from ..sets import Interval
    if br is None or hasattr(br, "degree") or (breaks_for(domain, [D]) is None and D != domain):
        return None
    if isinstance(D, Interval):
        return sorted({float(p) for p in np.ravel(br) if D.a < p < D.b}) or None
    return tuple(None if b is None else tuple(np.ravel(b)) for b in br)


def members(S, X):
    """Whether the points X (tuple of coordinate arrays) lie in the closed set S (with the domains' own tolerance)."""
    from ..sets import Interval
    if isinstance(S, Interval):
        t = np.asarray(X[0], float)
        tol = TOL * max(1.0, abs(S.a), abs(S.b))
        return (t >= S.a - tol) & (t <= S.b + tol)
    inside = getattr(S, "_inside_closed", S._inside)
    return np.asarray(inside(*[np.asarray(x, float) for x in X]), bool)
