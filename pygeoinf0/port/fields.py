"""What a plotting or export plugin needs from an element: its pieces, fields and members, each piece's chart, the values of
each (piece, field, member) at points of that piece, and how accurate those values are (plan 056 P1, PVPLOT-001).

    F = pg.port.fields(f)        f an element, a family or a list of elements (a list stays formulas: nothing is fitted)
    F.members, F.fields          member names; field names ("value" unless f lives on a pg.PiecewiseSpace of fields)
    F.pieces                     [Piece]: name, domain, chart (reference box, periodic axes, breaks, to_cartesian), fields
    F.values(piece, field, member, points)    (m,) values at points (m, cartesian_dim) of that closed piece
    F.accuracy(piece, field, member)          {"kind": "formula" | "realisation" | "symbolic", "error", "norm", "text"}
    F.absent                     what is not gridded, with the reason: pieces without a field, PiecewiseSpace summands
                                 outside the partition (a boundary's heights), Euclidean summands

Pieces. On pg.PiecewiseSpace the partition's leaves that carry a field, each with the fields it carries
(a glued Sobolev field is one element of the whole domain, evaluated on every leaf); on another pg.DirectSum each summand that
is a function space (Euclidean summands are listed in F.absent); otherwise the domain's own pieces (a polyline: its segments).
Each piece's values are those of its own part, so at a seam each side gives its own limit and a jump between pieces is kept:
nothing here evaluates a glued function on a seam.

Accuracy. A formula part is exact at every point (the formula is the element). A realised part carries its realisation's
estimated error in the norm of its space (Function.error), which bounds no single value: a pointwise bound is not known
here and the text says so. A symbolic part (the representer of a measure) is evaluated through its density.
"""
import numpy as np

from ..core.errors import NotDefined, Unsupported
from ..domains.base import Domain
from ..spaces.identity import require_realised

VALUE = "value"


class Chart:
    """A piece's coordinates: reference box, periodic axes, breaks and the map to Cartesian points (domains/base.py)."""

    def __init__(self, domain):
        ref = tuple(getattr(domain, "reference", ()) or ())
        if not ref or type(domain).to_cartesian is Domain.to_cartesian:
            raise Unsupported(f"{domain!r} has no chart (a reference box and to_cartesian), so it cannot be gridded: "
                              "export the pieces it is made of")
        self.domain, self.reference = domain, tuple((float(a), float(b)) for a, b in ref)
        self.dim = len(self.reference)
        self.cartesian_dim = int(domain.cartesian_dim)
        self.periodic = tuple(bool(p) for p in (domain.periodic or (False,) * self.dim))
        self.names = tuple(domain.internal_names or tuple(f"xi{k}" for k in range(self.dim)))
        br = domain.reference_breaks or (None,) * self.dim
        self.breaks = tuple(tuple(sorted(float(b) for b in (br[k] or ()) if lo < b < hi))
                            for k, (lo, hi) in enumerate(self.reference))

    def to_cartesian(self, *xi):
        """Cartesian points (..., cartesian_dim) of internal coordinate arrays."""
        out = self.domain.to_cartesian(*[np.asarray(a, float) for a in xi])
        return np.stack(np.broadcast_arrays(*out), axis=-1)

    def __repr__(self):
        return f"Chart({self.domain!r}: {', '.join(f'{n} in [{a:g}, {b:g}]' for n, (a, b) in zip(self.names, self.reference))})"


class Piece:
    def __init__(self, name, domain, fields):
        self.name, self.domain, self.fields = name, domain, tuple(fields)
        self.chart = Chart(domain)

    def __repr__(self):
        return f"Piece({self.name!r} on {self.domain!r}: {', '.join(self.fields)})"


def _members(f):
    from ..spaces.elements import Family
    if isinstance(f, Family):
        return [f[i] for i in range(len(f))], [f"{f.name}[{i}]" for i in range(len(f))]
    if isinstance(f, (list, tuple)):
        if not f:
            raise ValueError("no elements to export")
        return list(f), [u.name if sum(v.name == u.name for v in f) == 1 else f"{u.name}[{i}]" for i, u in enumerate(f)]
    return [f], [f.name]


class Fields:
    """The pieces, fields and members of f (module docstring)."""

    def __init__(self, f):
        from ..spaces.direct_sum import DirectSum
        from ..spaces.base import Euclidean
        from ..spaces.piecewise import PiecewiseSpace
        self.elements, self.members = _members(f)
        X = self.elements[0].space
        if any(u.space != X for u in self.elements):
            raise ValueError("the members of an export must lie in one space")
        for u in self.elements:
            require_realised(u.space, X, getattr(u, "name", None))
        self.space, self.absent, self._get, self._cache = X, [], {}, {}
        if isinstance(X, PiecewiseSpace):
            P = X.partition
            self.fields = tuple(X.fields)
            self.pieces = []
            for n in P.names:
                here = [fl for fl in X.fields if (fl, n) in X.keys]
                if not here:
                    self.absent.append((n, "no field on this piece"))
                    continue
                self.pieces.append(Piece(n, P[n], here))
                for fl in here:
                    key = X.keys[(fl, n)]
                    self._get[(n, fl)] = (lambda u, key=key: X.part(u, key))
            used = set(X.keys.values())
            self.absent += [(k, f"a summand outside the partition ({H!r}): export it on its own, pg.port.fields(x[{k!r}])")
                            for k, H in zip(X.names, X.summands) if k not in used]
        elif isinstance(X, DirectSum):
            self.fields, self.pieces = (VALUE,), []
            for n, H in zip(X.names, X.summands):
                if isinstance(H, Euclidean) or getattr(H, "domain", None) is None:
                    self.absent.append((n, f"{H!r} is not a space of functions on a domain"))
                    continue
                self.pieces.append(Piece(n, H.domain, [VALUE]))
                self._get[(n, VALUE)] = (lambda u, n=n: X.part(u, n))
        else:
            D = getattr(X, "domain", None)
            if D is None:
                raise NotDefined(f"{X!r} is not a space of functions on a domain: there is nothing to grid")
            parts = D.pieces()
            self.fields = (VALUE,)
            self.pieces = [Piece(str(i) if len(parts) > 1 else "whole", p, [VALUE]) for i, p in enumerate(parts)]
            for p in self.pieces:
                self._get[(p.name, VALUE)] = (lambda u: u)
        if not self.pieces:
            raise Unsupported(f"no piece of {X!r} carries a function: {self.absent}")

    def piece(self, name):
        for p in self.pieces:
            if p.name == name:
                return p
        raise KeyError(f"{name!r} is not a piece here (pieces: {', '.join(p.name for p in self.pieces)})")

    def _member(self, i):
        return self.members.index(i) if isinstance(i, str) else int(i)

    def element(self, piece, field, member):
        """The element of the piece's space whose values F.values gives."""
        name = piece.name if isinstance(piece, Piece) else piece
        key = (name, field, self._member(member))
        if key not in self._cache:
            if (name, field) not in self._get:
                raise KeyError(f"the field {field!r} is not on the piece {name!r}")
            self._cache[key] = self._get[(name, field)](self.elements[key[2]])
        return self._cache[key]

    def values(self, piece, field, member, points):
        """Values at points (m, cartesian_dim) of the closed piece: that piece's own part, so a seam gives this side's limit."""
        u = self.element(piece, field, member)
        P = np.asarray(points, float).reshape(-1, self.piece(piece if isinstance(piece, str) else piece.name).chart.cartesian_dim)
        return np.asarray(u(*P.T) if P.shape[1] > 1 else u(P[:, 0]), float).reshape(-1)

    def accuracy(self, piece, field, member):
        from ..core import lazy
        from ..spaces.representers import DensityRepresenter
        u = self.element(piece, field, member)
        norm = repr(u.space)
        if isinstance(u, DensityRepresenter):
            return dict(kind="symbolic", error=0.0, norm=norm,
                        text="the representer of a measure, evaluated through its density (no realisation)")
        if lazy.has_formula(u):
            return dict(kind="formula", error=0.0, norm=norm, text="exact at every point: the formula is the element")
        e = float(u.error)
        return dict(kind="realisation", error=e, norm=norm,
                    text=f"realised; estimated error {e:.2g} in the norm of {norm}, which bounds no single value "
                         "(no pointwise bound is known here)")

    def __repr__(self):
        return (f"Fields({len(self.members)} member(s) {', '.join(self.members[:3])}{'...' if len(self.members) > 3 else ''}; "
                f"fields {', '.join(self.fields)}; pieces {', '.join(p.name for p in self.pieces)})")


def fields(f):
    """The pieces, fields and members of f, an element, a family or a list of elements (module docstring)."""
    return Fields(f)
