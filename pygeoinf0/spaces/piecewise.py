"""Spaces of fields on the pieces of a partition: one class, glued or broken (plan 069 P7, review B9; it replaces
spaces/layered.Layered and operators/pieces.OnPieces, plans 007 and 035).

    P = pg.Partition(pg.Ball(1.0), {"core": pg.Ball(0.55), "mantle": pg.Shell(0.55, 1.0)})   (or a LayeredBall: its
                                                                                             radial partition)
    M = pg.PiecewiseSpace(P, X)                     one field: X a space on P's domain (or a constructor, pg.L2)
    M = pg.PiecewiseSpace(P, X, broken=True)        the broken form (+)_k X(Omega_k) (= X.restrictions(P).codomain)
    M = pg.PiecewiseSpace(P, {"vp": pg.L2, "s": H}, where={"vp": ["mantle"]})    several fields, keyed "<field>_<piece>"
                                                                                 (a glued Sobolev field: "<field>")
    M = pg.PiecewiseSpace.from_pieces(P, {"mantle": pg.Sobolev(P["mantle"], 2, length_scale=0.1), "crust": ...})
    M = pg.PiecewiseSpace.lebesgue(P), pg.PiecewiseSpace.sobolev(P, s, length_scale, pieces=...)     on each piece

The mathematics. Let Omega = union_k Omega_k be a partition (the pieces tile Omega up to a null set, the partition's
check) and X a space of functions on Omega with spaces X(Omega_k) on the pieces (X restricted: pg.Restrict).

  broken   (+)_k X(Omega_k), the Hilbert direct sum of the fields on the pieces: an element is a family (x_k) with no
           relation across the seams, and ||x||^2 = sum_k w_k ||x_k||^2_(X(Omega_k)) (unit weights unless given): the
           norm of the broken space is the sum of its pieces' norms.
  glued    the subspace of the broken sum whose families are the restrictions (u|Omega_k) of one u in X(Omega), which
           for X = H^s(Omega) is stated by interface conditions on the seams (spaces/gluing.py): none for s < 1/2;
           continuity of the trace for 1/2 < s < 3/2 (the trace theorem); and the normal derivatives of order
           j < s - 1/2 as well beyond. Its summand is X itself (the field's key), with X's norm: the pieces are views
           R_k, and the norm is not the sum of the pieces' norms when X's norm is non-local (the restriction
           construction, a fractional power: the claim "additive").
  L2       has no trace: L2(Omega) = (+)_k L2(Omega_k) isometrically, u -> (u|Omega_k)_k (the seams are null sets), so for
           L2 (and L2(rho dx)) glued and broken are the same space; its summands are the pieces either way.

X.restrictions(P) is u -> (u|Omega_k)_k into the broken space: for L2 an isometry onto it (the glued one), for H^s
injective, bounded and not an isometry, onto the glued subspace (operators/pieces.py, operators/pieces_sobolev.py).
M.glued() is the glued space of the same fields; M.glue(x) carries an element to the fields on the whole domain, and
for a broken H^s field checks the interface condition first (refused when the traces differ, or where the condition
is not checked here); M.seam_jumps(x) reports the jumps of the traces, M.evaluate(x, points, side) values anywhere.

    M.riesz(mu, side=)   one field: the representer of x -> sum_k int x_k d(mu|Omega_k) (P.split); a dict
                         {summand: measure} or a list of them as for pg.DirectSum
    M.functional(field, mu, side=), M.parts(field, mu, side=), M.piece_functionals(field, mu, side=)
    M.split({field: u}), M.glue(x, field=None), M.restriction(field, piece), M.key(field, piece), M.pieces_of(field)

L^p and C(K) fields are refused (the package's direct sums are Hilbert sums: spaces/gluing.refuse_product). The one
field of PiecewiseSpace(P, X) is named "value".
"""
import numpy as np

from . import gluing
from .base import VectorSpace
from .direct_sum import DirectSum
from .elements import Family, Function
from .identity import require_realised
from ..core import fits
from ..core.claims import THEOREM, Claim, Claims
from ..core.errors import Refused, SupportOutsideDomain

VALUE = "value"          # the name of the one field of PiecewiseSpace(P, X)


class _ByPieces:
    """One field given piece by piece (PiecewiseSpace.from_pieces)."""

    def __init__(self, spaces):
        self.spaces = dict(spaces)


def _partition(P):
    """The flat partition of P, or of a LayeredBall's radial partition."""
    from ..domains.layered import LayeredBall
    return (P.partition if isinstance(P, LayeredBall) else P).flat()


ON = 1e-9               # distance from a seam or the boundary, relative to the domain's size, within which a point is on it


def _ring(S):
    """(centre, inner radius, outer radius) of a Ball or Shell."""
    inner, outer = (0.0, S.radius) if hasattr(S, "radius") else (S.inner, S.outer)
    return np.asarray(S.centre, float), float(inner), float(outer)


def locate(P, X, side=None):
    """The index (in P.names) of the closed piece holding each point of X (a tuple of coordinate arrays), -1 outside the
    domain. A point within ON times the domain's size of a seam belongs to the pieces on both sides: side="below" or
    "above" picks one, None refuses. Concentric pieces and intervals are located by the radius or the coordinate (as
    LayeredBall.layer_of); boxes and bands by closed membership."""
    from ..domains.partition import members
    if P.kind in ("interval", "round"):
        if P.kind == "interval":
            r = np.asarray(X[0], float)
            ring = {n: (P.pieces[n].a, P.pieces[n].b) for n in P.order}
            size = max(1.0, abs(P.domain.a), abs(P.domain.b))
        else:
            c, _, size = _ring(P.domain)
            r = np.linalg.norm(np.stack([np.asarray(x, float) for x in X], axis=-1) - c, axis=-1)
            ring = {n: _ring(P.pieces[n])[1:] for n in P.order}
        tol = ON * size
        lo, R = ring[P.order[0]][0], np.array([ring[n][1] for n in P.order])
        k = np.searchsorted(R, r, side="left")
        near = np.abs(r[:, None] - R[:-1]) <= tol
        on = near.any(axis=1)
        if np.any(on):
            if side not in ("below", "above"):
                j = int(np.argmax(near[on][0]))
                raise ValueError(f"a point is on the seam {P.order[j]}/{P.order[j + 1]} of {P}, where a field on pieces "
                                 "may have two values: say side='below' or side='above'")
            k = np.where(on, np.argmax(near, axis=1) + (side == "above"), k)
        k = np.where((r > R[-1] + tol) | (r < lo - tol), -1, np.minimum(k, len(R) - 1))
        return np.array([P.names.index(P.order[i]) if i >= 0 else -1 for i in k], int)
    H = np.array([members(P[n], X) for n in P.names])                 # (pieces, points): in the closed piece
    count = H.sum(axis=0)
    owner = np.where(count > 0, np.argmax(H, axis=0), -1)
    for i in np.flatnonzero(count > 1):                                # on a seam: two closed pieces
        hit = [P.names[k] for k in np.flatnonzero(H[:, i])]
        seam = P.seam(*hit[:2])
        if side not in ("below", "above"):
            raise ValueError(f"a point is on the seam {seam.name!r}, where a field on pieces may have two values: "
                             "say side='below' or side='above'")
        owner[i] = P.names.index(getattr(seam, side))
    return owner


class PiecewiseSpace(DirectSum):
    """Fields on the leaves of a partition, glued or broken (module docstring).

    fields: one space on the partition's domain (or a constructor taking the domain: pg.L2, lambda D: pg.Sobolev(D, 2,
    length_scale=0.1)), its summands keyed by piece; or {field: such a space}, keyed "<field>_<piece>". where: the pieces
    of a field that lives on some only ({field: names}; for one field a list). weights: by key or by field. broken:
    False (Sobolev fields glued, the default), True (all broken) or the fields to break. L2 fields are the same either
    way."""

    def __init__(self, partition, fields, where=None, weights=None, realisation=None, broken=False):
        P = _partition(partition)
        single = not isinstance(fields, dict)
        if single:
            fields = {VALUE: fields.spaces if isinstance(fields, _ByPieces) else fields}
            where = {VALUE: where} if where is not None else {}
            broken = {VALUE} if broken else set()
        where = where or {}
        unknown = set(where) - set(fields)
        if unknown:
            raise KeyError(f"where= names fields that are not given: {sorted(unknown)}")
        broken = set(fields) if broken is True else set(broken or ())
        if broken - set(fields):
            raise KeyError(f"broken= names fields that are not given: {sorted(broken - set(fields))}")
        self.partition, self.fields, self._R, self.single = P, {}, {}, single
        self.glued_fields, keys, spaces = set(), {}, {}
        for f, X in fields.items():
            if isinstance(X, dict):                          # a field given piece by piece (from_pieces): broken
                self._given_by_pieces(P, f, X, keys, spaces)
                continue
            if isinstance(X, type) or not isinstance(X, VectorSpace):
                X = X(P.domain)                              # a constructor: the space on the whole domain
            if X.domain != P.domain:
                raise ValueError(f"the field {f!r} lives on {X.domain!r}, not on {P.domain!r}")
            self.fields[f] = X
            names = list(P.names if f not in where else where[f])
            bad = [n for n in names if n not in P.names]
            if bad:
                raise KeyError(f"{bad} are not pieces of {P}")
            if gluing.is_sobolev(X) and f not in broken:     # glued: the field's space itself
                if names != list(P.names):
                    raise Refused(f"the glued field {f!r} on some pieces only: H^s of their union is another domain",
                                  needs=(f"a space on the union for {f!r}", f"broken={{{f!r}}}: independent pieces"))
                self.glued_fields.add(f)
                for n in names:
                    keys[(f, n)] = f
                spaces[f] = X
                continue
            gluing.refuse_product(X, "a PiecewiseSpace with the field")
            for n in names:
                R = X.restrict_to(P[n], realisation, label=n)
                key = n if single else f"{f}_{n}"
                keys[(f, n)], self._R[(f, n)], spaces[key] = key, R, R.codomain
        if isinstance(weights, dict):                        # a field's weight goes to all its pieces
            w = {k: v for k, v in weights.items() if k not in self.fields}
            for (f, n), key in keys.items():
                if f in weights and key not in w:
                    w[key] = weights[f]
            weights = w
        super().__init__(spaces, weights)
        self.keys = keys

    def _given_by_pieces(self, P, f, spaces_by_piece, keys, spaces):
        unknown = [n for n in spaces_by_piece if n not in P.names]
        if unknown:
            raise KeyError(f"{unknown} are not pieces of {P}")
        self.fields[f] = None                                # no space on the whole domain: the field is broken
        for n in sorted(spaces_by_piece, key=P.names.index):
            Xn = spaces_by_piece[n]
            if Xn.domain != P[n]:
                raise ValueError(f"the space for {n!r} lives on {Xn.domain!r}, not on {P[n]!r}")
            gluing.refuse_product(Xn, "a PiecewiseSpace with the field")
            key = n if self.single else f"{f}_{n}"
            keys[(f, n)], spaces[key] = key, Xn

    # ---------------------------------------------------------------- constructors on each piece
    @classmethod
    def from_pieces(cls, partition, spaces, weights=None):
        """One field given piece by piece, {piece: a space on that piece} (pieces left out carry none): broken, keyed
        by piece."""
        return cls(partition, _ByPieces(spaces), weights=weights)

    @classmethod
    def lebesgue(cls, partition, pieces=None, realisation=None, weights=None):
        """L2 of each of the given pieces (default: all): L2 of the domain, as a direct sum, when all are given.
        realisation: as for pg.L2 on each piece ("radial_harmonics": radial x harmonic factors)."""
        from .base import L2
        P = _partition(partition)
        return cls.from_pieces(P, {n: L2(P[n], realisation=realisation) for n in (P.names if pieces is None
                                                                                    else pieces)}, weights)

    @classmethod
    def sobolev(cls, partition, s, length_scale, pieces=None, weights=None, **kw):
        """H^s_l on each of the given pieces (default: all), broken; s and length_scale one value, a list (one per
        piece) or a dict by piece."""
        from .sobolev import Sobolev
        P = _partition(partition)
        names = list(P.names if pieces is None else pieces)
        pick = lambda v, i, n: v[n] if isinstance(v, dict) else v[i] if isinstance(v, (list, tuple)) else v  # noqa
        return cls.from_pieces(P, {n: Sobolev(P[n], pick(s, i, n), length_scale=pick(length_scale, i, n), **kw)
                                   for i, n in enumerate(names)}, weights)

    # ---------------------------------------------------------------- structure
    @property
    def broken_fields(self):
        return set(self.fields) - self.glued_fields

    def _field(self, field):
        if field is None:
            if not self.single:
                raise TypeError(f"{self} has the fields {', '.join(self.fields)}: name one")
            return VALUE
        if field not in self.fields:
            raise KeyError(f"{field!r} is not a field of {self} (fields: {', '.join(self.fields)})")
        return field

    def key(self, field, piece):
        try:
            return self.keys[(field, piece)]
        except KeyError:
            raise KeyError(f"{field!r} has no summand on {piece!r} in {self}") from None

    def pieces_of(self, field=None):
        """The pieces carrying field, in the partition's order."""
        field = self._field(field)
        return [n for n in self.partition.names if (field, n) in self.keys]

    def restriction(self, field, piece):
        """R: X_field(Omega) -> the summand (field, piece); for a glued field R: X_field(Omega) -> X_field(piece), the
        view of that piece. Refused for a field given piece by piece (it has no space on the whole domain)."""
        if (field, piece) not in self._R:
            self.key(field, piece)
            if self.fields[field] is None:
                raise Refused(f"the field {field!r} of {self} is given piece by piece: there is no space on the whole "
                              "domain to restrict from", needs=("pg.PiecewiseSpace(P, X) with X on the whole domain",))
            self._R[(field, piece)] = self.fields[field].restrict_to(self.partition[piece], label=piece)
        return self._R[(field, piece)]

    def piece_element(self, x, field, piece):
        """x's field on one piece: the summand's part (broken) or the view R_k of the glued element."""
        if field in self.glued_fields:
            return self.restriction(field, piece)(self.part(x, self.key(field, piece)))
        return self.part(x, self.key(field, piece))

    def glued(self):
        """The glued space of the same fields, partition and weights (module docstring): itself when no field is
        broken; fields given piece by piece have no glued form (refused)."""
        given = [f for f, X in self.fields.items() if X is None]
        if given:
            raise Refused(f"the fields {given} of {self} are given piece by piece: their glued space needs a space on "
                          "the whole domain", needs=("pg.PiecewiseSpace(P, X) with X on the whole domain",))
        if not any(gluing.is_sobolev(self.fields[f]) for f in self.broken_fields):
            return self                                      # L2 fields: glued is broken
        if getattr(self, "_glued", None) is None:
            where = {f: self.pieces_of(f) for f in self.fields}
            fields = self.fields[VALUE] if self.single else dict(self.fields)
            self._glued = PiecewiseSpace(self.partition, fields, where=where[VALUE] if self.single else
                                         {f: w for f, w in where.items() if w != list(self.partition.names)},
                                         weights=self._field_weights())
        return self._glued

    def _field_weights(self):
        w = {}
        for (f, n), key in self.keys.items():
            w.setdefault(f, set()).add(self.weights[self.index(key)])
        if all(len(v) == 1 and v == {1.0} for v in w.values()):
            return None
        if any(len(v) > 1 for f, v in w.items() if f in self.broken_fields):
            raise Refused(f"{self} weighs the pieces of a field differently; its glued form has one weight per field",
                          needs=("one weight per field",))
        return {f: v.pop() for f, v in w.items()}

    # ---------------------------------------------------------------- functionals
    def parts(self, field, measure, side=None):
        """{key: measure restricted to the piece} over the pieces of field that measure charges (a dict for M.riesz);
        a measure charging a piece without the field is refused (the functional would depend on a field this space does
        not have). A glued Sobolev field takes the whole measure ({field: measure}, = the sum of its parts), side=
        optional; the result's .claims say how it split."""
        field = self._field(field)
        if field in self.glued_fields:
            return gluing.glued_parts(self, field, measure, side)
        split = self.partition.split(measure, side)
        missing = [n for n in split if (field, n) not in self.keys]
        if missing:
            raise SupportOutsideDomain(f"{measure} charges the pieces {missing}, which have no summand of {field!r} "
                                       f"in {self}: the functional would depend on a field this space does not have (leave those "
                                       "pieces out of the measure if the field there is known)")
        return gluing.Parts({self.keys[(field, n)]: m for n, m in split.items()}, split.claims)

    def riesz(self, spec, side=None):
        """The representer of x -> sum_k int x_k d(mu restricted to piece k) for a measure mu (one field; see the
        module docstring); a dict {summand: measure} gives the parts directly; a list gives a family."""
        if isinstance(spec, dict):
            return super().riesz(spec)
        if isinstance(spec, (list, tuple)) or (isinstance(spec, np.ndarray) and spec.ndim == 2):
            return super().riesz([m if isinstance(m, dict) else self._parts_one(m, side) for m in spec])
        return super().riesz(self._parts_one(spec, side))

    def _parts_one(self, mu, side):
        from ..measures import DiracMeasure, LebesgueMeasure
        from ..sets import Region
        if isinstance(mu, Region):
            mu = LebesgueMeasure(mu)
        elif not hasattr(mu, "restricted") and np.ndim(mu) <= 1:
            mu = DiracMeasure(np.asarray(mu, float))
        if not self.single:
            raise TypeError(f"{self} has the fields {', '.join(self.fields)}: M.parts(field, mu) or "
                            "M.functional(field, mu) says which one mu acts on")
        return dict(self.parts(VALUE, mu, side))

    def functional(self, *args, side=None):
        """x -> sum_k int x_(field, k) d(measure|Omega_k), zero on the other fields: a pg.Functional. Arguments
        (field, measure), or (measure) for one field. A list (or tuple) of measures gives the bounded pg.Functionals
        into R^n, row i the functional of measure i: pg.Functionals(M.riesz([M.parts(field, m) for m in measures]))."""
        from .dual import Functional
        field, measure = (None, args[0]) if len(args) == 1 else args
        field = self._field(field)
        if isinstance(measure, (list, tuple)):
            return self.functionals_of(self.riesz([self.parts(field, m, side) for m in measure]))
        parts = self.parts(field, measure, side)
        out = Functional.from_representer(self.riesz(dict(parts)), name=f"∫ {field} d({measure})")
        out._claims = list(getattr(parts, "claims", Claims(()))) + out._claims
        return out

    def piece_functionals(self, field, measure, side=None):
        """{piece: l_k}, l_k(x) = int x_field|Omega_k d(measure|Omega_k), Functionals on M with sum_k l_k the
        functional (for a glued field they pair in H^s(Omega), cross terms kept)."""
        return gluing.piece_functionals(self, self._field(field), measure, side)

    # ---------------------------------------------------------------- elements
    def split(self, fields, name="x", resolution=None):
        """The element of M with the parts R_k u_field: fields {field: element of the field's space on the whole domain,
        or a formula (evaluated on each piece)}; missing fields are zero. resolution=w: the formulas' narrowest feature
        width (core/fits.py); a formula's pieces share one record (one ResolutionWarning for all of them)."""
        parts = {}
        with fits.gathered(name):
            for f, u in fields.items():
                f = self._field(f)
                if f in self.glued_fields:
                    X = self.fields[f]
                    parts[f] = u if isinstance(u, (Function, Family)) else X.element(u, resolution=resolution)
                    continue
                first = None
                for n in self.pieces_of(f):
                    key = self.keys[(f, n)]
                    if isinstance(u, (Function, Family)):
                        parts[key] = self.restriction(f, n)(u)
                        continue
                    e = self.summand(key).element(u, resolution=resolution)
                    parts[key] = e if first is None else fits.piece(e, first)
                    first = e if first is None else first
            return self(parts, name=name)

    def glue(self, x, field=None):
        """{field: G x_field}, each an element of the field's space on the whole domain (pieces without the field: 0);
        field= (or one field) gives the element itself. A broken H^s field glues only when its pieces satisfy the
        interface conditions (spaces/gluing.py: the traces checked for 1/2 < s < 3/2, refused otherwise)."""
        pl = self._pl_of(x)
        out = {}
        names = [self._field(field)] if field is not None or self.single else list(self.fields)
        for f in names:
            if f in self.glued_fields:
                out[f] = self.part(x, f)
                continue
            X = self.fields[f]
            if X is None:
                raise Refused(f"the field {f!r} of {self} is given piece by piece: there is no space on the whole "
                              "domain to glue into", needs=("pg.PiecewiseSpace(P, X) with X on the whole domain",))
            if gluing.is_sobolev(X):
                gluing.require_matching_traces(self, x, f, X.s)
            parts = []
            for n in self.pieces_of(f):
                key = self.keys[(f, n)]
                parts.append((self.partition[n], None if pl.parts[self.index(key)] is None else self.part(x, key)))
            out[f] = gluing.extend(X, parts, name=f"{x.name}[{f}]")
        return out[names[0]] if len(names) == 1 and (field is not None or self.single) else out

    def seam_jumps(self, x, field=None):
        """{seam: max |x_below - x_above|} of x's field at points of each seam both pieces carry (spaces/gluing.py),
        with .match a Truth (the precision rule) and .claims: zero for a glued field to the rounding of its views,
        anything for a broken one."""
        return gluing.trace_jumps(self, x, self._field(field))

    def evaluate(self, x, points, side=None, field=None):
        """Values of the field x at points ((m, d) or (d,)) anywhere in the domain: each point in its piece (NaN on
        pieces without the field and outside the domain); on a seam side="below" or "above" must say which value is
        meant (P.seams: below is nearer the centre, the lower coordinate)."""
        if not isinstance(x, Function) or x.space != self:
            raise TypeError(f"expected an element of {self}")
        require_realised(x.space, self, x.name)
        field = self._field(field)
        P = self.partition
        pts = np.asarray(points, float)
        d = getattr(P.domain, "cartesian_dim", P.domain.dim)
        flat = pts.reshape(-1, d)
        cols = tuple(flat.T)
        owner = locate(P, cols, side)
        out = np.full(len(flat), np.nan)
        for n in self.pieces_of(field):
            sel = owner == P.names.index(n)
            if np.any(sel):
                v = self.piece_element(x, field, n)
                out[sel] = np.asarray(v(*[c[sel] for c in cols]), float).reshape(-1)
        if d == 1:
            return float(out[0]) if pts.ndim == 0 else out.reshape(pts.shape)
        return float(out[0]) if pts.ndim == 1 else out.reshape(pts.shape[:-1])

    # ---------------------------------------------------------------- claims and identity
    @property
    def claims(self):
        base = list(super().claims)
        out = [Claim(f"each field's summands are its spaces on the {len(self.partition)} pieces (a glued field: its "
                     "space itself)", THEOREM, "the pieces tile the domain up to a null set (the partition's check)",
                     key="on pieces")]
        if any(not gluing.is_sobolev(self.summand(k)) for k in self.keys.values()):
            out.append(Claim("L2 fields: glued is broken, L2(Omega) = (+)_k L2(Omega_k) isometrically, u -> "
                             "(u|Omega_k)_k unitary (unit weights)", THEOREM, "L2 has no trace and the seams are "
                             "null sets, so integrals of squares add over the pieces", key="glued is broken"))
        if self.broken_fields:
            out.append(Claim("the norm of the broken fields is the sum of their pieces' norms: ||x||^2 = sum_k w_k "
                             "||x_k||^2_(X(Omega_k))", THEOREM, "the definition of the Hilbert direct sum",
                             key="broken norm"))
        for f, X in self.fields.items():
            if f in self.glued_fields:
                out += gluing.glued_claims(f, X)
                continue
            Y = X if X is not None else next(self.summand(k) for (g, _), k in self.keys.items() if g == f)
            if gluing.is_sobolev(Y):
                out += gluing.broken_claims(f, Y)
        return Claims(base + out)

    def __eq__(self, other):
        return other is self or (isinstance(other, PiecewiseSpace) and other.partition == self.partition
                                 and other.glued_fields == self.glued_fields and DirectSum.__eq__(self, other))

    def __hash__(self):
        return hash(("PiecewiseSpace", self.partition, self.names, self.weights))

    def __repr__(self):
        how = "glued" if self.glued_fields and not self.broken_fields else "broken" if not self.glued_fields else \
            "glued and broken"
        what = ", ".join(self.fields) if not self.single else repr(self.fields[VALUE] or "fields on pieces")
        return f"PiecewiseSpace({what} on {', '.join(self.partition.names)}: {len(self)} summands, {how})"
