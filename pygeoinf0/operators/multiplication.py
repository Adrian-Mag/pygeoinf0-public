"""Multiplication operators f -> a f on L2 spaces (plan 009 G6; EXP-030 DESIGN G6, NOTES.md section 6).

    M = pg.Multiplication(space, a)             a: an element of a function space on the same domain (a Function),
                                                a pg.Tabulated function (of x on an interval, of the radius r on a
                                                ball or shell), or a callable with a declared bound
    M = pg.Multiplication(space, f, bound=2.0, radial=True)     callables: |f| <= bound is the user's statement

Mathematics (standard; Conway, A Course in Functional Analysis, II.1.5 and IX.2): for a in L^inf(mu), M_a f = a f is
bounded on L2(mu) with ||M_a|| = ess sup |a|, and self-adjoint for real a; it is positive semidefinite iff a >= 0 a.e.
On a ball, a shell or a sphere M_a commutes with every rotation iff a o g = a a.e. for every rotation g, i.e. a is
radial (a constant on a sphere). A radial multiplication maps each degree-l harmonic subspace into itself and acts there
by a radial operator, the same for every order (Schur; EXP-030 Theorem G2.1).

What is claimed about the number ||M_a||: for a Tabulated function the sup of its table (exact: the reading is piecewise
linear); for a Function the sup bound of its realisation (the sum of the absolute Legendre coefficients per piece, times
the largest value of a normalised harmonic): a bound on the operator that is applied (the multiplication by the realised
a), derived; for a callable the declared bound, asserted. The operator applied is the multiplication by the realised a;
products are exact on the realisation when a is radial (degree 0) on a radial x harmonic space or on an interval, and
refitted adaptively otherwise (realisation error).
"""
import math

import numpy as np

from ..core.claims import ASSERTED, DERIVED, ORDER, THEOREM, Claim
from ..spaces.elements import Family, Function
from ..core.errors import Unsupported
from ..realisations.series import harmonic_degrees as H
from .linear import LinearOperator
from ..spaces.identity import identical, require_realised

CONWAY = "Conway, A Course in Functional Analysis, II.1.5 (multiplication operators on L2)"


def _is_l2(space):
    from ..spaces.base import L2
    return isinstance(space, L2)


def _sup_of_realisation(space, R):
    """A bound on sup |a| over the domain for the realised a (one column), from its coefficients."""
    k = H.kind(space.backend)
    if k == "radial":
        from ..realisations.series.radial_harmonic import dense
        R = dense(R)
        l = np.floor(np.sqrt(np.arange(R.nlm))).astype(int)
        ymax = np.sqrt((2 * l + 1) / (4 * math.pi))                   # |Y_lm| <= sqrt((2l + 1) / 4 pi)
        return float(max((np.abs(C[:, :, 0]).sum(axis=0) * ymax).sum() for C in R.coefs))
    if k == "sphere":
        from ..spaces.harmonic import rows
        l = rows(R.L)[0]
        return float((np.abs(R.C[:, 0]) * np.sqrt((2 * l + 1) / (4 * math.pi))).sum() / R.R)
    if hasattr(R, "sup_bound"):
        return float(np.atleast_1d(R.sup_bound())[0])
    return None


def _is_radial(space, R):
    """Whether the realised a has degree 0 only (radial; a constant on a sphere), or None if not a harmonic space."""
    k = H.kind(space.backend)
    if k is None:
        return None
    if k == "radial":
        from ..realisations.series.radial_harmonic import dense
        R = dense(R)
        return R.L == 0 or all(float(np.abs(C[:, 1:, :]).max(initial=0.0)) == 0.0 for C in R.coefs)
    return R.L == 0 or float(np.abs(R.C[1:]).max(initial=0.0)) == 0.0


class Multiplication(LinearOperator):
    """f -> a f on an L2 space (module docstring)."""

    why_linear = "pointwise multiplication by a fixed function"

    def __init__(self, space, a, bound=None, radial=None, name=None):
        if not _is_l2(space):
            raise Unsupported(f"pg.Multiplication acts on L2 spaces; on {space} the multiplication by a is bounded only "
                              "for a in a multiplier space, and its adjoint is not the multiplication by a (it involves "
                              "the space's own inner product): not offered")
        nm = name or f"M_{getattr(a, 'name', 'a')}"
        super().__init__(space, space, name=nm)
        self._claims = []
        self.a_source = a
        k = H.kind(space.backend)
        if hasattr(a, "nodes") and hasattr(a, "rule"):                  # a pg.Tabulated
            self._from_table(space, a, k)
        elif isinstance(a, Function):
            self._from_function(space, a, k)
        elif callable(a):
            if bound is None:
                raise ValueError("a callable multiplier needs a declared bound: pg.Multiplication(space, f, bound=B) "
                                 "with |f| <= B on the domain (it cannot be verified from finitely many values)")
            fa = space.element(a, name=getattr(a, "__name__", "a"))
            self._R, self.bound = fa.realisation, float(bound)
            self._bound_claim = Claim(f"bounded, norm <= {self.bound:.6g}", ASSERTED,
                                      "the declared bound |a| <= B (ess sup |a| = ||M_a||)", key="bounded")
            self._radial = (bool(radial), ASSERTED, "declared") if radial is not None and k is not None else None
            self._nonneg = None
        else:
            raise TypeError("the multiplier is a Function, a pg.Tabulated or a callable with bound=")
        if bound is not None and float(bound) < self.bound:             # a tighter declared bound (asserted)
            self.bound = float(bound)
            self._bound_claim = Claim(f"bounded, norm <= {self.bound:.6g}", ASSERTED, "the declared bound",
                                      key="bounded")
        self._sa = (True, THEOREM, f"a real multiplier on L2: <a f, g> = <f, a g> ({CONWAY})")
        self._claims += [Claim("linear", THEOREM, "pointwise multiplication", key="linear"), self._bound_claim,
                         Claim("||M_a|| = ess sup |a|", THEOREM, CONWAY)]
        if self._nonneg is not None and self._nonneg[0]:
            self._pos = ("psd", self._nonneg[1], self._nonneg[2])
        if self._radial is not None:
            holds, st, why = self._radial
            self._claims.append(Claim("commutes with rotations" if holds else "does not commute with rotations", st,
                                      ("a is radial: " if holds else "a is not radial: ") + why + "; M_a commutes with "
                                      "every rotation iff a is rotation invariant (EXP-030 NOTES.md section 6)",
                                      key="rotation invariant", holds=bool(holds)))

    # the multiplier ---------------------------------------------------------------------------
    def _from_table(self, space, t, k):
        vals = np.concatenate([t._v0, t._v1])
        self.bound = float(np.abs(vals).max())
        self._bound_claim = Claim(f"bounded, norm = {self.bound:.6g}", DERIVED, "the sup of a piecewise-linear reading "
                                  "is the largest absolute value at its breakpoints", key="bounded")
        self._nonneg = (True, DERIVED, "the table's reading is >= 0") if vals.min() >= 0 else None
        if k == "radial":
            from ..realisations.series.radial_harmonic import RadialHarmonicSeries, SQ4PI
            b = space.backend
            if abs(t.domain.a - b.a) > 1e-12 * b.b or abs(t.domain.b - b.b) > 1e-12 * b.b:
                raise ValueError(f"the table is on {t.domain}, the radial interval is [{b.a:g}, {b.b:g}]")
            pl = t.pl
            self._R = RadialHarmonicSeries(0, pl.breaks, [SQ4PI * C[:, None, :] for C in pl.coefs], np.zeros(1))
            self._radial = (True, DERIVED, "a tabulated function of the radius")
        elif space.domain_dim == 1:
            self._R = t.element(space).realisation
            self._radial = None
        else:
            raise Unsupported(f"a table is a function of one coordinate: of x on an interval, of r on a ball or shell "
                              f"realised in radial x harmonic factors; {space} is realised {space.realisation}")

    def _from_function(self, space, f, k):
        if not identical(f.space, space):           # another space, or the same realised otherwise: refitted here
            if f.space.domain != space.domain:
                raise ValueError(f"{f.name} is a function on {f.space.domain}, not on {space.domain}")
            f = space.element(lambda *X: f(*X), breaks=f.space.backend.breaks(f.realisation), name=f.name)
        self._R = f.realisation
        b = _sup_of_realisation(space, self._R)
        if b is None:
            raise Unsupported(f"no sup bound for elements realised {space.realisation}: give pg.Multiplication(space, "
                              "f, bound=B) with a callable f")
        self.bound = b
        self._bound_claim = Claim(f"bounded, norm <= {b:.6g}", DERIVED, "the sup bound of the realised multiplier "
                                  "(absolute Legendre coefficients per piece; |Y_lm| <= sqrt((2l + 1)/4 pi))",
                                  key="bounded")
        r = _is_radial(space, self._R)
        self._radial = None if r is None else (r, DERIVED, "its realisation has degree 0 only" if r else
                                               "its realisation has harmonics of degree >= 1")
        self._nonneg = None

    def multiplier(self):
        """a as an element of the space (its realisation)."""
        return Function(self.domain, pl=self._R, name=getattr(self.a_source, "name", "a"))

    def radial_factor(self):
        """a(r) as a one-column legendre.PiecewiseLegendre when the multiplier is radial on a radial x harmonic space, else None."""
        if self._radial is None or not self._radial[0] or H.kind(self.domain.backend) != "radial":
            return None
        from ..realisations.series import legendre as LG
        from ..realisations.series.radial_harmonic import SQ4PI, dense
        R = dense(self._R)
        return LG.PiecewiseLegendre(R.breaks, [C[:, 0, :] / SQ4PI for C in R.coefs], R.err)

    def norm_bound(self):
        """An upper bound on ||M_a|| (see the claims for its status)."""
        return self.bound

    # action ------------------------------------------------------------------------------------
    def _apply(self, x):
        if not isinstance(x, (Function, Family)) or x.space != self.domain:
            raise TypeError(f"{self.name} acts on elements of {self.domain}")
        require_realised(x.space, self.domain, x.name)
        R = self.domain.backend.multiply(self._R, x.realisation)
        if isinstance(x, Function):
            return Function(self.domain, pl=R, name=f"{self.name}{x.name}")
        return Family(self.domain, R, name=x.name)

    def _make_adjoint(self):
        return self

    def _compose(self, other):
        if isinstance(other, Multiplication) and identical(other.domain, self.domain):
            out = Multiplication.__new__(Multiplication)
            LinearOperator.__init__(out, self.domain, self.domain, name=f"{self.name}{other.name}")
            out.a_source, out._R = None, self.domain.backend.multiply(self._R, other._R)
            out.bound = self.bound * other.bound
            st = max((self._bound_claim.status, other._bound_claim.status), key=ORDER.index)
            out._bound_claim = Claim(f"bounded, norm <= {out.bound:.6g}", st, "||M_a M_b|| <= ||M_a|| ||M_b||",
                                     key="bounded")
            out._claims = [Claim("linear", THEOREM, "pointwise multiplication", key="linear"), out._bound_claim]
            out._sa = (True, THEOREM, "M_a M_b = M_(ab), a real multiplier")
            ra, rb = self._radial, other._radial
            out._radial, out._nonneg = None, None
            if ra is not None and rb is not None and ra[0] and rb[0]:
                out._radial = (True, ra[1], "a product of radial multipliers")
                out._claims.append(Claim("commutes with rotations", max((ra[1], rb[1]), key=ORDER.index),
                                         "a product of radial multipliers", key="rotation invariant"))
            return out
        return super()._compose(other)

    def __repr__(self):
        return f"{self.name}: multiplication on {self.domain} (norm <= {self.bound:.4g})"
