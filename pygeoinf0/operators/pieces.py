"""Elements carried to the pieces of a domain: restriction, extension by zero and gluing (plan 035 P3, design D3 and
D5's L2, weighted L2 and L^p rows); the spaces of fields on the pieces are pg.PiecewiseSpace (spaces/piecewise.py,
plan 069 P7), the extension and the glue of parts spaces/gluing.py.

    R = pg.Restrict(X, D)        (= X.restrict_to(D))  R: X(Omega) -> X(D), u -> u|_D; R.codomain is the new space X(D)
    E = R.extension              extension by zero X(D) -> X(Omega); on L2 and L2(rho dx) it is R.adjoint
    R = X.restrictions(P)        (R_k)_k: X(Omega) -> (+)_k X(Omega_k) over the leaves of a Partition P, into the broken
                                 pg.PiecewiseSpace(P, X, broken=True) (for L2 also the glued one: the seams are null)
    G = P.glue(X)                its inverse; on L2 also its adjoint (G is unitary)

X(D) is a new space (X is never changed): L2(mu|_D) with mu|_D the restricted measure (measures/restrict.py; on L2(rho dx)
the weight restricted), L^p(mu|_D) of the same p, with X's tolerance and realisation. R goes through evaluation
semantics: a symbolic representer d nu / d lambda restricts to that of nu|_D (exact), a formula (a bank function too) is
evaluated on D (exact: the formula is the element), a combination termwise, and a realised element by cutting its
realisation (Backend.restricted, exact: piecewise Legendre on intervals, radial x harmonic on concentric balls and shells,
tensor Legendre on smaller reference boxes of the same chart with Fourier axes kept whole; the source's error estimate is
carried, and bounds the result's since R does not raise a norm). Where no cut exists (spherical harmonics to a band, a piece
that is no coordinate box, a piece space with another realisation) it is evaluated on D and fitted there, and the fit's own
error is the element's. The claim "realisation" says so.

Extension by zero of realised parts joins their coefficients where the realisation allows it (Backend.glued, the mirror of the
cut, with no fit): the result is the piecewise function, so a jump between neighbouring pieces stays a jump (seams are null
sets in L2), and what no part covers is zero. Otherwise it is a piecewise formula with D's boundary as breaks when it is a coordinate surface of Omega (ends of
intervals, faces of boxes, concentric spheres: domains/partition.breaks_for), so its fit is exact for piecewise
polynomials; symbolic parts stay symbolic (the representer of the measure, now on Omega); otherwise on L2 the part is
kept as the representer of the measure v rho dx on D (paired exactly, realised only where that is exact). The claims
(D5): on L2 and L2(rho dx) R is a co-isometry with adjoint extension by zero, and over a partition sum_k ||R_k u||^2 =
||u||^2 and G is unitary (theorems: the seams are null sets). On L^p ||u||_p^p = sum_k ||R_k u||_p^p and the dual map
is extension by zero in L^(p'), stated as claims; the package has no direct sum of Banach spaces (with the l^p norm) nor
dual-space objects, so X.restrictions(P) on L^p and R.adjoint on L^p are refused by name. Sobolev spaces (plan 035 P4):
operators/pieces_sobolev.py (R* the minimal-norm extension; PiecewiseSpace glued by default, broken=True by name); C(K)
(plan 035 P5): operators/pieces_continuous.py. On a sphere and its bands (domains/bands.py) L2 is restricted as here,
the band realised in the theta x phi tensor basis; the glue into spherical harmonics, which take no breaks, keeps
non-symbolic parts as representers of the measures v dA on the bands (paired exactly; a fit would ring at the seams).

From pieces_realised.py (merged here by plan 063):
Restriction of elements to a piece, the operator's _apply (plan 035 P3, plan 043); the module docstring of pieces.py says what
holds. A realised element is cut by its backend (Backend.restricted) where its realisation allows it, and otherwise fitted.
"""


import functools

import numpy as np

from ..core import fits
from ..core.claims import CHECKED, THEOREM, Claim
from ..spaces.direct_sum import DirectSum
from ..spaces.elements import Family, Function
from ..core.errors import Refused, SpaceMismatch, SupportOutsideDomain, Unsupported
from ..measures import LebesgueMeasure, SignedMeasure
from ..spaces.gluing import extend, is_sobolev, refuse_product
from ..spaces.representers import DensityRepresenter, DensityRepresenters
from . import pieces_continuous as continuous
from .linear import LinearOperator
from .pieces_sobolev import SobolevExtension, no_extension_by_zero, sobolev_claims
from .pieces_sobolev import piece_space as sobolev_piece_space
from .pieces_sobolev import restrict_element as restrict_sobolev


# ---------------------------------------------------------------- from pieces_realised.py (plan 063)
def _carried_breaks(XO, XD, u):
    """The breaks of u (on Omega) for a fit on D (domains/partition.carry_breaks)."""
    from ..domains.partition import carry_breaks
    br = u._breaks if u._src is not None else XO.backend.breaks(u._pl)
    if isinstance(br, np.ndarray) and br.ndim == 1 and XO.domain.dim > 1:      # radial x harmonic: the radial axis's
        br = (br,) + (None,) * (XO.domain.dim - 1)
    return carry_breaks(XO.domain, XD.domain, br)


def _cut(XO, XD, R):
    """R cut to XD's domain by XO's backend, or None: only into the same realisation, so a piece space that asks for another
    one (Restrict's realisation=) is fitted as it asked."""
    A, B = XO.backend, XD.backend
    if type(A) is not type(B) or not A.same_realisation(B):
        return None
    return A.restricted(R, B)


def restrict_element(XO, XD, u, label):
    """R u (module docstring): u an element or family of XO, the result in XD."""
    D = XD.domain
    name = f"{u.name}|{label}"
    if isinstance(u, DensityRepresenters):
        ms = [m.restricted(D) for m in u.measures]
        return XD.riesz([m if m is not None else SignedMeasure([(0.0, LebesgueMeasure(D))]) for m in ms])
    if isinstance(u, Family):
        cut = _cut(XO, XD, u.realisation)
        if cut is not None:
            return Family(XD, cut, name=name)
        return XD.family([restrict_element(XO, XD, u[i], label) for i in range(len(u))], name=name)
    if isinstance(u, DensityRepresenter):
        m = u.measure.restricted(D)
        if m is None:
            return XD.zero
        out = XD.riesz(m)
        out.name = name
        return out
    if u._terms is not None and u._pl is None:                    # a combination: termwise (R is linear)
        out = functools.reduce(lambda a, b: a + b, [restrict_element(XO, XD, t, label) * c for c, t in u._terms])
        out.name = name
        return out
    if u._exact is not None:                                      # a constant, a coordinate: still a formula
        return Function(XD, source=u._exact, exact=u._exact, name=name)
    if u._src is None:                                            # realised: cut exactly where the realisation allows
        cut = _cut(XO, XD, u.realisation)
        if cut is not None:
            return Function(XD, pl=cut, name=name)
    br = _carried_breaks(XO, XD, u)
    if u._src is not None:                                        # a formula (a bank function: its breaks on D)
        return fits.piece(XD.element(u._src, name=name, breaks=br), u)                  # u's record (core/fits.py)
    return fits.refit(XD.element(lambda *X: np.asarray(u(*X), float), name=name, breaks=br), u)   # realised: fitted on D


def _realisation_claim():
    return Claim("R u of a formula is the formula on D; of a realised element, its realisation cut exactly on D where it can "
                 "be, otherwise evaluated on D and fitted there", CHECKED, "symbolic representers restrict exactly (nu -> "
                 "nu|_D); a formula is the element; a cut keeps the coefficients (the source's error estimate is carried); a "
                 "refit, with the breaks it shares, has its own error, which is the result's", key="realisation")


def _check(X):
    """The kind of X: "sobolev" (pieces_sobolev.py), "lp" or "l2" (L2 and L2(rho dx)) on an interval, box, ball or
    shell; anything else is refused by name."""
    from ..spaces.continuous import ContinuousSpace
    from ..spaces.base import L2, LpSpace
    if is_sobolev(X):
        return "sobolev"
    if isinstance(X, ContinuousSpace):
        return "continuous"                                       # operators/pieces_continuous.py
    if isinstance(X, DirectSum):
        raise Refused(f"restricting the direct sum {X}: restrict each summand", needs=("pg.Restrict(summand, D)",
                                                                                       "pg.PiecewiseSpace(P, fields)"))
    if not isinstance(X, (L2, LpSpace)) or type(X.domain).__name__ not in ("Interval", "Box", "Ball", "Shell", "Sphere",
                                                                          "SphereBand"):
        raise Unsupported(f"restricting {X} to a piece (L2, L2(rho dx) and L^p on intervals, boxes, balls, shells, spheres "
                          "and their bands are restricted)")
    return "lp" if isinstance(X, LpSpace) else "l2"


def piece_space(X, D, realisation=None):
    """X(D): the space of X on the piece D (module docstring); D must lie in X's domain."""
    from ..sets import Interval
    from ..spaces.base import L2, LpSpace
    from ..domains.pieces import meet
    kind = _check(X)
    if kind in ("sobolev", "continuous"):
        return sobolev_piece_space(X, D, realisation) if kind == "sobolev" else continuous.piece_space(X, D)
    lp = kind == "lp"
    D = Interval(D) if isinstance(D, str) else D
    J = meet(X.domain, D)
    if J is None or not (J is D or J == D):
        raise SupportOutsideDomain(f"{D!r} is not contained in {X.domain!r}")
    mu = X.measure.restricted(D)
    real = X.realisation if realisation is None else realisation
    real = None if real == "harmonics" else real                  # a band: the theta x phi tensor basis
    if realisation is None and hasattr(X.backend, "cells"):
        found = X.backend.cells.on(D)
        if found is None:
            raise Refused(f"{D!r} is not a union of the cells of {X}", needs=("a piece tiled by whole cells", "realisation= for the piece"))
        real = found[0]
    out = LpSpace(mu, X.p, X.tol, real) if lp else L2(mu, X.tol, real)
    out._measure_given = X._measure_given
    return out


def _l2_claims(X, D, what):
    return [Claim("linear", THEOREM, what, key="linear"),
            Claim("bounded, norm <= 1", THEOREM, "int_D |u|^p dmu <= int_Omega |u|^p dmu", key="bounded")]


class Restrict(LinearOperator):
    """R: X(Omega) -> X(D), u -> u|_D (module docstring); R.codomain is X(D), R.extension the extension by zero (on
    Sobolev spaces refused: R.adjoint is the minimal-norm extension, pieces_sobolev.py; on C(K) refused: R.dual(mu) is
    the dual map, pieces_continuous.py)."""

    why_linear = "restriction to a subdomain, u -> u|_D"

    def __init__(self, X, piece, realisation=None, label=None):
        from ..domains.partition import Partition
        if isinstance(piece, Partition):
            raise TypeError("restrict to one piece here; X.restrictions(P) restricts to all the pieces of a partition")
        XD = piece_space(X, piece, realisation)
        kind = _check(X)
        self.lp, self.sobolev, self.continuous = kind == "lp", kind == "sobolev", kind == "continuous"
        self.piece, self.label = XD.domain, label or repr(XD.domain)
        super().__init__(X, XD, name=f"R[{self.label}]")
        if self.sobolev or self.continuous:
            self._claims = sobolev_claims(X, XD.domain) if self.sobolev else continuous.claims(X, XD.domain)
            return
        self._claims = _l2_claims(X, XD.domain, "u -> u|_D")
        if self.lp:
            p = X.p
            q = "1" if p == float("inf") else ("∞" if p == 1 else f"{p / (p - 1):g}")
            self._claims += [Claim(f"the dual map L^{q}(D) -> L^{q}(Omega) is extension by zero", THEOREM,
                                   "int_D (u|_D) v dx = int_Omega u (E v) dx for v in L^p'(D)", key="adjoint"),
                             Claim("R E = I with E extension by zero, ||E v||_p = ||v||_p", THEOREM,
                                   "E v vanishes outside D", key="right inverse")]
        else:
            self._claims += [Claim("the adjoint is extension by zero", THEOREM,
                                   "<R u, v>_D = int_D u v dmu = <u, E v>_Omega", key="adjoint"),
                             Claim("a co-isometry: R R* = I on X(D)", THEOREM, "extension by zero, then restriction",
                                   key="co-isometry")]
        self._claims.append(_realisation_claim())

    def _apply(self, u):
        if self.sobolev:
            return restrict_sobolev(self.domain, self.codomain, u, self.label)
        return restrict_element(self.domain, self.codomain, u, self.label)

    @property
    def extension(self):
        """E: X(D) -> X(Omega), extension by zero (R.adjoint on L2; refused on Sobolev spaces, where R* is the
        minimal-norm extension)."""
        if self.sobolev or self.continuous:
            raise no_extension_by_zero(self) if self.sobolev else continuous.refuse_extension(self)
        if getattr(self, "_ext", None) is None:
            self._ext = Extension(self)
            if not self.lp:
                self._adjoint, self._ext._adjoint = self._ext, self
        return self._ext

    @property
    def adjoint(self):
        if self.sobolev:
            if self._adjoint is None:
                self._adjoint = SobolevExtension(self)
                self._adjoint._adjoint = self
            return self._adjoint
        if self.continuous or self.lp:
            raise continuous.refuse_adjoint(self)
        return self.extension

    def _make_adjoint(self):
        return SobolevExtension(self) if self.sobolev else self.extension

    def dual(self, mu):
        """On C(K): the dual map, a measure on the piece as the functional u -> int u dmu on C(K)."""
        if not self.continuous:
            raise Refused(f"{self.name}.dual is for C(K); here use R.adjoint", needs=("R.adjoint",))
        return continuous.dual(self, mu)


class Extension(LinearOperator):
    """E: X(D) -> X(Omega), extension by zero (module docstring); built by Restrict.extension."""

    why_linear = "extension by zero"

    def __init__(self, R):
        super().__init__(R.codomain, R.domain, name=f"E[{R.label}]")
        self.R = R
        self._claims = [Claim("linear", THEOREM, "extension by zero", key="linear"),
                        Claim("an isometry, norm 1", THEOREM, "E v vanishes outside D", key="bounded")]
        if not R.lp:
            self._claims.append(Claim("the adjoint of R", THEOREM, "<R u, v>_D = <u, E v>_Omega", key="adjoint"))

    def _apply(self, v):
        return extend(self.codomain, [(self.domain.domain, v)], name=f"E({v.name})")

    def _make_adjoint(self):
        return self.R


class Restrictions(LinearOperator):
    """R = (R_k)_k: X(Omega) -> (+)_k X(Omega_k) over the leaves of a partition (X.restrictions(P)); R.glue is G."""

    why_linear = "restrictions to the pieces of a partition"

    def __init__(self, X, partition, realisation=None):
        _check(X)
        refuse_product(X, "restricting to the pieces of a partition")
        M = broken_space(X, partition, realisation)
        self.partition = M.partition
        self.parts = {n: M.restriction("value", n) for n in M.partition.names}
        super().__init__(X, M, name="R")
        self._claims = [Claim("linear", THEOREM, "u -> (u|Omega_k)_k", key="linear"),
                        Claim("an isometry onto the broken space (+)_k X(Omega_k), which for L2 is the glued one: "
                              "sum_k ||R_k u||^2 = ||u||^2", THEOREM,
                              "the pieces tile the domain up to a null set (the partition's check), so integrals add",
                              key="isometry"),
                        Claim("the adjoint is the glue G (extension by zero of each part, summed), G = R^-1", THEOREM,
                              "an isometry onto is unitary", key="adjoint"), _realisation_claim()]
        self.glue = Glue(self)
        self._adjoint, self.glue._adjoint = self.glue, self

    def _apply(self, u):
        return self.codomain({n: R(u) for n, R in self.parts.items()}, name=f"R({u.name})") if isinstance(u, Function) \
            else self.codomain.family({n: R(u) for n, R in self.parts.items()}, name=f"R({u.name})")

    def _make_adjoint(self):
        return self.glue


class Glue(LinearOperator):
    """G: (+)_k X(Omega_k) -> X(Omega), x -> sum_k E_k x_k: the inverse and the adjoint of Restrictions (unitary)."""

    why_linear = "a sum of extensions by zero"

    def __init__(self, R):
        super().__init__(R.codomain, R.domain, name="G")
        self.R = R
        self._claims = [Claim("linear", THEOREM, "a sum of extensions by zero", key="linear"),
                        Claim("unitary: G = R^-1 = R*", THEOREM, "the pieces tile the domain up to a null set",
                              key="unitary"),
                        Claim("G x of realised parts is the piecewise function: their coefficients joined where the realisation "
                              "allows it, otherwise a formula that picks the piece; a jump at a seam stays a jump", CHECKED,
                              "seams are null sets in L2, so no value at a seam is claimed", key="realisation")]

    def _apply(self, x):
        M, XO = self.domain, self.codomain
        pl = M._pl_of(x)
        parts = [(self.R.parts[n].piece, None if pl.parts[k] is None else M.part(x, n)) for k, n in enumerate(M.names)]
        return extend(XO, parts, name=f"G({x.name})")

    def _make_adjoint(self):
        return self.R


def broken_space(X, partition, realisation=None):
    """pg.PiecewiseSpace(P, X, broken=True), the codomain of X.restrictions(P), after checking that P partitions X's
    domain."""
    from ..spaces.piecewise import PiecewiseSpace
    P = partition.flat()
    if P.domain != X.domain:
        raise SpaceMismatch(f"{partition} partitions {P.domain!r}, not the domain {X.domain!r} of {X}")
    return PiecewiseSpace(P, X, broken=True, realisation=realisation)


def restrictions(X, partition, realisation=None):
    """X.restrictions(P): Restrictions, or SobolevRestrictions into the broken sum for a Sobolev space."""
    if _check(X) == "sobolev":
        from .pieces_sobolev import SobolevRestrictions
        return SobolevRestrictions(X, partition, realisation)
    return Restrictions(X, partition, realisation)
