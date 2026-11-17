"""Sobolev spaces on the pieces of a domain (plan 035 P4, design D3-D5, the Sobolev rows).

    R = pg.Restrict(H, D)        (= H.restrict_to(D)) for H = pg.Sobolev(Omega, s, length_scale=l), the restriction
                                 construction: R: H^s(Omega) -> H^s(D), the same construction on D (same kernel)
    R.adjoint                    the minimal-norm extension E: H^s(D) -> H^s(Omega), K(., mu)|_D -> K(., mu)|_Omega
    R.extension                  refused: extension by zero leaves H^s for s >= 1/2 and is not R* for s < 1/2
    H.restrictions(P)            (R_k)_k into the broken pg.PiecewiseSpace(P, H, broken=True) = (+)_k H^s(Omega_k):
                                 injective, bounded, not an isometry, onto the glued subspace
    pg.PiecewiseSpace(P, {"vp": H})                 glued (default): the summand of "vp" is H itself, the pieces views R_k
    pg.PiecewiseSpace(P, {"vp": H}, broken=True)    broken: the summands H^s(Omega_k), independent across the seams

The restriction construction R^s_l(D) = {U|_D : U in H^s_l(R^d)} with the quotient norm has the reproducing kernel K
of H^s_l(R^d) restricted to D (Aronszajn 1950, §I.5). So for D in Omega, H^s(D) is the quotient of H^s(Omega) by the
functions vanishing on D (a quotient of a quotient), and R is a co-isometry: R R* = I, R* v the minimal-norm element of
H^s(Omega) with restriction v, R* K(., mu)|_D = K(., mu)|_Omega for mu in the closure of D (the same kernel function),
and the energy of a measure in the closure of D is int int K dmu dmu in both spaces. What the backend represents
exactly (combinations of representers of measures, realisations/rkhs/backend.py) is carried exactly both ways: an element whose
measures lie in the closure of D keeps its realisation; a formula stays a formula; anything else (a representer of a
measure outside D) is evaluated on D and fitted there, the claim "realisation" saying so (its norm in H^s(D) is the
quotient norm: computed on intervals for integer orders, NormUnavailable elsewhere). R* of a formula needs the exterior
problem and is refused by name.

Over a partition Omega = union Omega_k the norms do not add: c ||u||^2 <= sum_k ||R_k u||^2 <= N ||u||^2 (c > 0 except
at s = 1/2), and the Grams over Omega carry cross terms <K mu_j, K mu_k>_Omega that the broken sum drops. The glued
space H^s(Omega) (the author's default) splits a functional as l = sum_k l_k (l_k the restricted measures), and a mass
on a seam gives the same functional from either side (the trace of u in H^s(Omega) is two-sided); the broken space
(+)_k H^s(Omega_k), by name only, is larger for s > 1/2 (no matching across seams) and needs side= on the seams.
The intrinsic and spectral constructions keep their glued view (functionals and Grams in H itself); their piece space,
the quotient space, is not realised and refused by name. The claims and functionals of glued and broken fields, and the
interface conditions of the glued space, are in spaces/gluing.py (plan 069 P7).
"""
import numpy as np

from ..core.claims import ASSERTED, CHECKED, THEOREM, Claim
from ..core import fits
from ..spaces.elements import Family, Function
from ..core.errors import Refused, SupportOutsideDomain, Unsupported
from ..spaces.gluing import LIONS
from ..core.logic import Truth
from .linear import LinearOperator

ARONSZAJN = "Aronszajn, Theory of reproducing kernels (1950), §I.5: the restriction of a reproducing-kernel space " \
    "to a subset has the restricted kernel and the quotient norm"
TRACE = "trace theorem (Jonsson-Wallin 1984; Adams-Hedberg 1996): an m-dimensional measure is in H^-s iff s > (d - m)/2"
ROUTES = "pg.PiecewiseSpace(P, {field: H}) (glued, the default): functionals and Grams in H itself, exact"


def _refuse_construction(X, D):
    c = X.construction
    raise Refused(
        f"the piece space of {X} on {D!r}: {X} is the {c.label()} construction; its canonical piece space is the "
        "quotient (restriction) space of X to the piece, with X's kernel restricted (Aronszajn), which is not realised "
        "here. Per-piece intrinsic or spectral spaces with conditions chosen at the seams are a broken model (a "
        "different, larger space), not built",
        needs=(ROUTES, "the restriction construction pg.Sobolev(Omega, s, length_scale=l), whose pieces are "
               "restricted (X.restrictions(P), broken=True)", "plan 035: the quotient space of an intrinsic or spectral "
               "space (not built)"))


def piece_space(X, D, realisation=None):
    """H^s(D): the restriction construction on the piece D, of X's order, length scale and tolerance."""
    from ..spaces.constructions import RestrictionSpace
    from ..sets import Interval
    from ..spaces.sobolev import SobolevSpace
    from ..domains.pieces import meet
    D = Interval(D) if isinstance(D, str) else D
    if X.construction.kind != "restriction":
        _refuse_construction(X, D)
    if X._measure_given:
        raise Unsupported(f"restricting {X}: a Sobolev space with a given measure (not Lebesgue's) is not restricted")
    if type(X.domain).__name__ not in ("Interval", "Box", "Ball", "Shell"):
        raise Unsupported(f"restricting {X} to a piece (intervals, boxes, balls and shells are restricted)")
    J = meet(X.domain, D)
    if J is None or not (J is D or J == D):
        raise SupportOutsideDomain(f"{D!r} is not contained in {X.domain!r}")
    return SobolevSpace(RestrictionSpace(D, X.s, X.length_scale), X.tol, X.realisation if realisation is None else
                        realisation)


# -------------------------------------------------------------------------------------------- what a realisation holds
def _atoms_in(A, D):
    from ..realisations.rkhs.atoms import atoms_inside
    return atoms_inside(A, D)


def _kernel_part(u):
    """The kernel part (RepresenterCombinations) of a realised element or family whose realisation has no formula part, else None."""
    pl = u.realisation
    return pl.kx if getattr(pl, "fx", 1) is None and getattr(pl, "kx", None) is not None else None


def _measures_of(u):
    """The measures a representer was made from (SobolevSpace.riesz records them), or None."""
    return u._memo.get("riesz measures")


def _inside_part(m, D):
    """m restricted to the closed set D (a mass on D's boundary kept: it lies in the closure), None when m does not
    charge D; raises Unsupported when the restriction is not constructed."""
    try:
        return m.restricted(D)
    except ValueError:
        below, above = m.restricted(D, "below"), m.restricted(D, "above")
        return below if below is not None else above


def _in_closure(m, D):
    """Whether the measure m lies in the closed set D."""
    try:
        return _inside_part(m, D) is m
    except Unsupported:
        return False


def _carry(XO, XD, u):
    """u (a realised element of XO) to XD with the same realisation: valid when XD has XO's kernel and flat coordinates
    (the restriction construction on a subset, or back) and its measures lie in XD's closed domain."""
    if isinstance(u, Family):
        return Family(XD, u.realisation, name=u.name)
    return Function(XD, pl=u.realisation, name=u.name)


# ---------------------------------------------------------------------------------------------- R and R*
def restrict_element(XO, XD, u, label):
    """R u for X = H^s(Omega) of the restriction construction (module docstring)."""
    from .pieces import _carried_breaks
    D, name = XD.domain, f"{u.name}|{label}"
    ms = _measures_of(u)
    if isinstance(u, Family):
        kx = _kernel_part(u)
        if kx is not None and _atoms_in(kx.atoms, D) is Truth.TRUE:     # the same kernel functions on D
            return _carry(XO, XD, u)
        if ms is not None and all(_in_closure(m, D) for m in ms):
            out = XD.riesz(list(ms))
            out.name = name
            return out
        return XD.family([restrict_element(XO, XD, u[i], label) for i in range(len(u))], name=name)
    if u._terms is not None and u._pl is None:                           # a combination: termwise
        out = None
        for c, t in u._terms:
            r = restrict_element(XO, XD, t, label) * c
            out = r if out is None else out + r
        out.name = name
        return out
    if u._exact is not None:
        return Function(XD, source=u._exact, exact=u._exact, name=name)
    if u._src is not None:                                               # a formula: it is the element
        return fits.piece(XD.element(u._src, name=name, breaks=_carried_breaks(XO, XD, u)), u)
    kx = _kernel_part(u)
    if kx is not None and _atoms_in(kx.atoms, D) is Truth.TRUE:          # the same kernel function on D
        out = _carry(XO, XD, u)
        out.name = name
        return out
    if ms is not None and _in_closure(ms[0], D):                         # a measure in the closure (volumes)
        out = XD.riesz(ms[0])
        out.name = name
        return out
    if ms is not None:                                                   # the part inside exact, the rest fitted
        out = _split_representer(XO, XD, ms[0], name)
        if out is not None:
            return out
    return fits.refit(XD.element(lambda *X: np.asarray(u(*X), float), name=name, breaks=_carried_breaks(XO, XD, u)), u)


def _split_representer(XO, XD, m, name):
    """R k_m = k^D_(m|D) + (K(., m - m|D) on D, fitted: smooth inside D) for a measure m charging D and its outside;
    None when m|D is not constructed or m does not charge D."""
    from ..measures import SignedMeasure
    try:
        m_in = _inside_part(m, XD.domain)
    except Unsupported:
        return None
    if m_in is None:
        return None
    rest = XO.riesz(SignedMeasure([(1.0, m), (-1.0, m_in)]))
    out = XD.riesz(m_in) + fits.refit(XD.element(lambda *X: np.asarray(rest(*X), float), name=f"{name} (outside part)"),
                                      how="internal")         # the kernel's part from outside D: smooth inside D
    out.name = name
    return out


def extend_minimal(XD, XO, v):
    """R* v, the minimal-norm extension, for v a combination of representers of measures in the closure of D (the same
    kernel function on Omega); refused for formulas (the exterior problem is not solved here)."""
    ms = _measures_of(v)
    if ms is not None:
        out = XO.riesz(list(ms) if isinstance(v, Family) else ms[0])
        out.name = f"E({v.name})"
        return out
    if isinstance(v, Function) and v._terms is not None and v._pl is None:
        out = None
        for c, t in v._terms:
            r = extend_minimal(XD, XO, t) * c
            out = r if out is None else out + r
        return out
    if isinstance(v, Function) and (v._src is not None or v._exact is not None) and v._pl is None:
        kx = None
    else:
        kx = _kernel_part(v)
    if kx is None:
        raise Refused(f"R* of {v.name} in {XD}: the minimal-norm extension of a function given by a formula solves the "
                      "exterior problem (1 - l^2 Delta)^s U = 0 outside the piece with matching traces, which is not "
                      "realised here", needs=("combinations of representers of measures in the piece (exact: "
                                              "K(., mu)|_D -> K(., mu)|_Omega)", "R.adjoint applied to representers "
                                              "(XD.riesz(mu))"))
    out = _carry(XD, XO, v)
    out.name = f"E({v.name})"
    return out


def sobolev_claims(X, D):
    from ..spaces.dual import trace_excess              # the one trace rule
    s, d = X.s, X.domain.dim
    return [Claim("linear", THEOREM, "u -> u|_D", key="linear"),
            Claim(f"the codomain H^{s:g}(D) is the restriction construction on D with the same kernel", THEOREM,
                  ARONSZAJN + "; the energy of a measure in the closure of D is int int K dmu dmu in both spaces",
                  key="piece space"),
            Claim("a co-isometry: R R* = I on H^s(D), ||R|| = 1", THEOREM, "H^s(D) is a quotient of H^s(R^d), and so a "
                  "quotient of H^s(Omega) (a quotient of a quotient): the quotient map onto it", key="co-isometry"),
            Claim("bounded, norm <= 1", THEOREM, "a co-isometry", key="bounded"),
            Claim("the adjoint is the minimal-norm extension: R* K(., mu)|_D = K(., mu)|_Omega", THEOREM,
                  "R* v is the element of (ker R)^perp with R R* v = v; the closure of the span of K(., mu), mu in the "
                  "closure of D, is (ker R)^perp", key="adjoint"),
            Claim("not extension by zero", THEOREM, "extension by zero leaves H^s for s >= 1/2 (" + LIONS + ") and for "
                  "s < 1/2 is not orthogonal to ker R", key="extension by zero", holds=False),
            Claim(f"an m-dimensional measure in the closure of D is bounded on H^{s:g}(D) iff s > ({d} - m)/2 (points "
                  f"{trace_excess(s, d, 0) > 0}, surfaces {trace_excess(s, d, d - 1) > 0}, paths "
                  f"{trace_excess(s, d, 1) > 0})", THEOREM,
                  TRACE + "; the same threshold as on Omega (the same kernel)", key="measures"),
            Claim("R u of a representer of measures in the closure of D: the same measures (exact); of a formula: the "
                  "formula on D; of anything else: evaluated on D and fitted there", CHECKED, "the realisation of R u "
                  "is exact for representers and formulas; a fit's error is the result's", key="realisation")]


class SobolevExtension(LinearOperator):
    """R*: H^s(D) -> H^s(Omega), the minimal-norm extension (module docstring); built by R.adjoint."""

    why_linear = "the adjoint of a restriction (a sum of minimal-norm extensions)"

    def __init__(self, R):
        super().__init__(R.codomain, R.domain, name=f"E[{R.label}]")
        self.R = R
        self._claims = [Claim("linear", THEOREM, "the adjoint of R", key="linear"),
                        Claim("an isometry onto (ker R)^perp: R E = I, ||E v|| = ||v||", THEOREM,
                              "R is a co-isometry", key="bounded"),
                        Claim("the minimal-norm extension: K(., mu)|_D -> K(., mu)|_Omega", THEOREM, ARONSZAJN,
                              key="adjoint")]

    def _apply(self, v):
        return extend_minimal(self.domain, self.codomain, v)

    def _make_adjoint(self):
        return self.R


def no_extension_by_zero(R):
    s = R.domain.s
    why = ("for s >= 1/2 the extension by zero of an element with a nonzero trace on the piece's boundary is not in "
           "H^s(Omega)" if s >= 0.5 else "for s < 1/2 it is bounded (Lions-Magenes) but it is not R*: R* is the "
           "minimal-norm extension")
    return Refused(f"extension by zero H^{s:g}(D) -> H^{s:g}(Omega) for {R.name}: {why}",
                   needs=("R.adjoint: the minimal-norm extension (exact on representers of measures)", LIONS))


# ---------------------------------------------------------------------------------------------- over a partition
class SobolevRestrictions(LinearOperator):
    """R = (R_k)_k: H^s(Omega) -> (+)_k H^s(Omega_k), the broken sum (X.restrictions(P) on a Sobolev space)."""

    why_linear = "restrictions to the pieces of a partition"

    def __init__(self, X, partition, realisation=None):
        from .pieces import broken_space
        M = broken_space(X, partition, realisation)
        P = self.partition = M.partition
        self.parts = {n: M.restriction("value", n) for n in P.names}
        super().__init__(X, M, name="R")
        N, s = len(P), X.s
        self._claims = [
            Claim("linear", THEOREM, "u -> (u|Omega_k)_k", key="linear"),
            Claim("injective", THEOREM, "the closed pieces cover Omega: u|Omega_k = 0 for all k gives u = 0",
                  key="injective"),
            Claim(f"bounded: sum_k ||R_k u||^2 <= {N} ||u||^2", THEOREM, "each R_k is a co-isometry", key="bounded"),
            Claim("not an isometry: sum_k ||R_k u||^2 != ||u||^2 in general", THEOREM,
                  "for u = K(., y), y inside Omega_j: ||R_j u|| = ||u|| and R_k u != 0 for the neighbours (K > 0)",
                  key="isometry", holds=False),
            Claim("c ||u||^2 <= sum_k ||R_k u||^2 with c > 0" + (" fails: s = 1/2" if s == 0.5 else ""), ASSERTED,
                  LIONS + ": for s != 1/2 the broken norm is equivalent on H^s(Omega); at s = 1/2 the image is not "
                  "closed",
                  key="lower bound", holds=s != 0.5),
            Claim("the image is the glued subspace: proper for s > 1/2 (traces match across seams), all of the broken "
                  "sum for s < 1/2", ASSERTED, LIONS, key="image"),
            Claim("the adjoint is x -> sum_k R_k* x_k (sum of minimal-norm extensions), not the glue", THEOREM,
                  "<R u, x> = sum_k <R_k u, x_k> = <u, sum_k R_k* x_k>", key="adjoint")]

    def _apply(self, u):
        parts = {n: R(u) for n, R in self.parts.items()}
        return self.codomain(parts, name=f"R({u.name})") if isinstance(u, Function) else \
            self.codomain.family(parts, name=f"R({u.name})")

    def _make_adjoint(self):
        R = self

        class Sum(LinearOperator):
            def _apply(self, x):
                out = None
                for n, Rk in R.parts.items():
                    v = Rk.adjoint(R.codomain.part(x, n))
                    out = v if out is None else out + v
                return out

            def _make_adjoint(self):
                return R
        out = Sum(self.codomain, self.domain, name="R*")
        out._claims = [Claim("x -> sum_k R_k* x_k, the minimal-norm extensions summed", THEOREM, "the adjoint of R",
                             key="adjoint")]
        return out

    @property
    def glue(self):
        raise Refused("gluing a broken element of (+)_k H^s(Omega_k) as an operator: it is an element of H^s(Omega) "
                      "only when its pieces match across the seams (s > 1/2), so G is defined on the glued subspace "
                      "only, not on the broken space",
                      needs=("R.codomain.glue(x): one element, its traces checked (spaces/gluing.py)",
                             "the glued space R.codomain.glued() = pg.PiecewiseSpace(P, H), which keeps H itself",
                             "R.adjoint: the sum of minimal-norm extensions (not an inverse)"))
