"""Elements and functionals across the seams of a partition (plan 069 P7, review B9; from operators/pieces.py and
operators/pieces_sobolev.py, plan 035 P3-P5): extension by zero and the glue of parts, the traces of a piecewise
element on the seams, and the claims of glued and broken fields. PiecewiseSpace (spaces/piecewise.py) and the
restriction operators (operators/pieces.py) use these; nothing here imports an operator.

    extend(X, [(D_k, v_k)])       sum_k E_k v_k in X(Omega): v_k in X(D_k) extended by zero and summed (module docstring of
                                  operators/pieces.py: coefficients joined where the realisation allows it, otherwise a
                                  piecewise formula with the pieces' boundaries as breaks, symbolic parts kept symbolic)
    seam_points(P, seam)          points of a seam where traces are compared: the point of an interval's seam, 14 points
                                  of a sphere between concentric pieces, a grid on the face shared by two boxes
    trace_jumps(M, x, field)      {seam: max |x_below - x_above|} over those points, with .match a Truth (the precision
                                  rule: core.logic.decide against the largest value there)
    require_matching_traces(...)  the condition for gluing a broken H^s element into H^s(Omega) (refused otherwise)

The glued space. For X = H^s(Omega) and a partition Omega = union Omega_k with Lipschitz seams, u in H^s(Omega) gives
u|Omega_k in H^s(Omega_k), and a family (u_k) of H^s(Omega_k) is the restriction of one u in H^s(Omega) iff, on every
seam, the traces of the normal derivatives d_n^j u_k of order j < s - 1/2 agree from both sides (s - 1/2 not an
integer; the trace theorems of Lions-Magenes, ch. 1, and Grisvard, ch. 1: stated as ASSERTED, the claim "interface
conditions", since this package does not check the theorem's premises for a given partition): no condition for s < 1/2 (no
trace: the broken sum is all of H^s(Omega), with an equivalent norm), continuity of the trace for 1/2 < s < 3/2, and
the normal derivative as well from s > 3/2 on. At s - 1/2 an integer the condition is an integral one (the H^1/2_00
case), not checked here. L2 has no trace: glued and broken L2 are the same space, L2(Omega) = (+)_k L2(Omega_k)
isometrically (the seams are null sets).
"""
import numpy as np

from ..core import fits
from ..core.claims import ASSERTED, CHECKED, THEOREM, Claim, Claims
from ..core.errors import Refused, SupportOutsideDomain, Unsupported
from ..core.logic import Truth, decide
from ..measures import DensityMeasure, LebesgueMeasure, SignedMeasure
from .elements import Family, Function
from .representers import DensityRepresenters, symbolic

LIONS = "Lions-Magenes, Non-Homogeneous Boundary Value Problems I (1972), ch. 1, Theorems 11.4 and 11.7 (extension " \
    "by zero bounded on H^s iff s < 1/2; H^1/2_00 at s = 1/2); Grisvard, Elliptic Problems in Nonsmooth Domains (1985), " \
    "Theorems 1.4.4.4 and 1.5.1.2 (traces)"
ROUTES = "pg.PiecewiseSpace(P, {field: H}) (glued, the default): functionals and Grams in H itself, exact"


def is_sobolev(X):
    from .sobolev import SobolevSpace
    return isinstance(X, SobolevSpace)


def kind(X):
    """"sobolev", "continuous", "lp" or "l2": the kinds of space whose pieces are restricted (operators/pieces.py)."""
    from .base import LpSpace
    from .continuous import ContinuousSpace
    if is_sobolev(X):
        return "sobolev"
    if isinstance(X, ContinuousSpace):
        return "continuous"
    return "lp" if isinstance(X, LpSpace) else "l2"


def refuse_product(X, what):
    """X.restrictions(P) or a PiecewiseSpace for a Banach space X (L^p, C(K)): the package's direct sums are Hilbert
    sums."""
    k = kind(X)
    if k == "lp":
        raise Refused(f"{what} {X}: L^p(Omega) = (+)_k L^p(Omega_k) with the l^p sum (||u||_p^p = sum_k ||R_k u||_p^p, "
                      "a theorem), and the package's direct sums are Hilbert sums",
                      needs=("a direct sum of Banach spaces with the l^p norm (not in the package)",
                             "pg.Restrict(X, D) piece by piece"))
    if k == "continuous":
        raise Refused(f"{what} {X}: C(K) is the subspace of (+)_k C(K_k) with equal values on the seams and the norm "
                      "max_k ||R_k u|| (a theorem, Restrict(X, K_k).claims), and the package's direct sums are Hilbert "
                      "sums", needs=("a direct sum of Banach spaces with the max norm (not in the package)",
                                     "pg.Restrict(X, K_k) piece by piece",
                                     "P.functionals(X, mu, side=): the split of a functional over the pieces"))


# ---------------------------------------------------------------- extension by zero and the glue (operators/pieces.py)
def glued(XO, parts, name):
    """sum_k E_k v_k for realised parts, their coefficients joined without a fit (Backend.glued), or None: a part that is a
    formula keeps the formula, and another realisation or an overlap is left to the caller."""
    B = XO.backend
    if any(v._src is not None or type(v.space.backend) is not type(B) or not B.same_realisation(v.space.backend)
           for _, v in parts):
        return None
    R = B.glued([(v.space.backend, v.realisation) for _, v in parts])
    return None if R is None else Function(XO, pl=R, name=name)


def _values(v, X):
    return np.asarray(v.rep(*X), float)


def _density_part(XO, D, v):
    """The measure v rho dx on D (rho the density of XO's measure): its representer in XO is v extended by zero."""
    lam = XO.measure
    rho = (lambda P: lam.weight.values(P)) if isinstance(lam, DensityMeasure) else (lambda P: np.ones(len(P)))

    def w(*X):
        P = np.stack([np.asarray(x, float) for x in X], axis=-1)
        return _values(v, X) * rho(P.reshape(-1, P.shape[-1])).reshape(P.shape[:-1])
    return DensityMeasure(LebesgueMeasure(D), w, name=f"{v.name}·ρ")


def _sum(ms):
    return ms[0] if len(ms) == 1 else SignedMeasure([(1.0, m) for m in ms])


def extend(XO, parts, name="Gx"):
    """sum_k E_k v_k in XO: parts a list of (D_k, v_k) with v_k an element or family of X(D_k), or None (zero); see
    the module docstring of operators/pieces.py for the three routes."""
    from .base import L2
    from ..domains.partition import breaks_for, members
    parts = [(D, v) for D, v in parts if v is not None]
    if not parts:
        return XO.zero
    fam = [isinstance(v, Family) for _, v in parts]
    if any(fam):
        N = {len(v) for _, v in parts}
        if len(N) != 1 or not all(fam):
            raise ValueError("the parts must all be single elements or families of one size")
        n = N.pop()
        if all(isinstance(v, DensityRepresenters) for _, v in parts):
            return XO.riesz([_sum([v.measures[i] for _, v in parts]) for i in range(n)])
        return XO.family([extend(XO, [(D, v[i]) for D, v in parts], name) for i in range(n)], name=name)
    sym = [symbolic(v) for _, v in parts]
    if all(sym):
        out = XO.riesz(_sum([v.measure for _, v in parts]))
        out.name = name
        return out
    br = breaks_for(XO.domain, [D for D, _ in parts]) if XO.realisation != "harmonics" else None   # (no breaks there)
    whole = len(parts) == 1 and parts[0][0] == XO.domain
    if not any(sym):
        joined = glued(XO, parts, name)
        if joined is not None:
            return joined
    if not any(sym) and (br is not None or whole):
        def src(*X):
            X = [np.asarray(x, float) for x in X]
            shape = np.broadcast(*X).shape
            flat = [np.broadcast_to(x, shape).ravel() for x in X]
            out, taken = np.zeros(len(flat[0])), np.zeros(len(flat[0]), bool)
            for D, v in parts:
                sel = members(D, flat) & ~taken
                if sel.any():
                    out[sel] = _values(v, [x[sel] for x in flat])
                    taken |= sel
            return out.reshape(shape)
        return fits.refit(Function(XO, source=src, name=name, breaks=br), *[v for _, v in parts])
    if isinstance(XO, L2):
        out = XO.riesz(_sum([v.measure if s else _density_part(XO, D, v) for (D, v), s in zip(parts, sym)]))
        out.name = name
        return out
    raise Unsupported(f"extending by zero into {XO}: the pieces' boundaries are not coordinate surfaces of "
                      f"{XO.domain!r}, so a fit would cross them")


# ---------------------------------------------------------------- traces on the seams
_DIRS = np.vstack([np.eye(3), -np.eye(3), np.array([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)])
                   / np.sqrt(3.0)])


def seam_points(P, seam):
    """Points of the seam (a tuple of coordinate arrays, as elements are called) where traces are compared: the point
    of an interval's seam; 14 points (axes and diagonals) of the sphere between concentric pieces; a 3^(d-1) grid
    inside the face shared by two boxes. Bands of a sphere are refused (Unsupported)."""
    if P.kind == "interval":
        return (np.array([float(seam.at)]),)
    if P.kind == "round":
        S = seam.set
        X = np.asarray(S.centre, float) + S.radius * _DIRS
        return tuple(X.T)
    if P.kind == "box":
        k, value = seam.at
        A, B = P.pieces[seam.below], P.pieces[seam.above]
        axes = []
        for j in range(A.dim):
            if j == k:
                axes.append(np.array([value]))
                continue
            lo, hi = max(A.sides[j].a, B.sides[j].a), min(A.sides[j].b, B.sides[j].b)
            axes.append(lo + (hi - lo) * np.array([0.25, 0.5, 0.75]))
        G = np.meshgrid(*axes, indexing="ij")
        return tuple(g.ravel() for g in G)
    raise Unsupported(f"traces on the seams of {P} (intervals, concentric balls and shells, boxes are compared)")


class Jumps(dict):
    """{seam name: max |u_below - u_above|} over the seam's points (seam_points), with .scale (the largest |value|
    there) and .match, a Truth: TRUE when every jump is at most DECIDE_TOL times the scale (core.logic.decide)."""

    @property
    def claims(self):
        return Claims([Claim(f"the traces {'match' if self.match is Truth.TRUE else 'differ'} on the seams: max jump "
                             f"{max(self.values(), default=0.0):.1e} against values up to {self.scale:.1e}", CHECKED,
                             "the pieces evaluated at points of each seam (seam_points), decided by the precision rule "
                             "(core.logic.decide); a check at those points, not over the whole seam", key="traces")])


def trace_jumps(M, x, field):
    """Jumps of x's field across the seams of M.partition carried by that field on both sides (module docstring)."""
    P = M.partition
    have = set(M.pieces_of(field))
    out, scale, parts = Jumps(), 0.0, {}
    for seam in P.seams:
        if seam.below not in have or seam.above not in have:
            continue
        X = seam_points(P, seam)
        vals = []
        for n in (seam.below, seam.above):
            if n not in parts:
                parts[n] = M.piece_element(x, field, n)
            vals.append(np.asarray(parts[n](*X), float).reshape(-1))
        out[seam.name] = float(np.max(np.abs(vals[0] - vals[1])))
        scale = max(scale, float(np.max(np.abs(vals[0]))), float(np.max(np.abs(vals[1]))))
    out.scale = scale
    tests = [decide(j, max(scale, np.finfo(float).tiny)) for j in out.values()]
    out.match = Truth.FALSE if Truth.FALSE in tests else Truth.UNKNOWN if Truth.UNKNOWN in tests else Truth.TRUE
    return out


def glue_conditions(s):
    """The interface conditions of H^s(Omega) among broken families (module docstring), in words."""
    if s < 0.5:
        return "none: H^s has no trace for s < 1/2, and the broken sum is all of H^s(Omega) (an equivalent norm)"
    if s == 0.5 or float(s - 0.5).is_integer():
        return f"at s = {s:g} (s - 1/2 an integer) an integral condition of H^1/2_00 type on the seams"
    m = int(np.ceil(s - 0.5)) - 1
    if m == 0:
        return "the traces of the pieces agree on every seam (continuity of the trace)"
    return (f"the traces of the pieces and of their normal derivatives of order <= {m} agree on every seam "
            "(continuity of the normal derivative from s > 3/2 on)")


def require_matching_traces(M, x, field, s):
    """Refuses a broken H^s element whose pieces do not glue into H^s(Omega) (module docstring): no condition for
    s < 1/2; for 1/2 < s < 3/2 the traces must agree (checked at seam_points by the precision rule); at s - 1/2 an
    integer and for s > 3/2 the conditions are not checked here, so gluing is refused naming them."""
    if s < 0.5:
        return
    if s == 0.5 or s > 1.5 or float(s - 0.5).is_integer():
        raise Refused(f"gluing the broken field {field!r} of {M} into H^{s:g}(Omega): its condition is "
                      f"{glue_conditions(s)}, which is not checked here (only the traces of values, for "
                      "1/2 < s < 3/2)", needs=("the glued space M.glued(): its elements are elements of H^s(Omega)",
                                                "a field of order 1/2 < s < 3/2, where the trace is checked"))
    J = trace_jumps(M, x, field)
    if J.match is not Truth.TRUE:
        bad = {k: v for k, v in J.items() if decide(v, max(J.scale, np.finfo(float).tiny)) is not Truth.TRUE}
        raise Refused(f"gluing the broken field {field!r} into H^{s:g}(Omega): its traces differ on the seams "
                      f"({', '.join(f'{k}: jump {v:.3g}' for k, v in bad.items())}, values up to {J.scale:.3g}), so it "
                      "is no element of H^s(Omega) (the trace theorem: " + LIONS + ")",
                      needs=("pieces that agree on the seams (M.seam_jumps(x) says where they do not)",
                             "stay in the broken space (+)_k H^s(Omega_k), which does not require them"))


# ---------------------------------------------------------------- glued and broken fields (from pieces_sobolev.py)
def additivity_claim(X):
    """Whether ||u||^2 splits as a sum over pieces, by construction (plan 035 D5)."""
    c = X.construction
    if c.kind == "spectral" and c.is_form_space:
        return Claim(f"additive over pieces: a(u, u) = sum_k a_(Omega_k)(u|Omega_k, u|Omega_k), the boundary terms on "
                     "the parts of the boundary each piece touches", THEOREM, "an integer-order intrinsic form is an "
                     "integral of a local density: integrals add over pieces meeting in null seams (on the glued "
                     "space; the piece forms are not norms of the pieces' quotient spaces)", key="additive")
    if c.kind == "spectral" and c.power() is not None:
        return Claim("the norm is an integral of a local density, so it adds over pieces as integrals (not as the "
                     "pieces' own norms)", THEOREM, "||A^(p/2) u||^2 for an integer power p of a differential operator "
                     "is local", key="additive")
    return Claim("not additive over pieces: sum_k ||u|Omega_k||^2 != ||u||^2", THEOREM,
                 "a quotient (restriction) norm or a fractional power is non-local: Grams over Omega keep cross terms "
                 "<K mu_j, K mu_k>", key="additive", holds=False)


def glued_claims(f, X):
    return [Claim(f"{f}: glued: the summand is {X} itself; piece k is seen through R_k, a functional splits as l = "
                  "sum_k l_k (l_k the restricted measures) and its Gram over Omega keeps the cross terms "
                  "<K mu_j, K mu_k>_Omega", THEOREM, "u -> int u dmu = sum_k int u|Omega_k d(mu|Omega_k); the inner "
                  "product is H's", key="glued"),
            Claim(f"{f}: as a subspace of the broken sum, the glued space is the families whose pieces satisfy the "
                  f"interface conditions: {glue_conditions(X.s)}", ASSERTED, "for Lipschitz seams and s - 1/2 not an "
                  "integer (" + LIONS + "); M.glue checks the traces at points of the seams for 1/2 < s < 3/2",
                  key="interface conditions"),
            Claim(f"{f}: a mass on a seam (bounded: s > (d - m)/2) gives the same functional from either side", THEOREM,
                  "the trace of u in H^s(Omega) on an interior surface is two-sided (" + LIONS + ")", key="seams"),
            additivity_claim(X)]


def broken_claims(f, X):
    return [Claim(f"{f}: broken: (+)_k H^{X.s:g}(Omega_k), independent across the seams (no interface condition); a "
                  "mass on a seam needs side=", ASSERTED, "a different model from H^s(Omega): larger for s > 1/2 (the "
                  "pieces need not match), with the norm sum_k ||u|Omega_k||^2, not ||u||^2 (" + LIONS + ")",
                  key="broken")]


def seam_check(X, partition, mu):
    """The claim that a seam mass mu gives the same functional on the glued space X from both sides: checked through
    the piece spaces (the restriction construction: ||k_below - k_above|| in X), a theorem otherwise."""
    below, above = partition.split(mu, "below"), partition.split(mu, "above")
    if X.construction.kind != "restriction":
        return Claim(f"{mu} on a seam: both sides give the same functional", THEOREM,
                     "the trace of u in H^s(Omega) is two-sided (" + LIONS + ")", key="seam sides")

    def rep(split):
        out = None
        for n, m in split.items():
            R = X.restrict_to(partition[n], label=n)
            v = R.adjoint(R.codomain.riesz(m))
            out = v if out is None else out + v
        return out
    a, b = rep(below), rep(above)
    G = np.array([[X.inner(a, a), X.inner(a, b)], [X.inner(b, a), X.inner(b, b)]])
    rel = float(np.sqrt(max(G[0, 0] + G[1, 1] - 2 * G[0, 1], 0.0) / max(G[0, 0], 1e-300)))
    if rel > 1e-8:
        raise RuntimeError(f"{mu}: the two sides of the seam give different functionals on {X} (relative {rel:.1e})")
    return Claim(f"{mu} on a seam: both sides give the same functional: ||k_below - k_above|| = {rel:.1e} ||k||",
                 CHECKED,
                 "R_below* and R_above* of the pieces' representers, compared in H^s(Omega)", key="seam sides")


class Parts(dict):
    """{summand key: measure} for M.riesz (PiecewiseSpace.parts), with the claims of the split."""

    def __init__(self, items, claims=()):
        super().__init__(items)
        self.claims = Claims(claims)


def glued_parts(M, f, measure, side=None):
    """PiecewiseSpace.parts for a glued Sobolev field: {f's key: measure} (= the sum of the restricted measures), with
    the split's claims; side= is optional (both sides give the same functional, a claim)."""
    X, P = M.fields[f], M.partition
    try:
        split = P.split(measure, side)
        claims = list(split.claims)
    except ValueError as e:
        if "side=" not in str(e):
            raise
        claims = [seam_check(X, P, measure)]
    return Parts({M.keys[(f, P.names[0])]: measure}, claims + [c for c in glued_claims(f, X) if c.key == "glued"])


def piece_functionals(M, field, measure, side=None):
    """PiecewiseSpace.piece_functionals: {piece: Functional of the restricted measure on that piece}; for a glued field
    they pair in H^s(Omega) (their Gram keeps the cross terms), and a seam mass goes to the side given (below by default:
    on the glued space both sides agree, glued_parts' claim)."""
    from .dual import Functional
    P = M.partition
    try:
        split = P.split(measure, side)
    except ValueError as e:
        if field not in M.glued_fields or "side=" not in str(e):
            raise
        split = P.split(measure, "below")
    missing = [n for n in split if (field, n) not in M.keys]
    if missing:
        raise SupportOutsideDomain(f"{measure} charges the pieces {missing}, which carry no {field!r} in {M}")
    return {n: Functional.from_representer(M.riesz({M.keys[(field, n)]: m}), name=f"∫_{n} {field} d({m})")
            for n, m in split.items()}

