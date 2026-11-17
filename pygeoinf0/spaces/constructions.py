"""Constructions of Sobolev spaces: the Hilbert structure is part of a space's identity (plan 006).

On a bounded domain D the set H^s(D) carries several inner products, equivalent as norms but not equal, and the choice
selects different representers, adjoints, minimum-norm solutions and posteriors. Four mathematical objects (no
realisation here) say which one a space has:

    RestrictionSpace(D, s, l)
                           R^s_l(D) = {U|_D : U in H^s_l(R^d)}, ||u|| = inf{||U|| : U|_D = u} (McLean, Strongly
                           Elliptic Systems, 2000, ch. 3): pg.Sobolev(D, s, length_scale=l) on intervals, boxes,
                           disks, cylinders, balls, paths and flat patches
    SobolevForm(D, k, length_scale=l, dirichlet=..., robin=...)
                           the intrinsic form of integer order k with boundary data,
                               a_k(u, v) = sum_{j=0..k} C(k, j) l^2j int_D D^j u : D^j v + sum_R beta_R int_R u v dS
                           on V = {u in H^k(D) : u = 0 on the Dirichlet parts}, D^j u the tensor of all j-th derivatives
                           (not powers of the Laplacian: with (Delta u)^2 the natural domain on a bounded domain is larger
                           than H^2). On R^d it is the Bessel norm, sum_j C(k, j) l^2j |xi|^2j = (1 + l^2 |xi|^2)^k.
    FormOperator           a.operator(): the self-adjoint A_a >= 1 of the closed, coercive, densely defined symmetric
                           form a (Kato, Perturbation Theory for Linear Operators, 2nd ed. 1976, Theorem VI.2.1,
                           first representation theorem): D(A_a) in V and <A_a u, v> = a(u, v) for v in V. The natural
                           boundary conditions belong to D(A_a), not to V. An unbounded operator: it lives in
                           operators/unbounded.py (plan 068 P3), a closed densely defined linear relation on L2.
    Spectral(A, s)         D(A^(s/2k)) with ||A^(s/2k) u||, A = A_a of order 2k, s the Sobolev order: pg.Sobolev(A, s).
                           Spectral(A_a, k) = (V, a) by the second representation theorem (Kato, Theorem VI.2.23),
                           so a.space() is pg.Sobolev(a.operator(), k).

On closed manifolds (tori, circles, spheres) pg.Sobolev(X, s, length_scale=l) is Spectral(A, s) for A = 1 - l^2 Delta,
the operator of SobolevForm(X, 1, length_scale=l). Constructions with the same set and the same inner product have the
same key, so their spaces are equal: the restriction of H^1_l to an interval is the form a_1 + l [u(a)^2 + u(b)^2]
(Robin with beta = l; experiments/EXP-026-sobolev-constructions/NOTES.md §3), and on a flat closed manifold
A_(a_k) = A_(a_1)^k. Any other difference in the inner product makes different spaces.

Boundary parts are named subsets of the boundary (domain.boundary_parts): "left", "right" on an interval; "x-", "x+",
"y-", "y+", "z-", "z+" on a box. Boundary conditions are polynomials c in the outward normal derivative d_n
(sum_r c_r d_n^r u = 0 at the part), which is how the realisations read them.
"""
import math

import numpy as np

from ..core.claims import DERIVED, THEOREM, Claim, Claims
from ..core.errors import Unsupported
from ..sets import Interval

KATO1 = "Kato, Perturbation Theory for Linear Operators, Theorem VI.2.1 (first representation theorem)"
KATO2 = "Kato, Perturbation Theory for Linear Operators, Theorem VI.2.23 (second representation theorem)"
FRACTIONAL = ("integer powers: the definition of A (Kato, Theorem VI.2.1) applied p times; fractional orders by interpolation "
              "(Lions-Magenes I, ch. 1 §11; Grisvard 1967; Fujiwara 1967); on intervals and boxes from the cosine/sine "
              "series of the even/odd extensions (EXP-026 NOTES §5)")


def as_domain(D):
    return Interval(D) if isinstance(D, str) else D


def domain_key(D):
    if isinstance(D, Interval):
        return ("Interval", D.a, D.b)
    return D.key() if hasattr(D, "key") else (type(D).__name__, repr(D))


def boundary_parts(D):
    """The named parts of the boundary of D (() for a closed manifold)."""
    return tuple(getattr(D, "boundary_parts", ("boundary",)))


def _curved(D):
    from ..domains import Sphere
    return isinstance(D, Sphere)


def poly_mul(p, q):
    return tuple(float(c) for c in np.convolve(np.asarray(p, float), np.asarray(q, float)))


def L_poly(l, k=1):
    """(1 - l^2 d^2)^k as a polynomial in the derivative d (coefficients of d^0, d^1, ...)."""
    out = (1.0,)
    for _ in range(k):
        out = poly_mul(out, (1.0, 0.0, -l * l))
    return out


def order(c):
    """The order of a boundary condition c (its highest derivative with a nonzero coefficient)."""
    nz = [r for r, x in enumerate(c) if x != 0]
    return nz[-1] if nz else -1


def describe_condition(c):
    terms = []
    for r, x in enumerate(c):
        if x == 0:
            continue
        u = "u" if r == 0 else "d_n u" if r == 1 else f"d_n^{r} u"
        terms.append(u if x == 1 else f"{x:.6g} {u}")
    return " + ".join(terms).replace("+ -", "- ") + " = 0"


# ------------------------------------------------------------------------------------------ forms and operators

class SobolevForm:
    """The intrinsic Sobolev form a_k of integer order k >= 1 on a domain, with boundary data (module docstring):

        SobolevForm("[0, 1]", 1, length_scale=0.1)                         int u v + l^2 u' v' on H^1 (natural Neumann)
        SobolevForm(I, 1, length_scale=l, dirichlet="left")                on {u in H^1 : u(0) = 0}
        SobolevForm(I, 1, length_scale=l, robin={"right": beta})           + beta u(1) v(1)
        SobolevForm(I, 2, length_scale=l)                                  int u v + 2 l^2 u' v' + l^4 u'' v'' on H^2

    dirichlet: a part name, a list of them or "all" (essential u = 0 there); robin: {part: beta >= 0} or one beta for
    every part that is not Dirichlet (beta = 0 is the natural Neumann condition). Boundary data are offered for k = 1;
    forms of order 2 are unconstrained (their essential conditions, u = 0 or u = d_n u = 0, would be a choice)."""

    def __init__(self, domain, k, length_scale, dirichlet=(), robin=None):
        D = as_domain(domain)
        if int(k) != k or k < 1:
            raise ValueError("an intrinsic form has a positive integer order k (fractional intrinsic norms, "
                             "Sobolev-Slobodeckij, are a different construction and are not offered)")
        l = float(length_scale)
        if not l > 0:
            raise ValueError("the length scale must be positive")
        parts = boundary_parts(D)
        if isinstance(dirichlet, str):
            dirichlet = parts if dirichlet == "all" else (dirichlet,)
        dirichlet = tuple(dirichlet or ())
        if robin is None:
            robin = {}
        elif not isinstance(robin, dict):
            robin = {p: float(robin) for p in parts if p not in dirichlet}
        for p in list(dirichlet) + list(robin):
            if p not in parts:
                raise ValueError(f"{D!r} has the boundary parts {parts or '(none: a closed manifold)'}; not {p!r}")
        if set(dirichlet) & set(robin):
            raise ValueError(f"a part cannot carry a Dirichlet and a Robin condition: {set(dirichlet) & set(robin)}")
        if any(float(b) < 0 for b in robin.values()):
            raise ValueError("Robin coefficients must be >= 0 (a negative beta can destroy coercivity)")
        if int(k) > 1 and (dirichlet or any(float(b) for b in robin.values())):
            raise Unsupported("boundary data are offered for forms of order 1; a form of order 2 here is unconstrained "
                              "(its essential conditions, simply supported u = 0 or clamped u = d_n u = 0, are a choice "
                              "not made yet)")
        if int(k) > 1 and _curved(D):
            raise Unsupported(f"on {D!r} the form of order {int(k)} involves the curvature (Bochner: int |Hess u|^2 = "
                              "int (Delta u)^2 - R^-2 int |grad u|^2 on a sphere of radius R); not offered")
        self.domain, self.k, self.length_scale = D, int(k), l
        self.dirichlet = tuple(p for p in parts if p in dirichlet)
        self.robin = tuple((p, float(robin[p])) for p in parts if p in robin and float(robin[p]) != 0)

    @property
    def parts(self):
        return boundary_parts(self.domain)

    def key(self):
        return ("form", domain_key(self.domain), self.k, self.length_scale, self.dirichlet, self.robin)

    def __eq__(self, other):
        return isinstance(other, SobolevForm) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def operator(self):
        """A_a, the self-adjoint operator of this form (Kato's first representation theorem): an
        operators.unbounded.FormOperator, a closed densely defined operator on L2 of the domain."""
        from ..operators.unbounded import FormOperator
        return FormOperator(self)

    def space(self, tol=1e-14, realisation=None):
        """(V, a): the Hilbert space of this form, pg.Sobolev(a.operator(), k)."""
        from .sobolev import Sobolev
        return Sobolev(self.operator(), self.k, tol=tol, realisation=realisation)

    def __call__(self, u, v):
        """a(u, v) for elements u, v of the form's space."""
        return u.space.inner(u, v) if u.space.construction.key() == Spectral(self.operator(), self.k).key() \
            else self.space().inner(u, v)

    def conditions(self, part):
        """The boundary conditions of D(A_a) at a part: [(c, essential)], c a polynomial in d_n (module docstring).
        Order 1: u = 0 (Dirichlet, essential) or l^2 d_n u + beta u = 0 (natural; beta = 0 Neumann). Order 2 on an
        interval: d_n^2 u = 0 and 2 l^2 d_n u - l^4 d_n^3 u = 0 (natural; EXP-026 NOTES §2)."""
        if part not in self.parts:
            raise ValueError(f"{self.domain!r} has no boundary part {part!r}")
        l = self.length_scale
        if part in self.dirichlet:
            return [((1.0,), True)]
        if self.k == 1:
            return [((dict(self.robin).get(part, 0.0), l * l), False)]
        if self.k == 2 and self.domain.dim == 1:
            return [((0.0, 0.0, 1.0), False), ((0.0, 2 * l * l, 0.0, -l ** 4), False)]
        raise Unsupported(f"the natural conditions of a form of order {self.k} on {self.domain!r} are not written out")

    def _bits(self):
        bits = [f"Dirichlet at {', '.join(self.dirichlet)}"] if self.dirichlet else []
        bits += [f"Robin {b:g} at {p}" for p, b in self.robin]
        free = [p for p in self.parts if p not in self.dirichlet and p not in dict(self.robin)]
        if free and self.k == 1:
            bits.append(f"natural (Neumann) at {', '.join(free)}")
        return "; ".join(bits)

    def label(self):
        return f"a_{self.k}" + (f" ({self._bits()})" if self._bits() else "")

    @property
    def claims(self):
        l = self.length_scale
        terms = " + ".join(f"{math.comb(self.k, j) * l ** (2 * j):.6g} int D^{j} u : D^{j} v" for j in range(self.k + 1))
        cl = [Claim(f"a(u, v) = {terms}" + "".join(f" + {b:g} int_{p} u v" for p, b in self.robin), DERIVED,
                    "the intrinsic form of order k with length scale l (full derivative tensors)"),
              Claim("form domain V = H^k(D)" + (f" with u = 0 on {', '.join(self.dirichlet)}" if self.dirichlet else ""),
                    THEOREM, "essential conditions belong to V; natural ones to D(A_a)"),
              Claim("symmetric, densely defined in L2, closed and coercive: a(u, u) >= ||u||_L2^2", THEOREM,
                    "a is equivalent to the H^k inner product on V (a closed subspace of H^k); Robin terms with beta >= 0 "
                    "are bounded on H^1 by the trace theorem")]
        return Claims(cl)

    def __repr__(self):
        return f"SobolevForm({self.domain}, {self.k}, l={self.length_scale:g}" + (
            f"; {self._bits()})" if self._bits() else ")")


# ------------------------------------------------------------------------------------------ constructions

class RestrictionSpace:
    """R^s_l(D): the restriction of H^s_l(R^d) to D with the quotient norm (module docstring)."""

    kind = "restriction"

    def __init__(self, domain, s, length_scale):
        self.domain, self.s, self.length_scale = as_domain(domain), float(s), float(length_scale)
        self.operator = None

    def key(self):
        if isinstance(self.domain, Interval) and self.s == 1:          # = the form a_1 + l [u(a)^2 + u(b)^2]
            return Spectral(interval_restriction_form(self.domain, self.length_scale).operator(), 1).key()
        return ("restriction", domain_key(self.domain), self.s, self.length_scale)

    def element_conditions(self, part):
        return []

    def vanishes(self, mu):
        return False

    def label(self):
        return "restriction of H^s_l(R^d)"

    @property
    def claims(self):
        d = self.domain.dim
        cl = [Claim(f"definition: H^{self.s:g}_l(R^{d}) restricted to the domain (quotient norm), l = {self.length_scale:g}",
                    THEOREM, "McLean 2000, ch. 3 (flat coordinates of the domain)"),
              Claim("Hilbert space (complete)", THEOREM, "a quotient of a Hilbert space by a closed subspace"),
              Claim("equivalent (not equal) to the intrinsic Sobolev-Slobodeckij norm", THEOREM,
                    "Lipschitz domains (Stein extension)"),
              Claim("no boundary condition on elements; the reproducing kernel is the stationary Matern kernel", THEOREM,
                    "the restriction of a reproducing-kernel space to a subset has the restricted kernel (Aronszajn)")]
        if isinstance(self.domain, Interval) and self.s == int(self.s):
            k = int(self.s)
            cl.append(Claim(f"||u||^2 = a_{k}(u, u) + exterior energy of the boundary jet (u, ..., d_n^{k - 1} u) at both "
                            f"ends" + (": l [u(a)^2 + u(b)^2]" if k == 1 else ""), THEOREM,
                            "the H^k_l(R) norm is the local form a_k; the minimal extension solves (1 - l^2 d^2)^k U = 0 "
                            "outside (EXP-026 NOTES §3)"))
        return Claims(cl)


class Spectral:
    """D(A^(s/2k)) with ||A^(s/2k) u|| for A = A_a of order 2k (module docstring); s > 0 the Sobolev order."""

    kind = "spectral"

    def __init__(self, operator, s):
        from ..operators.unbounded import FormOperator
        if not isinstance(operator, FormOperator):
            raise TypeError("a spectral Sobolev space is built from the operator of a form: pg.Sobolev(a.operator(), s)")
        if not float(s) > 0:
            raise ValueError("the order of a spectral Sobolev space is positive (s = 0 is L2)")
        self.operator, self.s = operator, float(s)
        self.form, self.domain, self.length_scale = operator.form, operator.form.domain, operator.form.length_scale

    @property
    def is_form_space(self):
        return self.s == self.form.k

    def power(self):
        """s/k: the integer power p with ||u||^2 = <A^p u, u> when s/k is an integer (else None)."""
        p = self.s / self.form.k
        return int(p) if p == int(p) else None

    def key(self):
        f = self.form
        if f.k > 1 and not f.parts and not _curved(f.domain):            # flat closed manifold: A_(a_k) = A_(a_1)^k
            f = SobolevForm(f.domain, 1, f.length_scale)
        from ..operators.unbounded import FormOperator
        return ("spectral", FormOperator(f).key(), self.s)

    def kernel_conditions(self, part):
        """The conditions of D(A^p), p = s/k: those of the reproducing kernel (Green function of A^p) at a part."""
        p = self.power()
        if p is None:
            raise Unsupported("the kernel conditions of a fractional power are not local")
        return self.operator.power_conditions(p, part)

    def element_conditions(self, part):
        """The conditions every element smooth up to the part satisfies there: those of D(A^p) (p = ceil(s/2k)) of order
        m <= s - 1/2 (at m = s - 1/2 the Lions-Magenes borderline: smooth elements still need it)."""
        p = max(1, math.ceil(self.s / self.operator.order))
        return [(c, ess) for c, ess in self.operator.power_conditions(p, part) if order(c) + 0.5 <= self.s]

    def vanishes(self, mu):
        """Whether u -> int u dmu is zero on this space: mu (a point, or a segment) lies in a Dirichlet part, where every
        element vanishes (s > 1/2: the trace exists and is zero)."""
        if not self.form.dirichlet or self.s <= 0.5:
            return False
        from ..measures import DiracMeasure
        from ..domains import Flat
        if isinstance(mu, DiracMeasure):
            X = [np.atleast_1d(np.asarray(mu.x0, float))]
        elif isinstance(getattr(mu, "set", None), Flat) and mu.set.dim == 1:
            X = [np.asarray(p, float) for p in mu.set.endpoints]          # a face is flat: both ends in it suffice
        else:
            return False
        return any(all(_on_part(self.domain, part, x) for x in X) for part in self.form.dirichlet)

    def label(self):
        if self.is_form_space:
            return f"intrinsic {self.form.label()}"
        return f"spectral, {self.operator.label()}"

    @property
    def claims(self):
        A, s, k = self.operator, self.s, self.form.k
        t = f"{s / (2 * k):g}"
        cl = [Claim(f"definition: D(A^{t}) with ||A^{t} u||, {A.label()}, l = {self.length_scale:g}", THEOREM,
                    "spectral theorem: A self-adjoint, A >= 1, compact resolvent"),
              Claim("Hilbert space (complete)", THEOREM, f"A^{t} is closed and >= 1, so its domain is complete in ||A^{t} u||")]
        if self.is_form_space:
            cl.append(Claim(f"equal to the form space: ||u||^2 = a(u, u) on V", THEOREM, KATO2))
        if not self.form.parts:
            cl.append(Claim("sum (1 + l^2 lambda_k)^s |u_k|^2, lambda_k the eigenvalues of -Laplacian", THEOREM,
                            "spectral (Laplace-Beltrami on a closed manifold)"))
        else:
            for part in self.form.parts:
                try:
                    conds = self.element_conditions(part)
                except Unsupported:
                    continue
                text = "; ".join(describe_condition(c) for c, _ in conds) or "none"
                cl.append(Claim(f"conditions on elements at {part}: {text}", THEOREM, FRACTIONAL))
        return Claims(cl)


def _on_part(D, part, x):
    """Whether the point x lies on the named boundary part of an interval or a box."""
    if isinstance(D, Interval):
        e = D.a if part == "left" else D.b
        return abs(float(x[0]) - e) <= 1e-12 * max(1.0, abs(e))
    sides = getattr(D, "sides", None)
    if sides is None:
        return False
    k = D.internal_names.index(part[0])
    e = sides[k].a if part[1] == "-" else sides[k].b
    return abs(float(x[k]) - e) <= 1e-12 * max(1.0, abs(e))


def interval_restriction_form(I, l):
    """The form a_1 + l [u(a)^2 + u(b)^2] on H^1(I), whose space is the restriction of H^1_l(R) (NOTES §3)."""
    return SobolevForm(I, 1, l, robin=l)


def default_construction(domain, s, length_scale):
    """What pg.Sobolev(D, s, length_scale=l) means: the restriction of H^s_l(R^d) on domains with a boundary, the
    spectral space of 1 - l^2 Delta on closed manifolds."""
    D = as_domain(domain)
    if boundary_parts(D) == ():
        return Spectral(SobolevForm(D, 1, length_scale).operator(), s)
    return RestrictionSpace(D, s, length_scale)

