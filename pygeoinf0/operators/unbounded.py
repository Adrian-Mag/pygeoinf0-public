"""Closed, densely defined (unbounded) operators (plan 068 P3; review B5, A4; from core/sturm_liouville.py and
spaces/constructions.py). Each is a single-valued linear relation on its ambient Hilbert space X
(relations.linear.ClosedOperator): dom = D(A), an OperatorDomain that is dense in X and, A being unbounded, not closed
(a closed operator with a closed domain is bounded: the closed graph theorem); the graph-norm space is reached only by
domain_space(). Three objects are kept apart (AGENTS.md): A.domain (X), A.dom (D(A), a subset of X) and
A.domain_space() (D(A) with the graph norm, another Hilbert space). A is a descriptor: its claims, its adjoint, its
graph-norm space and its spectrum, not an action on elements.

    FormOperator(a)     a.operator(): the self-adjoint A_a >= 1 of a SobolevForm a on L2(D) (spaces/constructions.py),
                        interior expression (1 - l^2 Delta)^k; domain_space() = pg.Sobolev(A_a, 2k)
    SturmLiouville(interval, p, q, w, left, right, interfaces)
        L u = w^-1 (-(p u')' + q u) on (a, b), the self-adjoint operator of the closed form
            a(u, v) = int p u'v' + q u v dx + sum over Robin ends beta u(e) v(e)
        on L2((a, b), w dx), with u and p u' continuous across the interfaces (the natural transmission conditions of the
        form). Ends: "dirichlet" (u = 0, essential), "neumann" (p d_n u = 0, natural), ("robin", beta) (p d_n u + beta u =
        0, beta >= 0, natural) and, at an end where p and w vanish, "regular" (the singular end of a ball's centre).
        L.spectrum(): the realised spectrum (eigenvalue brackets, eigenfunctions, lower bounds for tails; realisations/
        registry.spectrum: bessel.py closed forms for the per-degree radial Laplacian, sl_galerkin.py brackets otherwise)
    Laplacian(D, bc)    -Delta on L2(D) with boundary conditions, one class for every domain whose spectrum is built:
        interval    bc {"left": c, "right": c}                      closed-form modes (realisations/spectra/laplace_spectra.py)
        box         bc {"x-": c, "x+": c, ...} or one c for every face   sums over the axes (separation of variables)
        ball        bc {"boundary": c}, centre="regular" for degree 0    (+)_s L_s (x) I_(H_s) (Schur, EXP-030 G2.1)
        shell       bc {"inner": c, "outer": c}                          the same, degree by degree
        circle, sphere, torus    no bc (closed manifolds)                (m/R)^2, l(l + 1)/R^2, sums over the periods
        c as in SturmLiouville ("dirichlet", "neumann", ("robin", beta)); one c may stand for every boundary part.
        Elsewhere (a disk, a cylinder, ...) it is refused with the choices: the spectrum is not built.
        Lap.spectrum(): eigenvalues(n) with multiplicity, brackets(n), claims; on balls and shells Lap.radial(s) is L_s
        and Lap.bessel_potential(kappa, q, degrees, modes) the covariance (realisations/spectra/laplacian_covariance.py)

Facts (THEOREM unless said; Zettl, Sturm-Liouville Theory, AMS 2005; Kato, Perturbation Theory, VI.2): with p, w > 0
bounded away from 0 on [a, b], q bounded below and beta >= 0, the form is closed and bounded below, L is self-adjoint with
compact resolvent, its eigenvalues are simple (separated conditions), and the n-th eigenfunction has n - 1 zeros in
(a, b) (Sturm oscillation). At r = 0 for L_s: the solutions behave as r^s and r^(-s-1); for s >= 1 the second is not in
L2(r^2 dr) near 0, so the end is limit point (Weyl's alternative) and nothing is imposed; for s = 0 both are, the end is
limit circle and the regular (Friedrichs) condition, u bounded, must be named: Laplacian(..., centre="regular").
Unbounded: a self-adjoint operator with compact resolvent on an infinite-dimensional space has eigenvalues of finite
multiplicity accumulating only at infinity, so it is not bounded.
"""
import math

import numpy as np

from ..core.claims import CHECKED, DERIVED, THEOREM, Claim
from ..core.errors import Refused, Unsupported
from ..relations.linear import ClosedOperator
from ..sets import Interval
from ..spaces.constructions import KATO1, L_poly, poly_mul

ZETTL = "Zettl, Sturm-Liouville Theory (AMS 2005), ch. 4 and 10 (regular and limit-point problems)"
KINDS = ("dirichlet", "neumann", "robin", "regular")
UNBOUNDED = ("self-adjoint with compact resolvent on an infinite-dimensional space: its eigenvalues accumulate only at "
             "infinity")


def as_condition(c):
    """(kind, beta) from "dirichlet", "neumann", ("robin", beta), {"robin": beta} or "regular"."""
    if isinstance(c, dict) and len(c) == 1:
        c = tuple(c.items())[0]
    if isinstance(c, str):
        c = (c, 0.0)
    kind, beta = c[0], float(c[1]) if len(c) > 1 else 0.0
    if kind not in KINDS:
        raise ValueError(f"a boundary condition is one of {KINDS}; not {kind!r}")
    if kind == "robin" and beta < 0:
        raise ValueError("Robin coefficients are >= 0 (a negative beta can make the form unbounded below)")
    if kind == "robin" and beta == 0:
        kind = "neumann"
    return kind, (beta if kind == "robin" else 0.0)


def describe(c):
    kind, beta = c
    return f"Robin beta = {beta:g}" if kind == "robin" else kind


def _coef(f, name):
    """A coefficient: a number, a pg.Tabulated or a vectorised callable of x."""
    if isinstance(f, (int, float, np.number)):
        v = float(f)
        return lambda x: np.full(np.shape(x), v)
    if callable(f):
        return f
    raise TypeError(f"the coefficient {name} is a number, a pg.Tabulated or a callable of x")


def _self_adjoint(why, status=THEOREM):
    return Claim("self-adjoint", status, why, key="self-adjoint")


SAMPLES = 4097               # points at which a callable coefficient is sampled on [a, b] (measures/weights.SAMPLES_1D)


def _extremes(src, f, a, b, extra=()):
    """(inf, sup, exact) of a coefficient on [a, b]: exact for a number and a pg.Tabulated (exactly piecewise linear:
    its extremes are at the ends of its pieces), sampled at SAMPLES points (and extra) for a callable (exact False)."""
    if isinstance(src, (int, float, np.number)):
        return float(src), float(src), True
    if all(hasattr(src, k) for k in ("_x0", "_x1", "_v0", "_v1")):
        lo, hi = np.maximum(src._x0, a), np.minimum(src._x1, b)
        keep = hi >= lo
        s0, s1 = (lo - src._x0) / (src._x1 - src._x0), (hi - src._x0) / (src._x1 - src._x0)
        dv = src._v1 - src._v0
        v = np.r_[(src._v0 + dv * s0)[keep], (src._v0 + dv * s1)[keep], np.asarray(f(np.array([a, b])), float)]
        return float(v.min()), float(v.max()), True
    x = np.unique(np.r_[np.linspace(a, b, SAMPLES), np.asarray(extra, float)])
    v = np.asarray(f(x), float)
    if v.shape != x.shape:
        v = np.broadcast_to(v, x.shape) if v.ndim == 0 else np.array([float(f(t)) for t in x])
    if not np.all(np.isfinite(v)):
        return -math.inf, math.inf, False
    return float(v.min()), float(v.max()), False


# ---------------------------------------------------------------------------------------------- FormOperator
class FormOperator(ClosedOperator):
    """A_a: the self-adjoint operator of a SobolevForm a on L2(D) (spaces/constructions.py), A_a >= 1, with compact
    resolvent (bounded D: Rellich; closed manifolds). Its interior expression is (1 - l^2 Delta)^k; its domain D(A_a)
    carries the natural conditions (conditions(part)). The graph norm of D(A_a) is equivalent to ||A_a u|| (A_a >= 1), so
    the restricted operator domain with its graph norm is the space domain_space() = pg.Sobolev(A_a, 2k)."""

    def __init__(self, form):
        self.form, self.order = form, 2 * form.k
        self.name = f"A[{form.label()}]"

    def ambient(self):
        from ..spaces.base import L2
        return L2(self.form.domain)

    @property
    def length_scale(self):
        return self.form.length_scale

    def key(self):
        return ("A",) + self.form.key()

    def __eq__(self, other):
        return isinstance(other, FormOperator) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def conditions(self, part):
        return self.form.conditions(part)

    def power_conditions(self, p, part):
        """The boundary conditions of D(A^p) at a part (integer p >= 1): B(L^i u) = 0 for the conditions B of D(A) and
        i < p, L = (1 - l^2 d_n^2)^k the interior expression (one-dimensional reading, d_n^2 = d^2): [(c, essential)]."""
        L = L_poly(self.length_scale, self.form.k)
        out, Li = [], (1.0,)
        for i in range(int(p)):
            out += [(poly_mul(c, Li), ess and i == 0) for c, ess in self.conditions(part)]
            Li = poly_mul(Li, L)
        return out

    def domain_space(self, tol=1e-14, realisation=None):
        """D(A_a) with the norm ||A_a u|| (equivalent to the graph norm: ||u|| <= ||A_a u||): pg.Sobolev(A_a, 2k)."""
        from ..spaces.sobolev import Sobolev
        return Sobolev(self, self.order, tol=tol, realisation=realisation)

    def dense_reason(self):
        return f"D(A_a) contains the smooth functions with the natural conditions, dense in L2 ({KATO1})"

    def label(self):
        f = self.form
        L = "1 - l²Δ" if f.k == 1 else f"(1 - l²Δ)^{f.k}"
        return f"A = {L} of {f.label()}"

    def operator_claims(self):
        return [Claim("self-adjoint, A >= 1, compact resolvent", THEOREM,
                      f"{KATO1}; the form is coercive with constant 1; Rellich (bounded domain or closed manifold)"),
                _self_adjoint(KATO1),
                Claim("unbounded", THEOREM, f"{UNBOUNDED} (A of order {self.order})", key="unbounded"),
                Claim(f"interior expression (1 - l^2 Delta)^{self.form.k}", THEOREM,
                      "int D^j u : D^j v = (-1)^j int Delta^j u v for v in C_c^infinity; sum_j C(k, j) (-l^2 Delta)^j"),
                Claim("domain: u in V with A u in L2 and the natural conditions", THEOREM, KATO1)]

    def __repr__(self):
        return f"A[{self.form!r}]"


# ---------------------------------------------------------------------------------------------- Sturm-Liouville
class SturmLiouville(ClosedOperator):
    """L u = w^-1(-(p u')' + q u) on (a, b) with separated end conditions and interfaces (module docstring), on
    L2((a, b), w dx)."""

    def __init__(self, interval, p, q=0.0, w=1.0, left="dirichlet", right="dirichlet", interfaces=(), name="L",
                 _radial=None):
        if isinstance(interval, str):
            I = Interval(interval)
        elif isinstance(interval, (tuple, list)):
            I = Interval(float(interval[0]), float(interval[1]))
        else:
            I = interval                                           # an Interval
        self.interval, self.a, self.b, self.name = I, I.a, I.b, name
        self.p_src, self.q_src, self.w_src = p, q, w
        self.p, self.q, self.w = _coef(p, "p"), _coef(q, "q"), _coef(w, "w")
        self.left, self.right = as_condition(left), as_condition(right)
        ifs = np.unique(np.asarray(interfaces, float).ravel())
        self.interfaces = ifs[(ifs > self.a) & (ifs < self.b)]
        self.radial = _radial                 # (s,) for the per-degree radial Laplacian (p = w = r^2, q = s(s + 1))
        for side, c in (("left", self.left), ("right", self.right)):
            if c[0] == "regular" and not (_radial is not None and side == "left" and self.a == 0):
                raise ValueError("'regular' names the singular end r = 0 of a ball's radial operator; a regular end "
                                 "takes dirichlet, neumann or robin")
        self._premises = self._check_premises()

    def _check_premises(self):
        """(status, note) of the premises of the facts (module docstring): a bounded interval, p and w > 0 bounded away
        from 0 and bounded, q bounded (below), beta >= 0 (as_condition). THEOREM when decided exactly (numbers, tables;
        the radial Laplacian's own r^2 and s(s + 1)), CHECKED when a callable was sampled; a coefficient found not
        positive is refused with the choices (plan 071 P2, review item 3)."""
        if self.radial is not None:
            return THEOREM, ""
        a, b = self.a, self.b
        if not (math.isfinite(a) and math.isfinite(b) and a < b):
            raise Refused(f"{self.name}: a regular Sturm-Liouville problem is on a bounded interval, not ({a:g}, {b:g})",
                          needs=("a bounded interval (a, b)",))
        sampled = []
        for nm, src, f in (("p", self.p_src, self.p), ("w", self.w_src, self.w), ("q", self.q_src, self.q)):
            lo, hi, exact = _extremes(src, f, a, b, self.interfaces)
            if not exact:
                sampled.append(nm)
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise Refused(f"{self.name}: the coefficient {nm} is not finite on [{a:g}, {b:g}] (a regular problem "
                              "has bounded coefficients)", needs=(f"{nm} bounded on [a, b]",))
            if nm in ("p", "w") and not lo > 0:
                where = "takes" if exact else "is sampled at"
                raise Refused(
                    f"{self.name}: the facts of a Sturm-Liouville operator (self-adjoint, bounded below, compact "
                    f"resolvent) need {nm} > 0 bounded away from 0 on [{a:g}, {b:g}]; {nm} {where} {lo:.6g} there",
                    needs=(f"{nm} > 0 on [a, b] (p > 0 and w > 0: a regular problem)",
                           "for p < 0 throughout, -L instead: SturmLiouville(interval, -p, -q, w) with the Robin "
                           "coefficients negated (L is then bounded above)",
                           "a sign-changing p or w (an indefinite problem) is not built"))
        if not sampled:
            return THEOREM, ""
        return CHECKED, (f"; premises p, w > 0 bounded away from 0 and q bounded: {', '.join(sampled)} sampled at "
                         f"{SAMPLES} points of [{a:g}, {b:g}] (a passing check, not a proof)")

    def premises(self):
        return self._premises

    def ambient(self):
        """L2((a, b), w dx)."""
        from ..measures import DensityMeasure, LebesgueMeasure
        from ..spaces.base import L2
        if isinstance(self.w_src, (int, float, np.number)) and float(self.w_src) == 1.0:
            return L2(self.interval)
        return L2(DensityMeasure(LebesgueMeasure(self.interval), self.w, name="w"))

    @classmethod
    def radial_laplacian(cls, a, b, s, inner=None, outer="dirichlet", centre=None):
        """L_s on L2((a, b), r^2 dr) (module docstring); a = 0 is a ball's centre (inner must then be None)."""
        s = int(s)
        if s < 0:
            raise ValueError("degrees are >= 0")
        if a == 0:
            if inner is not None:
                raise ValueError("a ball has no inner boundary; its centre is r = 0")
            if s == 0 and centre != "regular":
                raise Refused("at r = 0 the degree-0 radial operator is limit circle (1 and 1/r are both square "
                              "integrable against r^2 dr): a condition must be stated", needs=["centre='regular' "
                              "(u bounded at the centre: the Friedrichs extension, the Laplacian of the ball)"])
            left = "regular"
        else:
            left = "dirichlet" if inner is None else inner
        ss = float(s * (s + 1))
        return cls((float(a), float(b)), lambda r: np.asarray(r, float) ** 2, ss, lambda r: np.asarray(r, float) ** 2,
                   left=left, right=outer, name=f"L_{s}", _radial=(s,))

    # spectrum -----------------------------------------------------------------------------------------
    def spectrum(self):
        """The realised spectrum (realisations/registry.py chooses: closed forms or Galerkin brackets)."""
        if getattr(self, "_spectrum", None) is None:
            from ..realisations.registry import spectrum
            self._spectrum = spectrum(self)
        return self._spectrum

    def bessel_potential(self, kappa, q, modes, space=None, name="C"):
        """(1 + kappa^-2 L)^(-q/2) as a pg.Covariance on L2((a, b), w dx) with its truncation bounds (regular problems;
        on balls and shells use pg.Laplacian(...).bessel_potential). realisations/spectra/laplacian_covariance.py."""
        from ..realisations.spectra.laplacian_covariance import interval_bessel_potential
        return interval_bessel_potential(self, float(kappa), float(q), modes, space, name)

    def free_ends(self):
        """The number of regular ends that are not Dirichlet (interlacing: lambda_n >= lambda_(n - free)^DD)."""
        return sum(1 for c in (self.left, self.right) if c[0] in ("neumann", "robin"))

    def domain_space(self):
        raise Unsupported(f"the graph-norm space of {self.name} (D(L) with ||u||^2 + ||L u||^2) is not built")

    def dense_reason(self):
        return f"the domain of the operator of a densely defined closed form (Kato VI.2.1; {ZETTL})"

    # description --------------------------------------------------------------------------------------
    def label(self):
        what = f"-r^-2 (r^2 u')' + {self.radial[0] * (self.radial[0] + 1)} r^-2 u" if self.radial else \
            "w^-1 (-(p u')' + q u)"
        return (f"{self.name} = {what} on ({self.a:g}, {self.b:g}); left {describe(self.left)}, right "
                f"{describe(self.right)}" + (f", interfaces {list(self.interfaces)}" if len(self.interfaces) else ""))

    def operator_claims(self):
        st, note = self.premises()
        cl = [Claim("self-adjoint, bounded below, compact resolvent; simple eigenvalues", st,
                    f"the closed semibounded form int p u'v' + q u v + Robin terms (Kato VI.2.1); separated conditions "
                    f"({ZETTL})" + note),
              _self_adjoint(f"the operator of a closed symmetric semibounded form (Kato VI.2.1; {ZETTL})" + note, st),
              Claim("unbounded", st, f"{UNBOUNDED} ({ZETTL})" + note, key="unbounded"),
              Claim("the n-th eigenfunction has n - 1 zeros in the open interval", st,
                    f"Sturm oscillation ({ZETTL})" + note)]
        if self.radial:
            s = self.radial[0]
            if self.a == 0:
                cl.append(Claim("no condition at r = 0" if s >= 1 else "the regular (Friedrichs) condition at r = 0",
                                THEOREM, "limit point for s >= 1 (r^(-s-1) is not in L2(r^2 dr) near 0)" if s >= 1 else
                                "limit circle at s = 0: stated, u bounded (the Laplacian of the ball)"))
            cl.append(Claim(f"lambda_n >= s(s + 1)/b^2 + ((n - {self.free_ends()}) pi / (b - a))^2 for n > "
                            f"{self.free_ends()}", DERIVED, "v = r u, min-max and interlacing (EXP-030 NOTES.md "
                            "section 5(d))"))
        return cl

    def __repr__(self):
        return f"SturmLiouville[{self.label()}]"


# ---------------------------------------------------------------------------------------------- the Laplacian
NOT_BUILT = {
    "disk": ("pygeoinf0.functions.disk.laplace_eigen(D, n=..., m=..., bc=...) (after import pygeoinf0.functions): the "
             "function bank's Dirichlet and Neumann eigenfunctions J_m(k r) cos/sin(m phi) with their eigenvalues "
             "(scipy's Bessel zeros)",
             "pg.Sobolev(D, s, length_scale=l): the restriction construction, without the Laplacian's spectrum"),
}


class Laplacian(ClosedOperator):
    """-Delta on L2(D) with boundary conditions (module docstring): Laplacian(D, bc), centre="regular" on a ball for
    degree 0. lap.region is D, lap.domain the ambient L2(D), lap.dom D(-Delta) (dense, not closed)."""

    def __init__(self, domain, bc=None, centre=None):
        from ..realisations.registry import laplacian_geometry
        kind = laplacian_geometry(domain)
        self.region, self.centre, self.geometry, self.name = domain, centre, kind, "-Δ"
        if kind not in ("interval", "box", "ball", "shell", "circle", "sphere", "torus"):
            exists = ("; what exists: " + "; ".join(NOT_BUILT[kind])) if kind in NOT_BUILT else ""
            raise Unsupported(f"the Laplacian of {domain!r} is not built: its spectrum is not realised (built on "
                              f"intervals, boxes, balls, shells, circles, spheres and flat tori){exists}")
        parts = tuple(getattr(domain, "boundary_parts", ("boundary",)))
        if kind in ("circle", "sphere", "torus"):
            if bc:
                raise ValueError(f"{domain!r} is a closed manifold: it has no boundary and takes no condition")
            self.bc = {}
        else:
            if bc is None:
                raise ValueError(f"state the condition on every boundary part of {domain!r}: {parts} (dirichlet, "
                                 "neumann, ('robin', beta)), or one condition for all of them")
            if not isinstance(bc, dict) or (len(bc) == 1 and next(iter(bc)) in KINDS):
                bc = {p: bc for p in parts}                   # one condition for every part
            unknown = set(bc) - set(parts)
            if unknown:
                raise ValueError(f"{domain!r} has the boundary parts {parts}; not {sorted(unknown)}")
            missing = [p for p in parts if p not in bc]
            if missing:
                raise ValueError(f"state the condition on every boundary part: missing {missing} (dirichlet, neumann, "
                                 "('robin', beta))")
            self.bc = {k: as_condition(bc[k]) for k in parts}
            if any(c[0] == "regular" for c in self.bc.values()):
                raise ValueError("'regular' names the centre of a ball (centre='regular'), not a boundary condition")
        if kind in ("ball", "shell"):
            a, b = (float(x) for x in domain.reference[0])
            self.a, self.b = a, b
            if a == 0:
                self.inner, self.outer = None, self.bc["boundary"]
            else:
                self.inner, self.outer = self.bc["inner"], self.bc["outer"]
        self._radial = {}

    def ambient(self):
        from ..spaces.base import L2
        return L2(self.region)

    def radial(self, s):
        """L_s, the degree-s radial operator of a ball or a shell (module docstring)."""
        if self.geometry not in ("ball", "shell"):
            raise Unsupported(f"radial operators are those of a ball or a shell; {self.region!r} is a {self.geometry}")
        s = int(s)
        if s not in self._radial:
            self._radial[s] = SturmLiouville.radial_laplacian(self.a, self.b, s, self.inner, self.outer, self.centre)
        return self._radial[s]

    def spectrum(self):
        """The realised spectrum: eigenvalues(n) (with multiplicity), brackets(n), claims (realisations/registry.py
        chooses, realisations/spectra/laplace_spectra.py)."""
        if getattr(self, "_spectrum", None) is None:
            from ..realisations.registry import laplacian_spectrum
            self._spectrum = laplacian_spectrum(self)
        return self._spectrum

    def bessel_potential(self, kappa, q, degrees, modes=None, rtol=None, weights=None, space=None, name="C"):
        """C = (+)_(s in degrees) c_s (1 + kappa^-2 L_s)^(-q/2) (x) I_(H_s) as a pg.Covariance on L2 of a ball or a shell
        in the radial x harmonic realisation (space, or built), with modes eigenpairs per degree (or as many as make
        the trace's omitted part <= rtol times the known part); weights c_s (default 1). See
        realisations/spectra/laplacian_covariance.py for the claims."""
        if self.geometry not in ("ball", "shell"):
            raise Unsupported(f"Laplacian.bessel_potential is built on balls and shells (degree by degree); on an "
                              f"interval use SturmLiouville(...).bessel_potential; {self.region!r} is a {self.geometry}")
        from ..realisations.spectra.laplacian_covariance import bessel_potential
        return bessel_potential(self, float(kappa), float(q), degrees, modes, rtol, weights, space, name)

    def dense_reason(self):
        return ("the domain of the operator of the closed Dirichlet/Neumann/Robin form, dense in L2 (Kato VI.2.1)" if
                self.bc else "the smooth functions are dense in L2 of a closed manifold and lie in D(-Delta)")

    def label(self):
        bits = ", ".join(f"{k} {describe(v)}" for k, v in self.bc.items())
        return f"-Δ on {self.region!r}" + (f" ({bits})" if bits else "")

    def operator_claims(self):
        if self.bc:
            why = ("the Dirichlet/Neumann/Robin (beta >= 0) Laplacian of a bounded Lipschitz domain, the operator of a "
                   "closed form (Kato VI.2; Rellich)")
        else:
            why = "the Laplace-Beltrami operator of a closed manifold (Kato VI.2; Rellich on a compact manifold)"
        cl = [Claim("self-adjoint, >= 0, compact resolvent", THEOREM, why), _self_adjoint(why),
              Claim("unbounded", THEOREM, f"{UNBOUNDED}", key="unbounded")]
        if self.geometry in ("ball", "shell"):
            cl.append(Claim("-Delta = (+)_s L_s (x) I_(H_s)", THEOREM, "it commutes with rotations (rotation-invariant "
                            "conditions on spheres r = const); Schur's lemma (EXP-030 Theorem G2.1)",
                            key="rotation invariant"))
        return cl

    def __repr__(self):
        return self.label()


__all__ = ["FormOperator", "SturmLiouville", "Laplacian", "as_condition", "describe"]
