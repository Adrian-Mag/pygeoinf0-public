"""Inverses of operators (plan 068 P3, review A12; from operators/linear.py and relations/linear.py): the inverse, the
pseudo-inverse A⁺ with the threshold the caller chose, and the minimum-norm map, its finite-rank case, live together.

    Inverse(A, solver)        A^-1 applied with a solver (A.inverse(solver), A.inv): the factorisation is computed on first
                              use and kept; its claims state the conditioning, what was dropped or refused, and what it
                              inherits from A's stated error (solvers.inherited)
    PseudoInverse(A, s, rcond) A⁺ = A.pseudo_inverse(rcond) of a square operator: the threshold is the caller's choice and
                              its claims state it with the number of eigenvalues dropped; a non-square operator is refused
                              when it is built, with the pseudo-inverses defined for it named
    MinimumNorm(G, mul)       y -> the element of least norm of {x : G x - y in mul} = G* P (P G G* P)^+ P y for G: X -> R^n
                              (P the projection onto mul^perp; mul = None or {0}: G*(G G*)^-1 y): the operator part of the
                              converse of a linear relation (relations/linear.py), with its threshold (the rounding floor)
                              and what it dropped

The stated error of an inverse built from an operator with a stated error follows the rule of operators/error.py
(inverse_error: n^2 e / (1 - n e), n = ||A~^-1||, when n e < 1; UNKNOWN otherwise; a pseudo-inverse: UNKNOWN).
"""
from ..core.claims import DERIVED, UNKNOWN, Claim, Claims, by_rule
from ..core.logic import Truth
from .linear import LinearOperator
from ..spaces.identity import require_realised


class Inverse(LinearOperator):
    """A^-1, applied with a solver; the factorisation is computed on first use and kept."""

    def __init__(self, A, solver=None):
        super().__init__(A.codomain, A.domain, name=f"({A.name})⁻¹")
        self.A = A
        self.recommendation = None
        if solver is None:
            from ..solvers import recommend
            self.recommendation = recommend(A)
            solver = self.recommendation.solver
        self.solver = solver
        self._fact, self._inherited = None, None
        self._sa = A._sa
        if A._pos and A._pos[0] == "pd":
            self._pos = ("pd", by_rule(A._pos[1]), "the inverse of a positive definite operator")

    def factorisation(self):
        if self._fact is None:
            self._fact = self.solver.factor(self.A)
            self._inherited = None
            pseudo = getattr(self._fact, "dropped", 0) or getattr(self.solver, "rcond", None)
            if pseudo:                                     # a pseudo-inverse: no rule here (plan 065 A6), stated as such
                from ..solvers.direct import stated_error
                st = stated_error(self.A)
                if st is not None:
                    self._inherited = (
                        f"inherits the stated error of {self.A.name}: ||dA|| <= {st[0]:.3g} ({st[2]}); through a "
                        "pseudo-inverse its effect on a solve is not bounded here: a perturbation of that size can move "
                        "eigenvalues across the threshold, and the perturbation bounds of pseudo-inverses need the rank "
                        "unchanged, which is not checked", UNKNOWN, "no rule applies (plan 065 A6)", float("nan"), True)
            else:                                          # inverses
                from ..solvers.direct import inherited
                self._inherited = inherited(self.A, self._fact)
                if self._inherited is not None and self._inherited[3] >= 1:
                    if self._inherited[4] and getattr(self.solver, "refuse", self.recommendation is not None):
                        from ..solvers.direct import inherited_refusal
                        fact, self._fact = self._fact, None
                        raise inherited_refusal(self.name, self._inherited[0], fact)
                    import warnings

                    from ..core.errors import PrecisionWarning
                    warnings.warn(f"{self.name}: {self._inherited[0]}", PrecisionWarning, stacklevel=3)
        return self._fact

    def _apply(self, y):
        return self.factorisation().solve(y)

    @property
    def claims(self):
        """With the factorisation's numbers: conditioning and digits lost, what was dropped, or why it is refused; and
        what it inherits from the stated error of the operator (plan 054 P3: solvers.inherited)."""
        from ..core.errors import Refused
        extra = []
        try:
            text, why = "; ".join(self.factorisation().info), f"{self.solver!r}"
            if self._inherited is not None:
                st, status, reason = self._inherited[:3]
                extra = [Claim(st, status, reason, key="inherited error")]
        except Refused as e:
            text, why = f"no inverse here: {e.reason}", "its refusal"
        return Claims(list(super().claims) + [Claim(text, DERIVED, why, key="inverse")] + extra)

    def _make_adjoint(self):
        return Inverse(self.A.adjoint, self.solver)

    def report(self):
        f = self.factorisation()
        return f"{self.name} with {self.solver!r}: " + "; ".join(f.info)

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [solver: {self.solver!r}]"


def _square_for_pseudo(A):
    """Refuse A.pseudo_inverse(rcond) for an operator whose domain and codomain differ, naming the pseudo-inverses that
    are defined for it (plan 065 A12: it was built as "(A)⁻¹" and failed on first use)."""
    if A.domain == A.codomain:
        require_realised(A.domain, A.codomain, A.name)
        return
    from ..core.errors import Refused
    raise Refused(f"{A.name}: {A.domain} → {A.codomain} is not square; A.pseudo_inverse(rcond) thresholds the eigenvalues "
                  "of a square operator", needs=[
                      f"{A.name}.adjoint @ ({A.name} @ {A.name}.adjoint).pseudo_inverse(rcond): the minimum-norm "
                      "pseudo-inverse G*(G G*)^+ with your threshold",
                      f"({A.name}.adjoint @ {A.name}).pseudo_inverse(rcond) @ {A.name}.adjoint: the least-squares "
                      "pseudo-inverse (G* G)^+ G*",
                      f"pg.LinearRelation({A.name}).converse().operator_part(): the minimum-norm map (into R^n, nothing "
                      "dropped beyond the rounding floor)"])


def pseudo_inverse(A, rcond):
    """A.pseudo_inverse(rcond): refused for a non-square A when built, else the PseudoInverse with that threshold, on A's
    structure (solvers.pseudo_solver)."""
    from ..solvers import pseudo_solver
    _square_for_pseudo(A)
    return PseudoInverse(A, pseudo_solver(A, rcond), rcond)


class PseudoInverse(Inverse):
    """A⁺ = A.pseudo_inverse(rcond) of a square operator: the threshold is the caller's choice and its claims state it,
    with the number of eigenvalues dropped (plan 065 A12). MinimumNorm (below) is the finite-rank case
    G*(G G*)^+ of a map into R^n."""

    def __init__(self, A, solver, rcond):
        _square_for_pseudo(A)
        super().__init__(A, solver)
        self.name, self.rcond = f"({A.name})⁺", rcond
        self._pos = A._pos and ("psd", by_rule(A._pos[1]), "the pseudo-inverse of a positive semidefinite operator")

    @property
    def claims(self):
        out = list(super().claims)
        f = self._fact
        if f is not None:
            n = getattr(f, "n", None)
            count = (f"{f.dropped} of {n} eigenvalue(s) dropped, |lam| <= {f.cut:.3g}" if n is not None else
                     "; ".join(f.info))
            out.append(Claim(f"the pseudo-inverse with threshold rcond = {self.rcond:g} relative to the largest "
                             f"eigenvalue: {count}", DERIVED, f"{self.solver!r}: the threshold is the caller's choice",
                             key="pseudo-inverse"))
        return Claims(out)

    def _make_adjoint(self):
        return PseudoInverse(self.A.adjoint, self.solver, self.rcond)

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [pseudo-inverse, rcond = {self.rcond:g}: {self.solver!r}]"


class MinimumNorm(LinearOperator):
    """y -> the element of least norm of {x : G x - y in mul} = G* P (P G G* P)^+ P y, P the projection onto mul^perp
    (G: X -> R^n bounded). With mul = {0} it is G*(G G*)^-1 y, computed exactly as G.adjoint((G @ G.adjoint).inv(y))."""

    why_linear = "G* after a solve with G G* (and projections)"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return [self.G]

    def __init__(self, G, mul=None, name=None):
        super().__init__(G.codomain, G.domain, name=name or f"{G.name}⁺")
        self.G, self.nuisance = G, mul
        if mul is None or getattr(mul, "dim", None) == 0:      # {0}: nothing to project away
            self.P, self.Ainv = None, (G @ G.adjoint).inv
        else:
            from ..solvers import Eigh
            self.P = mul.perp().projector()
            A = self.P @ (G @ G.adjoint) @ self.P             # the Gram G G*, with its stated error (plan 065 A6)
            A._sa = (True, DERIVED, "P G G* P")
            self.Ainv = A.inverse(Eigh("rounding"))
        from .claim_rules import status
        held, st = status(G, "linear")                      # linear: the rule, from G's claim (plan 065 A3)
        self._claims = [Claim("bounded", st, "linear on a finite-dimensional domain", key="bounded")] if (
            held is Truth.TRUE and st != UNKNOWN) else []

    @property
    def claims(self):
        """With mul != {0}: the threshold of (P G G* P)^+ and the count dropped, beside dim mul (the zero eigenvalues by
        construction); a count above dim mul means directions of G G* at the rounding floor were dropped too."""
        out = list(super().claims)
        inh = self.Ainv.claims.find("inherited error")      # what the solve inherits from G's stated error (plan 065 A6)
        out += [inh] if inh is not None else []
        if self.P is None:
            return Claims(out)
        from ..solvers.direct import rounding_claim
        dim = getattr(self.nuisance, "dim", "unknown")
        return Claims(out + [rounding_claim(self.Ainv, "(P G G* P)^+", f"; dim mul = {dim}")])

    def factors(self):
        """(G*, (G G*)^-1) or (G*, P (P G G* P)^+ P): what MinimumNorm is the product of (operators/error.py reads its
        stated error by the composition rule; a pseudo-inverse of a Gram with a stated error makes it unknown)."""
        return (self.G.adjoint, self.Ainv) if self.P is None else (self.G.adjoint, self.P @ self.Ainv @ self.P)

    def _apply(self, y):
        if self.P is None:
            return self.G.adjoint(self.Ainv(y))
        return self.G.adjoint(self.P(self.Ainv(self.P(y))))

    def _make_adjoint(self):
        if self.P is None:
            return LinearOperator(self.codomain, self.domain, apply=lambda x: self.Ainv(self.G(x)), name=f"{self.name}*")
        return LinearOperator(self.codomain, self.domain, apply=lambda x: self.P(self.Ainv(self.P(self.G(x)))),
                              name=f"{self.name}*")
