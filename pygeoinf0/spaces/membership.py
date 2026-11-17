"""When a function that carries a Regularity (a bank function) is an element of a space (plan 021; EXP-037 DESIGN.md 3).
In core since plan 065 (A1): every M.element(f) consults it, so f.element(M) and M(f) give one answer.

    check(f, M) -> (ok, reason, claim)
    has_rule(M)             whether a rule below applies to M (rule 5 is the absence of one)
    refusal(f, M, reason)   the Refused for a function the rules put outside M, with the choices

Rules (theorems; the regularity is the function's claim):
1. L2(mu) and Lp(mu) on a bounded domain: bounded measurable functions are members.
2. C(K): continuous functions (order >= 0).
3. H^s(D): C-infinity up to the boundary: every s. C^k with bounded derivatives on the closure of a bounded Lipschitz
   domain: s <= k. Piecewise smooth with derivatives continuous up to order m across finitely many smooth interfaces
   (m = -1: jumps): s < m + 3/2.
4. Boundary conditions of the space's construction (e.g. spectral Dirichlet: elements vanish on the boundary): the
   function must vanish there to all orders (interior support), or its claimed vanishing order must cover them.
5. Any other space: no rule; f.element(M) refuses with that reason, M(f) treats f as a plain callable.
"""
from ..core.claims import THEOREM, Claim

REF_SOBOLEV = ("Adams & Fournier 2003, Sobolev Spaces, 3.1-3.2 (C^k with bounded derivatives on the closure of a bounded "
               "Lipschitz domain lies in W^{k,2}); piecewise smooth functions: Grisvard 1985, Elliptic Problems in "
               "Nonsmooth Domains, 1.4")


def _yes(statement, why):
    return True, "", Claim(statement, THEOREM, why, key="membership")


def _no(why):
    return False, why, None


def has_rule(M):
    from .continuous import ContinuousSpace
    from .sobolev import SobolevSpace
    from .base import L2, LpSpace
    return isinstance(M, (L2, LpSpace, ContinuousSpace, SobolevSpace))


def refusal(f, M, reason):
    from ..core.errors import Refused
    return Refused(f"{f.name} is not shown to be in {M}: {reason}",
                   needs=("a space its regularity puts it in (L2 for a function with jumps; H^s with s below its bound)",
                          "a function of the regularity the space needs"))


def check(f, M):
    from .continuous import ContinuousSpace
    from .sobolev import SobolevSpace
    from .base import L2, LpSpace
    reg = f.regularity
    if isinstance(M, (L2, LpSpace)):
        return _yes(f"{f.name} is in {M}", "bounded measurable functions on a finite measure space are in every L^p")
    if isinstance(M, ContinuousSpace):
        if reg.order >= 0:
            return _yes(f"{f.name} is in {M}", "a continuous function on a compact set")
        return _no("it has jumps (regularity order -1); C(K) needs a continuous function")
    if isinstance(M, SobolevSpace):
        s = float(M.s)
        smax, strict = reg.sobolev_bound()
        inside = s < smax if strict else s <= smax
        if not inside:
            bound = f"s {'<' if strict else '<='} {smax:g}"
            return _no(f"its regularity ({reg.describe()}) puts it in H^s for {bound}, not s = {s:g}")
        cons = M.construction
        from .constructions import boundary_parts
        conds = []
        for part in boundary_parts(M.domain):
            conds += [(part, c) for c, _ in cons.element_conditions(part)]
        if conds:
            v = reg.vanishes_at_boundary
            needed = 1 + max(_order(c) for _, c in conds)
            if not (v == "all orders" or (isinstance(v, int) and v >= needed)):
                parts = sorted({p for p, _ in conds})
                return _no(f"the elements of {M} satisfy boundary conditions on {', '.join(parts)} (derivatives up to "
                           f"order {needed - 1}); {f.name} claims no such vanishing")
        return _yes(f"{f.name} is in {M}", REF_SOBOLEV + (f"; boundary conditions met ({reg.describe()})" if conds else ""))
    return _no(f"no membership rule for {type(M).__name__}; use M(f) and state the claim yourself")


def _order(c):
    from .constructions import order
    return int(order(c))
