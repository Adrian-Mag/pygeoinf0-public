"""C(K) on closed pieces (plan 035 P5, design D5's C(K) row), and the refusals that Banach spaces share (L^p).

    R = pg.Restrict(C, K_k)          (= C.restrict_to(K_k)) for C = pg.Continuous("[a, b]"): R: C(K) -> C(K_k), u -> u|K_k
    R.dual(mu)                       the dual map: a measure on K_k is the functional u -> int u dmu on C(K)
    P.functionals(C, mu, side=)      {piece: f -> int f d(mu|K_k) on C(K_k)} with sum_k l_k(R_k u) = int u dmu

The pieces K_k of a partition of K = [a, b] are closed and overlap at the seams. C(K) is the subspace of
(+)_k C(K_k) of the families with equal values on the seams, and ||u|| = max_k ||R_k u|| (the maximum over a finite
union is the largest of the maxima): R_k is linear with norm 1 (a theorem), and the glue of matching pieces is a
continuous function (theorem). The package's direct sums are Hilbert sums, so the max-norm product (+)_k C(K_k) is not
built: C.restrictions(P) and a PiecewiseSpace with a C field are refused by name, and R_k works piece by piece.

Dual side (Riesz-Markov: the dual of C(K) is the finite signed measures): the dual map of R_k carries a measure on K_k
to the same measure on K, and int u dmu = sum_k int R_k u d(mu|K_k) for a split that counts every seam mass once. A
point mass on a seam lies in both closed pieces, so the split needs side= (Partition.split; never halved, never
counted twice); on C(K) itself both sides give the same number (u is single-valued), stated as a claim. Extension by
zero leaves C(K) and is refused; a norm-preserving continuous extension exists (Tietze; constant continuation on an
interval) and is not built.
"""
from ..core.claims import CHECKED, THEOREM, Claim, Claims
from ..core.errors import Refused, SupportOutsideDomain


def piece_space(X, D):
    """C(D) for a closed subinterval D of X's interval, with X's tolerance."""
    from ..spaces.continuous import ContinuousSpace
    from ..sets import Interval
    from ..domains.pieces import meet
    D = Interval(D) if isinstance(D, str) else D
    J = meet(X.domain, D)
    if J is None or not (J is D or J == D):
        raise SupportOutsideDomain(f"{D!r} is not a subinterval of {X.domain!r} of positive length")
    if not (D.left_closed and D.right_closed):
        raise ValueError(f"C({D}) is not a normed space on a set that is not closed: restrict to [{D.a:g}, {D.b:g}]")
    return ContinuousSpace(D, X.tol)


def claims(X, D):
    return [Claim("linear", THEOREM, "u -> u|D", key="linear"),
            Claim("bounded, norm 1", THEOREM, "max_D |u| <= max_K |u|, with equality for constants", key="bounded"),
            Claim("over closed pieces K = union K_k: ||u|| = max_k ||R_k u||; C(K) is the subspace of (+)_k C(K_k) "
                  "with equal values on the seams", THEOREM, "the maximum over a finite union is the largest maximum; "
                  "continuous pieces that agree on the shared endpoints glue to a continuous function", key="pieces"),
            Claim("the dual map takes a measure nu on D to the same measure on K: int R u dnu = int u dnu", THEOREM,
                  "Riesz-Markov; the measures on D are the measures on K carried by D", key="adjoint"),
            Claim("R u: the formula evaluated on D, or the realisation of u evaluated on D and refitted there in the sup "
                  "norm", CHECKED, "realisations/series/sup.py (continuity and sup errors checked on the refit)",
                  key="realisation")]


def refuse_extension(R):
    return Refused(f"{R.name}: extension by zero from {R.codomain} is not continuous at the interior ends of the piece "
                   "(unless the function vanishes there), so it does not map into C(K)",
                   needs=("a norm-preserving continuous extension (Tietze; constant continuation on an interval): not "
                          "built", "R.dual(mu): the dual map on measures"))


def refuse_adjoint(R):
    if R.lp:
        return Refused(f"{R.name} maps between Banach spaces: its adjoint is the dual map L^p'(D) -> L^p'(Omega), "
                       "extension by zero (a theorem, in R.claims), and the package has no dual-space objects for L^p",
                       needs=("dual spaces of L^p (not in the package)", "R.extension: extension by zero "
                              "L^p(D) -> L^p(Omega), the isometric right inverse of R"))
    return Refused(f"{R.name} maps between Banach spaces without inner products: its adjoint is the dual map on finite "
                   "measures, nu on the piece -> nu on K (a theorem, in R.claims), not an operator between the spaces",
                   needs=("R.dual(mu): the functional u -> int u dmu on C(K) of a measure on the piece",))


def dual(R, mu):
    """The dual map of R: the measure mu on the piece as the functional u -> int u dmu on R.domain."""
    from ..measures import as_measures
    m = as_measures(mu, 1)[0][0]
    try:
        inside = m.restricted(R.piece)
    except ValueError:                           # on an end of the piece: in the closed piece
        inside = m.restricted(R.piece, "below")
        inside = m.restricted(R.piece, "above") if inside is None else inside
    if inside is not m:
        raise SupportOutsideDomain(f"{m} is not carried by the piece {R.piece!r}")
    return R.domain.functional(m, name=f"{R.name}'({m})")


class PieceForms(dict):
    """{piece: functional on C(K_k)}, from Partition.functionals; called on u in C(K) it gives sum_k l_k(R_k u)."""

    def __call__(self, u):
        return sum(l(self.restrictions[n](u)) for n, l in self.items())

    @property
    def claims(self):
        return Claims(list(self.split.claims) + [Claim(
            f"sum_k int R_k u d(mu|K_k) = int u dmu on {self.space}", THEOREM, "the parts sum to mu with every seam "
            "mass once (the split's claim), and R_k u = u on K_k; on C(K) a seam mass gives the same number from either "
            "side (u is single-valued)", key="functionals")])


def functionals(X, partition, mu, side=None):
    """Partition.functionals for X = C(K) (C.split_functional): {piece: u_k -> int u_k d(mu|K_k) on C(K_k)} over the
    pieces mu charges."""
    from ..spaces.continuous import ContinuousSpace
    from .pieces import Restrict
    if not isinstance(X, ContinuousSpace):
        raise Refused(f"P.functionals is for C(K) (functionals are measures); on {X} a functional is a Riesz "
                      "representer", needs=("pg.PiecewiseSpace(P, fields).functional(field, mu) and .piece_functionals",))
    P = partition.flat()
    split = P.split(mu, side)
    out = PieceForms()
    out.restrictions = {n: Restrict(X, P[n], label=n) for n in split}
    for n, m in split.items():
        out[n] = out.restrictions[n].codomain.functional(m, name=f"∫_{n} · d({m})")
    out.split, out.space = split, X
    return out
