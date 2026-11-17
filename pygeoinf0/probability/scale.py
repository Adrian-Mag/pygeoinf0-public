"""The posterior above settings.dense_max (plan 005 P5): variances and samples of post = prior.condition(G, d, noise)
from one randomised data-space decomposition, with the error of each variance bounded and stated.

With F the whitening of the noise, W = F G C G* F^T (C the prior covariance) and w_u = F G C u:

    Var_post <x, u> = <C u, u> - w_u^T (W + I)^-1 w_u.

Errors (plan 005 P5 design; the derivation in plans/active/005-scale-ladder.md). The products with G C G* may come from
a fast engine (settings.engine above dense_max) with an estimated relative error delta: the operator computed is
W^ = W + Delta, ||Delta|| <~ delta ||W^||, and w_u, <C u, u> carry errors eta <~ delta ||w^||, delta_t <~ delta t^. The
randomised decomposition gives a Nystrom W_k with ||W^ - W_k|| <= E (probability >= 1 - 1e-10); W_k <= W^ holds in
exact arithmetic but only to rounding in double (plan 054 P3, solvers/dataspace/partial.py), so E = W^ - W_k is of either sign.
With R = (W_k + I)^-1 and b = (W^ + I)^-1 w^, a = (W + I)^-1 w^:

    Nystrom      |w^T [R - (W^ + I)^-1] w| <= E ||R w||^2 / (1 - E)              (the resolvent identity, E < 1)
    engine       |w^T [(W + I)^-1 - (W^ + I)^-1] w| <= ||Delta|| ||a|| ||b||     ((W + I)^-1 Delta (W^ + I)^-1)
                 ||b|| <= min(||w||, ||R w|| / (1 - E)),  ||a|| <= min(||w||, (1 + ||Delta||) ||b||)
    data image   2 ||eta|| ||a|| + ||eta||^2 + delta_t

so the stated variance error is the sum (an estimate where an engine's delta is: its error is estimated, not bounded;
the Nystrom part alone is a probabilistic bound). A functional whose Nystrom std bound exceeds settings.accuracy times
its std is computed by deflated block CG on (W^ + I) z = w^ (exact for W^; the engine part remains and is stated).

Route by cost (plan 005 P5 review): the decomposition's rank is capped at about the number of functionals p first asked
for (or the memory cap). When the bound on ||W^ - W_k|| is still above the tail there (a spectrum that does not fall
below it), every functional is computed by block CG deflated by that rank-p approximation, whose cost is a few products
per functional; the claims say which route was taken. Samples follow Matheron's rule, x0 - C G* S^-1 (G x0 + e0),
S = F^-1 (W^ + I) F^-T, every solve by the same block CG.
"""
import numpy as np

from ..core.claims import CHECKED, DERIVED, Claim, Claims

EXACT_MAX = 256       # functionals recomputed exactly when their bound fails the rule (plan 005 P5 review, question 2)


def product_error(G):
    """(delta, how): the estimated relative error of the products with G G* behind the decomposition (0 for exact
    energies or a plain matrix), from the engine that settings.engine chose for the representers' atoms."""
    from ..settings import settings
    K = getattr(G, "kernels", None)
    engine = getattr(getattr(G.domain, "backend", None), "engine", None)
    atoms = getattr(getattr(getattr(K, "realisation", None), "kx", None), "atoms", None)
    if engine is None or atoms is None or not hasattr(engine, "route") or getattr(atoms, "n", 0) <= settings.dense_max:
        return 0.0, "exact products"
    name, eng, _ = engine.route(atoms)
    if name == "exact":
        return 0.0, "exact energies streamed"
    # the spectral engine's estimate lives on the plan it stores on the atom set (Atoms.spectral), not on the engine
    est = getattr(getattr(atoms, "spectral", None), "est", None) if name == "spectral" else \
        getattr(getattr(atoms, "hmatrix", None), "rel_error", None)
    if est is None:
        est = settings.accuracy
        return float(est), f"{name} engine, not yet measured: its target settings.accuracy {est:g} taken"
    return float(est), f"{name} engine, estimated relative product error {est:.1e}"


class AtScale:
    """Variances and samples of a Gaussian posterior through a randomised decomposition (module docstring)."""

    def __init__(self, prior, G, noise):
        self.prior, self.G, self.noise = prior, G, noise
        self._dec = None
        self.last_bound = None

    def decomposition(self, p=None):
        """The decomposition (built once; its rank capped at about p, the number of functionals first asked for)."""
        if self._dec is None:
            from ..solvers.dataspace import DataDecomposition
            from ..solvers.dataspace.partial import BLOCK, KMAX, operator
            from .gaussian import noise_metric
            cap = None if p is None else min(KMAX, max(BLOCK, int(p)))
            self._dec = DataDecomposition(self.G, noise_metric(self.noise, self.G.codomain), self.prior.covariance,
                                          route="randomised", rank_cap=cap)
            self.W, self.wt = operator(self._dec)
        return self._dec

    dec = property(decomposition)

    def _pairings(self, fam):
        """(plain cross (n, p): G C u, tt (p,): <C u, u>) for a family of model elements."""
        Cu = self.prior.covariance(fam)
        cross = np.asarray(self.G(Cu).values, float)
        cross = cross.T if cross.shape[0] == len(fam) and cross.ndim == 2 else cross
        tt = np.atleast_1d(np.diag(np.atleast_2d(self.prior.space.inner(Cu, fam))))
        return cross.reshape(self.G.codomain.dim, -1), tt

    def _solve(self, B, tol=1e-12):
        from ..solvers.dataspace.partial import pcg
        dec = self.dec
        return pcg(self.W, dec._U[0], dec.lam, B, tol=tol)

    def variance(self, fam, exact=False):
        """(variances, std bounds) of the members of fam; the route and the total variance bounds in self.last and
        self.last_bound (module docstring)."""
        from ..settings import settings
        dec = self.decomposition(len(fam))
        cross, tt = self._pairings(fam)
        lam, C, _ = dec._spec(cross)
        C = C[:, None] if C.ndim == 1 else C
        E = dec.error
        nw = np.sqrt(np.einsum("kp,kp->p", C, C))                      # ||w^||
        nRw = np.sqrt(np.einsum("kp,k->p", C * C, 1.0 / (lam + 1.0) ** 2))   # ||R w^||
        var = tt - np.einsum("kp,k->p", C * C, 1.0 / (lam + 1.0))
        from ..solvers.dataspace.partial import std_error, two_sided
        fac = float(two_sided(E))
        nys = E * nRw ** 2 * fac                                        # two-sided (plan 054 P3)
        nb = np.minimum(nw, nRw * fac)                                  # ||b|| bound
        std = np.sqrt(np.maximum(var, 0.0))
        capped = E > dec.tail                                           # the rank cap reached first: route by cost
        if exact or capped:
            redo, route = np.arange(len(var)), "cg"
        else:
            fail = np.flatnonzero(std_error(var, nys) > settings.accuracy * std)
            redo, route = fail[:EXACT_MAX], "low rank"
        left = 0 if route == "cg" else len(np.flatnonzero(std_error(var, nys) > settings.accuracy * std)) - len(redo)
        its = 0
        if len(redo):
            w = self.wt.fwd(cross[:, redo])
            Z, its, rel = self._solve(w)
            var[redo] = tt[redo] - np.einsum("kp,kp->p", w, Z)
            nys[redo] = 0.0
            nb[redo] = np.linalg.norm(Z, axis=0)                      # ||b|| itself
        delta, how = product_error(self.G)
        D = delta * float(lam.max(initial=0.0))                         # ||Delta|| ~ delta ||W^||
        na = np.minimum(nw, (1.0 + D) * nb)
        eng = D * na * nb + 2.0 * delta * nw * na + (delta * nw) ** 2 + delta * np.abs(tt)
        self.last_bound = nys + eng
        self.last_yn = _resolvent_factor(self.wt, self.G.codomain) * nb    # ||S^-1 G C u||_D bounds (input_error.py)
        sb = std_error(var, self.last_bound)
        what = (f"every functional by deflated block CG ({its} iterations): the rank cap {len(dec.lam)} (about the "
                f"number of functionals) was reached with ||W - W_k|| <= {E:.3g} above the tail {dec.tail:g}, so CG is "
                "cheaper than a larger rank" if route == "cg" and not exact else
                "every functional exactly by deflated block CG (exact=True)" if exact else
                f"{len(var) - len(redo)} by the low-rank closed form" + (f" and {len(redo)} by deflated block CG "
                                                                         f"({its} iterations)" if len(redo) else "")
                + (f"; {left} beyond {EXACT_MAX} keep their low-rank bound" if left else ""))
        cl = [Claim(f"posterior variances above settings.dense_max (randomised decomposition, rank {len(dec.lam)}): "
                    f"{what}; stated error per functional: largest std bound {float(sb.max(initial=0.0)):.3g} "
                    f"({how})", CHECKED,
                    "Nystrom W_k with ||W^ - W_k|| bounded with probability >= 1 - 1e-10, two-sided via the resolvent "
                    "identity (W_k <= W^ only to rounding in double, plan 054 P3); "
                    "plus the products' error ||Delta|| ~ delta ||W^|| propagated through (W + I)^-1 - (W^ + I)^-1 = "
                    "(W + I)^-1 Delta (W^ + I)^-1 and the data image's (probability/scale.py)"
                    + ("; delta is the engine's estimate, not a bound" if delta else ""))]
        self.last = Claims(list(dec.claims) + cl)
        return var, sb

    def sampler(self, N, rng):
        """N centred posterior draws by Matheron's rule with block CG (module docstring)."""
        from ..settings import settings
        prior, G, noise = self.prior, self.G, self.noise
        self.decomposition()
        X0, E0 = prior._centred(N, rng), noise._centred(N, rng)
        Y = G(X0) + E0
        V = np.asarray(Y.values, float)
        V = V.reshape(N, -1).T if N > 1 or V.ndim == 2 else V[:, None]
        b = self.wt.fwd(V)
        Z, its, rel = self._solve(b, tol=1e-3 * settings.accuracy)
        Dsp = G.codomain
        plain = self.wt.back(Z)                                   # F^T Z: S^-1 y in D's coordinates
        Yi = Dsp.family(Dsp.solve_metric(plain).T) if N > 1 else Dsp(Dsp.solve_metric(plain[:, 0]))
        self.last_sample = Claims([Claim(
            f"posterior samples by Matheron's rule with deflated block CG ({its} iterations, largest relative residual "
            f"{rel:.2g} in whitened coordinates)", DERIVED, "x0 - C G* S^-1 (G x0 + e0) has the posterior law (exact draws "
                                                            "x0, e0); the solve's error is in the residual")])
        return X0 - prior.covariance(G.adjoint(Yi))


def _resolvent_factor(wt, D):
    """c with ||S^-1 y||_D <= c ||(W + I)^-1 F y|| (S^-1 y = W_D^-1 F^T (W + I)^-1 F y in D's coordinates): 1 for D's
    own whitening (W_D^-1/2 F^T is orthogonal), ||F|| ||W_D^-1/2|| for a diagonal noise."""
    if wt.kind == "metric":
        return 1.0
    from ..optimisers.dual.exactness import whitening_norm
    w = 1.0 if D.is_standard else float(np.min(D.weights)) if D.weights is not None else \
        float(np.linalg.eigvalsh(D.metric_matrix()).min())
    return whitening_norm(wt, D.dim) / np.sqrt(max(w, 1e-300))


def applies(G, noise, prior):
    """Whether condition() takes this route: R^n data above settings.dense_max, noise diagonal or the data space's own
    (a cylindrical prior too: its variances are defined; it only has no samples)."""
    from ..spaces.base import Euclidean
    from ..settings import settings
    from .gaussian import noise_metric
    D = G.codomain
    if not isinstance(D, Euclidean) or D.dim <= settings.dense_max:
        return False
    try:
        return noise_metric(noise, D).kind in ("standard", "diag")
    except Exception:
        return False
