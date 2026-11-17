"""One data-space decomposition per problem, reused across hyperparameters, targets and methods (plan 009 G3;
experiments/EXP-030-cmb-thesis DESIGN.md G3, NOTES.md section 4). The internal engine of the methods (plan 070 P3,
review D4, decision Q7): the public route is pg.decompose (inference/dataspace.py), which reads the noise law of a
model-data relation and states the metric here; the methods are the public routes to its closed forms.

For a linear map G: M -> D = R^n with representers g_i, a noise covariance C on D (its coordinate covariance; the
standard Gaussian of D by default) and a middle operator S on M (self-adjoint, positive semidefinite: none for the
model metric, as in SOLA and a norm-ball prior; the prior covariance for PLI; A^-1 for an ellipsoid <A m, m> <= a^2),
the whitened data Gram matrix is decomposed once,

    W = F K F^T = U diag(Lam) U^T,     K_ij = <g_i, S g_j>,     F C F^T = I,

per block when K and C are block diagonal. With beta = U^T F <g, S t> (one column per target t, the representers of
a property map T), the whitened datum delta = U^T F (d - noise mean - G m0) and Lam, every method is a sum over k of
rational functions of Lam_k (proofs in NOTES.md section 4):

    dec = pg.decompose(G, noise, middle)      the public route: noise a GaussianMeasure, a covariance (operator,
                                              vector or matrix of coordinates), a NormBall or Ellipsoid of D (its
                                              shape) or None (D's own inner product), read by inference/dataspace.py
    DataDecomposition(G, metric, middle)      the engine: the metric stated (D.whitened(C), the noise covariance C,
                                              or D), never inferred; offset= the noise mean
    dec.sola(T, gamma, d)                     SOLA for an array of gamma: X d = sum beta delta / (Lam + gamma), the
                                              noise std, the delivered mismatch ||t - G* X*||^2 (the TRM), the
                                              delivered norm, their derivatives in gamma, per-summand components
    dec.posterior(T, d, a, nu)                PLI with prior covariance a^2 S and noise nu^2 C for arrays of (a, nu):
                                              mean T m0 + a^2 sum beta delta / (a^2 Lam + nu^2), covariance
                                              a^2 T S T* - a^4 sum beta beta^T / (a^2 Lam + nu^2)
    dec.ellipsoid_bound(T, a, r, d, q)        BGP: sup <q, T m> over {||m - m0||_(S^-1) <= a} and {||F(d - G m)|| <= r},
                                              with a certificate (the two-multiplier dual, inf:1001-1072) evaluated
                                              exactly; O(n) per direction
    dec.feasibility(d, a=..., r=...)          the least misfit rho_min(a) over the prior set, and a*(r), the least
                                              prior radius that fits the data to r (the secular equation)
    dec.evidence(d, a, nu)                    log p(d | a, nu), the Gaussian evidence, and its gradient
    dec.damping_for_misfit(d, target)         alpha with squared damped residual = target (solvers.damping_for_misfit)
    dec.predictive(d, a, nu)                  the prior-predictive check: z_k, chi^2, tr H, the residual law's weights

Routes (the claims say which was taken):
    blocks    K given as a structured operator (operators/blocks.py: pg.BlockDiagonal with shared blocks and shifts,
              the structure claimed by whoever built it), or found from the values of a dense K and C (a CHECKED
              claim); one small eigendecomposition per block, equal blocks decomposed once, the blocks run through
              pygeoinf0.execution.map_blocks when sum n_b^3 exceeds PARALLEL_FLOPS
    dense     one eigendecomposition of the n x n whitened Gram (n <= settings.dense_max)
    randomised  above settings.dense_max with no structure (the default; plan 005 P5): a Nystrom approximation W_k of W
              from a randomised range finder, grown until a probabilistic a posteriori bound on ||W - W_k|| is at most
              `tail` (default settings.accuracy, in whitened units: a signal-to-noise variance ratio when S is a prior
              covariance); W_k <= W holds only to rounding in double, so the posterior closed forms state a
              two-sided error bound per functional (solvers/dataspace/partial.py, plan 054 P3)
    lanczos   route="lanczos": the leading eigenpairs by Lanczos (ARPACK) until the next would be below `tail`; the
              rest of the spectrum is not bounded
    A partial decomposition (randomised or lanczos) treats the rest of the spectrum as 0 and serves the quadratic
    closed forms (SOLA, PLI, evidence, damping, predictive); BGP and feasibility need a complete one
Each result carries claims: the closed form (DERIVED), the route and the eigen residual (CHECKED, three Gaussian probes
of ||W U - U Lam|| per block). pg.Problem keeps a cache of decompositions keyed by the identities and value hashes of
(G, noise covariance, middle), so that the methods it runs share them (Problem.decomposition, cached()).

Modules: this one builds the decomposition (routes, coordinates, pairings, the cache); closed_forms.py has the closed
forms, whitening.py the metric in coordinates and the per-block factors (solvers/dataspace/), optimisers/dual/spectral.py
BGP in the eigenbasis.
"""
import math

import numpy as np

from ...core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, Claim, Claims
from ...core.errors import Refused, SpaceMismatch, Unsupported
from ...spaces.base import Euclidean
from .closed_forms import ClosedForms, Result  # noqa: F401  (Result: what the closed forms return)
from .factored import COVARIANCE_DENSE
from . import whitening
from .whitening import (PROBES, NoiseMetric, _White, _cc, _cols, _digest, _eig, _eig_task, _representers,
                        _same_partition, _split, _values, _white_block)

DETECT = 1e-14          # off-block entries below DETECT * sqrt(K_ii K_jj) count as zero when blocks are detected
PARALLEL_FLOPS = 1e9    # per-block eigendecompositions go through execution.map_blocks above this (sum of n_b^3)


class _Targets:
    def __init__(self, cross, tt, T=None, tt_diag=None, tt_op=None):
        self.cross, self.T = np.atleast_2d(np.asarray(cross, float).T).T, T
        self.tt = None if tt is None else np.atleast_2d(tt)
        self.tt_diag = np.diag(self.tt).copy() if self.tt is not None else np.asarray(tt_diag, float)
        self.tt_op, self.beta = tt_op, None

    def diag(self):
        """<t_l, S t_l> for every property."""
        return self.tt_diag


def _correlations(A):
    """|A_ij| / sqrt(A_ii A_jj), so that small kernels are not lost against the noise when blocks are detected."""
    d = np.sqrt(np.clip(np.abs(np.diag(A)), 0.0, None))
    d = np.where(d > 0.0, d, 1.0)
    return np.abs(A) / np.outer(d, d)


# ------------------------------------------------------------------------------------------------ the object
class DataDecomposition(ClosedForms):
    """The whitened data Gram matrix, decomposed once (module docstring). pg.decompose(G, noise, middle) builds one."""

    def __init__(self, G=None, metric=None, middle=None, *, offset=None, gram=None, blocks="auto", tail=None,
                 normal=None, space=None, name="dec", route=None, rank_cap=None, target_rule=None):
        from ...operators.blocks import BlockDiagonal
        if normal is not None:
            D = normal.domain
        elif G is not None:
            D = G.codomain
        else:
            D = space if space is not None else getattr(gram, "domain", None)
        if not isinstance(D, Euclidean):
            raise SpaceMismatch("a data-space decomposition needs a finite-dimensional data space (pg.Euclidean); give "
                                "G, or the space with gram=")
        if gram is not None and isinstance(gram, BlockDiagonal) and gram.domain.dim != D.dim:
            raise SpaceMismatch(f"the Gram operator acts on R^{gram.domain.dim}, not on R^{D.dim}")
        self.G, self.D, self.n, self.name = G, D, D.dim, name
        self.middle = getattr(middle, "covariance", middle)
        self.noise = whitening.metric_of(metric, D, offset)    # stated, never inferred (plan 070 P3, Q1)
        self.normal, self.target_rule = normal, target_rule
        self._targets, self._claims = {}, []
        self.partial, self.error, self._route, self._rank_cap = False, None, route, rank_cap
        if route not in (None, "randomised", "lanczos"):
            raise ValueError("route is None (by size and structure), 'randomised' or 'lanczos'")
        self._build(gram, blocks, tail)

    # ---------------------------------------------------------------------------------------------- building
    def _gram(self, gram):
        """K (plain Gram <g_i, S g_j>): an array, a BlockDiagonal, or None (matrix-free: products only)."""
        from ...operators.blocks import BlockDiagonal
        from ...settings import settings
        D, S = self.D, self.middle
        if gram is not None:
            if isinstance(gram, BlockDiagonal):
                own = [c for c in gram._claims if c.key == "block diagonal" or "block" in c.statement]
                self._claims.append(Claim("the Gram matrix given as a structured operator", ASSERTED,
                                          "gram= (its structure: " + ("; ".join(f"{c.statement} [{c.status}]" for c in own)
                                                                      or "stated by its constructor") + ")"))
                return gram
            K = np.asarray(gram, float)
            if K.shape != (self.n, self.n):
                raise ValueError(f"gram: an {self.n} x {self.n} matrix")
            self._claims.append(Claim("the Gram matrix given", ASSERTED, "gram="))
            return K
        if self.normal is not None:
            K = getattr(self.normal, "K", self.normal)            # plain_gram.WeightedGram: its plain blocks
            if isinstance(K, BlockDiagonal):
                return K
            if self.n > settings.dense_max:
                return None
            return D.solve_metric(np.asarray(self.normal.matrix(), float).T).T
        G = self.G
        hook = getattr(G, "structured_gram", None)              # plan 009 P2: separable functionals (G2)
        if hook is not None:
            out = hook(S)
            if out is not None:
                return out
        if self.n > settings.dense_max:
            return None
        if S is None and type(G).__name__ == "Functionals":
            Gr = G.gram()
            if isinstance(Gr, np.ndarray):
                return Gr
        A = G @ G.adjoint if S is None else G @ S @ G.adjoint
        return D.solve_metric(np.asarray(A.matrix(), float).T).T

    def _build(self, gram, blocks, tail):
        from ...operators.blocks import BlockDiagonal, coupled_components
        from ...settings import settings
        from ..direct import orthonormal_form, _square_matrix
        n, D, noise = self.n, self.D, self.noise
        # the discrepancy route of solvers.py: the orthonormal form of G G* itself, bitwise as before
        if self.normal is not None and noise.kind == "standard" and gram is None and blocks in (None, "none") \
                and n <= settings.dense_max and not isinstance(getattr(self.normal, "K", self.normal), BlockDiagonal):
            W = orthonormal_form(self.normal, _square_matrix(self.normal))
            lam, U = np.linalg.eigh(0.5 * (W + W.T))
            Z = np.random.default_rng(0).standard_normal((n, PROBES))
            res = float(np.linalg.norm(W @ (U @ Z) - U @ (lam[:, None] * Z)) / math.sqrt(PROBES)
                        / max(float(np.sqrt(np.sum(lam ** 2))), 1e-300))
            self._finish([np.arange(n)], [_White("metric", None, D)], [(lam, U, res)], "dense", 1)
            return
        K = self._gram(gram)
        if K is None:
            from . import partial
            if self._route == "lanczos":
                partial.lanczos(self, tail)
            else:
                partial.nystrom(self, tail, kmax=self._rank_cap)
            return
        how, ratio = None, None
        if isinstance(K, BlockDiagonal) and K.n_blocks == 1:
            K = K.matrix()                                       # one block: its matrix in D's own row order
        if noise.kind == "blocks" and not (isinstance(K, BlockDiagonal) and _same_partition(K, noise.data)):
            noise = self.noise = NoiseMetric("dense", noise.data.matrix(), D, noise.offset, noise.what)
        if isinstance(K, BlockDiagonal):
            parts = K.partition
            dense_metric = noise.kind == "standard" and not (D.is_standard or D.weights is not None)
            if (noise.kind == "dense" or dense_metric) and len(parts) > 1:
                Kd = K.matrix()
                Cd = noise.data if noise.kind == "dense" else D.metric_matrix()
                sizes, order, ratio = coupled_components(np.abs(Kd) + np.abs(Cd), 0.0, _cc())
                if len(sizes) < len(parts):
                    K, parts = Kd, _split(sizes, order)
                    how = "detected"
            if how is None:
                how = "structured"
        elif isinstance(blocks, (list, tuple)):
            parts = [np.asarray(p, int) for p in blocks]
            if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(n)):
                raise ValueError("blocks: row sets that partition the data")
            how = "given"
        elif blocks == "auto" and n > 1:
            M = _correlations(K) + (_correlations(noise.data) if noise.kind == "dense" else 0.0)
            if noise.kind == "standard" and not (D.is_standard or D.weights is not None):
                M = M + _correlations(D.metric_matrix())
            sizes, order, ratio = coupled_components(M, DETECT, _cc())
            parts = _split(sizes, order)
            how = "detected" if len(parts) > 1 else "dense"
        else:
            parts, how = [np.arange(n)], "dense"
        full = len(parts) == 1
        if full:
            parts = [np.arange(n)]
        whites = [_white_block(noise, p, D, full) for p in parts]
        jobs, keys, which = [], {}, []
        for b, (p, wt) in enumerate(zip(parts, whites)):
            if full:
                key = 0
            elif isinstance(K, BlockDiagonal) and how == "structured":
                key = ("s", int(K.shared[b]), _digest(K.shift(b)), wt.kind, _digest(wt.data) if wt.data is not None else 0)
            else:
                key = None
            if key is None:
                Kb = K[np.ix_(p, p)] if not full else K
                key = ("v", _digest(np.ascontiguousarray(Kb)), wt.kind, _digest(wt.data) if wt.data is not None else 0)
            if key not in keys:
                keys[key] = len(jobs)
                jobs.append(b)
            which.append(keys[key])

        def make(b):
            p, wt = parts[b], whites[b]
            if isinstance(K, BlockDiagonal) and how == "structured":
                pos = K.partition[b]
                Kb = K.matrices[K.shared[b]] + (np.diag(K.shift(b)) if K._shift is not None else 0.0)
                if not np.array_equal(pos, p):
                    raise AssertionError("block rows")
            else:
                Kb = K if full else K[np.ix_(p, p)]
            return _eig(wt.both(Kb))
        flops = float(sum(len(parts[b]) ** 3 for b in jobs))
        from ...execution import map_blocks, workers
        if len(jobs) > 1 and flops > PARALLEL_FLOPS and workers() > 1:
            done = dict(map_blocks(_eig_task, (make,), [(b,) for b in jobs]))
            eig = [done[(b,)] for b in jobs]
            ran = f"{min(workers(), len(jobs))} worker processes (pygeoinf0.execution.map_blocks)"
        else:
            eig = [make(b) for b in jobs]
            ran = "serially"
        if how == "structured":
            self._claims.append(Claim(f"block diagonal: {len(parts)} blocks of rows from the structured Gram operator, "
                                      f"{len(jobs)} distinct eigendecompositions ({ran})", DERIVED,
                                      "the operator's blocks; equal blocks (same matrix, shift and whitening) share one"))
        elif how == "detected":
            self._claims.append(Claim(f"block diagonal: {len(parts)} blocks of rows found from the values of K and C, "
                                      f"{len(jobs)} distinct eigendecompositions ({ran})", CHECKED,
                                      f"entries below {DETECT:g} sqrt(K_ii K_jj) treated as zero (largest such ratio "
                                      f"{ratio:.2g}); equal blocks share one eigendecomposition"))
        elif how == "given":
            self._claims.append(Claim(f"{len(parts)} blocks of rows as given (blocks=)", ASSERTED, "blocks="))
        self._finish(parts, whites, [eig[w] for w in which], "blocks" if len(parts) > 1 else "dense", len(jobs))

    def _finish(self, parts, whites, eigs, route, distinct):
        self._rows, self._white = parts, whites
        self._lam_b = [e[0] for e in eigs]
        self._U = [e[1] for e in eigs]
        self.residual = max(e[2] for e in eigs)
        self.lam = np.concatenate(self._lam_b)
        self.offsets = np.r_[0, np.cumsum([len(l) for l in self._lam_b])]
        self._block_of = np.repeat(np.arange(len(parts)), [len(l) for l in self._lam_b])
        self.route = route
        sizes = [len(p) for p in parts]
        self._claims.insert(0, Claim(
            "W = U diag(Lam) U^T, W the whitened data Gram matrix F K F^T (K_ij = <g_i, S g_j>, F C F^T = I)", THEOREM,
            "the spectral theorem for a symmetric matrix"))
        if route not in ("lanczos", "randomised"):
            self._claims.append(Claim(
                ("one dense eigendecomposition of the " + f"{self.n} x {self.n} whitened Gram matrix") if route == "dense"
                else f"{len(parts)} blocks of sizes {min(sizes)}-{max(sizes)} ({distinct} distinct eigendecompositions)",
                DERIVED, f"route {route}; noise: {self.noise.what}"))
            self._claims.append(Claim(f"eigen residual ||W U - U Lam||_F <= about {self.residual:.1e} ||W||_F", CHECKED,
                                      f"{PROBES} Gaussian probes per block (LAPACK eigh)"))

    def positive(self, rcond):
        """The eigenvalues that count as positive: Lam > rcond * (the largest eigenvalue of the same block). eigh's error
        is relative to the block it factorised, so a block's small eigenvalues are resolved even when another block's
        scale is many orders larger (a cut relative to the global maximum would discard them)."""
        return np.concatenate([l > rcond * max(float(l.max(initial=0.0)), 0.0) if len(l) and l.max() > 0
                               else np.zeros(len(l), bool) for l in self._lam_b])

    def above_floor(self):
        """The eigenvalues above the rounding floor: Lam > eps n_block (their block's largest) (plan 054)."""
        return np.concatenate([b > np.finfo(float).eps * len(b) * float(b.max(initial=0.0)) for b in self._lam_b])

    def zero_count(self, rcond):
        """(counted as zero, kept below the rounding floor, n): eigenvalues at most rcond times their block's largest,
        and those kept above it but at most eps n_block times it (no correct digit; plan 054 P2)."""
        pos, floor = self.positive(rcond), self.above_floor()
        return int((~pos).sum()), int((pos & ~floor).sum()), len(pos)

    def at_floor(self, rcond):
        """(unexplained, at the floor, structural): zero_count's two counts together, less those exactly zero by
        structure (n - dim M: optimisers/dual/reduction.structural_zeros); the rest cannot be told from exact dependence (plan 054)."""
        from ...optimisers.dual.reduction import structural_zeros
        zero, below, n = self.zero_count(rcond)
        st = min(structural_zeros(n, self.G), zero + below)
        return zero + below - st, zero + below, st

    # ---------------------------------------------------------------------------------------------- coordinates
    @property
    def claims(self):
        return Claims(self._claims)

    def describe(self):
        return (f"{self.name}: {self.route} route, n = {self.n}" + (f", {len(self._rows)} blocks" if len(self._rows) > 1
                                                                       else "")
                + (f", {len(self.lam)} pairs (partial)" if self.partial else "") + f"; noise {self.noise.what}"
                + ("; middle " + getattr(self.middle, "name", "S") if self.middle is not None else ""))

    def __repr__(self):
        return f"DataDecomposition({self.describe()})"

    def _coef(self, V):
        """U^T F V (eigen coordinates, blocks concatenated) for V of shape (n,) or (n, m) in D's coordinates."""
        V = np.asarray(V, float)
        out = np.empty((len(self.lam),) + V.shape[1:])
        for p, wt, U, a, b in zip(self._rows, self._white, self._U, self.offsets[:-1], self.offsets[1:]):
            Vb = V if len(self._rows) == 1 else V[p]
            out[a:b] = U.T @ wt.fwd(Vb)
        return out

    def _back(self, X):
        """F^T U X (plain multipliers in D's coordinates) for X of shape (n_eff,) or (n_eff, m)."""
        X = np.asarray(X, float)
        out = np.zeros((self.n,) + X.shape[1:])
        for p, wt, U, a, b in zip(self._rows, self._white, self._U, self.offsets[:-1], self.offsets[1:]):
            out[p] = wt.back(U @ X[a:b])
        return out

    def _spec(self, X):
        """(Lam, C, extra): the spectrum and eigen coordinates of the whitened columns X (plain, (n,) or (n, q)). For
        a partial decomposition the parts of X outside the computed eigenvectors are appended as coordinates with
        Lam = 0 (exact for the operator whose remaining spectrum is 0); extra counts the further zero eigenvalues."""
        X = np.asarray(X, float)
        if not self.partial:
            return self.lam, self._coef(X), 0
        one = X.ndim == 1
        X2 = X[:, None] if one else X
        FX = self._white[0].fwd(X2)
        U = self._U[0]
        H = U.T @ FX
        P = FX - U @ H
        Q, R = np.linalg.qr(P)
        d = np.abs(np.diag(R))
        keep = d > 1e-14 * max(float(np.linalg.norm(FX)), 1e-300)
        C = np.r_[H, R[keep]]
        lam = np.r_[self.lam, np.zeros(int(keep.sum()))]
        return lam, (C[:, 0] if one else C), self.n - len(self.lam) - int(keep.sum())

    def _delta(self, datum, centre=None):
        """The plain residual d - noise offset - G m0 (D's coordinates)."""
        e = _values(self.D, datum)
        if e is None:
            raise ValueError("this needs the datum")
        if self.noise.offset is not None:
            e = e - self.noise.offset
        if centre is not None:
            if self.G is None:
                raise Unsupported("a prior centre needs G")
            e = e - self.G(centre).values
        return e

    def logdet_noise(self):
        """log det C (the noise covariance, coordinates)."""
        return float(sum(wt.logdet() for wt in self._white))

    # ---------------------------------------------------------------------------------------------- targets
    def targets(self, T=None, pairings=None):
        """The pairings <g_i, S t_l> (n x p) and <t_l, S t_m> (p x p) of the representers t_l of T (an operator into
        R^p) or of one model element c; or given as pairings=(cross, tt). Cached per T."""
        if pairings is not None:
            cross, tt = pairings
            return _Targets(cross, tt)
        if T is None:
            raise ValueError("give the property map T (or a model element, or pairings=(cross, tt))")
        key = getattr(T, "_id", None)
        if key is not None and key in self._targets:
            return self._targets[key]
        G, S = self.G, self.middle
        if G is None:
            raise Unsupported("pairings with a property map need G (or give pairings=)")
        from ...operators.linear import LinearOperator
        if isinstance(T, LinearOperator):
            P = T.codomain
            if not isinstance(P, Euclidean):
                raise SpaceMismatch(f"the property map maps into {P}, not into R^p")
            if getattr(T, "factors", None) is not None:          # plan 009 G4: T = Phi tau, no p x p pairings
                from .factored import factored_targets
                out = factored_targets(self, T, S)
                if key is not None:
                    self._targets[key] = out
                return out
            hook = getattr(G, "structured_cross", None)          # plan 009 P2: separable functionals
            if hook is not None and getattr(G, "lm", None) is not None:          # plan 044: a target kept symbolic
                from ...realisations.series.symbolic_harmonics import is_symbolic
                if is_symbolic(T):
                    if S is not None:
                        raise Refused("a target kept symbolic with a prior needs its degree, at least the prior's: use "
                                      "pg.harmonic_target(T, degree) or pg.RotatedFamily(T0, centres, degree=)", needs=("degree",))
                    if self.target_rule is None:
                        raise Refused("a target kept symbolic needs its degree: pg.harmonic_target(T, degree) (the data "
                                      f"reach degree {math.isqrt(int(G.lm.max()))}) or pg.RotatedFamily(T0, centres, "
                                      "degree=); pg.decompose and pg.Problem take the data's degree without a prior",
                                      needs=("degree",))
                    return self.targets(self.target_rule(G, T))
            pair = hook(T, S) if hook is not None else None
            if pair is None and P.dim > COVARIANCE_DENSE and P.is_standard:
                out = _many_targets(G, T, S, P)
                if key is not None:
                    self._targets[key] = out
                return out
            if pair is None:
                GT = G @ T.adjoint if S is None else G @ S @ T.adjoint
                TT = T @ T.adjoint if S is None else T @ S @ T.adjoint
                cross = P.solve_metric(_cols(GT).T).T
                tt = P.solve_metric(_cols(TT).T).T
                pair = (cross, 0.5 * (tt + tt.T))
            out = _Targets(pair[0], pair[1], T)
        else:
            c = T
            Sc = c if S is None else S(c)
            out = _Targets(G(Sc).values, [[float(G.domain.inner(c, Sc))]])
        if key is not None:
            self._targets[key] = out
        return out

    def _beta(self, tg):
        if tg.beta is None:
            tg.beta = self._coef(tg.cross)
        return tg.beta

    def _result_claims(self, what, why):
        return [Claim(what, DERIVED, why), Claim(f"from the data-space decomposition {self.describe()}", DERIVED,
                                                 "one decomposition reused (plan 009 G3)")] + list(self._claims)


def _many_targets(G, T, S, P):
    """Pairings of a plain map T with more than COVARIANCE_DENSE properties: the cross pairings (n x p), the diagonal
    <t_l, S t_l> and T S T* as an operator (no p x p matrix)."""
    reps = _representers(T)
    M = T.domain
    GT = G @ T.adjoint if S is None else G @ S @ T.adjoint
    cross = _cols(GT)
    diag = np.empty(P.dim)
    for a in range(0, P.dim, 256):
        chunk = reps[a:a + 256]
        Sc = chunk if S is None else S(chunk)
        diag[a:a + 256] = np.einsum("ii->i", np.atleast_2d(M.inner(chunk, Sc)))
    TT = T @ T.adjoint if S is None else T @ S @ T.adjoint
    return _Targets(cross, None, T, tt_diag=diag, tt_op=TT)


# ------------------------------------------------------------------------------------------------ cache
def decomposition_key(G, metric, middle, D=None, offset=None):
    """The cache key of a decomposition: the identities of G and of the middle operator (LinearOperator ids are never
    reused) and the value hash of the noise covariance in coordinates (and of G and the middle when they are matrices).
    The metric is stated as for DataDecomposition (whitening.metric_of)."""
    D = G.codomain if D is None else D
    spec = whitening.metric_of(metric, D, offset)
    S = getattr(middle, "covariance", middle)

    def ident(A):
        if A is None:
            return None
        h = _digest(A.A) if hasattr(A, "A") and isinstance(getattr(A, "A"), np.ndarray) else None
        return (getattr(A, "_id", id(A)), h)
    return (ident(G), spec.key, ident(S))


def cached(cache, G, metric=None, middle=None, target_rule=None, **kw):
    """The decomposition of (G, metric, middle) from a cache (a dict, e.g. pg.Problem's), built on first use. A
    target_rule given (the public routes' rule for targets kept symbolic) is attached to a cached one without."""
    if cache is None:
        return DataDecomposition(G, metric, middle, target_rule=target_rule, **kw)
    key = decomposition_key(G, metric, middle, offset=kw.get("offset"))
    if key not in cache:
        cache[key] = DataDecomposition(G, metric, middle, target_rule=target_rule, **kw)
    elif target_rule is not None and cache[key].target_rule is None:
        cache[key].target_rule = target_rule
    return cache[key]
