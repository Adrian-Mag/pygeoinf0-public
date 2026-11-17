"""Covariances that act degree by degree on radial x harmonic spaces and on direct sums of them (plan 009 P4b: G2 x G5
x G6; EXP-030 DESIGN G5, G6, NOTES.md sections 3, 5, 6).

    K = pg.Laplacian(B, bc=...).bessel_potential(kappa, q, degrees, modes=N)    sum_n lam_ln phi_ln (x) phi_ln per degree
    KJ = K.congruence(pg.Multiplication(X, j))       j radial: per degree sum_n lam_ln (j phi_ln) (x) (j phi_ln)
    A = pg.Covariance.angular(S, {l: c_l})          isotropic on a sphere space: c_l on the degree-l harmonics
    C = pg.Covariance.glued(M, {("a_0", "a_1"): KJ, "h": A})
                                                    on a direct sum M whose summands a_0, a_1 are the pieces of KJ's
                                                    domain (shells tiling a ball or shell, one weight), h a sphere summand
    C.congruence(D)                                 D acting by a number per degree and summand (degree_scalars: a
                                                    pg.BlockDiagonal of multiples of identities, functions of -Delta_S)
    C.degree_spectrum(), K.radial_variance(r)       eigenvalues with multiplicities (pg.WeightedChiSquare); Var u(r, xi)

Mathematics. (a) A radial multiplication M_j commutes with rotations, so M_j K M_j acts on the degree-l harmonics of
every order by the radial operator sum_n lam_n (j phi_n) (x) (j phi_n) (Schur; EXP-030 Theorem G2.1); its trace is
sum_l (2l + 1) sum_n lam_ln ||j phi_ln||^2 (Parseval) and its spectrum on degree l that of diag(lam^1/2) Gamma
diag(lam^1/2), Gamma_nm = <j phi_n, j phi_m> (the non-zero eigenvalues of A A* and A* A agree). (b) If B is the union
of pieces B_k (shells), L2(B) = (+)_k L2(B_k) isometrically (u -> (u|B_k)_k; one weight on all pieces), and an
operator on L2(B) is transported to the direct sum by that isometry: it couples the pieces. A functional g = (g_k)_k
then pairs with the degree-l part as sum_n lam_n (sum_k <g_k, j phi_n>_k)(sum_k <g'_k, j phi_n>_k), the inner sums over
the pieces of one part (the "group" of the degree structure, operators/separable.py). (c) A congruence by D with
D = (+)_k d_k(-Delta_S) on the summands multiplies the radial factor on piece k by d_k(l) and a sphere part's c_l by
d_k(l)^2. (d) The pointwise variance of u ~ N(0, C) at (r, xi) is sum_l (2l + 1)/(4 pi) sum_n lam_ln (j phi_ln(r))^2
(the addition theorem; independent of xi).

Realisation. The radial integrals (realisations/series/radial_integrals.py, plan 070 P7) are Gauss rules on the union of the factors' breakpoints, with an order per piece from
the factors' degrees (scaled for a polynomial factor much longer than the piece), or a declared quadrature rule
(quadrature=: e.g. pg.realisations.Trapezoid(n, over=(a, b))), which is then a realisation option named in the claims.
The covariance is applied only through its degree structure (structured Grams of separable functionals, traces,
spectra, variances); applying it to an arbitrary element is not offered (Unsupported).
"""
import math

import numpy as np

from ..core.numbers import Enclosure
from ..core.claims import DERIVED, ORDER, THEOREM, Claim, by_rule
from ..core.errors import Unsupported
from ..quadrature import gauss
from .operator_covariance import OperatorCovariance
from ..spaces.identity import identical
from ..realisations.series.radial_integrals import mgram

SCHUR = "a function of the Laplacian with rotation-invariant conditions times radial multipliers (Schur; EXP-030 " \
        "Theorem G2.1)"


def _weakest(*st):
    return max(st, key=ORDER.index)


def _vals(p, x):
    from ..realisations.series import legendre as LG
    return LG.evaluate(p, x)


def _interval(space):
    B = space.backend
    return float(B.a), float(B.b)


# ------------------------------------------------------------------------------------------------ one space
def _sharp_tail(C, m, a, b, factor):
    """sum_l (2l + 1) [sum_(n = N+1)^(M) lam_ln ||m phi_ln||^2 + ||m||_inf^2 sum_(n > M) lam_ln] with M = factor N: the
    first part from the closed-form eigenfunctions (Gauss rules per piece of m with an order from k_M h), the second
    from the spectra's tail bounds (realisations/spectra/laplacian_covariance.matern_tail)."""
    from ..realisations.spectra.laplacian_covariance import matern_tail
    msup = float(np.max(np.abs(m.sup_bound())))
    total = 0.0
    for l, d in C.degree_spectra.items():
        sp, N = d["spectrum"], d["N"]
        M = int(factor * N)
        ev = sp.eigenvalues(M)
        lam = d["c"] * d["f"](ev)[N:]
        phi = sp.eigenfunctions(M)
        kM = math.sqrt(float(ev[-1]))
        br = np.unique(np.r_[a, m.breaks[(m.breaks > a) & (m.breaks < b)], b])
        xs, ws = [], []
        for lo, hi in zip(br[:-1], br[1:]):
            t, w = gauss(max(8, int(kM * (hi - lo)) + 8))
            x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t
            xs.append(x)
            ws.append(0.5 * (hi - lo) * w * x * x * _vals(m, x)[:, 0] ** 2)
        x, W = np.concatenate(xs), np.concatenate(ws)
        norms = np.zeros(M - N)
        for i in range(0, len(x), 4096):
            V = phi(x[i:i + 4096])[:, N:]
            norms += W[i:i + 4096] @ (V * V)
        rest = d["c"] * matern_tail(d["f"], sp.lower_bound, M, (math.pi / d["L"]) ** 2, d["theta"], 0.0, d["kappa"],
                                    d["q"])
        total += (2 * l + 1) * (float(lam @ norms) + msup ** 2 * rest)
    return total


def congruence_radial(C, B, quadrature=None, name=None, tail_modes=4):
    """B C B* for C with radial degree parts and B a radial multiplication on C's space (module docstring (a)); the
    trace from the parts, exactly or by the declared rule."""
    mB = B.radial_factor()
    a, b = _interval(C.domain)
    parts, known = {}, 0.0
    from ..realisations.series import legendre as LG
    for l, part in C.degree_parts.items():
        kind, P, lam = part[:3]
        m0 = part[3] if len(part) > 3 else None
        m = mB if m0 is None else LG.multiply(m0, mB)
        parts[l] = (kind, P, lam, m)
        known += (2 * l + 1) * float(lam @ mgram(P, m, a, b, quadrature, diag=True))
    nb, st = B.norm_bound(), B.claims.find("bounded").status
    sharp = _sharp_tail(C, mB, a, b, tail_modes) if tail_modes and getattr(C, "degree_spectra", None) and \
        C.rest is not None \
        else None
    nm = name or f"{B.name}{C.name}{B.name}*"
    op = B @ C.realised @ B
    rest = None
    if C.rest is not None:
        r = C.rest
        rest = dict(norm=nb ** 2 * r["norm"], trace=nb ** 2 * r["trace"], status=_weakest(r["status"], st),
                    reason=f"||B||^2 times the omitted part's bounds ({r['reason']})", orthogonal=False)
        if sharp is not None and sharp < rest["trace"]:
            rest["trace"] = sharp
            rest["reason"] = (f"the omitted part's trace: sum_n lam_n ||j phi_n||^2 explicitly up to {tail_modes} times "
                              "the modes kept (closed-form eigenfunctions, Gauss per piece of j), ||B||^2 times the tail "
                              f"bound beyond; its norm ||B||^2 ||R|| ({r['reason']})")
    how = "exactly (Gauss per piece)" if quadrature is None else f"by the {quadrature.name} (a realisation option)"
    if rest is None:
        tr = Enclosure(known, known * (1 - 1e-12), known * (1 + 1e-12), DERIVED,
                       f"sum_l (2l + 1) sum_n lam_ln ||j phi_ln||^2, the radial integrals {how}")
    else:
        tr = Enclosure(known + 0.5 * rest["trace"], known, known + rest["trace"], rest["status"],
                       f"the known part's sum_l (2l + 1) sum_n lam_ln ||j phi_ln||^2 (radial integrals {how}) plus "
                       "||B||^2 times the omitted part's trace bound")
    out = OperatorCovariance(op, tr, name=nm, _rest=rest,
                             selfadjoint=(True, by_rule(C._sa[1]), "of the form B C B*"),
                             positive=("psd", by_rule(C._pos[1]), "a congruence of a positive operator"), _claims=[
        Claim("commutes with rotations", _weakest(DERIVED, B.claims.find("rotation invariant").status),
              "a congruence of a rotation-invariant covariance by a radial multiplication (EXP-030 Theorem G2.1)",
              key="rotation invariant")])
    out.degree_parts = parts
    out.quadrature = quadrature
    return out


def radial_variance(C, r):
    """Var u(r, xi) for u ~ N(0, C), C with radial degree parts (module docstring (d)); r: radii."""
    r = np.atleast_1d(np.asarray(r, float))
    out = np.zeros(r.shape)
    for l, part in C.degree_parts.items():
        P, lam = part[1], part[2]
        V = _vals(P, r) ** 2
        if len(part) > 3 and part[3] is not None:
            V = V * _vals(part[3], r)[:, :1] ** 2
        out += (2 * l + 1) / (4 * math.pi) * (V @ lam)
    return out


def part_spectrum(P, lam, m, a, b, rule=None):
    """The eigenvalues of sum_n lam_n (m phi_n) (x) (m phi_n) on L2((a, b), r^2 dr) (module docstring (a))."""
    sq = np.sqrt(np.maximum(lam, 0.0))
    if m is None and rule is None:
        return np.sort(lam)[::-1]
    G = mgram(P, m, a, b, rule)
    return np.maximum(np.linalg.eigvalsh(sq[:, None] * G * sq[None, :])[::-1], 0.0)


def angular(space, spectrum, name="C"):
    """The isotropic covariance f(-Delta_S) with f(l) = c_l on the given degrees and 0 elsewhere, on a sphere space."""
    from ..operators.band import AngularFunction
    from ..realisations.series import harmonic_degrees as H
    if H.require(space, "an angular covariance") != "sphere":
        raise Unsupported("pg.Covariance.angular is for spaces on a sphere (spherical harmonics)")
    c = {int(l): float(v) for l, v in spectrum.items()}
    if any(v < 0 for v in c.values()):
        raise ValueError("an angular power spectrum is >= 0")
    ls = np.array(sorted(c))
    vals = np.array([c[l] for l in ls])

    def f(l):
        l = np.asarray(l)
        out = np.zeros(l.shape)
        idx = np.searchsorted(ls, l)
        ok = (idx < len(ls)) & (ls[np.minimum(idx, len(ls) - 1)] == l)
        out[ok] = vals[idx[ok]]
        return out
    op = AngularFunction(space, f, name=name)
    t = float(np.sum((2 * ls + 1) * vals))
    out = OperatorCovariance(op, Enclosure(t, t * (1 - 1e-13), t * (1 + 1e-13), DERIVED,
                                           "sum_l (2l + 1) c_l (a finite band)"), name=name)
    out.angular_spectrum = c
    return out


# ------------------------------------------------------------------------------------------------ direct sums
def glued(M, parts, name="C"):
    """The covariance on the direct sum M built from parts (module docstring (b)): {summand name or tuple of the names of
    the pieces of one domain, in radial order: Covariance}."""
    names = list(M.names)
    acts, groups, traces, rests, quad, partitions = {}, [], [], [], {}, {}
    for key, Cp in parts.items():
        ks = [M.index(k) for k in ((key,) if isinstance(key, str) else key)]
        if getattr(Cp, "angular_spectrum", None) is not None:
            if len(ks) != 1:
                raise ValueError("an angular covariance is one sphere summand")
            acts[ks[0]] = {l: ("scalar", c) for l, c in Cp.angular_spectrum.items()}
        elif getattr(Cp, "degree_parts", None) is not None:
            a, b = _interval(Cp.domain)
            iv = [_interval(M.summands[k]) for k in ks]
            w = {M.weights[k] for k in ks}
            part = _partition(Cp.domain, [M.summands[k] for k in ks], [names[k] for k in ks])
            if part is not None:                  # pg.Partition decides the tiling (plan 035 P3); the order is radial
                ok = part.order == tuple(names[k] for k in ks)
            else:
                tol = 1e-12 * max(abs(a), abs(b), 1.0)
                ok = abs(iv[0][0] - a) <= tol and abs(iv[-1][1] - b) <= tol and \
                    all(abs(iv[i][1] - iv[i + 1][0]) <= tol for i in range(len(iv) - 1))
            if not ok:
                raise ValueError(f"the summands {[names[k] for k in ks]} do not tile [{a:g}, {b:g}] in order")
            partitions[tuple(ks)] = part
            if len(w) != 1:
                raise ValueError("the pieces of one domain carry one weight (L2(B) = (+)_k L2(B_k) isometrically)")
            g = len(groups)
            groups.append(tuple(ks))
            for k in ks:
                acts[k] = {l: ("radial", p[1], p[2], dict(m=p[3] if len(p) > 3 else None, d=1.0, group=g))
                           for l, p in Cp.degree_parts.items()}
                quad[k] = getattr(Cp, "quadrature", None)
        else:
            raise Unsupported(f"{Cp.name}: no degree structure (a radial covariance per degree or an angular one)")
        traces.append(Cp.trace())
        if Cp.rest is not None:
            rests.append(Cp.rest)
    out = GluedCovariance(M, acts, groups, traces, rests, quad, name=name)
    out.partitions = partitions                   # {summand positions: the pg.Partition they tile, or None}
    return out


def _partition(X, summands, names):
    """The pg.Partition of X's domain into the summands' domains (plan 035 P3), None when the domains are not of a kind
    it decides (then the radial-interval check above stands); NotATiling (a ValueError) when they do not tile it."""
    from ..core.errors import Unsupported
    from ..domains.partition import Partition
    D = getattr(X, "domain", None)
    try:
        return Partition(D, dict(zip(names, [getattr(H, "domain", None) for H in summands])))
    except (Unsupported, AttributeError, TypeError):
        return None


class GluedCovariance(OperatorCovariance):
    """A covariance on a direct sum applied through its degree structure (module docstring); built by glued()."""

    def __init__(self, M, acts, groups, traces, rests, quad, name="C", scale=1.0, dscal=None):
        from ..operators.linear import LinearOperator
        op = LinearOperator(M, M, name=name)
        op._sa = (True, THEOREM, "the transport of self-adjoint parts by an isometry, and congruences of them")
        op._pos = ("psd", THEOREM, "positive parts")
        op._apply = self._refuse
        rest = None
        if rests:
            rest = dict(norm=max(r["norm"] for r in rests), trace=sum(r["trace"] for r in rests),
                        status=_weakest(*(r["status"] for r in rests)), reason="the parts' omitted parts",
                        orthogonal=False)
        lo, hi = sum(t.lower for t in traces), sum(t.upper for t in traces)
        st = _weakest(*(t.status for t in traces)) if traces else DERIVED
        tr = Enclosure(sum(float(t) for t in traces), lo, hi, st, "the sum of the parts' traces (trace is basis free)")
        super().__init__(op, tr, name=name, _rest=rest, _claims=[Claim("commutes with rotations", THEOREM, SCHUR,
                                                                        key="rotation invariant")])
        self.acts, self.groups, self._traces, self._rests, self.quad = acts, groups, traces, rests, quad

    @staticmethod
    def _refuse(x):
        raise Unsupported("this covariance is applied through its degree structure only (structured Grams of separable "
                          "functionals, traces, spectra); applying it to an element is not offered")

    def _make_adjoint(self):
        return self

    def degree_scalars(self, space):
        return None

    def degree_operators(self, space):
        if not identical(space, self.domain):
            return None
        n = len(self.domain.summands)
        return [(lambda l, a=self.acts.get(k, {}): a.get(int(l))) for k in range(n)]

    def _scaled(self, c, name):
        acts = {k: {l: (("scalar", c * a[1]) if a[0] == "scalar" else ("radial", a[1], c * a[2], a[3]))
                    for l, a in A.items()} for k, A in self.acts.items()}
        rests = [dict(r, norm=c * r["norm"], trace=c * r["trace"]) for r in self._rests]
        out = GluedCovariance(self.domain, acts, self.groups, [t.scaled(c) for t in self._traces], rests, self.quad,
                              name=name)
        out._claims = list(self._claims)
        return out

    def congruence(self, B, name=None):
        """B C B* for B acting by a number per degree and summand (module docstring (c))."""
        from ..operators.equivariance import degree_scalars
        f = degree_scalars(B, self.domain)
        if f is None:
            raise Unsupported(f"{B.name}: a congruence of a degree-structured covariance needs B acting by a number per "
                              "degree and summand (degree_scalars)")
        acts = {}
        for k, A in self.acts.items():
            out = {}
            for l, a in A.items():
                d = float(np.asarray(f[k](np.array([l])), float).ravel()[0])
                out[l] = ("scalar", d * d * a[1]) if a[0] == "scalar" else \
                    ("radial", a[1], a[2], dict(a[3], d=a[3]["d"] * d))
            acts[k] = out
        traces = self._congruent_traces(acts)
        dmax = max((abs(a[3]["d"]) for A in acts.values() for a in A.values() if a[0] == "radial"), default=1.0)
        rests = [dict(r, norm=dmax ** 2 * r["norm"], trace=dmax ** 2 * r["trace"]) for r in self._rests]
        out = GluedCovariance(self.domain, acts, self.groups, traces, rests, self.quad,
                              name=name or f"{B.name}{self.name}{B.name}*")
        return out

    def _congruent_traces(self, acts):
        """Per part, sum_l (2l + 1) (sum_n lam_n sum_k d_k^2 ||m phi_n||_k^2 or c_l)."""
        out = []
        done = set()
        for k, A in acts.items():
            if any(a[0] == "scalar" for a in A.values()):
                t = sum((2 * l + 1) * a[1] for l, a in A.items())
                out.append(Enclosure(t, t * (1 - 1e-13), t * (1 + 1e-13), DERIVED, "sum_l (2l + 1) c_l"))
                continue
            g = next(iter(A.values()))[3]["group"]
            if g in done:
                continue
            done.add(g)
            t = 0.0
            for kk in self.groups[g]:
                a_, b_ = _interval(self.domain.summands[kk])
                for l, a in acts[kk].items():
                    t += (2 * l + 1) * a[3]["d"] ** 2 * float(a[2] @ mgram(a[1], a[3]["m"], a_, b_, self.quad.get(kk),
                                                                          diag=True))
            out.append(Enclosure(t, t * (1 - 1e-12), t * (1 + 1e-12), DERIVED, "the known part after the congruence"))
        return out

    def degree_spectrum(self):
        """(eigenvalues, multiplicities) of the known part: per degree and part the spectrum of the radial operator
        (module docstring (a), with the pieces' scalars), each with multiplicity 2l + 1; and (tail, tail_max) of the
        omitted part (for pg.WeightedChiSquare)."""
        lam, mult = [], []
        for k, A in self.acts.items():
            for l, a in A.items():
                if a[0] == "scalar":
                    lam.append(np.array([a[1]]))
                    mult.append(np.array([2 * l + 1.0]))
        for g, ks in enumerate(self.groups):
            for l in self.acts[ks[0]]:
                P, lw = self.acts[ks[0]][l][1], self.acts[ks[0]][l][2]
                G = 0.0
                for kk in ks:
                    a = self.acts[kk][l]
                    a_, b_ = _interval(self.domain.summands[kk])
                    G = G + a[3]["d"] ** 2 * mgram(P, a[3]["m"], a_, b_, self.quad.get(kk))
                sq = np.sqrt(np.maximum(lw, 0.0))
                ev = np.maximum(np.linalg.eigvalsh(sq[:, None] * G * sq[None, :]), 0.0)
                lam.append(ev)
                mult.append(np.full(len(ev), 2 * l + 1.0))
        tail = sum(r["trace"] for r in self._rests) if self._rests else 0.0
        tmax = max((r["norm"] for r in self._rests), default=None)
        return np.concatenate(lam), np.concatenate(mult), tail, tmax

    def __repr__(self):
        return f"{self.name}: degree-structured covariance on {self.domain} ({len(self.groups)} radial parts), " \
               f"tr {self.trace()!r}"
