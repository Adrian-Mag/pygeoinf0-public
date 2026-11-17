"""Potentials and energies of interfaces: flat patches (o + a e1 + b e2, a, b in [0, 1]) and spheres in R^3, and
circles in R^2 (whose potential is a graded quadrature of the kernel, see circle_potential).

The potential of a patch, g(x) = int_patch k(|x - y|) dA(y), is exact at every point: kernels.polygon_potential
reduces it to one-dimensional analytic integrals (the foot of the perpendicular from x as origin, h its distance to the
plane). It is continuous when s > 1/2 (the trace condition for surfaces in R^3), with cusps where x crosses the patch
(|h|^(2s-1)) and, in its plane, across its edges. The energies are Gauss quadratures of g or of k:

    point - patch     g(x): exact
    segment - patch   int_0^L g(p + t u) dt, Gauss on cells of length <= l with breaks where the segment crosses the
                      plane or (near the plane) an edge line, graded geometrically towards every break near the patch
    patch - patch     int_0^1 [segment - patch for the line b of the first patch] |e1 x f1| / |e1| db, with breaks at
                      the lines through the singular points of the second patch (its corners when the planes are
                      parallel, else the points where its edges cross the first plane)
    a patch itself    2 J^2 int_0^1 int_0^1 (1 - a)(1 - b) [k(|a e1 + b e2|) + k(|a e1 - b e2|)] da db, J = |e1 x e2|
                      (the autocorrelation of the parallelogram), by a Duffy split at the corner a = b = 0, graded
                      towards it, the innermost cell exactly (int_0^d u k(u r) du = G(0, d r) / r^2)

The potential of a sphere of radius R at distance r from its centre is exact in closed form (rho^2 = r^2 + R^2 -
2 r R cos(theta)): g(r) = (2 pi R / r) int_|r-R|^(r+R) k(rho) rho drho = (2 pi R / r) G(|r - R|, r + R), with g(0) =
4 pi R^2 k(R); a sphere's own energy is 4 pi R^2 g(R) = 8 pi^2 R^2 G(0, 2 R). With segments and patches the same graded
quadratures apply (breaks where a segment crosses the sphere, lines tangent to its trace on a plane); two spheres:
(2 pi R1 / D) int_|D-R1|^(D+R1) g2(rho) rho drho, D the distance of the centres (exact when they are concentric).

The quadratures take their order n (Gauss nodes per cell); pairs.py evaluates them at LOW and HIGH and reports the
difference as the error estimate.
"""
import numpy as np

from . import kernels as KR
from .kernels import polygon_potential
from ...quadrature import gauss

RATIO, DEPTH = 0.15, 1e-12           # geometric grading towards singular points (bounded integrands), DEPTH * cell
LOW, HIGH = 12, 16                   # Gauss nodes per cell


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def point_patch(kern, X, O, E1, E2):
    """int over patch i of k(|x_i - y|) dA(y), rowwise (arrays (m, 3))."""
    N = _unit(np.cross(E1, E2))
    U1 = _unit(E1)
    U2 = np.cross(N, U1)
    rel = X - O
    h = np.sum(rel * N, axis=1)
    C = np.stack([np.zeros_like(E1), E1, E1 + E2, E2], axis=1) - (rel - h[:, None] * N)[:, None, :]
    V = np.stack([np.einsum("mkd,md->mk", C, U1), np.einsum("mkd,md->mk", C, U2)], axis=-1)
    return polygon_potential(kern, V, np.abs(h))


def _rule(breaks, singular, cell, n, depth=DEPTH):
    """Composite Gauss-n nodes and weights on [breaks[0], breaks[-1]]: cells of length <= cell between consecutive
    breaks, graded geometrically (ratio RATIO, down to depth * cell) towards the breaks flagged singular."""
    edges = []
    for a, b, sa, sb in zip(breaks[:-1], breaks[1:], singular[:-1], singular[1:]):
        L = b - a
        if L <= 0:
            continue
        h0 = min(0.5 * L if (sa and sb) else L, cell)
        pts = [a, b]
        k = int(np.ceil(np.log(depth) / np.log(RATIO)))
        geo = h0 * RATIO ** np.arange(k + 1)
        if sa:
            pts += list(a + geo)
        if sb:
            pts += list(b - geo)
        lo, hi = a + (h0 if sa else 0.0), b - (h0 if sb else 0.0)
        if hi > lo:
            pts += list(np.linspace(lo, hi, int(np.ceil((hi - lo) / cell)) + 1))
        edges += sorted(set(pts))
    e = np.unique(np.asarray(edges, float))
    t, w = gauss(n)
    mid, half = 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e)
    return (mid[:, None] + half[:, None] * t[None]).ravel(), (half[:, None] * w[None]).ravel()


def _affine(O, E1, E2):
    """The map x -> (a, b) with x - O = a E1 + b E2 + h N (projection onto the plane), as a (2, 3) matrix."""
    G = np.array([[E1 @ E1, E1 @ E2], [E1 @ E2, E2 @ E2]])
    return np.linalg.solve(G, np.stack([E1, E2]))


def _distance(x, O, E1, E2, N, A):
    """An upper bound of the distance from x to the patch (exact when the edges are orthogonal)."""
    ab = A @ (x - O)
    c = np.clip(ab, 0.0, 1.0)
    return float(np.hypot((x - O) @ N, np.linalg.norm((ab[0] - c[0]) * E1 + (ab[1] - c[1]) * E2)))


def segment_patch(kern, P, Q, O, E1, E2, n=HIGH):
    """int over the segment P -> Q (arc length) of the potential of the patch (O, E1, E2)."""
    L = float(np.linalg.norm(Q - P))
    u = (Q - P) / L
    N = _unit(np.cross(E1, E2))
    A = _affine(O, E1, E2)
    h0, h1 = float((P - O) @ N), float(u @ N)
    ab0, dab = A @ (P - O), A @ u
    cand = [-h0 / h1] if abs(h1) > 1e-15 else []
    for c0, dc in zip(ab0, dab):
        if abs(dc) > 1e-15:
            cand += [-c0 / dc, (1 - c0) / dc]
    l = kern.l
    br = [0.0, L] + [t for t in cand if 0 < t < L and _distance(P + t * u, O, E1, E2, N, A) < l]
    br = np.unique(br)
    sing = [_distance(P + t * u, O, E1, E2, N, A) < l for t in br]
    t, w = _rule(br, sing, 2 * l, n)
    m = len(t)
    X = P[None] + t[:, None] * u[None]
    return float(w @ point_patch(kern, X, np.broadcast_to(O, (m, 3)), np.broadcast_to(E1, (m, 3)),
                                 np.broadcast_to(E2, (m, 3))))


def patch_self(kern, E1, E2, n=HIGH):
    """int int over the patch twice of k(|x - y|) dA dA (the patch's energy)."""
    J = float(np.linalg.norm(np.cross(E1, E2)))
    l = kern.l
    rho_max = float(np.linalg.norm(E1) + np.linalg.norm(E2))
    v, wv = _rule(np.array([0.0, 1.0]), [False, False], l / rho_max, n)
    d = 1e-14 * min(l / rho_max, 1.0)                                    # the innermost cell [0, d] exactly
    u, wu = _rule(np.array([d, 1.0]), [True, False], l / rho_max, n, depth=1e-14)
    total = 0.0
    for r in (np.linalg.norm(E1[None] + v[:, None] * E2[None], axis=1), np.linalg.norm(E1[None] - v[:, None] * E2[None], axis=1),
              np.linalg.norm(v[:, None] * E1[None] + E2[None], axis=1), np.linalg.norm(v[:, None] * E1[None] - E2[None], axis=1)):
        Ku = kern(u[None, :] * r[:, None]) * (u * (1 - u))[None, :] * (1 - u[None, :] * v[:, None])
        total += wv @ (Ku @ wu) + wv @ (kern.G(np.zeros_like(r), d * r) / r ** 2)
    return 2 * J * J * total


def patch_patch(kern, O1, E1, F1, O2, E2, F2, n=HIGH):
    """int over the first patch of the potential of the second (lines of the first patch parallel to E1)."""
    if np.array_equal(O1, O2) and np.array_equal(E1, E2) and np.array_equal(F1, F2):     # the same patch
        return patch_self(kern, E1, F1, n)
    N1, N2 = _unit(np.cross(E1, F1)), _unit(np.cross(E2, F2))
    A1 = _affine(O1, E1, F1)
    corners = [O2, O2 + E2, O2 + E2 + F2, O2 + F2]
    if abs(N1 @ N2) > 1 - 1e-12:
        special = corners
    else:
        special = []
        for p, e in zip(corners, [E2, F2, -E2, -F2]):
            den = e @ N1
            if abs(den) > 1e-15:
                t = ((O1 - p) @ N1) / den
                if 0 <= t <= 1:
                    special.append(p + t * e)
    l = kern.l
    width = float(np.linalg.norm(np.cross(E1, F1)) / np.linalg.norm(E1))   # distance between the lines b = 0 and 1
    bs = [0.0, 1.0] + [float((A1 @ (p - O1))[1]) for p in special]
    if abs(F1 @ N2) > 1e-15:                    # where the second plane crosses the edges a = 0, 1 of the first patch
        bs += [float(-((O1 + a * E1 - O2) @ N2) / (F1 @ N2)) for a in (0.0, 1.0)]
    br = np.unique([b for b in bs if 0 <= b <= 1])
    near = _distance(O1 + 0.5 * (E1 + F1), O2, E2, F2, N2, _affine(O2, E2, F2)) \
        < l + 0.5 * (np.linalg.norm(E1 + F1) + np.linalg.norm(E1 - F1))
    sing = [bool(near)] * len(br)
    b, wb = _rule(br, sing, 2 * l / width, n, depth=1e-6)
    vals = [segment_patch(kern, O1 + bb * F1, O1 + bb * F1 + E1, O2, E2, F2, n) for bb in b]
    return float(np.linalg.norm(np.cross(E1, F1)) / np.linalg.norm(E1) * (wb @ np.array(vals)))


def sphere_potential(kern, r, R):
    """int over the sphere of radius R of k(|x - y|) dA(y), at distance r from its centre (elementwise)."""
    r, R = np.broadcast_arrays(np.asarray(r, float), np.asarray(R, float))
    centre = r < 1e-12 * R
    rr = np.where(centre, R, r)
    g = 2 * np.pi * R / rr * kern.G(np.abs(rr - R), rr + R)
    return np.where(centre, 4 * np.pi * R ** 2 * kern(R), g)


def segment_sphere(kern, P, Q, C, R, n=HIGH):
    """int over the segment P -> Q of the potential of the sphere (C, R): breaks where it crosses the sphere."""
    L = float(np.linalg.norm(Q - P))
    u = (Q - P) / L
    w0 = P - C
    b, c = float(w0 @ u), float(w0 @ w0) - R * R
    disc = b * b - c
    cand = [-b - np.sqrt(disc), -b + np.sqrt(disc)] if disc > 0 else []
    l = kern.l
    br = np.unique([0.0, L] + [t for t in cand if 0 < t < L])
    near = [abs(np.linalg.norm(P + x * u - C) - R) < l for x in br]
    t, w = _rule(br, near, 2 * l, n)
    return float(w @ sphere_potential(kern, np.linalg.norm(P[None] + t[:, None] * u[None] - C, axis=1), R))


def sphere_sphere(kern, C1, R1, C2, R2, n=HIGH):
    """int over the first sphere of the potential of the second."""
    D = float(np.linalg.norm(C1 - C2))
    if D < 1e-14 * max(R1, R2):
        return float(4 * np.pi * R1 ** 2 * sphere_potential(kern, R1, R2))
    a, b = abs(D - R1), D + R1
    br = np.unique([a, b] + ([R2] if a < R2 < b else []))
    rho, w = _rule(br, [abs(x - R2) < kern.l for x in br], 2 * kern.l, n)
    return float(2 * np.pi * R1 / D * (w @ (sphere_potential(kern, rho, R2) * rho)))


def patch_sphere(kern, O1, E1, F1, C, R, n=HIGH):
    """int over the patch of the potential of the sphere (lines of the patch parallel to E1)."""
    N1 = _unit(np.cross(E1, F1))
    width = float(np.linalg.norm(np.cross(E1, F1)) / np.linalg.norm(E1))
    h = float((C - O1) @ N1)
    bs = [0.0, 1.0]
    if abs(h) < R:                     # the trace of the sphere on the plane: lines tangent to it, and its crossings
        bm = float((_affine(O1, E1, F1) @ (C - O1))[1])        # with the edges a = 0, 1
        bs += [bm - np.sqrt(R * R - h * h) / width, bm + np.sqrt(R * R - h * h) / width]
        for a in (0.0, 1.0):
            w = O1 + a * E1 - C
            qa, qb, qc = F1 @ F1, 2 * (w @ F1), w @ w - R * R
            disc = qb * qb - 4 * qa * qc
            if disc > 0:
                bs += [(-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)]
    br = np.unique([x for x in bs if 0 <= x <= 1])
    near = abs(h) < R + kern.l
    b, wb = _rule(br, [near] * len(br), 2 * kern.l / width, n, depth=1e-6)
    vals = [segment_sphere(kern, O1 + bb * F1, O1 + bb * F1 + E1, C, R, n) for bb in b]
    return float(width * (wb @ np.array(vals)))


# ------------------------------------------------------------------------------------------ circles in R^2

def circle_potential(kern, r, R, n=HIGH):
    """int over the circle of radius R of k(|x - y|) ds(y) at distance r from its centre in R^2 (elementwise):
    2 R int_0^pi k(rho) dtheta, rho^2 = (r - R)^2 + 4 r R sin^2(theta / 2), theta measured from the nearest point of
    the circle; one rule for all r, graded towards theta = 0, its innermost cell [0, d] as the line integral it is to
    O(d^2) (rho^2 = delta^2 + r R theta^2 there)."""
    r = np.atleast_1d(np.asarray(r, float))
    cell = min(kern.l / R, np.pi / 4)
    d = DEPTH * cell
    th, w = _rule(np.array([d, np.pi]), [True, False], cell, n, depth=1e-14)
    delta = np.abs(r - R)
    body = kern(np.sqrt(delta[:, None] ** 2 + 4 * r[:, None] * R * np.sin(0.5 * th[None]) ** 2)) @ w
    s = np.sqrt(np.maximum(r * R, 1e-300))
    head = KR.line_integral(kern, 0.0, delta, 0.0, s * d) / s
    return np.where(r < 1e-12 * R, 2 * np.pi * R * kern(R), 2 * R * (body + head))


def segment_circle(kern, P, Q, C, R, n=HIGH):
    """int over the segment P -> Q of the potential of the circle (C, R): breaks where it crosses the circle."""
    L = float(np.linalg.norm(Q - P))
    u = (Q - P) / L
    w0 = P - C
    b, c = float(w0 @ u), float(w0 @ w0) - R * R
    disc = b * b - c
    cand = [-b - np.sqrt(disc), -b + np.sqrt(disc)] if disc > 0 else []
    br = np.unique([0.0, L] + [t for t in cand if 0 < t < L])
    near = [abs(np.linalg.norm(P + x * u - C) - R) < kern.l for x in br]
    t, w = _rule(br, near, 2 * kern.l, n)
    return float(w @ circle_potential(kern, np.linalg.norm(P[None] + t[:, None] * u[None] - C, axis=1), R, n))


def circle_circle(kern, C1, R1, C2, R2, n=HIGH):
    """int over the first circle of the potential of the second (exact reduction when they are concentric)."""
    D = float(np.linalg.norm(C1 - C2))
    if D < 1e-14 * max(R1, R2):
        return float(2 * np.pi * R1 * circle_potential(kern, R1, R2, n)[0])
    a0 = float(np.arctan2(*(C2 - C1)[::-1]))                    # angle of the direction to the other centre
    cos_x = (R1 * R1 + D * D - R2 * R2) / (2 * R1 * D)          # where |C1 + R1 u(phi) - C2| = R2
    br = [a0 - np.pi, a0 + np.pi]
    if abs(cos_x) < 1:
        br += [a0 - np.arccos(cos_x), a0 + np.arccos(cos_x)]
    br = np.unique(br)
    phi, w = _rule(br, [True] * len(br), kern.l / R1, n)
    X = C1[None] + R1 * np.stack([np.cos(phi), np.sin(phi)], axis=1)
    return float(R1 * (w @ circle_potential(kern, np.linalg.norm(X - C2, axis=1), R2, n)))
