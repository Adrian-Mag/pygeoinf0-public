"""Products E V for spectral spaces with Neumann/Dirichlet faces (reflection images, green.Reflections) by the spectral
engine of plan 005 on a torus (EXP-026 NOTES §15): a prototype, chosen in registry.py under settings.engine =
"spectral".

The images of y in the box [lo, hi] are, per axis, y + 2 n L with sign (a b)^n and 2 lo - y + 2 n L with sign
a (a b)^n (a, b the signs of the faces lo, hi: +1 Neumann, -1 Dirichlet; L = hi - lo). They are a group G of
isometries with the character g -> s_g. Its translations by P = 2 L (a = b) or P = 4 L (a != b: the extension is
antiperiodic with period 2 L) have sign 1, so G = G0 x (lattice P Z^d) with the finite set G0 of 2 (a = b) or 4
(a != b) copies per axis, all in one cell [2 lo - hi, 2 lo - hi + P):

    a = b     y (+1), 2 lo - y (a)                                      P = 2 L
    a != b    y (+1), 2 lo - y (a), y + 2 L (a b), 2 lo - y + 2 L (b)    P = 4 L

With K_per the lattice sum of the Matern kernel on the torus of periods P (the definition of the torus engine) and
the signed measure mu~ = sum_(g in G0) s_g g_# mu, since K_per is invariant under every g (they normalise the lattice)
and s_g s_h = s_(g^-1 h):

    <mu~_i, K_per mu~_j> = sum_(g, h in G0) s_g s_h <mu_i, K_per (g^-1 h)_# mu_j> = |G0| sum_(g in G) s_g <mu_i, K g_# mu_j>
                         = |G0| E_ij,

E the energies of the image sum. So E V = (E~ V) / |G0|, E~ the energies of the atom set A~ (|G0| signed copies of
every atom of A, owned by the same measure) on the torus, by spectral.SpectralEngine: its tapered NUFFT far field,
exact corrections on the segment pairs closer than r_corr (among them a segment and its mirror copy across a face it
touches or approaches), its two-resolution estimate of the relative error of products (the same relative error for
E, since the factor |G0| is exact) and its accuracy contract (settings.accuracy). Matrices and diagonals stay those
of the exact image sum (energy.ExactEngine with images).

Scope: segment atoms in 2-D and 3-D whose length + r_corr is below the smallest period (unsupported()); else the
registry falls back to the H-matrix (with images), as for the torus engine.
"""
import itertools

import numpy as np

from .atoms import Atoms, Kind

PROBES = 8               # Gaussian probes of the two-resolution estimate (the torus engine's default 1 underestimated
                         # the error of Dirichlet image sums up to 6 x at 1000 rays: EXP-026 NOTES §15)


def copies(images):
    """(G0 as a list of (sign, R, t) with R a vector of +-1, periods P, cell origin) for a green.Reflections."""
    axes, periods, origin = [], [], []
    for lo, hi, a, b in zip(images.lo, images.hi, images.s_lo, images.s_hi):
        L = hi - lo
        items = [(1.0, 1.0, 0.0), (a, -1.0, 2 * lo)]
        if a != b:
            items += [(a * b, 1.0, 2 * L), (b, -1.0, 2 * lo + 2 * L)]
        axes.append(items)
        periods.append(2 * L if a == b else 4 * L)
        origin.append(2 * lo - hi)
    G0 = [(float(np.prod([c[0] for c in combo])), np.array([c[1] for c in combo]), np.array([c[2] for c in combo]))
          for combo in itertools.product(*axes)]
    return G0, np.array(periods), np.array(origin)


def extend(A, G0):
    """A~: the signed copies g A (weights times s_g) of the atoms of A for g in G0, owned by the same measures."""
    parts = [Atoms(A.P * R + t, A.E1 * R, A.E2 * R, A.radius, A.kind, s * A.weight, A.owner, A.n, A.piece)
             for s, R, t in G0]
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])          # noqa: E731
    return Atoms(cat("P"), cat("E1"), cat("E2"), cat("radius"), cat("kind"), cat("weight"), cat("owner"), A.n,
                 cat("piece"))


def unsupported(engine, A):
    """Why the ImagesSpectralEngine engine does not compute products for the atom set A (None when it does)."""
    if np.any(A.kind != Kind.SEGMENT):
        return "atoms other than segments (the prototype handles segments only)"
    from .spectral import unsupported as torus_unsupported
    return torus_unsupported(engine.torus.exact, A)


class _Extended:
    """What the route keeps on an atom set A (as A.spectral): its extension A~, whose own .spectral is the torus plan."""

    def __init__(self, atoms, n_copies):
        self.atoms, self.n_copies, self.key = atoms, n_copies, None


class ImagesSpectralEngine:
    """Products E V of the image sum by the torus spectral engine on signed copies (module docstring); everything
    else from the exact engine with images. torus: a spectral.SpectralEngine on the exact engine of the torus of
    copies(exact.images) (both constructed in registry.py)."""

    name = "spectral on the extended torus (signed copies; tapered NUFFT far field, exact corrections)"

    def __init__(self, exact, torus):
        self.exact, self.kernel, self.torus = exact, exact.kernel, torus
        self.periods, self.origin, self.images = exact.periods, exact.origin, exact.images
        self.G0 = copies(exact.images)[0]

    def shifts(self, extent=0.0):
        return self.exact.shifts(extent)

    def extent(self, *As):
        return self.exact.extent(*As)

    def energy(self, A, B):
        return self.exact.energy(A, B)

    def self_energies(self, A, idx):
        return self.exact.self_energies(A, idx)

    def self_energies_all(self, A):
        return self.exact.self_energies_all(A)

    def extended(self, A):
        if not isinstance(A.spectral, _Extended):
            A.spectral = _Extended(extend(A, self.G0), len(self.G0))
        return A.spectral.atoms

    def plan(self, A):
        return self.torus.plan(self.extended(A))

    def apply(self, A, V):
        return self.torus.apply(self.extended(A), V) / len(self.G0)

    def estimate(self, A, V=None):
        return self.torus.estimate(self.extended(A), V)

    def estimate_data(self, A, c, sigma):
        return self.torus.estimate_data(self.extended(A), c, len(self.G0) * np.asarray(sigma, float))

    def claims(self, A):
        return self.torus.claims(self.extended(A))

    def describe(self, A):
        P = self.plan(A)
        per = " x ".join(f"{p:.3g}" for p in P.periods)
        return (f"{self.torus.describe(self.extended(A))}, on the torus of periods {per} with {len(self.G0)} signed "
                "copies of every segment (the reflection images as a lattice sum: EXP-026 NOTES §15)")
