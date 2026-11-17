"""The test problem shared by both notebooks: one square, the same rays, the same true model, the same noisy data.

Nothing here depends on how the model is later represented (pixels or functions): the data are line integrals of
the true model along each ray, computed accurately by Gauss-Legendre quadrature on the ray itself.
"""
import numpy as np

N_STATIONS, N_SOURCES = 20, 30
SIGMA = 0.01                        # standard deviation of the noise added to every datum


def geometry(seed=0):
    """Stations on the boundary of the unit square, sources inside; one ray from every source to every station."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 4, N_STATIONS)                       # position along the perimeter
    sta = np.empty((N_STATIONS, 2))
    for k, t in enumerate(s):
        if t < 1:   sta[k] = (t, 0)
        elif t < 2: sta[k] = (1, t - 1)
        elif t < 3: sta[k] = (3 - t, 1)
        else:       sta[k] = (0, 4 - t)
    src = rng.uniform(0, 1, (N_SOURCES, 2))
    p = np.repeat(src, N_STATIONS, axis=0)                  # ray i goes from p[i] to q[i]
    q = np.tile(sta, (N_SOURCES, 1))
    return src, sta, p, q


def true_model(x, y):
    """A positive and a negative anomaly on a gentle background."""
    return (np.exp(-((x - 0.3) ** 2 + (y - 0.65) ** 2) / 0.02)
            - 0.7 * np.exp(-((x - 0.7) ** 2 + (y - 0.3) ** 2) / 0.03)
            + 0.2 * np.sin(2 * np.pi * x) * np.cos(np.pi * y))


def ray_integrals(p, q, f, nodes=200):
    """The integral of f along each straight ray from p[i] to q[i] (Gauss-Legendre with `nodes` points per ray)."""
    t, w = np.polynomial.legendre.leggauss(nodes)
    t, w = 0.5 * (t + 1), 0.5 * w                            # nodes and weights on [0, 1]
    L = np.linalg.norm(q - p, axis=1)
    X = p[:, None, :] + t[None, :, None] * (q - p)[:, None, :]
    return L * (f(X[..., 0], X[..., 1]) @ w)


def data(seed=1):
    """(noise-free data, noisy data) for the geometry above; the noise is Gaussian with standard deviation SIGMA."""
    _, _, p, q = geometry()
    d_clean = ray_integrals(p, q, true_model)
    d = d_clean + SIGMA * np.random.default_rng(seed).standard_normal(len(d_clean))
    return d_clean, d
