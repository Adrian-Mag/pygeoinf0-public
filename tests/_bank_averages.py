"""What used to be separate families for an average: each is a plain family, normalised."""
import pygeoinf0.functions as F
from pygeoinf0.functions import geometry as geo


def target(D, shape, centre=None, scale=None, edge=0.0, k=None):
    on = F.on(D)
    if shape == "box":
        return on.plateau(centre=centre, half_width=scale, edge=edge).normalised()
    if shape == "gaussian":
        return on.gaussian(centre=centre, width=scale).normalised()
    if shape == "bump":
        return on.bump(centre=centre, radius=scale, k=1.0 if k is None else k).normalised()
    if edge:
        return on.smooth_indicator(centre=centre, radius=scale, inner=scale - edge).normalised()
    return on.indicator(centre=centre, radius=scale).normalised()


def radius_window(D, *, r1, r2, edge=0.0):
    if edge:
        return F.on(D).smooth_band(coordinate="radius", lower=r1, upper=r2, edge=edge).normalised()
    return F.on(D).band_indicator(lower=r1, upper=r2).normalised()


def depth_window(D, *, d1, d2, edge=0.0):
    if edge:
        return F.on(D).smooth_band(coordinate="depth", lower=d1, upper=d2, edge=edge).normalised()
    k = geo.kind(D)
    if k in ("disk", "ball", "shell"):
        outer = D.outer if k == "shell" else D.radius
        return F.on(D).band_indicator(lower=outer - d2, upper=outer - d1).normalised()
    if k == "interval":
        return F.on(D).box_indicator(lower=D.b - d2, upper=D.b - d1).normalised()
    top = D.sides[-1].b if k == "box" else D.z.b
    normal = tuple(float(i == geo.ambient(D) - 1) for i in range(geo.ambient(D)))
    return F.on(D).slab_indicator(normal=normal, lower=top - d2, upper=top - d1).normalised()


def cap_average(D, *, centre=None, angle=None, edge=0.0):
    if edge:
        return F.on(D).smooth_indicator(centre=centre, radius=D.radius * angle, inner=D.radius * (angle - edge)).normalised()
    return F.on(D).indicator(centre=centre, radius=D.radius * angle).normalised()
