"""Plots of elements, chosen by the dimension of the domain. matplotlib (and plotly for interactive 3D) are
imported only when a plot is asked for."""


def _dim(space):
    return getattr(getattr(space, "backend", None), "dim", None)


def _module(space):
    """The plotting module for a space, by (dimension, Cartesian dimension) of its domain: interval (1, 1);
    curves (1, 2-3: circle, segment, polyline, arc); planar (2, 2: box, disk, torus); surfaces in R^3 (2, 3: sphere,
    patch); volumes (3, 3); a field on the concentric pieces of a ball (a pg.PiecewiseSpace of one field: sections)."""
    from .layered import round_pieces
    if round_pieces(space) is not None:
        from . import layered as m
        return m
    d = _dim(space)
    c = getattr(space.domain, "cartesian_dim", d)
    if d == 1 and c == 1:
        from . import interval as m
    elif d == 1 or (d == 2 and c == 3):
        from . import manifold as m
    elif d == 2:
        from . import planar as m
    else:
        from . import volume as m
    return m


def slices(f, kind="plane", backend="window", **kw):
    """Interactive slices of a field on a 3-D domain, in their own window by default (plotting/slice_views.py)."""
    from .slice_views import slices as make
    return make(f, kind, backend, **kw)


def sphere_map(f, r=None, ax=None, **kw):
    """The latitude-longitude map of f on the sphere of radius r (plotting/slice_views.py)."""
    from .slice_views import sphere_map as draw
    return draw(f, r, ax, **kw)


def section(f, normal=(0.0, 0.0, 1.0), offset=0.0, ax=None, **kw):
    """The plane through a 3-D domain (plotting/slice_views.py)."""
    from .slice_views import section as draw
    return draw(f, normal, offset, ax, **kw)


def plot_function(f, ax=None, **kw):
    return _module(f.space).plot_function(f, ax=ax, **kw)


def plot_family(F, ax=None, **kw):
    return _module(F.space).plot_family(F, ax=ax, **kw)


def plot_vector(v, ax=None, **kw):
    from .interval import plot_vector
    return plot_vector(v, ax=ax, **kw)


def plot_vector_family(V, ax=None, **kw):
    from .interval import plot_vector_family
    return plot_vector_family(V, ax=ax, **kw)
