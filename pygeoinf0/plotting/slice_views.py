"""What the slices of a field share (plans 046, 048): its members, the domain's geometry, the grids of a plane and of a sphere, and the
colour scale. The live window, the notebook widget and the static maps only draw what this gives them.

clim: "robust" (default) the 99th percentile of |f| over a sample of the whole domain, ignoring zeros, so one strong layer does not pale
the rest, and colours saturate above it; "whole" its largest value; "view" the largest value in the view, moving with it; or (low, high).

From static_slices.py (merged here by plan 063):
Short static maps of a field, for figures (plan 048).

    pg.plotting.sphere_map(f, r, ax=ax)                       latitude-longitude map on the sphere of radius r; no r on a sphere's own space
    pg.plotting.section(f, normal=(0, 0, 1), offset=0, ax=ax)  the plane through a 3-D domain, boundaries= as circles

f may be a family or a list of elements: then a grid of axes, one per member (ncols=), returned as an array. clim as in pg.plotting.slices.

From live_slices.py (merged here by plan 063):
Interactive slices of a field on a 3-D domain (plans 046, 048): pg.plotting.slices picks the backend.

    pg.plotting.slices(f, "plane")      a disk or rectangle through the domain: two angles for its normal, and an offset
    pg.plotting.slices(f, "sphere")     the sphere of radius r: r on a slider (balls and shells)

f is an element, a family or a plain list of elements (which stay formulas; a selector picks the member), on a ball, shell, box or cylinder, or
a field of a pg.PiecewiseSpace of one field on concentric pieces; a field of one with several fields goes through M.glue. backend="window" (default) is SliceWindow below. backend="widget",
this module, draws inside a notebook, in 3-D and flat, with plotly and ipywidgets: it needs a live kernel and a notebook front end that shows
widgets, and each move is evaluated in Python and cached. clim as in SliceField; boundaries= draws spheres of those radii (a
field on concentric pieces brings its own). write_html saves the widget's current view as a page without sliders.

From window_slices.py (merged here by plan 063):
Slices of a field in their own window, with sliders (matplotlib; plan 048): what pg.plotting.slices opens.

From a script the call shows the window and waits until it is closed (block=False: it returns at once, and plt.show() is yours). From a
notebook or IPython it returns at once, the window is live while the kernel runs, and the object stays alive by itself. A notebook on
the inline backend is moved to a window backend (the magic %matplotlib osx or tk), which later plots follow too, and says so;
%matplotlib inline goes back. A backend that cannot draw a window (Agg) gives a warning and a figure whose sliders move only from code.
"""


import warnings

import numpy as np

SAMPLE = 20000
MODES = ("robust", "whole", "view")


def basis(normal):
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return n, e1, np.cross(n, e1)


def unit(polar, azimuth):
    return np.array([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)])


def angles(normal):
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    return float(np.arccos(np.clip(n[2], -1.0, 1.0))), float(np.arctan2(n[1], n[0]) % (2 * np.pi))


def members(f):
    """(elements, names) of an element, a family, or a plain list of elements: a list stays formulas, with no family to fit."""
    from ..spaces.elements import Family
    if isinstance(f, Family):
        return [f[i] for i in range(len(f))], [f"{f.name}[{i}]" for i in range(len(f))]
    if isinstance(f, (list, tuple)):
        if not f:
            raise ValueError("no elements to show")
        return list(f), [u.name if sum(v.name == u.name for v in f) == 1 else f"{u.name}[{i}]" for i, u in enumerate(f)]
    return [f], [f.name]


class SliceField:
    """The field and its domain (module docstring)."""

    def __init__(self, f, clim="robust", boundaries=None, side="below", allow_surface=False):
        self.members, self.names = members(f)
        self.space = self.members[0].space
        self.side, self.member = side, 0
        from .layered import round_pieces
        pieces = self._round = round_pieces(self.space)
        self.D = pieces[0] if pieces is not None else self.space.domain
        self.surface = type(self.D).__name__ == "Sphere"
        if self.surface and not allow_surface:
            from ..core.errors import Unsupported
            raise Unsupported(f"{self.D!r} is a surface, with nothing to slice: use pg.plotting.sphere_map(f)")
        if not self.surface and getattr(self.D, "dim", None) != 3:
            from ..core.errors import Unsupported
            raise Unsupported(f"slices are for fields on 3-D domains, not {self.D!r}")
        self.box = np.asarray(self.D.bounding_box, float)
        kind = type(self.D).__name__
        self.c = np.asarray(self.D.centre, float) if kind in ("Ball", "Shell", "Sphere") else self.box.mean(axis=1)
        self.radii = {"Ball": lambda: (0.0, float(self.D.radius)), "Shell": lambda: (float(self.D.inner), float(self.D.outer)),
                      "Sphere": lambda: (float(self.D.radius), float(self.D.radius))}.get(kind, lambda: None)()
        given = () if boundaries is None else boundaries
        self.boundaries = sorted({float(x) for x in given} | (set(pieces[2]) if pieces is not None else set()))
        # how far a plane reaches from the centre: the radius of a round domain, the half-diagonal of anything else
        self.extent = self.radii[1] if self.radii is not None else float(np.linalg.norm(self.box[:, 1] - self.box[:, 0]) / 2.0)
        if isinstance(clim, str):
            if clim not in MODES:
                raise ValueError(f"clim: one of {MODES} or (low, high), not {clim!r}")
            self.mode, self.fixed = clim, None
        else:
            self.mode, self.fixed = "fixed", (float(clim[0]), float(clim[1]))
        self._scales = {}

    def values(self, X, Y, Z, member=None):
        u = self.members[self.member if member is None else member]
        shape = np.shape(X)
        P = np.stack([np.ravel(X), np.ravel(Y), np.ravel(Z)], axis=1)
        inside = np.ones(len(P), bool) if self.surface else np.asarray(self.D.contains(P), bool)
        v = np.full(len(P), np.nan)
        if inside.any():
            Q = P[inside]
            if self._round is not None:
                v[inside] = self.space.evaluate(u, Q, side=self.side)
            else:
                v[inside] = np.asarray(u(Q[:, 0], Q[:, 1], Q[:, 2]), float).ravel()
        return v.reshape(shape)

    def _sample(self, member):
        """|f| at points of the domain, not zero, with points in the support of a narrow function, which a uniform sample misses."""
        rng = np.random.default_rng(0)
        if self.surface:
            d = rng.normal(size=(SAMPLE, 3))
            P = self.c + self.radii[1] * d / np.linalg.norm(d, axis=1, keepdims=True)
        else:
            P = rng.uniform(self.box[:, 0], self.box[:, 1], (SAMPLE, 3))
        support = getattr(getattr(self.members[member], "_src", None), "support", None)
        centre, radius = getattr(support, "centre", None), getattr(support, "radius", None)
        if centre is not None and radius is not None and np.size(centre) == 3:
            d = rng.normal(size=(SAMPLE, 3))
            d = d / np.linalg.norm(d, axis=1, keepdims=True) * radius * rng.uniform(0, 1, (SAMPLE, 1)) ** (1 / 3)
            P = np.vstack([P, np.asarray(centre, float) + d])
        v = np.abs(self.values(P[:, 0], P[:, 1], P[:, 2], member))
        return v[np.isfinite(v) & (v > 0)]

    def limits(self, shown=None):
        """(low, high) for the current member; shown: the values in view, for clim="view" and when a sample finds nothing."""
        if self.mode == "fixed":
            return self.fixed
        top = None
        if self.mode != "view":
            if self.member not in self._scales:
                v = self._sample(self.member)
                self._scales[self.member] = None if not v.size else float(np.percentile(v, 99) if self.mode == "robust" else v.max())
            top = self._scales[self.member]
        if top is None:
            v = np.array([]) if shown is None else np.abs(np.asarray(shown, float))
            v = v[np.isfinite(v)] if v.size else v
            top = float(v.max()) if v.size and v.max() > 0 else 1.0
        return -top, top

    @property
    def scale_label(self):
        return {"robust": "the sample's 99th percentile, saturated above", "whole": "the sample's largest value", "view": "this view",
                "fixed": "as given"}[self.mode]

    def plane(self, normal, offset, n):
        """(X, Y, Z, u, w, values, origin, e1, e2) of the plane {x . normal = offset} about the centre, on an n x n grid."""
        nrm, e1, e2 = basis(normal)
        g = np.linspace(-self.extent, self.extent, n)
        U, W = np.meshgrid(g, g)
        o = self.c + float(offset) * nrm
        X, Y, Z = (o[k] + U * e1[k] + W * e2[k] for k in range(3))
        return X, Y, Z, g, g, self.values(X, Y, Z), o, e1, e2

    def sphere(self, r, n):
        """(X, Y, Z, longitudes, latitudes in degrees, values) of the sphere of radius r: n latitudes, 2 n - 1 longitudes."""
        lat = np.linspace(-90.0, 90.0, n)
        lon = np.linspace(-180.0, 180.0, 2 * n - 1)
        LO, LA = np.meshgrid(np.radians(lon), np.radians(lat))
        X = self.c[0] + r * np.cos(LA) * np.cos(LO)
        Y = self.c[1] + r * np.cos(LA) * np.sin(LO)
        Z = self.c[2] + r * np.sin(LA)
        return X, Y, Z, lon, lat, self.values(X, Y, Z)

    def circles(self, offset):
        """(radius of each boundary's circle on a plane at this offset, or None where it misses)."""
        return [np.sqrt(r * r - offset * offset) if r * r > offset * offset else None for r in self.boundaries]

    def near(self, r, within):
        return [x for x in self.boundaries if abs(x - r) <= within]


# ---------------------------------------------------------------- from static_slices.py (plan 063)
TITLE = 48              # characters: a bank function's name is long, and grids sit side by side


def _grid(f, ncols, draw):
    els, names = members(f)
    if len(els) == 1:
        return None
    import matplotlib.pyplot as plt
    ncols = min(ncols, len(els))
    nrows = int(np.ceil(len(els) / ncols))
    _, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False, constrained_layout=True)
    for ax in axs.ravel()[len(els):]:
        ax.set_visible(False)
    for u, name, ax in zip(els, names, axs.ravel()):
        draw(u, name, ax)
    return axs


def _show(ax, a, b, v, F, cmap, colorbar, title, xlabel, ylabel, equal):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.8) if not equal else (5.2, 4.4))
    low, high = F.limits(v)
    im = ax.imshow(np.ma.masked_invalid(v), origin="lower", extent=(a[0], a[-1], b[0], b[-1]), cmap=cmap, vmin=low, vmax=high,
                   aspect="equal" if equal else "auto", interpolation="nearest")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.85, extend="both" if F.mode == "robust" else "neither")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if len(title) <= TITLE else title[:TITLE - 1] + "…", fontsize=10)
    return ax


def sphere_map(f, r=None, ax=None, *, n=91, clim="robust", cmap="RdBu_r", colorbar=True, title=None, ncols=3, side="below"):
    """The map of f on the sphere of radius r (module docstring). Returns the axes."""
    axs = _grid(f, ncols, lambda u, name, a: sphere_map(u, r, a, n=n, clim=clim, cmap=cmap, colorbar=colorbar, title=name, side=side))
    if axs is not None:
        return axs
    F = SliceField(f, clim, side=side, allow_surface=True)
    if F.radii is None:
        from ..core.errors import Unsupported
        raise Unsupported(f"sphere_map is for balls, shells and spheres, not {F.D!r}; use pg.plotting.section")
    if r is None:
        if not F.surface:
            raise ValueError("give r: the radius of the sphere to map")
        r = F.radii[1]
    if not F.surface and not F.radii[0] <= r <= F.radii[1]:
        raise ValueError(f"r = {r} is outside {F.D!r}")
    _, _, _, lon, lat, v = F.sphere(float(r), n)
    label = F.names[0] if F.surface else f"{F.names[0]}: r = {r:.5g}"
    return _show(ax, lon, lat, v, F, cmap, colorbar, title or label, "longitude", "latitude", False)


def section(f, normal=(0.0, 0.0, 1.0), offset=0.0, ax=None, *, n=181, clim="robust", cmap="RdBu_r", colorbar=True, title=None, boundaries=None,
            ncols=3, side="below"):
    """The plane {x . normal = offset} about the domain's centre (module docstring). Returns the axes."""
    axs = _grid(f, ncols, lambda u, name, a: section(u, normal, offset, a, n=n, clim=clim, cmap=cmap, colorbar=colorbar, title=name,
                                                    boundaries=boundaries, side=side))
    if axs is not None:
        return axs
    F = SliceField(f, clim, boundaries, side)
    out = F.plane(normal, offset, n)
    ax = _show(ax, out[3], out[4], out[5], F, cmap, colorbar, title or f"{F.names[0]}: offset {offset:.4g}", "along the plane", "along the plane", True)
    t = np.linspace(0.0, 2 * np.pi, 361)
    for h in F.circles(float(offset)):
        if h is not None:
            ax.plot(h * np.cos(t), h * np.sin(t), color="k", lw=0.7, ls=":")
    return ax


# ---------------------------------------------------------------- from live_slices.py (plan 063)
NEEDS = "the notebook widget needs plotly, ipywidgets and anywidget (the extra \"interactive\" of pygeoinf0) and a live notebook kernel"


class Slices(SliceField):
    """The notebook widget of pg.plotting.slices (module docstring): .widget to display, .figure the plotly figure, .sliders by name, .view(...) to
    move it from code, .write_html(path)."""

    def __init__(self, f, kind="plane", n=None, clim="robust", boundaries=None, cmap="RdBu_r", side="below", normal=None):
        try:
            import ipywidgets
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError(f"{NEEDS}: {e}") from e
        if kind not in ("plane", "sphere"):
            raise ValueError('kind: "plane" or "sphere"')
        super().__init__(f, clim, boundaries, side)
        self.kind = kind
        if kind == "sphere" and self.radii is None:
            from ..core.errors import Unsupported
            raise Unsupported(f'kind="sphere" is for balls and shells, not {self.D!r}; use kind="plane"')
        self.n = int(n) if n is not None else (121 if kind == "plane" else 91)
        self._cache = {}
        self.clim = self.limits()
        specs = [[{"type": "scene"}, {"type": "xy"}]]
        fig = make_subplots(rows=1, cols=2, specs=specs, column_widths=[0.55, 0.45], horizontal_spacing=0.08)
        fig.add_surface(x=[[0.0]], y=[[0.0]], z=[[0.0]], surfacecolor=[[0.0]], colorscale=cmap, cmin=self.clim[0], cmax=self.clim[1],
                        showscale=False, row=1, col=1)
        fig.add_heatmap(z=[[0.0]], x=[0.0], y=[0.0], colorscale=cmap, zmin=self.clim[0], zmax=self.clim[1], row=1, col=2)
        for r in self.boundaries:                        # one trace per boundary, in 3-D and flat
            fig.add_scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="black", width=2), showlegend=False, hoverinfo="skip", row=1, col=1)
            fig.add_scatter(x=[], y=[], mode="lines", line=dict(color="black", width=1), showlegend=False, hoverinfo="skip", row=1, col=2)
        fig.update_layout(width=980, height=520, margin=dict(l=10, r=10, t=40, b=10), scene=dict(aspectmode="data"))
        if kind == "plane":
            fig.update_yaxes(scaleanchor="x", row=1, col=2)
        self.figure = __import__("plotly.graph_objects", fromlist=["FigureWidget"]).FigureWidget(fig)
        make = lambda **kw: ipywidgets.FloatSlider(continuous_update=True, readout_format=".4g", layout=dict(width="480px"), **kw)
        self.sliders = {}
        if kind == "plane":
            polar, azimuth = (0.0, 0.0) if normal is None else angles(normal)
            self.sliders["polar"] = make(value=np.degrees(polar), min=0.0, max=180.0, step=5.0, description="polar °")
            self.sliders["azimuth"] = make(value=np.degrees(azimuth), min=0.0, max=360.0, step=5.0, description="azimuth °")
            self.sliders["offset"] = make(value=0.0, min=-self.extent, max=self.extent, step=self.extent / 50.0, description="offset")
        else:
            a, b = self.radii
            self.sliders["r"] = make(value=0.5 * (a + b), min=a, max=b, step=(b - a) / 100.0, description="r")
        boxes = list(self.sliders.values())
        if len(self.members) > 1:
            self.selector = ipywidgets.Dropdown(options=list(zip(self.names, range(len(self.names)))), value=0, description="member")
            self.selector.observe(self._member_changed, names="value")
            boxes.append(self.selector)
        for s in self.sliders.values():
            s.observe(lambda _change: self._draw(), names="value")
        self.widget = ipywidgets.VBox(boxes + [self.figure])
        self._draw()

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def _member_changed(self, change):
        self.member = int(change["new"])
        self._draw()

    def _plane(self, polar, azimuth, offset):
        return self.plane(unit(polar, azimuth), offset, self.n) + (offset,)

    def _current(self):
        s = self.sliders
        key = (self.member,) + tuple(round(float(x.value), 9) for x in s.values())
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = self._plane(np.radians(s["polar"].value), np.radians(s["azimuth"].value), float(s["offset"].value)) \
                if self.kind == "plane" else self.sphere(float(s["r"].value), self.n)
        return self._cache[key]

    def _circles(self, out):
        """Per boundary: (x, y, z, u, w) of its circle on the plane, empty where the plane misses it."""
        o, e1, e2, offset = out[6:10]
        t = np.linspace(0.0, 2 * np.pi, 181)
        for h in self.circles(offset):
            if h is None:
                yield [], [], [], [], []
                continue
            u, w = h * np.cos(t), h * np.sin(t)
            yield (o[0] + u * e1[0] + w * e2[0], o[1] + u * e1[1] + w * e2[1], o[2] + u * e1[2] + w * e2[2], u, w)

    def _draw(self):
        out = self._current()
        X, Y, Z, a, b, v = out[:6]
        f = self.figure
        with f.batch_update():
            s, h = f.data[0], f.data[1]
            s.x, s.y, s.z, s.surfacecolor = X, Y, Z, v
            s.cmin, s.cmax = self.limits(v)
            h.x, h.y, h.z = a, b, v
            h.zmin, h.zmax = s.cmin, s.cmax
            name = self.names[self.member]
            if self.kind == "plane":
                for k, (x, y, z, u, w) in enumerate(self._circles(out)):
                    f.data[2 + 2 * k].update(x=x, y=y, z=z)
                    f.data[3 + 2 * k].update(x=u, y=w)
                f.layout.title.text = name
            else:
                r = float(self.sliders["r"].value)
                near = self.near(r, 0.5 * float(self.sliders["r"].step))
                f.layout.title.text = f"{name}, r = {r:.5g}" + (f" (near the boundary at {near[0]:.5g})" if near else "")

    def view(self, **values):
        """Move the sliders from code: view(polar=30, azimuth=120, offset=0.1) or view(r=5000.0); member= too."""
        for key, value in values.items():
            if key == "member":
                self.selector.value = int(value)
            else:
                self.sliders[key].value = float(value)
        return self

    def write_html(self, path):
        """The current view as a page that needs no kernel (the camera still turns; no sliders)."""
        self.figure.write_html(path)
        return path


def slices(f, kind="plane", backend="window", **kw):
    """pg.plotting.slices: backend="window" (default) its own window (SliceWindow); "widget" live in a notebook (Slices)."""
    if backend == "widget":
        return Slices(f, kind, **kw)
    if backend == "window":
        from .slice_views import SliceWindow
        return SliceWindow(f, kind, **kw)
    raise ValueError('backend: "window" or "widget"')


# ---------------------------------------------------------------- from window_slices.py (plan 063)
WINDOW = ("macosx", "tkagg", "qtagg", "qt5agg", "gtk3agg", "gtk4agg", "wxagg", "nbagg", "ipympl", "widget")
OPEN = set()            # sliders live only as long as their object: an unassigned call must keep working


def _ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def prepare_backend():
    """Whether the current backend can draw a live window, after moving an inline notebook to one."""
    import matplotlib
    name = matplotlib.get_backend().lower()
    if any(w in name for w in WINDOW):
        return True
    ip = _ipython()
    if ip is not None and "inline" in name:
        for gui in ("osx", "tk"):
            try:
                ip.run_line_magic("matplotlib", gui)
            except Exception:
                continue
            if any(w in matplotlib.get_backend().lower() for w in WINDOW):
                print(f"pg.plotting.slices: matplotlib moved to a window backend (%matplotlib {gui}); later plots open in windows too. "
                      "%matplotlib inline goes back.")
                return True
    return False


class SliceWindow(SliceField):
    """The window (module docstring): .figure, .ax, .sliders by name, .view(...) to move it from code, .save(path)."""

    def __init__(self, f, kind="plane", n=None, clim="robust", boundaries=None, cmap="RdBu_r", side="below", normal=None, block=None):
        if kind not in ("plane", "sphere"):
            raise ValueError('kind: "plane" or "sphere"')
        super().__init__(f, clim, boundaries, side)
        self.kind = kind
        if kind == "sphere" and self.radii is None:
            from ..core.errors import Unsupported
            raise Unsupported(f'kind="sphere" is for balls and shells, not {self.D!r}; use kind="plane"')
        live = prepare_backend()
        if not live:
            import matplotlib
            warnings.warn(f"matplotlib's backend {matplotlib.get_backend()} cannot draw a window: this figure's sliders move only from code",
                          UserWarning, stacklevel=3)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        self.n = int(n) if n is not None else (181 if kind == "plane" else 91)
        self._cache = {}
        rows = (3 if kind == "plane" else 1) + (len(self.members) > 1)
        self.figure = fig = plt.figure(figsize=(9.0, 6.0 + 0.3 * rows))
        bottom = 0.08 + 0.05 * rows
        self.ax = fig.add_axes([0.10, bottom + 0.06, 0.72, 0.86 - bottom])
        self.image = self.ax.imshow(np.zeros((2, 2)), origin="lower", cmap=cmap, aspect="equal" if kind == "plane" else "auto",
                                    interpolation="nearest")
        self.bar = fig.colorbar(self.image, ax=self.ax, fraction=0.04, extend="both" if self.mode == "robust" else "neither")
        self.bar.set_label(self.scale_label, fontsize=8)
        self.rings = [self.ax.plot([], [], color="k", lw=0.8, ls=":")[0] for _ in self.boundaries]
        self.sliders = {}
        y = 0.03

        def slider(key, label, low, high, value, step):
            nonlocal y
            s = Slider(fig.add_axes([0.16, y, 0.56, 0.03]), label, low, high, valinit=value, valstep=step)
            y += 0.05
            s.on_changed(lambda _value: self._draw())
            self.sliders[key] = s
        if kind == "plane":
            polar, azimuth = (0.0, 0.0) if normal is None else angles(normal)
            slider("offset", "offset", -self.extent, self.extent, 0.0, self.extent / 100.0)
            slider("azimuth", "azimuth °", 0.0, 360.0, np.degrees(azimuth), 1.0)
            slider("polar", "polar °", 0.0, 180.0, np.degrees(polar), 1.0)
            self.ax.set_xlabel("along the plane")
            self.ax.set_ylabel("along the plane")
            self.ax.set_xlim(-self.extent, self.extent)
            self.ax.set_ylim(-self.extent, self.extent)
        else:
            a, b = self.radii
            slider("r", "r", a, b, 0.5 * (a + b), (b - a) / 200.0)
            self.ax.set_xlabel("longitude")
            self.ax.set_ylabel("latitude")
            self.ax.set_xlim(-180.0, 180.0)
            self.ax.set_ylim(-90.0, 90.0)
        if len(self.members) > 1:
            slider("member", "member", 0, len(self.members) - 1, 0, 1)
        self._button = Button(fig.add_axes([0.84, 0.03, 0.12, 0.05]), "Save PNG")
        self._button.on_clicked(lambda _event: print(f"saved {self.save(f'slice_{kind}.png')}"))
        OPEN.add(self)
        fig.canvas.mpl_connect("close_event", lambda _event: OPEN.discard(self))
        self._draw()
        if live:
            interactive = _ipython() is not None or plt.isinteractive()
            plt.show(block=(not interactive) if block is None else block)

    def _current(self):
        s = self.sliders
        self.member = int(s["member"].val) if "member" in s else 0
        key = (self.member,) + tuple(round(float(x.val), 9) for k, x in s.items() if k != "member")
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = self.plane(unit(np.radians(s["polar"].val), np.radians(s["azimuth"].val)), s["offset"].val, self.n) \
                if self.kind == "plane" else self.sphere(float(s["r"].val), self.n)
        return self._cache[key]

    def _draw(self):
        out = self._current()
        a, b, v = out[3], out[4], out[5]
        self.image.set_data(np.ma.masked_invalid(v))
        self.image.set_extent((a[0], a[-1], b[0], b[-1]))
        self.image.set_clim(*self.limits(v))
        name = self.names[self.member]
        if self.kind == "plane":
            t = np.linspace(0.0, 2 * np.pi, 361)
            for line, h in zip(self.rings, self.circles(float(self.sliders["offset"].val))):
                line.set_data(([], []) if h is None else (h * np.cos(t), h * np.sin(t)))
            s = self.sliders
            self.ax.set_title(f"{name}: plane, polar {s['polar'].val:.0f}°, azimuth {s['azimuth'].val:.0f}°, offset {s['offset'].val:.4g}", fontsize=10)
        else:
            r = float(self.sliders["r"].val)
            near = self.near(r, 0.5 * float(self.sliders["r"].valstep))
            self.ax.set_title(f"{name}: r = {r:.5g}" + (f" (near the boundary at {near[0]:.5g})" if near else ""), fontsize=10)
        self.figure.canvas.draw_idle()

    def view(self, **values):
        """Move the sliders from code: view(polar=30, azimuth=120, offset=0.1), view(r=5000.0), member=1."""
        for key, value in values.items():
            self.sliders[key].set_val(value)
        return self

    def save(self, path):
        self.figure.savefig(path, dpi=200)
        return path

    def _ipython_display_(self):
        """In a notebook the window is the display: the cell shows nothing more."""
