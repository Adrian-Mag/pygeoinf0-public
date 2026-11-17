"""Elements of spaces.

Function      an element of a function space on a domain: a class of functions equal almost everywhere; also an
              element of a direct sum (a tuple of parts, x[key]: spaces/direct_sum.py)
Family        N elements of one function space, realised together (fast batch arithmetic and application)
Vector        an element of a Euclidean space: a NumPy array that knows its space
VectorFamily  N vectors of one Euclidean space (an (N, n) array)

A Function has two ways to give values at points:
  f(x)      the canonical representative: the continuous one where the class has one, and at a jump the
            mean of the one-sided limits (the limit of averages over shrinking intervals centred at x).
            It depends only on the class, so f == g implies f(x) == g(x).
  f.rep(x)  the representative the element was built from (your callable), if there is one. Two equal
            elements can have different representatives: they may differ on a null set.

Values without a realisation, linear combinations kept as terms (plan 022 P6, P7): core/lazy.py.

From points.py (merged here by plan 063):
Points and user callables as arrays: the argument conventions of elements (split from spaces/elements.py, which
re-exports these names).

    as_array_function(f, probe)   a user's function of the coordinates as a callable on arrays (np.vectorize if needed)
    _coords(space, args)          f(x), f(x, y, z) or f(P) with P of shape (..., d) -> a tuple of d coordinate arrays
    hints(f, space, breaks)       a bank function's natural breaks, and why its fit cannot be zero
"""


import warnings

import numpy as np

from ..core import estimable, fits, lazy
from ..core.claims import CHECKED, Claim, Claims, sampled_claim
from ..core.errors import NotAnElement, Refused, SpaceMismatch
from ..core.logic import Truth
from ..core.numbers import element_values, field_scalar
from ..core.kinds import POINT
from .identity import require


# ---------------------------------------------------------------- from points.py (plan 063)
def as_array_function(f, probe):
    """Return (callable on arrays of points, vectorised?) for a Python function f of the coordinates.

    probe is a tuple of coordinate arrays (three points inside the domain). If f accepts arrays it is used
    as it is; otherwise it is wrapped in np.vectorize (same values, one Python call per point)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = np.asarray(f(*probe), dtype=float)
        if v.shape in ((len(probe[0]),), ()):
            return f, True
    except Exception:
        pass
    return np.vectorize(lambda *t: float(f(*t)), otypes=[float]), False


def _cdim(space):
    """The number of Cartesian coordinates of points of the domain (2 on a circle, 3 on a sphere)."""
    return getattr(space.domain, "cartesian_dim", space.backend.dim)


def _coords(space, args):
    """Accept f(x), f(x, y, z) or f(P) with P of shape (..., d); return a tuple of d coordinate arrays."""
    d = _cdim(space)
    if len(args) == d:
        return tuple(np.asarray(a, float) for a in args)
    if len(args) == 1 and d > 1:
        P = np.asarray(args[0], float)
        if P.shape[-1] != d:
            raise ValueError(f"points in {space} need {d} coordinates (last axis of length {d})")
        return tuple(P[..., k] for k in range(d))
    raise TypeError(f"{space} has {d} coordinate(s); got {len(args)} argument(s)")


def hints(f, space, breaks):
    """(breaks, expect) for space.element(f): a bank function on a ball or shell gives its natural-coordinate breaks
    and minimum angular degree (BankFunction.fit_breaks), and expect, a callable naming why its fit cannot be zero."""
    if not (callable(getattr(f, "fit_breaks", None)) and hasattr(f, "support")):
        return breaks, None
    D = getattr(space, "domain", None)
    if breaks is None and type(D).__name__ in ("Ball", "Shell"):
        breaks = f.fit_breaks(D)
    return breaks, lambda: f.nonzero_on(D)


def _name_of(f, default="f"):
    n = getattr(f, "__name__", None)
    return default if not n or n == "<lambda>" else n


def _wrap(name):
    return f"({name})" if any(c in name for c in " +-") else name


def _join(a, op, b):
    s = f"{a} {op} {b}"
    return s if len(s) <= 48 else f"{a.split(' ')[0]} {op} …"


class Function:
    """An element of a function space (see the module docstring for f(x) versus f.rep(x))."""

    kind = POINT                 # a point state (plan 065 A9: declared, not inferred from attributes)

    __slots__ = ("space", "_pl", "_src", "name", "_breaks", "_unresolved", "_memo", "_exact", "_terms", "_expect",
                 "_span", "_res", "__weakref__")    # _span: how it was made (core/estimable.py); _res: core/fits.Record
    __array_priority__ = 1000

    def __init__(self, space, pl=None, source=None, name="f", breaks=None, exact=None, terms=None, expect=None, res=None):
        self.space, self._pl, self._src, self.name, self._breaks = space, pl, source, name, breaks
        self._res = fits.Record("formula", name=name) if res is None and exact is not None else res
        self._unresolved, self._expect = 0, expect      # expect: () -> why the fit cannot be zero, or None (plan 035)
        self._memo = {}
        self._exact, self._terms = exact, terms      # the formula as canonical representative; terms (core/lazy.py)

    # ------------------------------------------------------------ realisation
    @property
    def realisation(self):
        """The numerical realisation (see the space's backend); computed on first use."""
        if self._pl is None and self._terms is not None:          # the merged realisation, as eager arithmetic made it
            self._pl = self.space.backend.lincomb([(c, t.realisation) for c, t in self._terms])
            self._unresolved = sum(t._unresolved for _, t in self._terms)
        if self._pl is None:
            self._pl, self._unresolved = fits.realise(self)
        return self._pl

    @property
    def error(self):
        """Estimated distance in the space's norm between this element and its realisation."""
        return float(self.space.backend.error(self.realisation)[0])

    @property
    def claims(self):
        rec = self._res
        fitted = rec is not None and (self._src is not None or rec.how == "inherited" or rec.parent is not None)
        if fitted:
            try:
                self.realisation                                      # the resolution is that of the fit's samples
            except NotAnElement:                                      # not in the space (plan 065 A1): no claim
                raise
            except Refused:                                           # kept symbolic: never fitted
                pass
        ok = self._unresolved == 0
        member = self._memo.get("membership")                         # the rule of a regularity (spaces/membership.py)
        rule = getattr(self.space, "_element_claim", None)
        if member is not None and ok:
            first = member
        elif rule is not None:
            first = rule(self, ok)
        else:
            first = Claim(f"{self.name} is in {self.space}", CHECKED if ok else "unknown",
                          "its realisation converged with finite norm" if ok else
                          "the realisation did not converge everywhere")
        out = [first]
        if fitted:
            out.append(sampled_claim(self.name, fits.sampling_note(rec, rec.col)))
            out.append(fits.resolution_claim(self.name, rec))
        elif self._src is not None:
            out.append(sampled_claim(self.name, ""))
        return Claims(out)

    def with_resolution(self, w):
        """The same element, its fit declared to resolve features of width >= w (core/fits.py): it samples at spacing
        w / 2. For an element given by a callable or made by pointwise arithmetic, before it is realised."""
        if self._src is None or self._exact is not None:
            raise TypeError(f"{self.name} is not fitted from samples of a callable: it has no resolution to declare")
        if self._pl is not None and self._res is not None and self._res.w == w:
            return self
        root = self._res.root if self._res is not None else None
        return Function(self.space, source=self._src, name=self.name, breaks=self._breaks, expect=self._expect,
                        res=fits.Record("declared", w, root=root, name=self.name))

    # ------------------------------------------------------------ values
    def __call__(self, *args):
        X = _coords(self.space, args)
        v = lazy.formula_values(self, X)
        if v is None:
            v = self.space.backend.evaluate(self.realisation, *X)[..., 0]
        return v.item() if np.ndim(v) == 0 else v               # a Python float (complex on a complex space)

    def __getitem__(self, key):
        """A part of an element of a direct sum (pg.DirectSum): x["mantle"], x[0]."""
        part = getattr(self.space, "part", None)
        if part is None:
            raise TypeError(f"{self.name} is an element of {self.space}, which is not a direct sum: it has no parts")
        return part(self, key)

    def rep(self, *args):
        """Values of the representative this element was built from (or the canonical one if none)."""
        if self._src is None:
            return self(*args)
        X = _coords(self.space, args)
        shape = np.broadcast(*X).shape
        flat = [np.broadcast_to(a, shape).ravel() for a in X]
        v = np.broadcast_to(np.asarray(self._src(*flat), float), flat[0].shape).reshape(shape)
        return float(v) if v.ndim == 0 else v

    def redefine_at(self, points, values):
        """The same element with a representative that takes the given values at the given points.
        A finite set has measure zero, so this changes the representative and not the element.
        points: shape (m,) on an interval, (m, d) on a d-dimensional domain."""
        if not getattr(self.space, "has_classes", True):
            raise ValueError(f"an element of {self.space} is a function, determined by its values: other values at "
                             "points make a different (in general discontinuous) function, not another representative")
        d = _cdim(self.space)
        pts = np.asarray(points, float).reshape(-1, d)
        vals = np.broadcast_to(np.asarray(values, float), (len(pts),))
        old = self.rep

        def src(*X):
            X = [np.asarray(a, float) for a in X]
            v = np.array(old(*X), float, copy=True)
            for p_, q_ in zip(pts, vals):
                hit = np.ones(v.shape, bool)
                for k in range(d):
                    hit &= X[k] == p_[k]
                v[hit] = q_
            return v

        return Function(self.space, pl=self.realisation, source=src, name=self.name, res=self._res)

    # ------------------------------------------------------------ arithmetic
    def _coerce(self, other):
        if isinstance(other, Function):
            require(other.space, self.space, f"{self.name} is in {self.space} but {other.name} is in {other.space}",
                    f"{self.name} and {other.name}")
            return other
        if np.ndim(other) == 0 and isinstance(other, (int, float, np.number)):
            return self.space.constant(float(other))
        return None

    def _lin(self, a, other, b, name):
        if lazy.lazy_ok(self) and lazy.lazy_ok(other):        # kept as terms (core/lazy.py); subclasses decide
            out = Function(self.space, terms=tuple(lazy.terms_of(self, a) + lazy.terms_of(other, b)), name=name)
            return estimable.meet(out, self, other)
        R = self.space.backend.lincomb([(a, self.realisation), (b, other.realisation)])
        return estimable.meet(Function(self.space, pl=R, name=name), self, other)

    def _scaled(self, a, name):
        if lazy.lazy_ok(self):
            return estimable.keep(Function(self.space, terms=tuple(lazy.terms_of(self, a)), name=name), self)
        out = Function(self.space, pl=self.space.backend.lincomb([(a, self.realisation)]), name=name)
        return estimable.keep(out, self)

    def __add__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else self._lin(1.0, o, 1.0, _join(self.name, "+", o.name))

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self                                  # so that sum([f, g, h]) works
        o = self._coerce(other)
        return NotImplemented if o is None else o._lin(1.0, self, 1.0, _join(o.name, "+", self.name))

    def __sub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else self._lin(1.0, o, -1.0, _join(self.name, "-", _wrap(o.name)))

    def __rsub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else o._lin(1.0, self, -1.0, _join(o.name, "-", _wrap(self.name)))

    def __neg__(self):
        return self._scaled(-1.0, f"-{_wrap(self.name)}")

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isinstance(other, Function):
            o = self._coerce(other)
            return Function(self.space, pl=self.space.backend.multiply(self.realisation, o.realisation),
                            name=f"{_wrap(self.name)}·{_wrap(o.name)}")
        if np.ndim(other) == 0 and isinstance(other, (int, float, np.number)) and not np.iscomplexobj(other):
            return self._scaled(float(other), f"{float(other):g}·{_wrap(self.name)}")
        if np.ndim(other) == 0 and isinstance(other, (complex, np.complexfloating)):
            scale = getattr(self.space, "scale", None)
            if getattr(self.space, "field", "R") == "C" and scale is not None:
                return scale(other, self)
            raise TypeError(f"a complex scalar times an element of the real space {self.space}: use pg.Complexified")
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if np.ndim(other) == 0 and isinstance(other, (int, float, np.number)):
            return self * (1.0 / float(other))
        if isinstance(other, Function):
            return np.divide(self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        return np.divide(other, self)

    def __pow__(self, k):
        if isinstance(k, (int, np.integer)) and k >= 0:
            out, base, e = self.space.constant(1.0), self, int(k)
            while e:
                if e & 1:
                    out = out * base
                e >>= 1
                if e:
                    base = base * base
            out.name = f"{_wrap(self.name)}^{k}"
            return out
        return np.power(self, k)

    def __abs__(self):
        return np.abs(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """np.sin(f), np.exp(-f**2), np.maximum(f, 0): the composition with the representative, refitted."""
        if method != "__call__" or kwargs:
            return NotImplemented
        if ufunc is np.add:
            return inputs[0] + inputs[1] if isinstance(inputs[0], Function) else inputs[1] + inputs[0]
        if ufunc is np.subtract:
            return inputs[0] - inputs[1] if isinstance(inputs[0], Function) else -(inputs[1] - inputs[0])
        if ufunc is np.multiply:
            return inputs[0] * inputs[1] if isinstance(inputs[0], Function) else inputs[1] * inputs[0]
        if ufunc is np.negative:
            return -inputs[0]
        if ufunc is np.true_divide and isinstance(inputs[0], Function) and np.ndim(inputs[1]) == 0 \
                and not isinstance(inputs[1], Function):
            return inputs[0] * (1.0 / float(inputs[1]))
        fs = [u for u in inputs if isinstance(u, Function)]
        space = fs[0].space
        for u in fs:
            require(u.space, space, f"{u.name} is in {u.space}, not {space}", u.name)
        parts = []
        for u in inputs:
            if isinstance(u, Function):
                parts.append(u.rep if u._src is not None else u.__call__)     # a combination: its terms' values
            elif np.ndim(u) == 0:
                parts.append(lambda *X, c=float(u): c)
            else:
                return NotImplemented
        B = space.backend
        breaks = B.merge_breaks([B.breaks(t.realisation) for u in fs for t in lazy.leaves(u)])

        def src(*X):
            return ufunc(*[p(*X) for p in parts])

        names = [u.name if isinstance(u, Function) else f"{float(u):g}" for u in inputs]
        name = f"{ufunc.__name__}({', '.join(names)})"
        return Function(space, source=src, name=name, breaks=breaks, res=fits.Record("arithmetic", name=name))

    # ------------------------------------------------------------ geometry
    def norm(self):
        return self.space.norm(self)

    def inner(self, other):
        return self.space.inner(self, other)

    def __matmul__(self, other):
        if isinstance(other, Function):
            return self.space.inner(self, other)
        return NotImplemented

    def integral(self, over=None):
        """The integral of f against the space's measure, over the whole domain or a part of it."""
        return self.space.integrate(self, over)

    def distance(self, other):
        return (self - other).norm()

    def _gap(self, other, rtol):
        """(distance, rounding level, ten times the stated realisation errors) of f and other."""
        o = self._coerce(other)
        return (self - o).norm(), rtol * (self.norm() + o.norm()), 10 * (self.error + o.error)

    def equals(self, other, rtol=1e-12):
        """f = other as elements of the space, a Truth (core/logic.py; plan 067 P1, was (bool, distance, tolerance)):
        TRUE when the distance computes to exactly 0 and the stated realisation errors are at most the rounding level
        rtol (||f|| + ||g||); FALSE when the distance exceeds both the rounding level and ten times the stated errors;
        UNKNOWN between: the distance is within what the realisations and rounding resolve (f == other reads that as
        equal). A distance at the rounding level is not equality (plan 071 P1, Mac review item 9: constant(1) and
        constant(1 + 1e-13) were TRUE)."""
        d, rounding, err = self._gap(other, rtol)
        if d > max(rounding, err):
            return Truth.FALSE
        return Truth.TRUE if d == 0 and err <= rounding else Truth.UNKNOWN

    def __eq__(self, other):
        """The tolerance ==: equal when the distance is below what the realisations can resolve (a bool; f.equals(g)
        says which of the two it is)."""
        if not isinstance(other, (Function, int, float, np.number)):
            return NotImplemented
        try:
            d, rounding, err = self._gap(other, 1e-12)
        except SpaceMismatch:
            return False
        return bool(d <= max(rounding, err))

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = None

    # ------------------------------------------------------------ display
    def on_grid(self, *axes):
        """Values on the tensor grid axes[0] x axes[1] x ... of natural coordinates (Box: x, y, z; Disk: r, phi;
        Sphere: theta, phi; ...), shape (len(axes[0]), ...). Much faster than evaluating the grid as points when the
        element is an expansion in a product family. On an interval: values at axes[0]."""
        return self.space.backend.grid_values(self.realisation, [np.asarray(a, float) for a in axes])[..., 0]

    def plot(self, ax=None, **kw):
        from ..plotting import plot_function
        return plot_function(self, ax=ax, **kw)

    def info(self):
        B = self.space.backend
        d = B.describe(self.realisation)
        if getattr(self.space, "has_classes", True):
            what = ["  an equivalence class of functions equal almost everywhere",
                    f"  representative you gave: {'yes (f.rep)' if self._src is not None else 'none (built by arithmetic)'}",
                    "  canonical representative f(...): the realised polynomial, continuous inside each piece/cell"]
        else:
            what = ["  a function: f(x) is its value at x (no representatives)",
                    "  f(...) evaluates its realisation, a polynomial on each piece, continuous across pieces"]
        lines = [f"{self.name} ∈ {self.space}"] + what + [
                 f"  realisation: {B.summary(self.realisation)}, {d['size']} coefficients (f.realisation)"]
        lines += [f"  {k}: {v}" for k, v in d.items() if k not in ("kind", "size", "pieces", "degree")]
        lines.append(f"  estimated error of the realisation: {self.error:.2g} (in the norm of the space)")
        return "\n".join(lines)

    def under_the_hood(self):
        print(self.info())

    def __repr__(self):
        if self._pl is None:
            what = f"a combination of {len(self._terms)} terms, " if self._terms is not None else ""
            return f"{self.name} ∈ {self.space}  [{what}not yet realised]"
        return f"{self.name} ∈ {self.space}  [{self.space.backend.summary(self._pl)}, est. error {self.error:.1g}]"


class Family:
    """N elements of one function space, realised together. F[i] is an element; G(F) applies G to all."""

    __slots__ = ("space", "_pl", "name", "_memo", "_span", "_res", "__weakref__")

    def __init__(self, space, pl, name="F", res=None):
        self.space, self._pl, self.name, self._res = space, pl, name, res    # _res: core/fits.Record, from samples
        self._memo = {}

    @property
    def realisation(self):
        return self._pl

    def __len__(self):
        return self._pl.N

    def __getitem__(self, i):
        B = self.space.backend
        if isinstance(i, (int, np.integer)):
            i = int(i) % len(self)
            res = None if self._res is None else fits.inherited(self, f"{self.name}[{i}]", col=i)
            return estimable.keep(Function(self.space, pl=B.take(self._pl, i), name=f"{self.name}[{i}]", res=res), self)
        idx = np.arange(len(self))[i]
        return estimable.keep(Family(self.space, B.take(self._pl, idx), name=f"{self.name}[…]", res=fits.sub(self._res, idx)),
                              self)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def sum(self):                                    # a combination of members that all share F's record: kept
        return estimable.keep(Function(self.space, pl=self.space.backend.colsum(self._pl), name=f"Σ {self.name}"), self)

    def mean(self):
        return self.sum() * (1.0 / len(self))

    def errors(self):
        """Estimated distances (in the space's norm) between the members and their realisations."""
        return np.asarray(self.space.backend.error(self.realisation), float)

    def norms(self):
        return self.space.norm(self)

    def _broadcast(self, other, a, b):
        if isinstance(other, (Function, Family)):
            require(other.space, self.space, f"{other.name} is in {other.space}, not {self.space}",
                    f"{self.name} and {other.name}")
            R = self.space.backend.lincomb([(a, self._pl), (b, other.realisation)])
            return estimable.meet(Family(self.space, R, name=self.name), self, other)
        if np.ndim(other) == 0:
            return self._broadcast(self.space.constant(float(other)), a, b)
        return NotImplemented

    def __add__(self, other):
        return self._broadcast(other, 1.0, 1.0)

    __radd__ = __add__

    def __sub__(self, other):
        return self._broadcast(other, 1.0, -1.0)

    def __mul__(self, c):
        if np.ndim(c) == 0:
            R = self.space.backend.lincomb([(float(c), self._pl)])
            return estimable.keep(Family(self.space, R, name=self.name), self)
        return NotImplemented

    __rmul__ = __mul__

    def plot(self, ax=None, max_lines=12, **kw):
        from ..plotting import plot_family
        return plot_family(self, ax=ax, max_lines=max_lines, **kw)

    def info(self):
        B = self.space.backend
        d = B.describe(self._pl)
        return (f"Family {self.name}: {len(self)} elements of {self.space}\n"
                f"  realised together: {B.summary(self._pl)}, {d['size']} coefficients (F.realisation)\n"
                f"  largest estimated error: {B.error(self._pl).max():.2g}")

    def __repr__(self):
        return f"Family {self.name} of {len(self)} elements of {self.space}  [{self.space.backend.summary(self._pl)}]"


class Vector:
    """An element of a Euclidean space: v.values is a read-only NumPy array of length n."""

    kind = POINT                 # a point state (plan 065 A9: declared, not inferred from attributes)

    __slots__ = ("space", "values", "err", "name", "_memo", "_span", "__weakref__")
    __array_priority__ = 1000

    def __init__(self, space, values, err=None, name="v"):
        v = element_values(space, values)
        if v.shape != (space.dim,):
            raise ValueError(f"an element of {space} needs {space.dim} numbers, got shape {v.shape}")
        v.flags.writeable = False
        self.space, self.values, self.err, self.name = space, v, err, name
        self._memo = {}

    def __array__(self, dtype=None, copy=None):
        return self.values if dtype is None else self.values.astype(dtype)

    def __len__(self):
        return self.space.dim

    def __getitem__(self, i):
        return self.values[i]

    def _other(self, other):
        if isinstance(other, Vector):
            if other.space != self.space:
                raise SpaceMismatch(f"{other.name} is in {other.space}, not {self.space}")
            return other.values
        return None

    def __add__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else \
            estimable.meet(Vector(self.space, self.values + o, name=_join(self.name, "+", other.name)), self, other)

    def __sub__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else \
            estimable.meet(Vector(self.space, self.values - o, name=_join(self.name, "-", other.name)), self, other)

    def __neg__(self):
        return estimable.keep(Vector(self.space, -self.values, name=f"-{_wrap(self.name)}"), self)

    def __mul__(self, c):
        if np.ndim(c) == 0 and not isinstance(c, Vector):
            c = field_scalar(self.space, c)
            return estimable.keep(Vector(self.space, c * self.values, name=f"{c:g}·{_wrap(self.name)}"), self)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, c):
        return self * (1.0 / field_scalar(self.space, c))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self.space.inner(self, other)
        return NotImplemented

    def inner(self, other):
        return self.space.inner(self, other)

    def norm(self):
        return self.space.norm(self)

    def __eq__(self, other):
        return isinstance(other, Vector) and other.space == self.space and np.array_equal(self.values, other.values)

    __hash__ = None

    def plot(self, ax=None, **kw):
        from ..plotting import plot_vector
        return plot_vector(self, ax=ax, **kw)

    def info(self):
        return (f"{self.name} ∈ {self.space}\n  stored as a NumPy array of shape {self.values.shape} "
                f"(v.values): {np.array2string(self.values, precision=4, threshold=10)}"
                + ("" if self.err is None else f"\n  estimated error per component up to {np.max(self.err):.2g}"))

    def under_the_hood(self):
        print(self.info())

    def __repr__(self):
        return f"{self.name} ∈ {self.space}: {np.array2string(self.values, precision=4, threshold=8)}"


class VectorFamily:
    """N vectors of one Euclidean space; values has shape (N, n)."""

    __slots__ = ("space", "values", "err", "name", "_memo", "_span", "__weakref__")

    def __init__(self, space, values, err=None, name="V"):
        v = element_values(space, values)
        if v.ndim != 2 or v.shape[1] != space.dim:
            raise ValueError(f"expected an (N, {space.dim}) array, got {v.shape}")
        v.flags.writeable = False
        self.space, self.values, self.err, self.name = space, v, err, name
        self._memo = {}

    def __array__(self, dtype=None, copy=None):
        return self.values if dtype is None else self.values.astype(dtype)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return estimable.keep(Vector(self.space, self.values[i], name=f"{self.name}[{int(i)}]"), self)
        return estimable.keep(VectorFamily(self.space, self.values[i], name=self.name), self)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def _other(self, other):
        if isinstance(other, (Vector, VectorFamily)):
            if other.space != self.space:
                raise SpaceMismatch(f"{other.name} is in {other.space}, not {self.space}")
            return other.values if isinstance(other, VectorFamily) else other.values[None, :]
        return None

    def _made(self, values, other=None):         # a combination keeps what its terms share (core/estimable.py)
        out = VectorFamily(self.space, values, name=self.name)
        return estimable.keep(out, self) if other is None else estimable.meet(out, self, other)

    def __add__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._made(self.values + o, other)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._made(self.values - o, other)

    def __rsub__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._made(o - self.values, other)

    def __neg__(self):
        return self._made(-self.values)

    def __mul__(self, c):
        if np.ndim(c) == 0:
            return self._made(field_scalar(self.space, c) * self.values)
        return NotImplemented

    __rmul__ = __mul__

    def plot(self, ax=None, **kw):
        from ..plotting import plot_vector_family
        return plot_vector_family(self, ax=ax, **kw)

    def __repr__(self):
        return f"VectorFamily {self.name}: {len(self)} vectors of {self.space} (values: array {self.values.shape})"
