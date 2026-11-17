"""Values without a realisation and linear combinations kept as terms (plan 022 P6, P7; spaces/elements.Function).

An element whose formula is known to be its canonical representative (a continuous bank function, M.x, a constant:
Function._exact) gives f(x) from the formula, exactly. A linear combination of such elements (a + b, c * a, sum(...))
keeps its terms (Function._terms) and gives f(x) as the combination of their values: nothing is fitted or merged.
Inner products, integrals and functionals of a combination are sums over its terms (spaces/base.py,
operators/functionals.py), each term realised alone; <f, f> and norms use the merged realisation (expanded, they
cancel). The merged realisation, the same lincomb eager arithmetic made, is built on the first operation that needs it
(products, f.error, f.info(), solvers). Elements without a formula (a solver's iterate, a product) combine eagerly, as
before, so iterations do not accumulate terms.

Which elements combine by terms is a stated protocol, not an exact type test: a Function, or a subclass whose class
sets same_values = True (its values, formula and terms are its base's; it adds no evaluation of its own: the
CalibratedFunction Problem.run returns, inference/calibration.py, plan 070 P7). A subclass that evaluates in its own way
(spaces/representers.DensityRepresenter) does not set it and keeps its own arithmetic (its _lin).
"""
import numpy as np


def combines_by_terms(f):
    """f is a Function, or of a subclass that states it keeps its base's values (module docstring)."""
    from ..spaces.elements import Function
    return type(f) is Function or (isinstance(f, Function) and type(f).__dict__.get("same_values", False) is True)


def has_formula(f):
    if f._exact is not None:
        return True
    return f._terms is not None and all(combines_by_terms(t) and has_formula(t) for _, t in f._terms)


def formula_values(f, X):
    """Values from the formula(s) at the coordinate arrays X, or None when f has none."""
    if not has_formula(f):
        return None
    shape = np.broadcast(*X).shape
    if f._exact is not None:
        flat = [np.broadcast_to(a, shape).ravel() for a in X]
        v = np.asarray(f._exact(*flat), float)
        return np.broadcast_to(v, flat[0].shape).reshape(shape).copy()
    out = np.zeros(shape)
    for c, t in f._terms:
        out += c * formula_values(t, X)
    return out


def terms_of(f, a):
    """a * f as terms: a combination's own terms (scaled), else (a, f)."""
    if f._terms is not None and f._pl is None:
        return [(a * c, t) for c, t in f._terms]
    return [(a, f)]


def lazy_ok(f):
    """Kept as terms only when every term has a formula (module docstring)."""
    return combines_by_terms(f) and has_formula(f)


def leaves(f):
    """The elements a combination is made of (f itself otherwise): their breaks are the combination's."""
    if f._terms is not None and f._pl is None:
        return [leaf for _, t in f._terms for leaf in leaves(t)]
    return [f]


def is_lazy(u):
    """A linear combination kept as its terms and not merged (spaces/elements.py, plan 022 P7)."""
    from ..spaces.elements import Function
    return isinstance(u, Function) and u._terms is not None and u._pl is None


def by_terms(space, f, g, inner):
    """<f, g> by linearity over the terms of a combination f or g (no merged realisation), else None. Not for <f, f>:
    expanded, it cancels (a difference of nearly equal terms; a CG residual went negative), so it is merged."""
    if f is g:
        return None
    if is_lazy(g):
        return sum(c * inner(space, f, t) for c, t in g._terms)
    if is_lazy(f):
        return sum(c * inner(space, t, g) for c, t in f._terms)
    return None
