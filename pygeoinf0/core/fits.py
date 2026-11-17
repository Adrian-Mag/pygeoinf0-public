"""Fits from samples and the resolution they state (plan 054, narrow bumps, option C; the author's decision of 2026-11-06).

A callable is known only where it is sampled: a feature narrower than the samples' spacing that lies between them is not
seen, and the fit's own error estimate does not see it either (P3: a bump of width 1e-4 on [0, 1] lost entirely, estimate
1e-15). Every element fitted from samples of a callable, and every element made by pointwise arithmetic that is fitted
again from samples (np.exp(-((M.x - c) / w) ** 2)), therefore states in its claims (key "resolution") the resolution its
samples achieved:

    s        the largest spacing of the samples the fit took, along any natural axis of the domain, in lengths (an angle
             times the largest radius it is taken at): core/sampling.py, measured on the calls the fit made
    r = 2 s  its resolution: a feature of width >= r holds at least two samples along each axis, where the fit's own
             convergence test sees it; a narrower one may lie between samples and be lost without a warning

resolution=w (M.element(f, resolution=w), M.family(..., resolution=w), f.with_resolution(w), X(parts, resolution=w))
declares the narrowest feature: the fit samples at spacing <= w / 2 (interval: legendre's scout; spherical harmonics and
radial harmonics: the angular degree; cells: the quadrature; other realisations: a check against the callable on a lattice
of spacing w / 2, with breaks added around what it missed and the fit repeated, up to REPEATS times), and its claim states
w as met, or says that it is not (with a RealisationWarning). The cost: (2 L / w)^d samples on a domain of size L.

Warning period: a fit with neither breaks=, support= nor resolution= (bank functions bring their support; pygeoinf0's own
formulas, coordinates and constants are exact) gives a ResolutionWarning naming the choices, once per callable;
pg.settings.resolution_policy = "refuse" refuses it instead (ResolutionUnstated), the later default.

Bank functions (plan 054 fix2, the author's decision of 2026-11-06): where their support gives the fit no breaks (a
Gaussian's is the whole domain) the feature width their parameters state (BankFunction.feature_width) is the fit's
resolution, as if declared (how "bank"): sampled at w/2 and checked; where it is not met, or the realisation refuses it
(a spherical degree above the cap), the fit is made as before and a RealisationWarning says so (never a refusal).

Constructions inside pygeoinf0 (the record's how): a part or piece of a callable the user gave keeps the user's record
(one warning for all its pieces; a declared resolution is passed on), a re-fit of an element's realisation inherits that
element's stated resolution ("inherited", no warning), a kernel realised again at a tighter tolerance inherits the
kernel's ("inherited").
"""
import contextlib
import os
import sys
import warnings

import numpy as np

from .claims import CHECKED, ORDER, THEOREM, UNKNOWN, Claim
from .errors import RealisationError, RealisationWarning, Refused, ResolutionUnstated, ResolutionWarning
from . import sampling

REPEATS = 4                       # lattice checks, each followed by a fit with breaks around what it missed
BARE = ("bare", "arithmetic")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATHER = []
_BANK = []


class Record:
    """How an element or family fitted from samples comes to its resolution (module docstring). how: "declared",
    "breaks", "support", "bank", "formula", "bare", "arithmetic" or "inherited" (parent: the element whose stated
    resolution it inherits); w: the declared resolution; s: per column, the spacing measured at the fit; met: whether
    the declared resolution was met, and how (note)."""

    __slots__ = ("how", "w", "parent", "root", "warned", "s", "met", "note", "name", "col", "parts")

    def __init__(self, how, w=None, parent=None, root=None, name="f", col=0, parts=None):
        self.how, self.w, self.parent, self.name, self.col = how, (None if w is None else float(w)), parent, name, col
        self.parts = parts                                            # an element of a direct sum: {k: its part}
        self.root = self if root is None else root                   # whose warning covers this one
        self.warned, self.s, self.met, self.note = False, None, None, ""
        if self.w is not None and not self.w > 0:
            raise ValueError(f"resolution= is a positive feature width, not {w!r}")

    @property
    def bare(self):
        return self.how in BARE and self.w is None

    def piece(self, w=None):
        """The record of a piece or part of the same callable: the same warning, the same declared resolution."""
        return Record(self.how, self.w if w is None else w, self.parent, self.root, self.name)


@contextlib.contextmanager
def as_bank():
    """Elements made inside are a bank function's (BankFunction.element passes its formula on): not bare."""
    _BANK.append(True)
    try:
        yield
    finally:
        _BANK.pop()


def record_of(f, breaks, support, resolution, name, bank):
    """The record of space.element(f, breaks=, support=, resolution=) for a callable f."""
    bank = bank or bool(_BANK)
    how = "declared" if resolution is not None else "support" if support is not None else "bank" if bank else \
        "breaks" if breaks is not None else "bare"
    return Record(how, resolution, name=name)


def bank_record(rec, w):
    """rec of a bank function fitted with no breaks: its parameters' feature width w (None: none stated) as its
    resolution, how "bank" (module docstring)."""
    if w is not None and w > 0:
        rec.how, rec.w = "bank", float(w)
    return rec


def bank_width(e):
    """e, a bank function's element with its feature width as resolution (BankFunction.element): its record says so
    (how "bank": unmet or refused by the realisation, it warns instead of refusing)."""
    if getattr(e, "_res", None) is not None:
        e._res.how = "bank"
    return e


def inherited(parent, name=None, col=0):
    """The record of a re-fit of parent's realisation, a member of a family, or a kernel realised again: parent's
    stated resolution (column col of a family)."""
    return Record("inherited", parent=parent, name=name or getattr(parent, "name", "f"), col=col)


def piece(e, u):
    """e, u's formula fitted on a part of the domain (a restriction, a piece of a split), keeps u's record: one warning
    for all of u's pieces, and u's declared resolution is the piece's."""
    if getattr(u, "_res", None) is not None:
        e._res = u._res.piece()
    return e


def refit(e, *parents, how="inherited"):
    """e, fitted again from samples of the realisations of parents (an inclusion, a restriction of a realised element,
    a glued element): it inherits their stated resolutions (how="internal": a construction of pygeoinf0's own)."""
    e._res = Record(how, parent=parents if len(parents) > 1 else parents[0] if parents else None, name=e.name)
    return e


# ------------------------------------------------------------------------------------------------ fitting
def fit(B, fun, N, breaks, rec, family=False, sparse=None):
    """(R, unresolved) of fun (a callable of the Cartesian coordinates, fun(*X, idx) for a family) in backend B, with
    rec's resolution: refused or warned when rec is bare (settings.resolution_policy), sampled at spacing <= w / 2 when
    declared; rec.s holds the spacing measured per column."""
    if hasattr(B, "from_base"):                         # formulas realised by another backend (RKHS: its L2 base)
        R, u = fit(B.base, fun, N, breaks, rec, family, False if family else sparse)
        return (B.from_base(R, u)[0], u)
    if rec is not None and rec.bare:
        _policy(rec)
    name = rec.name if rec is not None else "F"

    def run(b, how=None):
        rc = sampling.Recorder(fun, B, family)
        if how == "native":
            R, u = (B.fit_family_at(rc, N, b, rec.w, name=name), 0) if family else B.fit_at(rc, N, b, rec.w)
        elif family:
            R, u = B.fit_family(rc, N, breaks=b, sparse=sparse, name=name), 0
        else:
            R, u = B.fit(rc, N=N, breaks=b)
        return R, u, sampling.spacing(rc, B, b, N)

    if rec is None or rec.w is None:
        R, u, s = run(breaks)
    else:
        R, u, s = _at(B, fun, N, breaks, rec, run, family)
    if rec is not None:
        rec.s = s
        if rec.bare:
            _warn(rec)
    return R, u


def _at(B, fun, N, breaks, rec, run, family):
    """The fit at the declared resolution w (module docstring); sets rec.met and rec.note."""
    w = rec.w
    native = hasattr(B, "fit_family_at" if family else "fit_at")
    if native:
        try:
            R, u, s = run(breaks, "native")
        except (Refused, RealisationError) as e:
            if rec.how != "bank":
                raise
            R, u, s = run(breaks)                       # a bank function's width the realisation cannot reach
            rec.met, rec.note = False, f"the realisation refused it ({str(e)[:160]}); fitted at its default"
            _unmet(rec)
            return R, u, s
    else:
        hint = getattr(B, "resolution_hint", None)
        try:
            hinted = hint(breaks, w) if hint else breaks
            R, u, s = run(hinted)
        except (Refused, RealisationError) as e:
            if rec.how != "bank":
                raise
            R, u, s = run(breaks)
            rec.met, rec.note = False, f"the realisation refused it ({str(e)[:160]}); fitted at its default"
            _unmet(rec)
            return R, u, s
        breaks = hinted
    if np.all(s <= 0.5 * w * (1 + 1e-12)):
        rec.met, rec.note = True, f"sampled at spacing <= {float(np.max(s)):.3g} <= w/2"
        return R, u, s
    if native:
        rec.met, rec.note = False, f"sampled at spacing {float(np.max(s)):.3g} > w/2 (the realisation's limit)"
        _unmet(rec)
        return R, u, s
    ch = sampling.chart(B, breaks)
    for _ in range(REPEATS):
        out = None if ch is None else sampling.lattice_misses(B, fun, R, N, w, breaks, family)
        if out is None:
            rec.met, rec.note = False, "no chart of the domain to check it on"
            break
        axes, cuts = out
        if all(c is None for c in cuts):
            rec.met, rec.note = True, (f"checked against the callable on a lattice of {int(np.prod([len(a) for a in axes]))} "
                                       "points at spacing <= w/2")
            break
        breaks = sampling.with_breaks(breaks, cuts, len(ch[0]))
        R, u, s = run(breaks)
    else:
        rec.met, rec.note = False, f"the lattice of spacing w/2 still differs from the fit after {REPEATS} refinements"
    if not rec.met:
        _unmet(rec)
    return R, u, s


def _unmet(rec):
    what = "the feature width its parameters state" if rec.how == "bank" else "the declared resolution"
    warnings.warn(f"{rec.name}: {what}, {rec.w:g}, is not met ({rec.note}); a feature narrower than "
                  "the fit's resolution may be lost. Pass breaks= at the features' supports, or another realisation",
                  RealisationWarning, stacklevel=_outside())


# ------------------------------------------------------------------------------------------------ warning period
def _choices(rec):
    return (f"resolution=w ({rec.name}'s narrowest feature width: the fit samples at w/2)",
            "breaks= at the features' supports", "support=((lo, hi), ...) where it vanishes outside",
            "a bank function (it brings its support)")


def _policy(rec):
    from ..settings import settings
    if settings.resolution_policy == "refuse":
        what = "pointwise arithmetic re-fitted from samples" if rec.how == "arithmetic" else "a callable"
        raise ResolutionUnstated(f"{rec.name}: a fit from samples of {what} with neither breaks=, support= nor "
                                 "resolution= is refused (pg.settings.resolution_policy = 'refuse'): its samples "
                                 "cannot show a feature narrower than their spacing", _choices(rec))


def _warn(rec):
    root = rec.root
    if root.warned:
        return
    root.warned = True
    s = float(np.max(rec.s)) if rec.s is not None and len(rec.s) else float("inf")
    what = "re-fitted from samples of pointwise arithmetic" if rec.how == "arithmetic" else \
        "fitted from samples of a callable"
    res = f"resolution {2 * s:.3g}" if np.isfinite(s) else "a resolution that is not known"
    msg = (f"{rec.name}: {what} with neither breaks=, support= nor resolution=; it has {res} (a feature narrower than "
           f"that may be lost without another warning: see its claims). Choices: {'; '.join(_choices(rec))}. "
           "This will become a refusal (pg.settings.resolution_policy = 'refuse' refuses it now)")
    if _GATHER:
        _GATHER[-1].append((rec.name, res))
        return
    warnings.warn(msg, ResolutionWarning, stacklevel=_outside())


@contextlib.contextmanager
def gathered(name):
    """One ResolutionWarning for all bare fits made inside (the parts of X({...}), the members of M.family([...]))."""
    _GATHER.append([])
    try:
        yield
    finally:
        got = _GATHER.pop()
        if got:
            msg = (f"{name}: {len(got)} part(s) fitted from samples of callables with neither breaks=, support= nor "
                   f"resolution= ({'; '.join(f'{n}: {r}' for n, r in got[:6])}{'; …' if len(got) > 6 else ''}), so "
                   "a feature narrower than that may be lost without another warning (see their claims). Pass "
                   "resolution=w (the narrowest feature width), breaks= or support=. This will become a refusal "
                   "(pg.settings.resolution_policy = 'refuse' refuses it now)")
            if _GATHER:
                _GATHER[-1].extend(got)
            else:
                warnings.warn(msg, ResolutionWarning, stacklevel=_outside())


def _outside():
    """The stacklevel of the first frame outside pygeoinf0 (the user's line), for warnings.warn called from here."""
    f, level = sys._getframe(1), 1
    while f is not None and (f.f_code.co_filename.startswith(_PKG) or f.f_code.co_filename.endswith("contextlib.py")):
        f, level = f.f_back, level + 1
    return level


# ------------------------------------------------------------------------------------------------ claims
def resolution_claim(name, rec, col=None):
    """The claim (key "resolution") of an element fitted from samples: the resolution its samples achieved (stated), or
    the declared one, met or not."""
    col = rec.col if col is None else col
    s = None if rec.s is None else float(np.asarray(rec.s).ravel()[min(col, np.size(rec.s) - 1)])
    if rec.how == "inherited" and rec.parent is not None:
        parents = rec.parent if isinstance(rec.parent, tuple) else (rec.parent,)
        names = ", ".join(getattr(p, "name", "its source") for p in parents)
        base = [resolution_claim(getattr(p, "name", "its source"), p._res, rec.col) for p in parents
                if getattr(p, "_res", None) is not None]
        what = "; ".join(c.statement for c in base) or f"{names}: realised, not fitted from samples of a callable"
        own = f"; re-fitted from their values, its own samples at most {s:.3g} apart" if s is not None and np.isfinite(s) \
            else ""
        status = max((c.status for c in base), key=ORDER.index, default=THEOREM)
        return Claim(f"{name} inherits the resolution of {names} ({what})", status,
                     "; ".join(c.reason for c in base) + own, key="resolution")
    if rec.how == "formula" and rec.s is None:
        return Claim(f"{name} is a formula realised exactly (a constant or a coordinate)", THEOREM,
                     "a polynomial of degree <= 1 in its coordinates has no feature its realisation can miss",
                     key="resolution")
    seen = f"its samples are at most {s:.3g} apart along each natural axis" if s is not None and np.isfinite(s) \
        else "not fitted (kept symbolic)" if rec.s is None else "the spacing of its samples is not known"
    r = f"{2 * s:.3g}" if s is not None and np.isfinite(s) else "not known"
    if rec.w is not None:
        what = "the feature width its parameters state" if rec.how == "bank" else "declared resolution"
        if rec.met:
            return Claim(f"{name} resolves features of width >= {rec.w:g} ({what}, met)", CHECKED,
                         f"{rec.note}; {seen}", key="resolution")
        what = "feature width (from its parameters)" if rec.how == "bank" else "declared resolution"
        return Claim(f"{name}'s {what} {rec.w:g} is not met (resolution {r})", UNKNOWN,
                     f"{rec.note}; a feature narrower than {r} may be lost without a warning", key="resolution")
    how = {"breaks": "with breaks= given", "support": "with support= given", "bank": "a bank function (its support)",
           "formula": "a formula of pygeoinf0 (a coordinate, a natural coordinate, a constant) or a continuous bank "
                      "function's",
           "internal": "made inside pygeoinf0", "arithmetic": "pointwise arithmetic re-fitted from samples"}.get(rec.how, "")
    return Claim(f"{name}'s resolution is {r} (stated from its sampling, none declared)", THEOREM,
                 f"{seen}{', ' + how if how else ''}: a feature narrower than {r} lying between samples may be lost "
                 "without a warning, and its error is not in the element's error; resolution=w declares and meets one",
                 key="resolution")


def sampling_note(rec, col=0):
    """The spacing part of the "sampled" claim (core/claims.sampled_claim)."""
    while rec is not None and rec.s is None and rec.how == "inherited" and getattr(rec.parent, "_res", None) is not None:
        rec = rec.parent._res                                     # a family's member: the family's spacing
    if rec is None or rec.s is None:
        return ""
    s = float(np.asarray(rec.s).ravel()[min(col, np.size(rec.s) - 1)])
    return f" (its samples at most {s:.3g} apart along each natural axis: resolution {2 * s:.3g})" if np.isfinite(s) else ""


# ------------------------------------------------------------------------------------------------ elements
def realise(f):
    """The realisation of an element f given by a callable (Function.realisation): the fit with f's record, the check
    that it is not zero when it cannot be, and the warnings of a fit that did not converge."""
    B = f.space.backend
    if f._res is None:
        f._res = Record("internal", name=f.name)
    pl, u = fit(B, f._src, 1, f._breaks, f._res)
    err = float(B.error(pl)[0])
    nrm = float(np.sqrt(max(B.sqnorms_or(pl, 1.0)[0], 0.0)))
    why = f._expect() if f._expect is not None and nrm == 0 else None
    if why:
        from .errors import RealisationError
        raise RealisationError(f"{f.name}: the {B.realisation_name} fit is zero ({B.summary(pl)}), but {why}: "
                               "its samples missed the function. Pass breaks= at its support in the domain's "
                               "natural coordinates (a bank function gives them: f.fit_breaks(D)), or keep the "
                               "measure symbolic (f.measure())")
    unknown = np.isnan(err) and not B.errors_known                # NaN: unknown in this norm (by design)
    note = getattr(B, "projection_note", None)                    # a realisation that projects by design
    big = err > max(1e-8, 10 * f.space.tol) * max(1.0, nrm)
    if note is not None and not u and np.isfinite(err) and big:
        warnings.warn(f"{f.name}: {note}; its distance to them is about {err:.2g} (the element's error)",
                      RealisationWarning, stacklevel=4)          # where Function.realisation's caller was
    elif u or (not np.isfinite(err) and not unknown) or big:
        hint = getattr(f.space, "membership_hint", "If the function is not square integrable it is not an element of {space}")
        warnings.warn(f"{f.name}: the realisation did not fully converge ({u} unresolved piece(s), estimated "
                      f"error {err:.2g}). {hint.format(space=f.space)}; otherwise pass breaks=[...] at its "
                      f"singular points or jumps.", RealisationWarning, stacklevel=4)
    return pl, u


def sub(rec, idx):
    """The record of the members idx of a family with record rec (their spacings)."""
    if rec is None:
        return None
    out = Record(rec.how, rec.w, rec.parent, rec.root, rec.name)
    out.s = None if rec.s is None else np.asarray(rec.s)[idx]
    out.met, out.note = rec.met, rec.note
    return out


def family_of(space, f, name, breaks, sparse, resolution, params):
    """space.family(f, ...) (spaces/base.py): a list of elements or callables, or one formula with parameter arrays."""
    from ..spaces.elements import Family, Function
    B = space.backend
    if not params:
        with gathered(name):
            items = [space.element(g, resolution=resolution) if callable(g) and not isinstance(g, Function)
                     else space.element(g) for g in f]
        N = len(items)

        def fun_idx(*X_and_idx):
            *X, idx = X_and_idx
            shape = np.broadcast(*X).shape
            return np.stack([np.broadcast_to(items[i].rep(*X), shape) for i in idx], axis=-1)

        if B.sparse_families and (sparse or sparse is None):
            pick = sparse
            if pick is None:
                probe = np.unique(np.linspace(0, N - 1, min(N, 64)).astype(int))
                pick = N >= 32 and sampling.localised(fun_idx(np.linspace(B.a, B.b, 257), probe)) and 4097 * N * 4 <= 4e8
            if pick:                                               # fitted again from the members' callables
                bare = any(u._res is not None and u._res.bare and u._pl is None for u in items)
                rec = Record("bare" if bare else "internal", resolution, name=name)
                R, _ = fit(B, fun_idx, N, None, rec, family=True, sparse=True)
                return Family(space, R, name=name, res=rec)
        with gathered(name):
            return Family(space, B.hstack([e.realisation for e in items]), name=name)
    arrays = {k: np.asarray(v, float).ravel() for k, v in params.items()}
    sizes = {len(v) for v in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("all parameter arrays must have the same length")
    N = sizes.pop()

    def fun_idx(*X_and_idx):
        *X, idx = X_and_idx
        return f(*[np.asarray(x)[..., None] for x in X], **{k: v[idx] for k, v in arrays.items()})

    bank = callable(getattr(f, "fit_breaks", None)) and hasattr(f, "support")
    rec = record_of(f, breaks, None, resolution, name, bank)
    R, _ = fit(B, fun_idx, N, breaks, rec, family=True, sparse=sparse)
    return Family(space, R, name=name, res=rec)
