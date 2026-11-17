"""Targets estimable by construction (plan 054; the author's decision, 2026-11-06).

With no prior, a bound of <c, m> over {m : d - G m in N} is finite only if c lies in the span of the data functionals,
ran G*. From rounded numbers that cannot be decided where c lies there to rounding: "exactly a combination of the data"
and "plus 1e-20 of an unseen direction" have the same Gram to the last digit, and their bounds are finite and +inf
(optimisers/dual/exactness.py, "No prior"). So it is decided by how the target was made, not by its numbers:

    estimable by construction   c was made inside pygeoinf0 from the data functionals themselves: G.adjoint(y) (G^T y,
                                a datum's functional G*(e_k)), G[k].representer of Functionals, linear combinations of
                                those; a property map T = L @ G (L any map on the data space; R.select(rows) of an
                                operator; sums and multiples of such maps), whose every direction T* q = G* (L* q) is so
    declared                    the caller states it: K.bound(c, estimable=True), K.image(T, estimable=True) or
                                P.estimable = True (FeasibleImage): the caller's responsibility (a target that is not
                                exactly in the span has bound +inf, and the declared value is then wrong)
    neither                     the bound has an infinite bracket where the target lies in the span only to rounding

The record is the range of what made an object, not its values. Every result of applying a linear operator A lies in
ran A; L @ R has ran(L R) in ran L and ran((L R)*) in ran R*; a sum's range lies in ran X where each term's does. So
each operator carries the ids of the operators X with ran A in ran X (`ranges`), and each element made by applying A
carries ranges(A) (`mark`). A linear combination keeps what all its terms have (marked + unmarked: nothing); anything
else (products of functions, a constant added, values copied into a new element) keeps nothing. c is estimable by
construction for data G iff G.adjoint's id is among c's: c = G* y for some y. Nothing reads G's structure or values.

The record states c in ran G*, the span of the data functionals. Where the noise set has free nuisances (a span added
to it), a bound is finite only for c in the span of the data combinations they leave: there a target in ran G* can be
UNBOUNDED, decided by the numbers as before (convex/feasible.solve), and its note says so.

The record holds what the constructions state: an adjoint given as a callable (LinearOperator(..., adjoint=f)) is
asserted to be the adjoint, as everywhere else in pygeoinf0. A realised target differs from the exact G* y by its
realisation error; the bound is that of the target meant, G* y, and its value carries that error as any other."""

DECLARED, BUILT = "declared", "by construction"


def ranges(A):
    """The ids of the operators X with ran A in ran X, by construction (A's own included); cached on A. Walked with a
    visited set: a composition may return an operator it was made from, so the records can form cycles."""
    out = A.__dict__.get("_ranges")
    if out is not None:
        return out
    ids, seen, stack = set(), set(), [A]
    while stack:
        X = stack.pop()
        if id(X) in seen:
            continue
        seen.add(id(X))
        ids.add(X._id)
        done = X.__dict__.get("_ranges")
        if done is not None and X is not A:
            ids |= done
            continue
        stack.extend(X.__dict__.get("_within", ()))
        terms = getattr(X, "terms", None)                       # Sum: ran(sum c_j A_j) in ran Y where every term's is
        if isinstance(terms, list) and terms and all(hasattr(T, "_id") for _, T in terms):
            common = None
            for c, T in terms:
                if c != 0 and T is not X:
                    r = ranges(T)
                    common = r if common is None else common & r
            ids |= common or set()
    out = A._ranges = frozenset(ids)
    return out


def within(out, left=None, right=None):
    """Record for out = left @ right (or any map made so): ran(out) in ran(left), ran(out*) in ran(right*)."""
    if left is not None:
        out.__dict__.setdefault("_within", []).append(left)
    if right is not None:
        out.__dict__.setdefault("_cowithin", []).append(right)
    out.__dict__.pop("_ranges", None)
    return out


def adjoint_made(A, adj):
    """A's adjoint adj was just made: ran(adj) = ran(A*) lies in ran(R*) for every right factor R of A."""
    for R in A.__dict__.get("_cowithin", ()):
        try:
            within(adj, R.adjoint)
        except (TypeError, NotImplementedError):                 # R has no adjoint: nothing to record
            pass


def of(x):
    """The range record of an element (empty if none)."""
    return getattr(x, "_span", None) or frozenset()


def _set(x, ids):
    try:
        x._span = ids
    except AttributeError:                                       # an object without the record (e.g. a number)
        pass
    return x


def mark(x, A):
    """x was made by applying A: it lies in ran A (and in ran X for every X in ranges(A))."""
    r = A.__dict__.get("_ranges") or ranges(A)
    old = getattr(x, "_span", None)
    try:
        x._span = old | r if old else r
    except AttributeError:                                       # an object without the record
        pass
    return x


def keep(out, x):
    """out is a multiple or a part of x: the same record."""
    return _set(out, of(x)) if of(x) else out


def meet(out, x, y):
    """out is a linear combination of x and y: what both have."""
    common = of(x) & of(y)
    return _set(out, common) if common else out


def target(c, G, declared=False):
    """How the target element c is estimable for data G: DECLARED, BUILT or None (neither)."""
    if declared:
        return DECLARED
    try:
        return BUILT if G.adjoint._id in of(c) else None
    except (TypeError, NotImplementedError):
        return None


def property_map(T, G, declared=False):
    """How every direction of the property map T is estimable for data G (T* q in ran G*): DECLARED, BUILT or None."""
    if declared:
        return DECLARED
    try:
        return BUILT if G.adjoint._id in ranges(T.adjoint) else None
    except (TypeError, NotImplementedError):
        return None


def status(how):
    """The claim status of `note(how)`: a declaration is asserted, a construction or its absence derived."""
    from .claims import ASSERTED, DERIVED
    return ASSERTED if how == DECLARED else DERIVED


def note(how):
    """The claim on a no-prior bound: which applied."""
    if how == DECLARED:
        return ("the target is declared estimable (estimable=True): taken exactly in the data's span; the caller's "
                "responsibility: a target not exactly there has bound +inf, and this value is then wrong")
    if how == BUILT:
        return ("the target is estimable by construction (made from the data functionals: G.adjoint, a datum's "
                "functional, a combination of them, or a map L @ G; core/estimable.py): taken exactly in the data's span")
    return "the target is neither declared estimable nor built from the data functionals (core/estimable.py)"
