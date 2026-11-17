"""Which spaces exist on which domains: ``pg.catalogue``.

    import pygeoinf0 as pg
    pg.catalogue.table()                         # the domain x space matrix, as text (fmt="markdown" for a notebook)
    pg.catalogue.plot()                          # the same as a coloured grid (matplotlib)
    pg.catalogue.how("Box", "Spec Mixed")        # the commands for one entry, in order: the domain, then the space on it
    pg.catalogue.how("Box")                      # a row: every space on that domain
    pg.catalogue.how(space="Spec Mixed")         # a column: every domain for that space
    pg.catalogue.probe()                         # try every combination again, now (seconds)

A domain comes first and a space is built on it, so ``how`` numbers its commands in that order and, for a refusal, also
names what does work on that domain. The matrix is *computed*, never written by hand: for each (domain, space) pair a recipe (a line of Python, shown by
``how``) is evaluated and the outcome recorded.

    ok           the space was constructed and a smoke test gave a finite norm: of a constant (L2, Lp, C), or of the
                 Riesz representer of the first bounded functional among a point evaluation and a ray (2-D), an arc
                 (sphere) or the surface of a small sphere (3-D: the intrinsic H^1 of a box or a 3-D torus, where point
                 evaluation and rays are unbounded; a flat patch is bounded in theory but did not finish in 100 s
                 there). This shows that the space is usable, not that it is correct.
    constructed  the space was constructed but none of the tried functionals is bounded on it (or none computes in the
                 time limit), so nothing was tested beyond its construction; ``how`` lists what was tried.
    refused      the package refused (``Unsupported`` and the like); ``how`` quotes the message, first line.
    n/a          the pair has no meaning: spectral boundary-condition spaces need a boundary (a closed manifold has
                 none) and a mixed condition needs two boundary parts.
    error        an unexpected exception or a time-out (none in the shipped snapshot; a test fails if there is one).

Only the abstract objects are attempted here: nothing in the matrix says that a space *is* its truncation. A realisation
is not a column: ``how`` names the one a space got, which is numerics rather than another space. A recipe fixes
illustrative parameters (length scale 0.2, the smallest integer order for which point evaluation is bounded); other
parameters change the numbers, not the status, except where ``how`` says an order must be large enough.

The result of ``probe()`` is shipped as a snapshot (``spaces_snapshot.json`` in the package), which ``table``, ``plot``
and ``how`` read, so they are instantaneous. ``tests/test_spaces_catalogue.py`` fails if the snapshot differs from a
fresh probe; after an intended change of the package run ``pg.catalogue.write_snapshot()`` and commit the file.

``probe()`` also records some checks beyond the matrix (``result["extras"]``): fractional orders, and a damped solve with
three functionals, for a few spaces.

Limits: a time limit per attempt uses ``signal.setitimer`` and therefore applies on the main thread of POSIX systems
only (elsewhere attempts run without limit); ``memory_mb=`` bounds the address space of each attempt where the
``resource`` module exists. Matplotlib is imported only by ``plot``.
"""
from ..core.errors import ProbeTimeout  # noqa: F401  (raised by probe's time limit)
from .probing import _limits, differences, probe, snapshot, stable, write_snapshot  # noqa: F401
from .registry import (BOUNDARY_SPACES, CAVEATS, DOMAIN_ALIASES, DOMAINS, FORMULAS, FRACTIONAL,  # noqa: F401
                       LENGTH_SCALE, SLOW, SNAPSHOT, SOLVES, SPACES, STATUSES)
from .views import Recipe, how, plot, table  # noqa: F401


# ------------------------------------------------------------------------------------------------- the object
class Spaces:
    """``pg.catalogue.spaces``: the catalogue of spaces on domains as one object (see the module docstring; its methods
    are pg.catalogue.how, .table, .plot, .probe). Calling it prints the table."""

    table = staticmethod(table)
    plot = staticmethod(plot)
    how = staticmethod(how)
    probe = staticmethod(probe)
    snapshot = staticmethod(snapshot)
    write_snapshot = staticmethod(write_snapshot)
    differences = staticmethod(differences)

    def __call__(self, fmt="text"):
        print(table(fmt))

    def __repr__(self):
        return table("text")


spaces = Spaces()
