"""The port: how external implementations meet pygeoinf0's objects (plan 019, EXT-001; the design is
experiments/EXP-034-external-implementations/DESIGN.md).

    Coordinates   a space's numbers for external code: a bounded export E and an import P (coordinates.py);
                  C.functional(g), the functional h -> g . E h of a tool's coordinate gradient g (a pg.Functional;
                  the derivative of a scalar map and its Frechet gradient are spaces/dual.py's, plan 068 P2)
    External      a map given by external code on numbers, through coordinates, with a declared dom and derivative
                  providers (jvp/vjp, jacobian, measures) (external.py)
    taylor_test   checks of a supplied derivative: Taylor rates and the adjoint identity, as checked claims (taylor.py)
    complex_linearity  its companion on complex spaces: A J = J A on random directions, as a checked claim
    plugins       the rules for plugins beside the package: Plugin, register, load, available, check (plugins.py)
    fields        what a plotting or export plugin needs: an element's pieces, fields and members, each piece's chart,
                  values per piece and their accuracy (fields.py; plan 056)
    generate      problems from profiles: counterparts of graph elements (core:Name, plugin:name/object@version)
                  composed into objects, with provenance (generate.py; plan 020 P5)

A classic discretisation by a basis is not external code: the span is pg.Span(basis, space=M) (convex/subspaces.py)
and its approximation role pg.Discretisation(V) (operators/galerkin.py; plan 068 P6).

pygeoinf0 contains no application code: ray tracers, PDE solvers and AD tools stay in user code or in plugins beside
the package, which connect through this port (plugins.py); the core nonlinear drivers are in pygeoinf0.optimisers.
"""
from . import plugins
from .coordinates import Coordinates
from .external import External
from .fields import Chart, Fields, Piece, fields
from .generate import Generated, generate, resolve
from .taylor import ComplexLinearityResult, TaylorResult, complex_linearity, taylor_test

__all__ = ["Chart", "ComplexLinearityResult", "Coordinates", "External", "Fields", "Generated",
           "Piece", "TaylorResult", "complex_linearity", "fields", "generate", "plugins", "resolve", "taylor_test"]
