"""The dual solvers of BGP (plan 070 P4, architecture review C5 and D6; moved out of convex/ by the author's amendment 1:
convex/ holds convex analysis, this package the numerics). convex/ calls them through the names below only (the solver
interface, __all__, with the public functions of its modules reduction, exactness and exactness_reduce, called through
the module so that one name is looked up at call time; tests/test_plan070_dual.py): it states the problem (convex/feasible.py: the feasible set, the dual
phi through the sets' own support functions) and turns its sets into forms (convex/forms.py); the solvers take arrays,
forms and callbacks and return Bounds. Nothing here imports convex/.

    bound.py              Bound, LazyBound (an Enclosure with certificate and witness), CERTIFIED
    reduction.py          the span reduction (gram_blocks, reduce, Reduction), whiten, data_svd, EPS,
                          structural_zeros, floor_refusal (the refusal at the rounding floor), and the reduced
                          problems: ball_ball, ball_ball_spectral, ball_exact, none_ball, ball_slab, slab_dual, general
    spectral.py           SpectralBall and the vectorised many-direction route in a data-space decomposition's eigenbasis
    exactness.py          the general rule for EXACT in the spectral route (plan 054)
    exactness_reduce.py   the general rule in the reduce route
    nonsmooth.py          box BGP: CellProblem, NoiseSide, the certified cell dual, interval cells (plan 009 P5)
    sphere_cells.py       latitude-longitude witness cells on spheres (plan 061)
    slabs.py              slab-type noise in box BGP: the reduced cell problem, one slab by its exact 1-D dual (plan 060)

The scalar searches they use are optimisers/scalar.py (bisect, trs, trs_columns); scipy.optimize is reached through
optimisers/minimise.py.
"""
from . import exactness, exactness_reduce, reduction
from .bound import CERTIFIED, Bound, LazyBound
from .nonsmooth import (CellDual, CellProblem, NoiseSide, auto_cells, emptiness, interval_cells, plain_interval,
                        solve_cells)
from .reduction import (EPS, Reduction, ball_ball, ball_ball_spectral, ball_exact, ball_slab, data_svd, floor_refusal,
                        general, gram_blocks, none_ball, reduce, residual_misfit, slab_dual, structural_zeros, whiten)
from .slabs import one_slab, reduced
from .spectral import SpectralBall
from .sphere_cells import sphere_cells

__all__ = ["exactness", "exactness_reduce", "reduction", "CERTIFIED", "EPS", "Bound", "LazyBound", "floor_refusal",
           "structural_zeros", "CellDual", "CellProblem", "NoiseSide", "auto_cells", "emptiness", "interval_cells",
           "plain_interval", "solve_cells", "Reduction", "ball_ball", "ball_ball_spectral", "ball_exact", "ball_slab",
           "data_svd", "general", "gram_blocks", "none_ball", "reduce", "residual_misfit", "slab_dual", "whiten",
           "one_slab", "reduced", "SpectralBall", "sphere_cells"]
