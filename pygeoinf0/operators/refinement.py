"""The a-posteriori Gram errors behind a later inverse (plan 054, the general rule; the verification levels).

Where a Gram's first-order statement leaves no correct digit for its inverse (solvers.inherited), the Gram is
formed again from more accurate inputs and its error stated a posteriori, by the triangle inequality

    ||G~ - G|| <= ||G~ - G^(d)|| + ||G^(d) - G||  <=  ||G~ - G^(d)|| + e^(d),

G^(d) the Gram at the deepest refinement reached and e^(d) its own first-order statement. pg.settings.verification = d
sets the depth (d >= 1 here; d = 0 forms nothing, solvers.inherited):
  - realised kernels (_refined): realised again from their callables at tol / REFINE^j, j = 1 .. d, one after the other;
    each must state errors at most 1/10 of the previous one's, or the chain stops at the previous level (stated);
  - compressed energies (_recompressed): one Gram with the energies compressed again at compression / REFINE^d (one
    H-matrix build; where it cannot be formed or is not more accurate, the next shallower level is tried).
Each level is about REFINE = 100 times tighter, so e^(d), a first-order statement (about 100 x the true error of the
kernels' Gram, plan 054 P3, a second-order quantity), falls by about that factor per level and the sum approaches
||G~ - G||. The chain's differences ||G^(j-1) - G^(j)|| are reported: they show whether the refinements settle.
A bound where the deepest level's stated errors are bounds: the kernels' are estimates (Family.errors()), the H-matrix's
an estimate, so the sum is an estimate there (status checked).
"""
import numpy as np

REFINE = 100.0          # each level: the kernels realised again at tol / REFINE, the energies compressed at comp / REFINE


def depth():
    """The verification depth asked (pg.settings.verification), at least 1 for an a-posteriori statement."""
    from ..settings import settings
    return max(1, int(settings.verification))


def _realise(G, M, tol):
    """(Gram, errors, norms) of G's kernels realised again from their callables on M at tol, or None."""
    import copy
    import warnings
    from ..spaces.elements import Function
    from ..core.errors import RealisationWarning
    M2 = copy.copy(M)
    M2._setup(M.measure, tol)
    if type(M2.backend) is not type(M.backend):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RealisationWarning)          # judged by the errors it states
        F = M2.family([Function(M2, source=f, breaks=b) for f, b in G._sources], name="k'")
        e2 = np.asarray(F.errors(), float)
        A2 = np.asarray(M2.inner(F, F), float)
    return A2, e2, np.sqrt(np.maximum(np.diag(A2), 0.0))


def _refined(G, Gr):
    """(||G~ - G^(d)||_2, tol^(d), e^(d), chain) with G^(d) the Gram of the kernels realised again from their callables
    at tol^(d) = tol / REFINE^d, d the deepest level reached of pg.settings.verification asked, e^(d) = 2 ||e_k|| ||n_k||
    + ||e_k||^2 its first-order statement from the errors its kernels state, and chain = [(tol^(j), ||G^(j-1) -
    G^(j)||_2)] for j = 1 .. d, with the depth asked and why the chain stopped short of it (module docstring); or None where not even one level can be formed
    here (no callables, another space than L2, a realisation that does not converge, tol^(1) below 1e-15, or errors
    not at most 1/10 of the first's)."""
    M = G.kernels.space
    if G._sources is None or type(M).__name__ != "L2" or not isinstance(Gr, np.ndarray):
        return None
    asked, prev, e_prev = depth(), np.asarray(Gr, float), np.asarray(G._kerr, float)
    tol, chain, best, why = M.tol, [], None, ""
    for _ in range(asked):
        tol = tol / REFINE
        if tol < 1e-15:
            why = f"tol {tol:.2g} is below 1e-15"
            break
        got = _realise(G, M, tol)
        if got is None:
            why = f"no realisation of the same kind at tol {tol:.2g}"
            break
        A2, e2, n2 = got
        if not (np.all(np.isfinite(e2)) and e2.max(initial=0.0) <= 0.1 * e_prev.max(initial=0.0)):
            why = f"the realisation at tol {tol:.2g} states errors not 10 x smaller"   # stop at the previous level
            break
        chain.append((tol, float(np.linalg.norm(prev - A2, 2))))
        best = (A2, tol, 2.0 * float(np.linalg.norm(e2) * np.linalg.norm(n2)) + float(e2 @ e2))
        prev, e_prev = A2, e2
    if best is None:
        return None
    return float(np.linalg.norm(Gr - best[0], 2)), best[1], best[2], (chain, asked, why)


def _recompressed(G, GrA, atoms, Hm):
    """(||G~ - G'||_2, e', compression', (level, asked)) with G' the Gram of the same kernels whose energies are
    compressed again at compression' = (the H-matrix's compression) / REFINE^level and e' = its H-matrix's estimate
    (relative error x ||G'||_F): the a-posteriori error of compressed energies (plan 054, the general rule). level is
    pg.settings.verification, or the next shallower one where that cannot be formed or is not much more accurate (its
    estimate above 1/10 of the first); None where no level can. The atoms' cached energies are restored afterwards."""
    from ..core.errors import Unsupported
    from ..settings import settings
    asked = depth()
    for level in range(asked, 0, -1):
        tol2 = Hm.compression / REFINE ** level
        saved = atoms.self_energy, atoms.hmatrix
        try:
            atoms.self_energy = atoms.hmatrix = None
            with settings.using(compression=tol2, dense_gram="compressed"):
                A2 = np.asarray(G.domain.inner(G.kernels, G.kernels), float)
            Hm2 = atoms.hmatrix
        except (MemoryError, Unsupported):
            continue
        finally:
            atoms.self_energy, atoms.hmatrix = saved
        if Hm2 is None or A2.shape != GrA.shape or not Hm2.rel_error <= 0.1 * Hm.rel_error:
            continue
        return (float(np.linalg.norm(GrA - A2, 2)), float(Hm2.rel_error) * float(np.linalg.norm(A2)), tol2,
                (level, asked))
    return None


def chain_text(chain, asked, why=""):
    """How deep the kernels' refinement went, why it stopped short of the depth asked, and its differences."""
    steps = ", ".join(f"{d:.2g} at tol {t:.2g}" for t, d in chain)
    head = f"verification level {len(chain)}" + ("" if len(chain) == asked else f" of the {asked} asked ({why})")
    return f"{head}; ||G^(j-1) - G^(j)|| = {steps}"
