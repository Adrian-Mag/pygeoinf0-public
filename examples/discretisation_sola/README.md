# SOLA: pixels versus a Sobolev formulation

Two notebooks on the original discretisation demo's 600 rays, truth and noise:

- `3_pixels_sola.ipynb`: area-weighted pixel/L2 kernel matching, fixed disk mean,
  unit response, eight mesh sizes through 2560 by 2560; unconstrained comparison.
- `4_sobolev_sola.ipynb`: H2 on the unit square, length scale 0.1, same physical
  target; refinement of quadrature, a separate target-quadrature check and smooth
  Riesz functions with their correct interpretation.

The default dimensionless trade-off is 0.1, fixed independently of observations.
The two spaces give different comparison geometries; the experiment tests
stability within each formulation, not equality of the resulting estimators.
Neither notebook reconstructs a model or asserts a prior norm bound/distribution.

Run from the package root `src/`, with pygeoinf0's dependencies, matplotlib,
nbformat, nbclient and ipykernel available:

```sh
python examples/discretisation_sola/run_notebooks.py
```

The runner creates a temporary kernel specification and explicitly imports this
package source. It does not install kernels or change environments. Generate clean
notebooks with `python examples/discretisation_sola/make_notebooks.py`.

The original `problem.py` is copied unchanged; `input_provenance.json` records its
source and hash. Generated JSON results, weights and execution records are under
ignored `output/`. Independent mathematical checks are in
`tests/test_sola_discretisation_demo.py`.

Interpretation distinguishes the finite data transformation, its true delivered
functional, a pixel density representation, and a Sobolev Riesz representer.
A smooth Sobolev representer is paired using the Sobolev inner product, not an
ordinary area integral. The reported noise standard deviation excludes target
mismatch. A single truth/noise draw is not a coverage experiment.

## Executed results (27 October 2026, europa)

Both notebooks have all code cells executed and embedded figures. The known disk
mean is **0.6336980412**. Unit-response results:

| Formulation | Estimate | Actual delivered truth | Propagated noise standard deviation |
|---|---:|---:|---:|
| Pixels 20 x 20 | 0.551334 | 0.553697 | 0.0073483 |
| Pixels 320 x 320 | 0.179277 | 0.180641 | 0.0014865 |
| Pixels 2560 x 2560 | 0.059583 | 0.060133 | 0.0006415 |
| H2, converged ray quadrature | 0.592966 | 0.593038 | 0.0043914 |

Pixel density norm first decreases (3.90 to 2.35), then rises (2.47, 2.95 on the
two finest meshes). Peak densities grow from 21.5 to 210.3. The relative L2 target
mismatch rises from 0.453 to 1.093. The weights and the delivered question change;
smaller noise error bars do not show recovery of the requested disk mean.

In H2, changing ray-pair quadrature from 16 to 24 nodes changes X by 3.22e-13
relatively. Separately, changing target quadrature from 32 to 64 chords changes X
by 9.58e-6. The latter is the larger tested numerical sensitivity; agreement at
these settings is a convergence check, not a rigorous quadrature error bound.
Under-resolved Gram matrices at 4, 6 and 8 nodes are flagged and not used for
claims about valid continuum SOLA solutions.

The H2 target bias for this truth is -0.0406603 despite a converged solve. The
weights accurately estimate their delivered property, not the requested target.
The chosen Sobolev metric preserves a useful localized question in this example;
it does not make every target identifiable.

The pixel notebook also includes the unconstrained case. The main result above
therefore does not rely on omitting the normalization used in SOLA papers.

## Follow-up: analytic limits and conforming projections

`unconstrained_limits.md` derives the orthogonal-projector formula for bounded
Hilbert data maps, its exact perturbation identity and the convergence of X_h.
It separately proves that fixed non-overlapping zero-width rays with a bounded
target give unconstrained pixel weights X_h=O(h)->0 and pixel densities tending
to zero in L2. Unit response gives a different limit: density norms must diverge
because their unit mass is confined to ray-touched cells of area O(h).

`5_unconstrained_limits.ipynb` follows these arguments numerically and adds
nested kernel-translate trial subspaces of H2, retaining their Hilbert mass
matrix. This refines actual conforming subspaces, whereas notebook 4 refines
quadrature. Generate this notebook separately with
`python examples/discretisation_sola/make_projection_notebook.py`, and execute
it by passing its filename to `run_notebooks.py`. Execution evidence for this
follow-up is recorded separately; the preceding results refer to notebooks 3/4.

The follow-up was executed locally with the author's explicit override of the
normal Europa-only guidance, one BLAS thread, a 768 MiB process-family RAM guard,
and trial grids through 33 per axis. It took 44 seconds in the notebook runner;
the monitored process family peaked at 395 MiB. An initial virtual-address cap
obstructed imports, was diagnosed and replaced; that failed startup is retained
in ignored output/attempt1-* and did not yield numerical results.

| H2 trial grid | Trial dimension | Relative K error | Relative X error |
|---|---:|---:|---:|
| 3 x 3 | 9 | 0.69361 | 0.98364 |
| 9 x 9 | 81 | 0.022154 | 0.23616 |
| 17 x 17 | 289 | 0.0020332 | 0.032312 |
| 33 x 33 | 1089 | 0.00019318 | 0.0028409 |

The unconstrained continuum-reference estimate is **0.5929619762**, delivered
truth **0.5930332850**, noise sigma **0.0043913593**. The 33 x 33 estimate is
0.5931171715. This is observed approach to the well-defined limit, with 0.284%
weight error still remaining, rather than machine-precision trial convergence.
The general projector theorem establishes the limit under the stated assumptions.
The reference retains the fixed 24-node/32-chord quadrature; refinement of trial
spaces alone does not remove those integration errors or target mismatch.

Unconstrained pixel weight norms drop from 0.734659 at n=20 to 0.0225661 at
n=2560; estimates drop from 0.551330 to 0.0366338. This sequence approaches the
zero question, not the fixed target mean. All five code cells ran without errors
and both embedded figures were inspected. Source for the successful run:
`6d0a17e68c26b14b9bd07de2d7974de68918d631` (local, no Europa transfer).

All **11 focused mathematical contracts pass** on the final example source.
The attempted full package run stopped at the conservative 768 MiB process-family
RSS guard, after 254 completed progress markers. Collection order identifies the
next test as the existing `test_blocked_and_parallel_energies_and_values` in
`test_sobolev_gaussian.py`; this identification is inferred from order, not a
pytest failure traceback. The new notebook and focused tests pass; a complete
full-suite result on the follow-up snapshot is not claimed. Earlier full-suite
results above belong to the preceding notebook task.

Checks: **564 passed, 1 skipped** in the full package suite; the seven experiment
contracts also pass on the final execution snapshot. Notebooks took 4.7 and 42.3
seconds on europa (excluding the full suite and initial environment correction).
No package core code was changed.

Execution source: local task commit `1f28d379032939af47d725f2d05ec49b313e16cc`,
exported with an identical `src/` tree as private Git commit
`b6fdf83b958c3ac903e2f102b52c6f2c36ebf4f1`. Only package code and synthetic
example files were transferred; no origin push or other checkout takeover occurred.
