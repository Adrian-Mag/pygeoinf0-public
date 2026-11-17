# SOLA presentation, part 1: inspectable notebooks

The author requested notebooks first; Slidev and later presentation parts are
deferred. Earlier notebooks 3/4/5 and the original input remain preserved.

Read and run in this order:

1. `01_classic_pixels.ipynb`: the 2D box, actual ray/cell matrix, correct L2 mass
   adjoint, constrained SOLA and numerical pixel refinement. Three fixed disk
   targets expose changes in the selected questions, not just one number.
2. `02_sobolev_galerkin.ipynb`: conforming H2 trial spaces with their full Hilbert
   mass matrix; finite G/T evaluations and a fully discrete constrained solve.
3. `03_sobolev_continuum.ipynb`: the same H2 objective assembled through native
   pygeoinf0 G G* and T G* operators, with separate quadrature checks.
4. `04_costs_and_comparison.ipynb`: actual assembly/elimination/solve timings and
   declared accuracy comparisons, including raw repetitions and matrix payloads.

Use the same original 600 rays, smooth truth, noise draw (sigma 0.01), disk radius
0.12, target centres (0.3,0.65), (0.7,0.3), (0.5,0.5), and lambda 0.1 throughout.
The parent input_provenance.json pins the original problem.py. Pixel/L2 and H2
use different matching geometries. The two H2 routes use exactly the same
restriction H2 norm, length scale 0.1, physical normalization and penalty.
The length scale is an explicit Experiment parameter; changing it changes the
SOLA comparison geometry and may require finer trial spaces/quadrature. The
author's existing scratch helper using 0.01 is preserved separately.

The Galerkin realization uses nested **dense kernel-translate** spaces. It does
not benchmark all finite elements. It constructs its own finite data/target/mass
matrices without using the continuum ray/ray Gram. Reference computations used
to measure errors and calibrate the fixed target norm are outside its timings.
Physical unit response v=G1 is known analytically and kept exact; projected
response is separately reported because constants need not lie in the trial span.
Riesz fields are paired in H2, not interpreted as Lebesgue averaging densities.

From package root `src/`, using an environment with pygeoinf0 dependencies plus
matplotlib, nbformat, nbclient and ipykernel:

```sh
python examples/discretisation_sola/part1/run_notebooks.py
```

The runner uses a temporary kernelspec and one BLAS thread, leaving environments
unchanged. Default maximum trial grid is 65 per axis; kernel Gram assembly can
use substantial memory there. Runtime controls (do not confuse smaller settings
with the full convergence/cost study):

```sh
SOLA_PART1_MAX_TRIAL=33 SOLA_PART1_MAX_PIXEL=1280 SOLA_PART1_BENCH_REPEATS=1 \
  python examples/discretisation_sola/part1/run_notebooks.py
```

Generate clean notebooks with `python examples/discretisation_sola/part1/make_notebooks.py`.
That overwrites these four notebooks, not the earlier notebook set. JSON/NPZ/CSV
results and execution records are in ignored output/. All four notebooks were originally
executed successfully on Europa on 2026-10-27, with 21 executed code cells and
nine embedded figures. Read them directly without rerunning the experiments.

The run used one actual BLAS thread, maximum trial grid 65, maximum pixel grid
2560 and three fresh benchmark repetitions. Numerical source commit:
`344c51eb7854b96c63f4443c575e834656247d76`; the private package-only execution
commit `a856a90075e897b13b91b4f619621161a44edf74` has the identical src tree.
Runtime versions, source hashes and thread counts are in `output/runtime.json`;
raw timings and medians are in `output/benchmark.json` and `benchmark.csv`.

Selected measured results (the three columns follow the target order above):

| Quantity | Positive disk | Negative disk | Central disk |
|---|---:|---:|---:|
| Requested true disk mean | 0.633698 | -0.658091 | 0.008925 |
| Pixel estimate, 2560 per axis | 0.059583 | -0.006406 | 0.029007 |
| Native H2 estimate | 0.592966 | -0.631234 | 0.018211 |
| Native H2 delivered truth | 0.593038 | -0.627827 | 0.017990 |

At trial grid 65 the Galerkin data weights differ from the native reference by
`2.6911e-4` in relative Frobenius norm; the delivered-functional distance divided
by the combined target norm is `3.0800e-5`. All physical constant responses
remain one. The native 32/64 target-chord comparison changes weights by
`8.4820e-6` with the penalty held fixed. This is a separate quadrature diagnostic.
Both Sobolev routes retain target mismatch: convergence of question selection
does not imply exact recovery of the requested target.

For a declared relative weight tolerance of `1e-3` against the 24-node/32-chord
reference, the cheapest **measured** qualifying settings are:

| Implementation | Setting | Median seconds | Relative weight error | Named matrix payload |
|---|---:|---:|---:|---:|
| Dense H2 Galerkin | 65 per axis | 28.7764 | 2.6911e-4 | 158.39 MiB |
| Native H2 | 10 ray nodes | 2.9465 | 1.8644e-7 | 2.76 MiB |

These times exclude common data generation, imports and reference/error
computations. Matrix payload is not peak RAM. The reference's own zero error is
only self-agreement; the separate quadrature study is needed to assess accuracy.
The mass-corrected pixel implementation is faster in these settings but solves
a different matching problem, so it is not an equal-accuracy competitor.

Validation on the same Europa source: 18 focused contracts passed; the full
package suite passed with 575 passed, 1 skipped and 3 realization warnings.
The Europa job finished with exit 0 and its owning mission was released.

Notebook 02 also includes a subsequent plot-only update: the first target's
finite resolving representer `P_h G* x_h` at trial grids 3, 9, 17 and 65, beside
the native resolving and target Riesz functions on a common colour scale. Panel
errors include trial projection as well as changing weights. These H2 functions
are not area densities. The new cell is executed separately on the laptop using
the saved, validated Europa weights; previous outputs and benchmark timings are
preserved. Its scope/host are recorded in notebook metadata and
`output/galerkin_fields_provenance.json`. That update brought the sequence to 22 code cells
and ten figures. Plot assembly is excluded from the benchmark.

All four notebooks now also have physical-measure and common-probe views. The
physical panels show an exact ray, the uniform disk and the signed weighted
rays, including their masses. The response panels ask what those same physical
functionals return for a known, moving unit-peak smooth anomaly. Its full shape
and profile are shown explicitly. The chosen length 0.1 is a display parameter,
not a physical ray width or an assumption on the unknown model.

The probe is the normalized restriction-H2 kernel,
`q_z(y) = K_H(y,z) / K_H(z,z)`; in 2D its profile is `rho * K_1(rho)`, with
`rho = |y-z| / length_scale` and value 1 at zero. Thus a full Riesz image divided
by the constant kernel peak is exactly the functional's response map for these
test models. Ray-integral responses have model-times-length units; target and
delivered averages have model units and share their colour scale. The response
maps are not area densities or blurred observations.

Notebook 01 shows actual pixel-SOLA response drift under refinement. Notebook 02
shows full `X_h G q_z` response evolution separately from the existing projected
`P_h G* x_h` fields, whose normalized values are `X_h G P_h q_z`. Notebook 03
compares requested and delivered responses for all three disk targets. Notebook
04 compares the actual questions selected by the three methods on one probe
family; its finest saved pixel level need not be one of the timed settings.
The same diagnostic probe is used even for the pixel/L2 method, without changing
that method's matching geometry.

To reproduce only these new physical/probe sections, after the original notebook
run and from package root src:

```sh
python examples/discretisation_sola/part1/run_probe_views.py
```

This runner preserves all previous cells/outputs, checks original input hashes
when the validation manifest is present, and records its separate scope in each
notebook and `output/probe_execution.json`. It reuses the saved weights and one
shared, bounded response-grid computation. `probe_views.py` implements these
untimed diagnostics; independent contracts compare its normalized Riesz pairings
to direct physical ray/disk integration. No new SOLA solve or benchmark is needed.

The physical/probe update adds four executed cells and ten figures, bringing the
sequence to **26 executed code cells and 20 figures**. All earlier cells and
outputs are preserved exactly. Its 21 focused contracts pass. Independent checks
on the original 600 rays agree with direct integration within 1.37e-7 absolute;
the disk chord-quadrature discrepancy is at most 4.23e-6 on the tested models.
The four new cells executed successfully locally with one BLAS thread, in
66.33/3.89/3.65/2.21 seconds (the first computes the shared grid). This is separate
plotting provenance, not a new performance benchmark or full-suite result.
