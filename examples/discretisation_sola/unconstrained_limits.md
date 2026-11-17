# Unconstrained SOLA under refinement

This derives the limits tested in `5_unconstrained_limits.ipynb`. The data are a
fixed finite collection of zero-width rays. The physical target, noise covariance
and trade-off are fixed; there is no response constraint in the main argument.
All adjoints and orthogonal projections below refer to the stated Hilbert metric.

## A bounded continuum problem

Let H be a real Hilbert space, G:H -> R^N and T:H -> R^q bounded linear maps,
and R=gamma C_d a fixed nonnegative data-space matrix. With Euclidean data/output
pairings, minimize

    J(X) = ||T-XG||_HS^2 + tr(X R X^T).

If A=GG*+R is positive definite, differentiating this finite quadratic gives

    K=GG*, B=TG*, X=B(K+R)^(-1).                         (1)

The constant ||T||_HS^2 does not affect the minimizer. This is a closed-form
solution, not a claim that X=0 or XG=T. Here X is chosen without using d.

For a finite conforming subspace V_h of H and its H-orthogonal projector P_h,
restrict G and T to V_h. The adjoint of G|V_h, embedded back in H, is P_h G*.
Consequently

    K_h=G P_h G*, B_h=T P_h G*, X_h=B_h(K_h+R)^(-1).      (2)

For a nonorthonormal trial basis phi_j, put
M_jk=<phi_j,phi_k>_H, E_ij=G_i phi_j, U_aj=T_a phi_j. Then

    K_h=E M^(-1) E^T, B_h=U M^(-1) E^T.                 (3)

Equation (3) is invariant under any invertible change of trial basis. Replacing
M by the identity without an orthonormal basis changes the problem.

Assume P_h g_i -> g_i in H for every data representer g_i=G*e_i. This follows,
for example, from nested spaces with dense union. Then K_h->K and B_h->B in matrix
norm: only finitely many representers are involved, and T is bounded. If also
P_h t_a->t_a, the sharper residual identities are

    (K-K_h)_ij = <(I-P_h)g_i,(I-P_h)g_j>,
    (B-B_h)_ai = <(I-P_h)t_a,(I-P_h)g_i>.

Write Delta K=K_h-K and Delta B=B_h-B. An exact identity, not a first-order
approximation, is

    X_h-X = (Delta B-X Delta K)(K_h+R)^(-1).             (4)

If R is positive definite, ||(K_h+R)^(-1)|| <= 1/lambda_min(R) on every mesh.
Equations (2)-(4) prove X_h->X. If R=0 but K is positive definite, convergence
still holds, since K_h is eventually positive definite with uniformly bounded
inverse. Rank-deficient zero-penalty problems require a specified selection rule
and rank control; inverse/pseudoinverse continuity is not automatic there.

Also X_h G->XG in operator norm, and P_h G* X_h^T->G* X^T in H. Thus both the
projected representers used in the computation and the actual delivered
functionals converge. Noise covariances X_h C_d X_h^T converge. This need not
produce the requested target: the limiting mismatch T-XG can remain nonzero.
If V_h already contains every g_i, K_h=K and B_h=B exactly, even when the target
representers are outside V_h. This exact invariance assumes R is fixed; rescaling
the penalty with a changing projected target norm can destroy it.

## Why the L2 ray/pixel case needs a different proof

A zero-width line integral does not define a bounded functional on L2 of area;
indeed it is not well defined on arbitrary L2 equivalence classes. There is no
L2 representer g_i and no bounded G* to substitute into (2). On pixel functions
with specified canonical values, each ray restriction is nevertheless a finite
functional. Let cell area be h^2 and let ell_iC be the ray length inside cell C.
Its L2 representer on this finite space is

    g_i,h = sum_C (ell_iC/h^2) 1_C.

Let t be the normalized fixed disk density, a bounded L2 function, and let P_h
now denote the L2 cell-average projector. Direct calculation gives

    (K_h)_ij=sum_C ell_iC ell_jC/h^2,
    (b_h)_i=<g_i,h,P_h t> = integral_ray_i P_h t,
    x_h=(K_h+R)^(-1)b_h.                                (5)

Here a single target is used, with X_h=x_h^T. Although P_h t->t in L2, this says
nothing by itself about the behavior of the unbounded ray functionals. A symbol
such as G P_h may be interpreted as the finite matrix extended by cell averaging,
but it is not a discretization of a bounded L2 operator G.

For the following limit assume fixed N, fixed positive-length straight rays in
the square, and no pair sharing a collinear segment of positive length. Exclude
rays lying along a mesh boundary, or specify a consistent assignment. These are
the generic conditions of the demo's randomly generated geometry.

Each ray meets at most C_i/h cells, and ell_iC <= sqrt(2)h. Since the lengths
sum to L_i>0, Cauchy-Schwarz and the upper bound give

    c_i/h <= (K_h)_ii <= C_i'/h.

Two distinct non-collinear intersecting rays share only O(1) cells as h->0:
any shared cell lies within O(h) of their fixed intersection, with a constant
depending on their fixed angle. Separated parallel rays eventually share none.
Hence (K_h)_ij=O(1) for i!=j. Gershgorin's theorem, with N fixed, yields

    lambda_min(K_h) >= c/h-C, so ||(K_h+R)^(-1)||=O(h).

Near-parallel rays can make the onset very late. This is an asymptotic theorem,
not an assertion that every computed grid is in the diagonal-dominant regime.
It does not cover ray counts increasing with refinement or coincident rays.

Because cell averaging preserves the L-infinity bound,
|(b_h)_i| <= L_i ||t||_infinity. Therefore (5) proves

    x_h=O(h)->0, x_h^T d->0,
    x_h^T C_d x_h=O(h^2).                               (6)

Moreover the selected pixel density r_h=sum_i (x_h)_i g_i,h satisfies

    ||r_h||_L2^2 = x_h^T K_h x_h
                 = x_h^T b_h-x_h^T R x_h = O(h).

Thus r_h->0 strongly in L2, and ||t-r_h||_L2 -> ||t||_L2. The unconstrained
method selects the zero question. Its exact continuum delivery sum_i x_hi G_i m
also tends to zero for every fixed m on which the finite ray values exist. The
noise calculation remains correct, but it pertains to that vanishing question.
This is not convergence to a non-L2 density.

The same scaling clarifies minimum-L2-norm ray interpolation in the least-squares
example. Once K_h is invertible, its minimum-norm exact interpolant is
m_h=sum_i (K_h^(-1)d)_i g_i,h. Thus ||m_h||_L2^2=d^T K_h^(-1)d=O(h), although
G_h m_h=d remains fixed and nonzero. In this setting the models can converge
strongly to zero in L2 while their ray predictions do not. The failure is the
absence of a continuous limiting L2 observation map, rather than necessarily a
model limit outside L2. This statement concerns the stated minimum-norm solve
and fixed-ray assumptions, not every discretized least-squares formulation.

For an exact elementary check, take N separated horizontal unit-length rays and
the uniform target t=1. For sufficiently fine pixels, K_h=h^(-1) I and b_h=1.
With R=tau I, x_hi=h/(1+tau h) and ||r_h||_L2^2=N h/(1+tau h)^2. The contract
test verifies these identities directly from the ray/pixel geometry.

## Why normalized pixel SOLA looks different

The earlier notebook's main calculation imposes v^T x_h=1, with v_i=L_i. Then
integral r_h=1, so (6) cannot apply. The support S_h of r_h lies in the union
of cells touched by the fixed rays, of total area O(h). Cauchy-Schwarz gives

    1=|integral_(S_h) r_h| <= |S_h|^(1/2)||r_h||_L2,
    hence ||r_h||_L2 >= c h^(-1/2) -> infinity.

This explains eventual growth of the normalized densities, whereas the
unconstrained densities vanish. It does not by itself prove convergence of the
normalized weights or identify their limiting line measure.

## The conforming Sobolev experiment

Use the same restriction H2 space on the square, length scale 0.1, as notebook 4.
On this bounded Lipschitz domain, the H2 embedding into continuous functions
bounds every finite-length ray functional and the disk mean. Thus (1)-(4) apply.
The metric is part of the SOLA criterion; no prior radius or probability law is
asserted. Ordinary piecewise constant pixels are not conforming H2 functions.

For nested dyadic point grids Z_h with 3,5,9,17,33,65 points per axis, take

    V_h = span { k_z : z in Z_h }, k_z=Riesz_H(point evaluation at z).

These are genuine subspaces of the continuum H; the points are trial-function
centres, not added data. Their union is dense: a function orthogonal to every
k_z vanishes at every point of the dense grid union and, by continuity, everywhere.
Hence P_h->I strongly. The numerical implementation uses (3), with the H2
mass/Gram matrix retained, and compares to the unconstrained full-representer
solution (1). The same fixed tau=lambda ||t||_H*^2 is used at every level.

This is a true trial-space refinement study, unlike notebook 4's quadrature-only
refinement. Matrix pairings still need numerical integration. We fix 24 ray
quadrature nodes and 32 disk chords here; the target in the computation is that
disk quadrature. Notebook 4 separately checks 32/64 chords. Projection refinement
alone cannot remove target-quadrature error, and inverse Gram matrices can become
ill-conditioned. No clipping, nugget or basis-dependent regularization is added.
