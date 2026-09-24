<p align="center">
  <img src="assets/logo.svg" alt="ChebyshevHelmoltzSolvers.jl logo" width="900">
</p>

# ChebyshevHelmoltzSolvers.jl

[![CI](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml)

ChebyshevHelmoltzSolvers.jl solves linear boundary-value problems on
`[-1, 1]`. It supports a second-order Helmholtz equation and a fourth-order
equation written as two Helmholtz equations. The coefficients multiplying
the derivatives and the solution are constant in space; the forcing may
vary with the wall-normal coordinate $y$.

The package is designed for repeated solves. Build a solver once, call
`update!` when the equation's coefficients change, and call `solve!` for
each new forcing. You can solve one problem on the CPU or many independent
problems together on the CPU or an NVIDIA GPU.

The numerical method has three parts:

- **Chebyshev tau discretisation:** approximate the solution by a polynomial,
  impose the differential equation on its lower-degree coefficients, and
  use the remaining equations for the wall conditions.
- **Integration and UL factorisation:** rewrite those equations as two small
  systems whose rows have at most three entries, apart from the boundary
  row. This makes the work for setup and each solve linear in the polynomial degree.
- **Influence-matrix correction:** for the fourth-order problem, combine
  solutions of two second-order problems, then correct their wall slopes
  by solving a $2\times2$ system.

The sections below develop these ideas and connect them to the code.
The tau method is described by [Canuto et al. (2006)](#references);
[Greengard (1991) and Viswanath (2014)](#references) provide background on
spectral integration. The boundary-correction idea is related to the
influence-matrix method of [Kleiser and Schumann (1980)](#references).

For a first solve, start with the [scalar example](#scalar-helmholtz-solver).
For the algorithms, see [tau discretisation](#the-chebyshev-tau-method),
[matrix structure](#where-the-structure-comes-from), and
[the influence matrix](#the-influence-matrix-method).

## Installation

Requires Julia 1.10 or later. Install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl")
```

The package uses [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) for
value/coefficient transforms, [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
for the small scalar influence matrix, and
[Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) to move solver storage between
CPU and GPU. [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is optional;
CPU use does not require a GPU or CUDA installation.

## Chebyshev representation

The solver works with polynomial **coefficients**, while input profiles are
often known as **values at points**. These are two representations of the
same polynomial, but their arrays are not interchangeable.

For example, $T_0(y)=1$, $T_1(y)=y$ and $T_2(y)=2y^2-1$. A function is
approximated by

$$
f_P(y)=\sum_{n=0}^{P}a_nT_n(y),
\qquad T_n(\cos\vartheta)=\cos(n\vartheta).
$$

Here $P$ is the highest polynomial degree, so the array has $P+1$ entries.
Julia index `n+1` stores $a_n$. The expansion is **ordinary**: there is no
implicit half-weight on $a_0$. For example, $1-y^2=(T_0-T_2)/2$ has
coefficients `a[1]=0.5`, `a[3]=-0.5`, with all other entries zero.
One-based vectors and vector views are accepted. Odd and even degrees are
both supported.

To convert a sampled profile, use the Chebyshev–Lobatto points

$$
y_j=\cos\left(\frac{j\pi}{P}\right),\qquad j=0,\ldots,P.
$$

`chebpoints(P)` returns these points in **descending order, from +1 to -1**.
They include both walls and cluster near them. `chebcoeffs(values)` converts
values in this order to coefficients; `chebvalues(a)` converts back.
Both allocate their result and preserve the input. Internally the forward
transform uses FFTW's DCT-I, divides by $P$, and halves the first and last
coefficients to obtain the ordinary expansion above.

| Operation | What it returns or changes |
| --- | --- |
| `chebpoints(P)` | The $P+1$ points, ordered from `+1` to `-1`. |
| `chebcoeffs(values)` | A new vector of Chebyshev coefficients. |
| `chebvalues(a)` | A new vector of values at the same Lobatto points. |
| `diff!(a)` | Replaces coefficients of a function by coefficients of its derivative. |
| `diff!(out, a)` | Writes derivative coefficients into separate storage, preserving `a`. |
| `diff(a, :left)` | The single value of the derivative at `y=-1`. |
| `diff(a, :right)` | The single value of the derivative at `y=+1`. |

All derivatives are with respect to $y$ on `[-1, 1]`. In particular,
`:left` is the last Lobatto point, even though it is the left wall in space.
The two-array derivative method requires non-overlapping arrays; use the
one-array method when you want to overwrite the input.

## Scalar Helmholtz solver

A Helmholtz problem asks for a function whose curvature and value balance
a prescribed forcing, while satisfying two wall conditions. Here the equation is

$$
\theta_0 u''(y)-\theta_1 u(y)=f(y), \qquad -1<y<1,
$$

$$
u(1)=u_+, \qquad u(-1)=u_-.
$$

The real constants $\theta_0$ and $\theta_1$ set the operator, with
$\theta_0\ne0$. Dirichlet data $u_+$ and $u_-$ prescribe the solution at
the upper and lower walls. Use the solver in three stages:

1. `HelmoltzSolver(P, T=Float64)` allocates storage for degree `P ≥ 3`.
2. `update!(h, θ₀, θ₁)` builds and factors the two coefficient matrices.
3. `solve!(h, u, f, u₊, u₋)` uses those factors to compute the solution.

Changing `f` or the wall values only requires another solve. Changing either
operator coefficient requires an update first. Construction alone does not
prepare the solver for use.

For example, choose $\theta_0=1$ and $\theta_1=4$, so the equation is
$u''-4u=f$. With $f(y)=-6+4y^2$ and homogeneous boundary values,
the solution is $u(y)=1-y^2$:

```julia
using ChebyshevHelmoltzSolvers

P = 16
h = HelmoltzSolver(P)
update!(h, 1.0, 4.0)

# Sample the right-hand side at Lobatto points, ordered from +1 to -1.
y = chebpoints(P)
f = chebcoeffs(-6 .+ 4 .* y.^2)
u = similar(f)
solve!(h, u, f, 0.0, 0.0)  # preserves f

# u stores the coefficients of (T₀ - T₂)/2.
# u[1] ≈ 0.5, u[3] ≈ -0.5
chebvalues(u)  # ≈ 1 .- y.^2
```

**Pass upper-wall data before lower-wall data:**
`solve!(h, u, f, u₊, u₋)`. Both `u` and `f` contain coefficients, not point
values. They must have `P+1` entries, matching element types, and separate
storage. `solve!` overwrites `u`, preserves `f`, and returns `u`.

The factors use `Float32` or `Float64`. The solution and forcing may use that
same real type or its complex counterpart, for example `ComplexF64` with
`Float64` factors. Complex coefficients are useful when these problems arise
from a Fourier transform in other spatial directions.

Use `HelmoltzSolver(P; neum=true)` for prescribed wall derivatives instead
of wall values. The [Neumann section](#neumann-boundary-conditions) explains
the sign convention and the special Poisson case.

### The Chebyshev tau method

A degree-$P$ approximation has $P+1$ unknown coefficients. Imposing every
coefficient equation as well as two boundary conditions would overdetermine
it. The tau method instead retains $P-1$ differential-equation conditions
and uses the two remaining equations for the boundary data. It enforces
conditions on the **spectral residual**, rather than requiring the equation
to hold separately at each Lobatto point.

The residual is the mismatch between the two sides of the equation.
Writing $[\cdot]_n$ for its coefficient of $T_n$, the tau conditions are

$$
\left[\theta_0 u_P''-\theta_1 u_P-f_P\right]_n=0,
\qquad n=0,\ldots,P-2,
$$

Equivalently, the residual may contain only the two highest modes:

$$
\theta_0u_P''-\theta_1u_P-f_P
=\tau_{P-1}T_{P-1}+\tau_PT_P.
$$

The amplitudes $\tau_{P-1}$ and $\tau_P$ are residual coefficients, not
additional inputs. The two boundary conditions are satisfied by the
polynomial solution, while these highest residual components are left
unconstrained. This also explains why the last two forcing coefficients
do not affect the computed solution.
See [Canuto et al. (2006)](https://doi.org/10.1007/978-3-540-30726-6)
for the general tau framework.

The [matrix derivation below](#where-the-structure-comes-from) shows how to
solve these equations without building a dense differentiation matrix.

For background on integration-based spectral boundary-value solvers, see
[Greengard (1991)](https://doi.org/10.1137/0728057) and
[Viswanath (2014)](https://arxiv.org/abs/1205.2717v2). These discuss related
formulations; the tau truncation and compact UL recurrences implemented here
are specified below.

### Neumann boundary conditions

A Neumann condition prescribes the slope rather than the value. With
`neum=true`, the same boundary arguments mean

$$
u'(1)=u_+,\qquad u'(-1)=u_-.
$$

Both are derivatives in the **positive $y$ direction**. They are not
outward-normal derivatives: no extra minus sign is applied at the lower wall.
For a nonsingular shifted equation, these slopes determine the solution.

The Poisson case $\theta_1=0$ needs special treatment. Integrating
$\theta_0u''=f$ from one wall to the other gives

$$
\int_{-1}^{1}f(y)\,\mathrm{d}y=\theta_0(u_+-u_-).
$$

This is a **compatibility condition**: the total forcing must agree with the
change in slope. If it does not, no solution exists. Even when it holds,
adding a constant to a solution changes neither its curvature nor its wall
slopes. We therefore choose the constant by requiring zero mean:

$$
\frac{1}{2}\int_{-1}^{1}u(y)\,\mathrm{d}y
=\sum_{\substack{n=0\\n\text{ even}}}^{P}
\frac{\widehat{u}_n}{1-n^2}=0.
$$

For the discrete tau problem, the compatibility check uses only the retained
forcing coefficients, of degrees `0:P-2`. It rejects incompatible data before
changing the output, with a tolerance scaled by machine precision and the
size of the forcing and wall contributions. It does not alter the forcing
to make the problem solvable.

For example, $u(y)=y^3+y^2-1/3$ has zero mean, $u''=6y+2$, and wall slopes
$u'(1)=5$ and $u'(-1)=1$:

```julia
h = HelmoltzSolver(16; neum=true)
update!(h, 1.0, 0.0)
y = chebpoints(16)
f = chebcoeffs(6 .* y .+ 2)
u = similar(f)
solve!(h, u, f, 5.0, 1.0)
chebvalues(u)  # ≈ y.^3 .+ y.^2 .- 1/3
```

Internally the even block temporarily fixes the constant coefficient; the
solution is then shifted to make its integral zero. The odd block keeps its
wall equation. This special case is available in the scalar and batched
**CPU** solvers, including CPU batches that mix Poisson and shifted systems.
Singular Neumann Poisson is not supported on CUDA and is rejected there.

## Quasi-tridiagonal matrices and UL factorisation

### Where the structure comes from

The matrix structure follows from three choices: use Chebyshev coefficients
as the unknowns, integrate the differential equation twice, and collect even
and odd coefficients separately. The differential equation then gives rows
with at most three entries; the wall conditions give one dense row in each
system. The steps below explain why.

#### 1. Start with the differential equation and its unknowns

Consider the Dirichlet problem on the package's fixed interval:

$$
\theta_0 u''(y)-\theta_1 u(y)=f(y),\qquad -1\le y\le1,
\qquad u(1)=u_+,\quad u(-1)=u_-.
$$

We approximate the solution and forcing by degree-$P$ expansions:

$$
u_P(y)=\sum_{n=0}^{P}\widehat{u}_nT_n(y),\qquad
f_P(y)=\sum_{n=0}^{P}\widehat{f}_nT_n(y).
$$

The $P+1$ numbers $\widehat{u}_0,\ldots,\widehat{u}_P$ are the unknowns.
They are **coefficients of polynomials**, not values of $u$ at grid points.
We therefore need $P+1$ equations to determine them.

The tau method supplies $P-1$ equations by requiring the Chebyshev
coefficients of $\theta_0u_P''-\theta_1u_P-f_P$ to vanish at degrees
$0,\ldots,P-2$. The remaining two equations are the wall conditions.
We do not also set the residual coefficients at degrees $P-1$ and $P$ to
zero: that would give too many equations.

#### 2. Integrate to obtain a short coefficient relation

A second derivative written directly in coefficient space couples many
coefficients. Integration has a simpler structure. For $n\ge2$,

$$
\int T_n(y)\,\mathrm{d}y
=\frac{T_{n+1}(y)}{2(n+1)}
-\frac{T_{n-1}(y)}{2(n-1)}+C.
$$

One integration changes the degree by one. Two integrations therefore
connect degrees differing by two, as well as the original degree.

To apply this to the **retained tau equations**, define

$$
g_n=
\begin{cases}
\theta_1\widehat{u}_n+\widehat{f}_n,&0\le n\le P-2,\\
0,&n>P-2.
\end{cases}
$$

Those equations are equivalent to the polynomial identity

$$
\theta_0u_P''(y)=\sum_{n=0}^{P-2}g_nT_n(y).
$$

Integrating this identity twice and comparing coefficients of degree
$n=2,\ldots,P$ gives

$$
\theta_0\widehat{u}_n
=\frac{c_{n-2}g_{n-2}}{4n(n-1)}
-\frac{g_n}{2(n^2-1)}
+\frac{g_{n+2}}{4n(n+1)},
\qquad c_0=2,\quad c_j=1\ \text{for }j\ge1.
$$

The factor $c_0$ accounts for integrating the constant polynomial $T_0=1$.
The two arbitrary integration constants multiply $1$ and $y$, that is,
$T_0$ and $T_1$. They do not appear in the equations for $n\ge2$;
we still need the two wall equations to close the system.

For a concrete example, take $n=4$ and $P\ge8$. Substituting the definition
of $g_n$ and moving the unknowns to the left gives

$$
-\frac{\theta_1}{48}\widehat{u}_2
+\left(\theta_0+\frac{\theta_1}{30}\right)\widehat{u}_4
-\frac{\theta_1}{80}\widehat{u}_6
=
\frac{\widehat{f}_2}{48}
-\frac{\widehat{f}_4}{30}
+\frac{\widehat{f}_6}{80}.
$$

This is one matrix row: it involves **only three unknowns**,
$\widehat{u}_2$, $\widehat{u}_4$ and $\widehat{u}_6$.
Near the highest degree, some terms vanish because $g_n=0$ for $n>P-2$.
This cutoff matters: integrating the full degree-$P$ forcing instead would
produce a different discretisation. In the code, the cached integration
weights include this cutoff through `_β`.

#### 3. Reorder the unknowns into two tridiagonal systems

Every integrated equation connects only $n-2$, $n$ and $n+2$.
An equation for an even degree therefore contains only even coefficients;
an equation for an odd degree contains only odd coefficients. Order them as

$$
\mathbf{u}_{\mathrm{even}}=(\widehat{u}_0,\widehat{u}_2,\widehat{u}_4,\ldots),
\qquad
\mathbf{u}_{\mathrm{odd}}=(\widehat{u}_1,\widehat{u}_3,\widehat{u}_5,\ldots).
$$

In either list, degrees differing by two are now **neighbours**. Thus each
integrated equation couples at most the previous, current and next entries
of its list: this is a tridiagonal row. For example, the $n=4$ equation above
connects three consecutive entries of the even list.

At this stage each block has one fewer equation than unknowns. For example,
with $P=6$, the even block has four unknowns (degrees 0, 2, 4 and 6) but
only three integrated equations, for $n=2,4,6$. The odd block has three
unknowns and two equations, for $n=3,5$.

#### 4. Add the boundary row to each block

At the two walls, $T_n(1)=1$ and $T_n(-1)=(-1)^n$. Consequently,

$$
\widehat{u}_0+\widehat{u}_1+\widehat{u}_2+\widehat{u}_3+\cdots=u_+,
$$

$$
\widehat{u}_0-\widehat{u}_1+\widehat{u}_2-\widehat{u}_3+\cdots=u_-.
$$

Adding and subtracting these equations separates the two parity blocks:

$$
\sum_{n\ \mathrm{even}}\widehat{u}_n=\frac{u_++u_-}{2},
\qquad
\sum_{n\ \mathrm{odd}}\widehat{u}_n=\frac{u_+-u_-}{2}.
$$

Each block now has its missing equation. Unlike an interior row, this row
contains **every coefficient in the block**, with weight one. The code
places it first. A dense first row followed by tridiagonal interior rows
is precisely the *quasi-tridiagonal* structure shown in the next section.

For Neumann data $u'(1)=g_+$ and $u'(-1)=g_-$, the same separation works
because $T_n'(1)=n^2$ and $T_n'(-1)=(-1)^{n+1}n^2$:

$$
\sum_{n\ \mathrm{even}}n^2\widehat{u}_n=\frac{g_+-g_-}{2},
\qquad
\sum_{n\ \mathrm{odd}}n^2\widehat{u}_n=\frac{g_++g_-}{2}.
$$

Only the boundary row changes: its entries are now $n^2$ rather than one.
Here derivatives are in the positive $y$ direction at both walls.
Pure Neumann Poisson is the singular exception: forcing and wall data must
satisfy a compatibility condition, and a zero-mean gauge fixes the otherwise
undetermined constant. See [Neumann boundary conditions](#neumann-boundary-conditions).

#### 5. Identify these matrices in the code

`HelmoltzSolver` stores the even and odd blocks as `Be` and `Bo`, of sizes
$\lfloor P/2\rfloor+1$ and $\lfloor(P+1)/2\rfloor$.
`CoupledHelmoltzSolver` uses two Helmholtz solvers, hence four parity blocks,
followed by its small influence-matrix correction. `BatchedHelmoltzSolver`
stores one pair of blocks per independent problem. Batching repeats this
same matrix structure; it does not introduce coupling between systems.

### Compact storage

`QuasiTridiagonal(M, T)` stores a matrix with a dense first row and a
tridiagonal interior. For example, when $M=5$,

$$
Q=
\begin{pmatrix}
b_1 & b_2 & b_3 & b_4 & b_5 \\
l_1 & d_2 & u_2 & 0 & 0 \\
0 & l_2 & d_3 & u_3 & 0 \\
0 & 0 & l_3 & d_4 & u_4 \\
0 & 0 & 0 & l_4 & d_5
\end{pmatrix}.
$$

This is the structure of each even/odd Helmholtz block: the dense row
imposes a boundary condition, and the other rows are the integrated
coefficient equations. Only four vectors are stored, rather than a dense
$M\times M$ array. The symbols $l_i$, $d_i$ and $u_i$ denote matrix
entries here; $u_i$ is unrelated to the solution function $u(y)$.

### Why UL rather than LU?

Factorisation rewrites the matrix as a product of two triangular matrices.
Triangular systems can be solved one entry at a time, and the factors can
be reused for every new right-hand side.

Here the dense row is at the top. Eliminating from the bottom upwards
keeps it confined to that row; eliminating downwards from the top would
spread its nonzero entries into the other rows. We therefore use the order
$Q=UL$, rather than the more familiar $Q=LU$.

`ul!(Q)` produces an upper factor $U$ and a lower factor $L$. For the
five-by-five example above, their forms are

$$
U=
\begin{pmatrix}
\beta_1 & \beta_2 & \beta_3 & \beta_4 & \beta_5 \\
0 & p_2 & u_2 & 0 & 0 \\
0 & 0 & p_3 & u_3 & 0 \\
0 & 0 & 0 & p_4 & u_4 \\
0 & 0 & 0 & 0 & p_5
\end{pmatrix},
\qquad
L=
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 \\
\ell_1 & 1 & 0 & 0 & 0 \\
0 & \ell_2 & 1 & 0 & 0 \\
0 & 0 & \ell_3 & 1 & 0 \\
0 & 0 & 0 & \ell_4 & 1
\end{pmatrix}.
$$

Below its first row, $U$ has only a diagonal and one upper diagonal.
$L$ has ones on its diagonal and one lower diagonal. The numbers $p_i$
are the pivots, $\ell_i$ are elimination multipliers, and $\beta_i$
are the updated entries of the dense row.

The bottom row starts the calculation with $p_M=d_M$. Working upwards,
for $i=M-1,\ldots,2$, compute

$$
\ell_i=\frac{l_i}{p_{i+1}},\qquad
p_i=d_i-u_i\ell_i,
$$

then set $\ell_1=l_1/p_2$. The dense row satisfies

$$
\beta_M=b_M,\qquad
\beta_i=b_i-\beta_{i+1}\ell_i,
\qquad i=M-1,\ldots,1.
$$

The implementation interleaves these two backward recurrences and overwrites
the assembly buffers. After elimination, it stores reciprocal pivots
$1/p_i$ and $1/\beta_1$, so repeated solves multiply instead of dividing.
There is no pivoting; all pivots and their reciprocals must be finite and
nonzero.

### Storage before and after factorisation

Factorisation overwrites the original entries. The four arrays store
the assembled matrix before `ul!`, and the reusable factors afterwards:

| Field | Before `ul!(Q)` | After `ul!(Q)` |
| --- | --- | --- |
| `b`, length $M$ | Dense first row $b_i$ | Reciprocal $1/\beta_1$ at index 1; entries $\beta_i$ for $i\ge2$ |
| `l`, length $M-1$ | Subdiagonal $l_i$ | Multipliers $\ell_i$ of the unit lower factor |
| `dᵢ`, length $M-1$ | Interior diagonal $d_{i+1}$ | Reciprocal pivots $1/p_{i+1}$ |
| `u`, length $M-2$ | Superdiagonal $u_{i+1}$ | Unchanged |

The size constructor allocates zero-filled assembly buffers. Alternatively,
`QuasiTridiagonal(b, l, dᵢ, u)` wraps existing vectors without copying.
The name `dᵢ` describes its contents **after factorisation**; ordinary diagonal
entries must be supplied when assembling a new matrix.

### Solving with the factors

After factorisation, a new right-hand side does not require another
elimination. Since $Qx=U(Lx)=r$, introduce $z=Lx$ and solve in two passes.

First solve $Uz=r$ from the bottom upwards. Each new entry depends only
on the one already found below it, until reaching the dense first row:

$$
z_M=\frac{r_M}{p_M},\qquad
z_i=\frac{r_i-u_i z_{i+1}}{p_i},\qquad i=M-1,\ldots,2,
$$

$$
z_1=\frac{r_1-\sum_{j=2}^{M}\beta_j z_j}{\beta_1}.
$$

It then solves $Lx=z$ forwards:

$$
x_1=z_1,\qquad x_i=z_i-\ell_{i-1}x_{i-1},\qquad i=2,\ldots,M.
$$

`ldiv!(Q, rhs)` performs these passes in the supplied array, replacing
$r$ by $x$. The first entry accumulates the dense-row contribution as the
other entries become available. No extra solution vector is needed, and
the factors are preserved for the next solve.

This array is an **assembled algebraic right-hand side**, including its
boundary equation. It is not the raw Chebyshev forcing vector `f`.
`solve!(h, u, f, ...)` performs that assembly for the Helmholtz problem.
Factorisation and substitution each require $\mathcal{O}(M)$ work and storage.

After `ul!`, indexing and `Matrix(Q)` expose the **compact factor storage**:
$U$ on and above the diagonal and the multipliers of $L$ below it, with
reciprocals converted back to pivots. They do not reconstruct the original
matrix $UL$. Before factorisation, access the assembly buffers directly.
Reassemble the original entries before calling `ul!` again; `update!` does
this automatically for Helmholtz solvers.

`BatchedQuasiTridiagonal` uses exactly the same recurrences and storage
convention, with a leading system index. CPU loops apply each recurrence
step across contiguous systems using SIMD; CUDA assigns a system to each
thread. Systems are independent, and no dense matrix is constructed.

For direct matrix work, `BatchedQuasiTridiagonal(B, M, T)` allocates
storage for `B` independent matrices of size `M ≥ 2`. Its four-array
constructor shares existing storage. Assemble the matrix entries, call
`ul!`, then use `ldiv!` on assembled right-hand sides. Both `B` and `M`
are fixed for the lifetime of this object. Most users should use the
Helmholtz solvers, which assemble these matrices and right-hand sides for them.

## Coupled Helmholtz solver

A fourth-order equation needs four boundary conditions. The coupled solver
handles equations that can be written as the product of two Helmholtz
operators:

$$
\mathcal{L}_1=\theta_0D^2-\theta_1,\qquad
\mathcal{L}_2=\theta_2D^2-\theta_3,\qquad D=\frac{\mathrm{d}}{\mathrm{d}y},
$$

$$
\mathcal{L}_1\mathcal{L}_2u=f,\qquad u(\pm1)=u'(\pm1)=0.
$$

The wall conditions are *clamped*: both the value and the slope vanish.
Introduce an intermediate field $w=\mathcal{L}_2u$. We can then solve

$$
\mathcal{L}_1w=f,\qquad \mathcal{L}_2u=w.
$$

This lets us reuse two second-order solvers. There is one difficulty:
all four wall conditions belong to $u$, while a second-order solve for $w$
needs its own two wall values. We do not know those values in advance.
The influence-matrix method finds them by measuring how they affect the
slopes of $u$.

### The influence-matrix method

**First find a particular solution.** Temporarily choose zero wall values
for $w$, then impose the required zero wall values for $u$:

$$
\mathcal{L}_1w_p=f,\quad w_p(\pm1)=0,
\qquad
\mathcal{L}_2u_p=w_p,\quad u_p(\pm1)=0.
$$

The result satisfies the two equations and the wall values of $u$, but its
wall slopes will generally be nonzero. We need to correct those slopes
without changing the forcing or the zero wall values.

**Next find two ways to change the slopes.** Set the forcing to zero and
solve twice: once with a unit upper-wall value of $w$, and once with a
unit lower-wall value. Denote these response pairs by $(w_+,u_+)$ and
$(w_-,u_-)$:

$$
\mathcal{L}_1w_\pm=0,\qquad
\mathcal{L}_2u_\pm=w_\pm,\qquad u_\pm(-1)=u_\pm(1)=0,
$$

$$
(w_+(1),w_+(-1))=(1,0),\qquad
(w_-(1),w_-(-1))=(0,1).
$$

Here the subscripts label entire response functions, rather than the scalar
boundary arguments used in the Helmholtz example. Each response has zero
forcing for the fourth-order problem and zero solution values at both walls.
It can therefore be added to $u_p$ without spoiling those conditions.

**Finally choose the two response amplitudes.** Write

$$
u=u_p+\delta_+u_++\delta_-u_-.
$$

Requiring zero slopes at both walls gives just two equations:

$$
\underbrace{\begin{pmatrix}
u_+'(1)&u_-'(1)\\
u_+'(-1)&u_-'(-1)
\end{pmatrix}}_{A}
\begin{pmatrix}\delta_+\\\delta_-\end{pmatrix}
=-\begin{pmatrix}u_p'(1)\\u_p'(-1)\end{pmatrix}.
$$

This is the **influence matrix**. Its first column records the effect of
unit upper-wall data for $w$ on the two slopes of $u$; its second column
records the effect of unit lower-wall data. Solving this $2\times2$ system
chooses the combination that cancels both unwanted slopes.

All second-order solves use the discrete tau equations already described.
Linearity makes the response combination valid for those discrete equations;
the correction enforces the remaining boundary conditions. The matrix $A$
must be invertible. This is the boundary-response idea used in
[Kleiser and Schumann (1980)](#references), applied here to a factored
fourth-order problem, rather than to a complete incompressible-flow solver.

### What is reused between solves?

The two response functions and their influence matrix depend on the four
operator coefficients, but not on the forcing. `update!` builds the two
factorisations and the responses once. Every subsequent `solve!` needs
only two particular Helmholtz solves, the small boundary solve, and the
response combination. The intermediate field $w$ is workspace; it is not
returned.

### Example

Set $(\theta_0,\theta_1,\theta_2,\theta_3)=(1,0,1,0)$. The equation is
then $u^{(4)}=f$. With $f=24$, the clamped solution is $u=(1-y^2)^2$:

```julia
P = 16
h = CoupledHelmoltzSolver(P)
update!(h, (1.0, 0.0, 1.0, 0.0))

y = chebpoints(P)
f = chebcoeffs(fill(24.0, length(y)))
u = similar(f)
solve!(h, u, f)  # writes the fourth-order solution into u; preserves f
chebvalues(u)   # ≈ (1 .- y.^2).^2

# u(y) = 3T₀/8 - T₂/2 + T₄/8.
# u[1] ≈ 0.375, u[3] ≈ -0.5, u[5] ≈ 0.125.
```

The constructor accepts `P ≥ 3`; degree at least four is needed to represent
a nonzero polynomial satisfying all four homogeneous wall conditions.
For complex coefficients, construct `CoupledHelmoltzSolver(P, ComplexF64)`;
its internal factors still use the corresponding real precision.
The input and output must match the chosen coefficient type and must not
overlap each other or solver storage. Each coupled solver owns mutable
workspace, so use separate instances for concurrent solves.

## Batched CPU Helmholtz solves

A batch is a collection of independent problems solved together. For example,
a Fourier discretisation can produce one wall-normal problem for each pair
of Fourier modes. The operator coefficients may differ between problems,
but every problem uses the same polynomial degree and boundary-condition
type. Label the systems by $s=1,\ldots,B$:

$$
\theta_{0,s}u_s''(y)-\theta_{1,s}u_s(y)=f_s(y),
\qquad s=1,\ldots,B.
$$

There is no interaction between systems in a batch. Each has its own forcing,
operator coefficients and two wall values. “Batched” describes how the work
is organised, not a different differential equation.

`BatchedHelmoltzSolver(P, B, T=Float64; neum=false)` allocates storage for
`B > 0` independent operators of degree `P ≥ 3`, with real factors of type `T`
and real or complex right-hand sides. Call `update!(h, θ₀, θ₁)` before solving
and whenever the operator coefficients change. `θ₀` and `θ₁` are vectors of
length `B` and element type `T`, with one coefficient per system.
The batch supports Dirichlet or Neumann conditions, with derivatives taken
in the positive y direction at **both** walls. Pure Neumann Poisson problems
require a compatibility condition and a choice of the additive constant.

Store coefficients in matrices of size `(B, P+1)`: each row is one
system, and each column is one Chebyshev coefficient:

$$
u_s(y)=\sum_{n=0}^{P}\widehat{u}_{s,n}T_n(y),
\qquad \texttt{u[s,n+1]}=\widehat{u}_{s,n}.
$$

Julia stores the first matrix index contiguously. Thus the coefficients
of a fixed degree for all systems lie next to each other in memory. This
is useful because, at a given step of elimination or substitution, the
same arithmetic can be applied independently to every system.

The solver assembles the equations directly in the output matrix and uses
views of its even and odd columns. You do not need to transpose the fields
or pack their coefficients into separate even/odd arrays. The forcing and
output must still occupy separate storage.

```julia
using ChebyshevHelmoltzSolvers

P, B = 32, 100
θ₀ = fill(1.0, B)
θ₁ = collect(range(1.0, 2.0; length=B))
h = BatchedHelmoltzSolver(P, B)
update!(h, θ₀, θ₁)

# Manufactured solution u = 1-y² = (T₀-T₂)/2 in every system.
# θ₀*u'' - θ₁*u = (-2θ₀-θ₁/2)*T₀ + (θ₁/2)*T₂.
f = zeros(ComplexF64, B, P+1)
f[:, 1] .= -2 .* θ₀ .- θ₁ ./ 2
f[:, 3] .= θ₁ ./ 2
u = similar(f)
bc = zeros(B)
solve!(h, u, f, bc, bc)
# Zero endpoint values; f is preserved.
```

Required `u₊, u₋` arguments are one-based vectors of length `B`, one entry
per system. They can be ordinary vectors or externally defined constant-valued
vectors with no backing storage. Both may be complex. Reuse the factors
for changing right-hand sides and wall data. After initialisation, call
`update!(h, θ₀, θ₁)` only when the operator changes. Updates assemble and factor
directly in the existing arrays; they do not construct another solver or copy
a new factor bank. On the CPU, coefficient vectors may be views of larger arrays.

On the CPU, the inner loop runs over systems. `@simd` allows the compiler
to process several systems with one vector instruction. The recurrence
within each system is still sequential. This is **vectorisation on one
CPU thread**, not multithreading and not a BLAS call.

## Batched coupled Helmholtz solves

Here “coupled” refers to the two second-order equations within each problem;
the different problems in the batch remain independent.
`BatchedCoupledHelmoltzSolver` applies the influence-matrix construction
above separately to every system. Each row stores one system, each column
one Chebyshev coefficient, exactly as for `BatchedHelmoltzSolver`.
The four operator coefficients are vectors, so every system may have a
different pair of Helmholtz operators:

$$
(\theta_{0,s}D^2-\theta_{1,s})(\theta_{2,s}D^2-\theta_{3,s})u_s=f_s,
\qquad u_s(\pm1)=u_s'(\pm1)=0.
$$

```julia
B, P = 256, 32
h = BatchedCoupledHelmoltzSolver(P, B, ComplexF64)
θs = (ones(B), zeros(B), ones(B), zeros(B))
update!(h, θs)                         # D⁴u = f in this example
f = zeros(ComplexF64, B, P+1)
f[:, 1] .= 24                         # f(y)=24
u = similar(f)
solve!(h, u, f)                        # u(y)=(1-y²)²; f is preserved
```

`update!` factors both operator banks and caches the two homogeneous
responses and inverse 2×2 influence matrix for every system. Each `solve!`
then needs only two particular Helmholtz solves and a wall-slope correction.
It reuses the allocated workspaces. CPU correction loops run across
contiguous systems for SIMD; the CUDA correction assigns one system to each
thread. Do not use the same solver concurrently from multiple tasks or streams.

## GPU Helmholtz solves

The GPU solves the same discrete equations in the same `(system, coefficient)`
layout. It gains parallelism from the number of independent systems, rather
than from splitting one coefficient recurrence across threads.

A solver contains arrays of factors and, for coupled problems, cached
responses and workspace. `adapt(CuArray, h)` creates a GPU version by
transferring all of that storage. It preserves the numerical precision;
it does not change the equation or rebuild the factors.

CUDA support is an optional package extension. Install
[CUDA.jl](https://cuda.juliagpu.org/stable/installation/overview/) in your Julia
environment together with `Adapt` (`Pkg.add(["CUDA", "Adapt"])`), and load
them before using device arrays:

```julia
using CUDA, Adapt, ChebyshevHelmoltzSolvers

B, P = 256, 32
h = BatchedHelmoltzSolver(P, B)
update!(h, ones(B), ones(B))       # u″ - u = f
f = zeros(ComplexF64, B, P+1)
f[:, 1] .= 1                     # f(y)=1
bc = zeros(B)                    # homogeneous Dirichlet walls

CUDA.functional() || error("A working NVIDIA CUDA device is required")
CUDA.allowscalar(false)
h_gpu = adapt(CuArray, h)  # transfer factors once; preserve Float64 precision
f_gpu = CuArray(f)
u_gpu = similar(f_gpu)

bc_gpu = CuArray(bc)
solve!(h_gpu, u_gpu, f_gpu, bc_gpu, bc_gpu)
CUDA.synchronize()  # needed for timings or explicit host-side completion
```

One GPU thread handles one system from beginning to end. Neighbouring
threads handle neighbouring rows, so their accesses to the same coefficient
are contiguous. The factors remain real even when the fields are complex.
A Dirichlet solve performs one kernel launch, with no host transfers or
temporary right-hand-side arrays. Neumann solves additionally check for
unsupported singular systems. A custom constant-valued boundary vector
must be usable inside CUDA kernels.

For a Helmholtz batch, GPU `update!` also assembles and factors on the
device. Device coefficient vectors are used directly; CPU coefficient
vectors are transferred first. Only the two coefficient vectors need
uploading, not the much larger factor arrays.
Updates reject zero or nonfinite UL pivots and nonfinite reciprocal pivots;
construction only allocates storage.

Keep factors, fields and boundary data on the device between solves to
avoid repeated transfers. GPU launches are asynchronous: the host can
continue before a solve finishes. `CUDA.synchronize()` waits for completion,
which is necessary when measuring elapsed time. The package chooses the
thread-block size; callers do not need to configure it.

GPU support covers `BatchedHelmoltzSolver`, `BatchedCoupledHelmoltzSolver`
and `BatchedQuasiTridiagonal`. Scalar solvers and FFTW profile transforms
retain their CPU interfaces. For a coupled solve, transfer the solver and
fields in the same way, then call `solve!(h_gpu, u_gpu, f_gpu)`.
To rebuild coupled operators on device, use
`update!(h_gpu, map(CuArray, θs))`. Boundary conditions remain clamped and
homogeneous. Singular Neumann Poisson batches remain CPU-only.

### CUDA verification and timing

Run from the repository root on the A100 (or another supported NVIDIA GPU):

```sh
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
julia --threads=1 --project=test/cuda perf/benchmark_devices.jl cuda gpu-results.csv
```

The CUDA tests require a functional device and fail explicitly if none is
available. They disable host scalar indexing and check manufactured solutions,
complex wall data, both boundary conditions, parity views and factor updates.
The benchmark warms up first and synchronizes GPU execution; it measures
complete batched solves with native y-last storage. Factorisation and data
transfers are setup costs and are excluded from solve timings.

The ordinary CPU test suite does not load CUDA. The repository includes a
GitHub Actions workflow for Julia 1.10 and the current stable Julia; that
CPU workflow is not a GPU verification.
The CUDA implementation has been validated on an NVIDIA A100 80 GB PCIe,
with scalar indexing disabled: 7,458 CUDA checks cover the Helmholtz, UL,
backend-contract and coupled paths. The GPU test target also runs CPU
reference checks. Recorded logs accompany the benchmark results.

## Batched CPU and A100 benchmarks

There are two distinct costs. An **update** prepares the operator, including
the homogeneous responses for coupled problems. A **solve** applies that
prepared operator to a new forcing. If many forcings share an operator,
setup is paid once, so solve time is the more relevant repeated cost.
Both are measured separately below.

The current benchmark measures **Helmholtz and coupled Helmholtz** solvers at
`N_y = 8,16,32,64,128,256,512,1024` coefficients and
`B = 64,256,1024,4096,16384,65536` systems. Fields use `ComplexF64` in native
`(system, coefficient)` storage. Every point is the **minimum of 500 warmed
samples**, with short calls repeated within each sample to reduce timer noise.
This is a minimum-time estimate, not a typical latency or confidence interval.

The measured solver revision is
[`92072b1`](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/commit/92072b170312604ed4f60cfcfcca4d5a08084af5).
The [raw results and environment records](perf/results/92072b1/README.md)
identify the hardware, software versions and benchmark script hash.

All figures use $N_y$ on the horizontal axis and one colour/marker per
system count. Legends such as $8^2$ mean 64 independent systems. Timing
panels show microseconds **per whole batch**, including all $B$ systems.
Speedup panels show a ratio, not a time.

### Local CPU

These measurements use an Apple M5 MacBook Air, Julia 1.12.6, and one Julia
and BLAS thread. Batching exposes SIMD across contiguous systems; the solver
kernels do not use BLAS. The plots show **time per whole batch**, without dividing by $B$. Coupled updates include both factorizations and
rebuilding the homogeneous responses and influence matrices.

The Mac completed 95 of 96 configurations. The coupled case with
$N_y=1024$ and $B=65536$ was stopped after memory pressure caused substantial
swapping on the 16 GB machine. That point is omitted, not extrapolated; the
compute-node sweep includes it.

![Local batched solve time per batch](perf/results/92072b1/local-cpu-solve.png)

![Local batched update time per batch](perf/results/92072b1/local-cpu-update.png)

### NVIDIA A100

The A100 80 GB PCIe measurements include a **batched CPU baseline on the same
compute node**, using one Julia and BLAS thread. The plotted speedup is

$$
S = \frac{t_{\mathrm{batched\ CPU}}}{t_{\mathrm{A100}}}.
$$

Values above one favour the GPU. The local Mac timings are separate results
and are not used in this ratio. CPU batching already uses SIMD; this is not
a comparison against a scalar loop of independent solvers.

GPU timings include public API checks, kernel launches and synchronization.
Solver/field construction, compilation and host/device transfers are excluded:
factors and fields remain on the device. Any work performed inside the public
solve or update call is included. GPU results are checked against the CPU result
before timing each configuration. Update timings are measured separately
from repeated solves.

At the largest Helmholtz case ($N_y=1024$, $B=65536$), the measured solve
minimum is **5.95 ms on the A100 versus 837 ms on the batched CPU**,
approximately **141× faster**. Updating the same operator bank takes
**9.39 ms versus 664 ms**, approximately **71× faster**. These ratios use
the single-threaded Xeon baseline, not a fully threaded CPU implementation.

At the same largest size, the **coupled** solve takes **15.31 ms versus
2.41 s**, approximately **157× faster**. Its update, including cached
homogeneous responses, takes **45.10 ms versus 6.75 s**, approximately
**150× faster**.

![Helmholtz solve timings and GPU speedup](perf/results/92072b1/a100-helmholtz-solve.png)

![Coupled solve timings and GPU speedup](perf/results/92072b1/a100-coupled-solve.png)

Small batches may be slower on the GPU because launch overhead and too few
independent systems limit useful parallelism. At larger batch sizes, threads
work on many independent systems with contiguous accesses. Increasing $N_y$
alone lengthens each thread's sequential recurrence; it does not create more
independent GPU work. These are solver measurements, not full DNS speedups.

Operator setup has a different cost from repeated solves. In particular,
coupled `update!` also recomputes the cached influence responses; its speedup
must not be inferred from the solve plot.

![Helmholtz update timings and GPU speedup](perf/results/92072b1/a100-helmholtz-update.png)

![Coupled update timings and GPU speedup](perf/results/92072b1/a100-coupled-update.png)

See [reproduction instructions](perf/README.md) for sample-count and size
controls. No layout conversion is part of these measurements.

<details>
<summary>Historical scalar-versus-batched CPU comparison (Float64)</summary>

Measured on an **Apple M5 MacBook Air**, Julia **1.12.6** (`apple-m1` LLVM
target), Float64, one Julia thread and one BLAS thread, on 2026-09-23.
These are **complete Helmholtz solves**, including RHS assembly, boundary
conditions, both parity substitutions and public API checks. Factors are
reused. Each point is the minimum of 100 warmed samples. The solve-time plot
shows the time to solve all $B$ systems, and the update figure shows the time
to update all $B$ systems.
Both speedup plots use $N_y$ on the horizontal axis, with one curve per
system count and matching colours and markers.

Across `N_y = 8, 16, 32, 64, 128, 256, 512` and batch sizes
`B = 64, 256, 1024, 4096, 16384`, native batched solves were
**3.21–8.09× faster** than independent scalar solves with contiguous
coefficient vectors. Colours and markers identify batch sizes.

The plotted solve speedup is

$$
S_{\mathrm{solve}}=\frac{t_{\mathrm{scalar\ loop}}}{t_{\mathrm{batched}}},
$$

so values greater than one favour batching. The update figure uses the same
convention: $S_{\mathrm{update}}=t_{\mathrm{scalar\ update}}/t_{\mathrm{batched\ update}}$.

The speedup comes from the batched organisation described above: contiguous
accesses across systems expose independent work to SIMD and reduce per-call
overhead. These measurements use one CPU thread and no BLAS kernels. They
do not isolate the contribution of vector instructions from cache behaviour
or other overheads.

![Full Helmholtz solve timings and speedups](perf/results/helmoltz-solves.png)

Both solve paths measured **zero allocations** after warm-up in this sweep.
Factors and RHS/output arrays are allocated before timing.

In that historical sweep, batched `update!` also measured **zero allocations**
after warm-up and was
**1.36–2.24× faster** than updating the independent scalar solvers.
Assembly and UL factorisation sweep contiguous systems with SIMD, while
retaining coefficient, alias and pivot validation. Factors and integration
weights are reused in place.

These update results include assembly, factorisation and validation.
They combine vectorisation, contiguous memory access and removal of temporary
wrappers; the gain varies with problem size.

![Helmholtz operator update costs](perf/results/helmoltz-updates.png)

See [benchmark scripts and reproduction instructions](perf/README.md),
[raw solve/update measurements](perf/results/helmoltz-cpu.csv), and
[environment details](perf/results/environment.txt). These are local CPU
measurements from the earlier Float64 implementation, not complete DNS
timings. The newer ComplexF64 CPU/A100 measurements are recorded separately.

</details>

## Cost, reuse and numerical limits

Each parity system has a fixed number of entries per interior row.
Factorisation and substitution therefore take $\mathcal{O}(P)$ work and
storage: doubling the polynomial degree roughly doubles the amount of
arithmetic and data. A batch of $B$ systems takes $\mathcal{O}(BP)$ work and
storage. The coupled solver has the same scaling, with additional solves
and cached responses. These operation counts do not imply that wall-clock
time scales perfectly; memory access and GPU launch overhead also matter.

For a scalar or batched Helmholtz solver, repeated solves preserve the
factors. Concurrent solves can share them if output arrays are disjoint
and no task calls `update!` at the same time. Coupled solvers also modify
their internal workspace, so each concurrent task needs its own solver.

The fast UL factorisation does **not pivot**: it does not swap equations to
avoid a bad division. `update!` rejects zero or nonfinite reciprocal pivots;
scalar `solve!` also checks its factors before changing the output.
A matrix may be invertible yet fail this particular elimination order.
For example, `P=4`, `θ₀=1`, `θ₁=-6` gives such a breakdown. The package raises
an error rather than switching to a general pivoted solver. After a failed
update, perform a successful update before reusing the solver.

Operator coefficients must be finite, with a nonzero second-derivative
coefficient for each Helmholtz operator. The coupled influence matrix must
also be invertible. The supported singular Neumann Poisson case has the
separate compatibility and gauge treatment described above.

Finally, a successful factorisation is not an accuracy guarantee for an
ill-conditioned problem. Near singularity, small input or rounding errors
can produce large solution errors. Multiplying by stored reciprocal pivots
also rounds differently from direct division; very small pivots can have
nonfinite reciprocals. Increasing the polynomial degree cannot remove
these floating-point limits.

## Tests

From a checkout:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The tests check both the mathematics and the array interface:

- **Known solutions:** prescribe a polynomial or smooth function, derive its
  forcing and wall data, and check that the solver recovers it.
- **Independent discretisation:** compare against a dense tau operator
  built from differentiation, rather than reusing the integration recurrence.
  Accuracy tests include convergence through degree 512 and nearly singular
  shifted Neumann equations.
- **Reuse and storage:** check real/complex and single/double precision,
  odd/even degrees, vector views, repeated updates, forcing preservation,
  cached responses, and rejection of invalid inputs.

The Poisson tests check compatibility and the zero-mean choice. Coupled
tests check all four wall conditions. CUDA tests use a real device with
host scalar indexing disabled; a CPU-only run cannot establish GPU correctness.
The [test guide](test/README.md) lists the files and commands. CPU tests use
Julia's `Test` standard library and the package dependencies; the separate
CUDA target additionally requires CUDA and a working NVIDIA device.

## References

- C. Canuto, M. Y. Hussaini, A. Quarteroni and T. A. Zang (2006).
  *Spectral Methods: Fundamentals in Single Domains*. Springer.
  [DOI: 10.1007/978-3-540-30726-6](https://doi.org/10.1007/978-3-540-30726-6).
  Background for Chebyshev approximation, spectral discretisation and algebraic solvers.
- L. Greengard (1991). “Spectral Integration and Two-Point Boundary Value
  Problems.” *SIAM Journal on Numerical Analysis*, **28**(4), 1071–1080.
  [DOI: 10.1137/0728057](https://doi.org/10.1137/0728057);
  [author-hosted paper](https://math.nyu.edu/~greengar/specint_sinum.pdf).
  An integration-based formulation for constant-coefficient boundary-value problems.
- D. Viswanath (2014 revision). “Spectral integration of linear boundary
  value problems.” [arXiv:1205.2717v2](https://arxiv.org/abs/1205.2717v2)
  (first submitted in 2012). Discusses spectral integration, bordered banded
  systems and numerical accuracy.
- L. Kleiser and U. Schumann (1980). “Treatment of incompressibility and
  boundary conditions in 3-D numerical spectral simulations of plane channel
  flows.” In *Proceedings of the Third GAMM Conference on Numerical Methods
  in Fluid Mechanics*, Notes on Numerical Fluid Mechanics, vol. 2, Vieweg.
  [Institutional record](https://publikationen.bibliothek.kit.edu/240013603).
  Historical reference for influence-matrix treatment of spectral boundary conditions.

The logo adapts the illustration and visual identity of FDHelmoltzSolver.jl.

## Licence

MIT licence. Copyright © 2026 Davide Lasagna. See [LICENSE](LICENSE).
