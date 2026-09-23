<p align="center">
  <img src="assets/logo.svg" alt="ChebyshevHelmoltzSolvers.jl logo" width="900">
</p>

# ChebyshevHelmoltzSolvers.jl

[![CI](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml)

In-place Chebyshev tau solvers for one-dimensional Helmholtz boundary-value
problems on $[-1,1]$. Scalar operators use even/odd quasi-tridiagonal UL
factorisations and cache the reciprocal pivots for repeated substitutions.
The coupled solver caches its homogeneous responses and
influence matrix, leaving only two scalar solves per right-hand side.

## Installation

Requires Julia 1.10 or later. Install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl")
```

## Chebyshev representation

Functions are represented by truncated expansions in Chebyshev polynomials
of the first kind. A degree-$P$ expansion has $P+1$ coefficients; these are
distinct from the function values at the collocation points.

A one-based `AbstractVector` stores the **ordinary**, unweighted expansion

$$
f_P(y)=\sum_{n=0}^{P}a_n T_n(y),
\qquad T_n(\cos\vartheta)=\cos(n\vartheta).
$$

Index `n+1` stores $a_n$, the coefficient of degree $n$. Use `zeros(T, P+1)`
to allocate an expansion or pass a one-based vector view directly. Both odd
and even polynomial degrees are supported.

The descending Chebyshev–Lobatto points returned by `chebpoints(P)` are

$$
y_j=\cos\left(\frac{j\pi}{P}\right), \qquad j=0,\ldots,P.
$$

Use `chebcoeffs(values)` to convert samples at these points into a coefficient
vector. This uses FFTW's DCT-I, divides the
transform by `P`, and halves the coefficients of degrees `0` and `P`. The input is preserved.

| Operation | Purpose |
| --- | --- |
| `chebpoints(P)` | Return the $P+1$ descending Chebyshev–Lobatto points. |
| `chebcoeffs(values)` | Convert descending Lobatto samples into ordinary Chebyshev coefficients. |
| `chebvalues(a)` | Evaluate coefficients at descending Lobatto points. |
| `diff!(a)` | Differentiate Chebyshev coefficients in place on [-1, 1]. |
| `diff!(out, a)` | Differentiate into distinct, non-aliasing output storage. |
| `diff(a, :left)` | Evaluate the derivative at -1 without a derivative workspace. |
| `diff(a, :right)` | Evaluate the derivative at +1. |

For a physical interval $x\in[a,b]$, use the affine map

$$
x=\frac{a+b}{2}+\frac{b-a}{2}y,\qquad
\frac{\mathrm{d}}{\mathrm{d}x}=\frac{2}{b-a}\frac{\mathrm{d}}{\mathrm{d}y},\qquad
\frac{\mathrm{d}^2}{\mathrm{d}x^2}=\left(\frac{2}{b-a}\right)^2\frac{\mathrm{d}^2}{\mathrm{d}y^2}.
$$

Multiply the physical second-derivative coefficient by $(2/(b-a))^2$ before
passing it to the solver. Multiply returned endpoint derivatives by
$2/(b-a)$ to obtain physical derivatives.


## Scalar Helmholtz solver

$$
\theta_0 u''(y)-\theta_1 u(y)=f(y), \qquad -1<y<1,
$$

$$
u(1)=u_+, \qquad u(-1)=u_-.
$$

`HelmoltzSolver(P, T=Float64)` allocates factors and integration weights for degree
`P ≥ 3`, with `P + 1` coefficients. Call `update!` before solving and whenever
the operator coefficients change. Right-hand sides and boundary values can
change without another update.

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

**Boundary arguments are upper then lower:** `solve!(h, u, f, u₊, u₋)`.
Scalar factors and coefficient vectors are real-valued (`Float32` or `Float64`).
The batched solver additionally supports complex coefficient matrices.
Use `HelmoltzSolver(P; neum=true)` to prescribe positive-y derivatives at both
walls. Pure Neumann Poisson (`θ₁=0`) is singular and is rejected; handle its
compatibility condition and additive constant separately.
The scalar tau equations are imposed through degree `P - 2`; the two
highest right-hand-side coefficients do not enter the solve.

Writing $u_P$ and $f_P$ for the degree-$P$ expansions, the tau conditions are

$$
\left[\theta_0 u_P''-\theta_1 u_P-f_P\right]_n=0,
\qquad n=0,\ldots,P-2,
$$

where $[\cdot]_n$ denotes the coefficient of $T_n$. The two boundary equations
complete the $P+1$ equations for the solution coefficients. Integrating the
interior equations twice produces a tridiagonal recurrence within each
parity; the wall conditions supply the dense first row of each block.

For `neum=true`, the boundary equations instead prescribe

$$
u'(1)=u_+, \qquad u'(-1)=u_-.
$$

Both derivatives use the positive-$y$ direction. When $\theta_1=0$ and
$\theta_0\ne0$, a Neumann solution would require

$$
\int_{-1}^{1}f(y)\,\mathrm{d}y=\theta_0(u_+-u_-),
$$

and would remain undetermined up to an additive constant. This singular case
is outside the solver's supported Neumann interface.

## Coupled Helmholtz solver

$$
\begin{aligned}
\theta_0 u''-\theta_1 u &= f,\\
\theta_2 v''-\theta_3 v &= u,\\
v(\pm1)=v'(\pm1)&=0.
\end{aligned}
$$

Equivalently, with $D=\mathrm{d}/\mathrm{d}y$,

$$
(\theta_0 D^2-\theta_1)(\theta_2 D^2-\theta_3)v=f.
$$

`CoupledHelmoltzSolver` solves this factored fourth-order problem and returns
`v`. Each `update!` factorises both operators, computes the two homogeneous
responses and stores their 2×2 influence matrix. Each `solve!` computes the
particular solution with two scalar Helmholtz solves, then adds the cached
responses to cancel the two wall derivatives.

For $v=(1-y^2)^2$, the problem $v^{(4)}=24$ has the required wall conditions:

```julia
P = 16
solver = CoupledHelmoltzSolver(P)
update!(solver, (1.0, 0.0, 1.0, 0.0))

y = chebpoints(P)
rhs = chebcoeffs(fill(24.0, length(y)))
solve!(solver, rhs)

v = chebvalues(rhs)  # v ≈ (1 .- y.^2).^2

# v = 3T₀/8 - T₂/2 + T₄/8
# rhs[1] ≈ 0.375, rhs[3] ≈ -0.5, rhs[5] ≈ 0.125
```

The source storage must be distinct from the solver's internal workspaces.
The intermediate `u` is not retained. The influence matrix must be nonsingular;
use degree at least four to represent a nonzero solution with four homogeneous
wall conditions.

## Batched CPU Helmholtz solves

The batch solves independent boundary-value problems

$$
\theta_{0,s}u_s''(y)-\theta_{1,s}u_s(y)=f_s(y),
\qquad s=1,\ldots,B.
$$

Each system has its own coefficients and boundary data; systems are not coupled.

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

The solver assembles
the integrated equations directly into the output and solves even/odd
coefficients in place. Source and output must be distinct; no transposition,
parity packing or additional RHS array is needed.

```julia
using ChebyshevHelmoltzSolvers

P, B = 32, 100
θ₀ = fill(1.0, B)
θ₁ = collect(range(1.0, 2.0; length=B))
h = BatchedHelmoltzSolver(P, B)
update!(h, θ₀, θ₁)

# Manufactured solution u = 1-y² = (T₀-T₂)/2 in every system.
# θ₀*u'' - θ₁*u = (-2θ₀-θ₁/2)*T₀ + (θ₁/2)*T₂.
rhs = zeros(ComplexF64, B, P+1)
rhs[:, 1] .= -2 .* θ₀ .- θ₁ ./ 2
rhs[:, 3] .= θ₁ ./ 2
u = similar(rhs)
bc = zeros(B)
solve!(h, u, rhs, bc, bc)
# Zero endpoint values; rhs is preserved.
```

Required `u₊, u₋` arguments are one-based vectors of length `B`, one entry
per system. They can be ordinary vectors or externally defined constant-valued
vectors with no backing storage. Both may be complex. Reuse the factors
for changing right-hand sides and wall data. After initialisation, call
`update!(h, θ₀, θ₁)` only when the operator changes. Updates assemble and factor
directly in the existing arrays; they do not construct another solver or copy
a new factor bank. On the CPU, coefficient vectors may be views of larger arrays.

The CPU path uses SIMD across all systems for each coefficient row.

For lower-level work, `BatchedQuasiTridiagonal(B, M, T)` allocates zero
storage for `B` matrices of size `M ≥ 2`. Assemble their entries and call
`ul!(batch)` before solving. The four-array constructor
`BatchedQuasiTridiagonal(b, l, dᵢ, u)` shares the supplied storage, whether
assembled or already factorised; no extra arrays are allocated.
`ldiv!(batch, rhs)` uses the factors to solve **assembled** sources,
which include the boundary equation. Raw Chebyshev forcing should instead go
through `BatchedHelmoltzSolver`. The batch size is encoded in both batched types. `BatchedQuasiTridiagonal`
also encodes the matrix size `M`; both sizes stay fixed for its lifetime.

## GPU Helmholtz solves

The following example continues from the CPU setup above.

CUDA support is an optional package extension. Install
[CUDA.jl](https://cuda.juliagpu.org/stable/installation/overview/) in your Julia
environment together with `Adapt` (`Pkg.add(["CUDA", "Adapt"])`), and load
them before using device arrays:

```julia
using CUDA, Adapt

CUDA.functional() || error("A working NVIDIA CUDA device is required")
CUDA.allowscalar(false)
h_gpu = adapt(CuArray, h)  # transfer factors once; preserve Float64 precision
rhs_gpu = CuArray(rhs)
u_gpu = similar(rhs_gpu)

bc_gpu = CuArray(bc)
solve!(h_gpu, u_gpu, rhs_gpu, bc_gpu, bc_gpu)
CUDA.synchronize()  # needed for timings or explicit host-side completion
```

The CUDA kernel assigns one thread to each system. Adjacent threads operate
on adjacent systems, using real factors for real or complex right-hand sides.
A solve performs one kernel launch, with no host transfers or temporary
RHS arrays. Keep stored fields, factors and boundary vectors on the GPU;
a custom storage-free boundary vector must be compatible with CUDA kernels.

A GPU `update!` also assembles and factors on the device. Device coefficient
vectors are used directly; CPU coefficient vectors are transferred first.
Only these two vectors need uploading, not the much larger factor arrays.
Updates reject zero or nonfinite UL pivots and nonfinite reciprocal pivots;
construction only allocates storage.

The CUDA path chooses its own thread-block size.

GPU support covers `BatchedHelmoltzSolver` and `BatchedQuasiTridiagonal`.
The existing scalar and coupled solvers and FFTW profile transforms retain
their CPU interfaces.

### CUDA verification and timing

Run from the repository root on the A100 (or another supported NVIDIA GPU):

```sh
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
julia --project=test/cuda perf/benchmark_cuda.jl
```

The CUDA tests require a functional device and fail explicitly if none is
available. They disable host scalar indexing and check manufactured solutions,
complex wall data, both boundary conditions, parity views and factor updates.
The benchmark warms up first and synchronizes GPU execution; it measures
complete batched solves with native y-last storage. Factorisation and data
transfers are setup costs and are excluded from solve timings.

The ordinary CPU test suite does not load CUDA. GitHub Actions runs it on
Julia 1.10 and the current stable Julia; that CI is not a GPU verification.
The reciprocal-pivot implementation has not yet been validated on a GPU or
benchmarked; the CUDA commands above provide the explicit verification path.

## CPU benchmarks

Measured on an **Apple M5 MacBook Air**, Julia **1.12.6** (`apple-m1` LLVM
target), Float64, one Julia thread and one BLAS thread, on 2026-09-23.
These are **complete Helmholtz solves**, including RHS assembly, boundary
conditions, both parity substitutions and public API checks. Factors are
reused. Each point is the minimum of 100 warmed samples. The solve-time plot
shows time per system (batch time divided by $B$); update times are per batch.

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

The batched algorithm exposes **SIMD vectorisation across systems**. Within
one system, each elimination or substitution step depends on neighbouring
coefficients, so those steps remain sequential. At a fixed coefficient,
however, the same operation can be applied independently to every system.
The batched loops put that system index innermost and use `@simd`, allowing
the compiler to process several systems with each vector instruction.
The `(system, coefficient)` layout makes these accesses contiguous.

This is single-threaded vectorisation, not multithreading or a BLAS speedup.
The arithmetic complexity remains $\mathcal{O}(BP)$. The measured solve gains combine
SIMD, memory-access behaviour and amortised per-call overhead; the benchmarks
do not isolate the contribution of SIMD alone. Both solve paths are
allocation-free, so their timing difference is not due to garbage collection.

![Full Helmholtz solve timings and speedups](perf/results/helmoltz-solves.png)

Both solve paths measured **zero allocations** after warm-up in this sweep.
Factors and RHS/output arrays are allocated before timing.

Batched `update!` also measured **zero allocations** after warm-up and was
**1.36–2.24× faster** than updating the independent scalar solvers.
Assembly and UL factorisation sweep contiguous systems with SIMD, while
retaining coefficient, alias and pivot validation. Factors and integration
weights are reused in place.

The same loop ordering exposes independent operations to SIMD during
assembly and factorisation. Compared with the previous system-by-system
batched update, this also removes strided row traversal and temporary
wrappers. Its improvement therefore combines vectorisation, contiguous
access and allocation removal. Speedup varies with problem size because
memory traffic, cache capacity and validation costs also affect runtime.

![Helmholtz operator update costs](perf/results/helmoltz-updates.png)

See [benchmark scripts and reproduction instructions](perf/README.md),
[raw solve/update measurements](perf/results/helmoltz-cpu.csv), and
[environment details](perf/results/environment.txt). These are local CPU
measurements, not complete DNS timings or A100 predictions. The CUDA extension
loads on this host, but GPU execution and speed remain **unverified**: no
functional NVIDIA device is available here.

## Quasi-tridiagonal matrices and UL factorisation

`QuasiTridiagonal(M, T)` stores a matrix with a dense first row and a
tridiagonal interior. For example, when $M=5$,

$$
Q=
\begin{pmatrix}
b_1 & b_2 & b_3 & b_4 & b_5 \\
c_1 & a_2 & e_2 & 0 & 0 \\
0 & c_2 & a_3 & e_3 & 0 \\
0 & 0 & c_3 & a_4 & e_4 \\
0 & 0 & 0 & c_4 & a_5
\end{pmatrix}.
$$

This is the structure of each even/odd Helmholtz block: the dense row
imposes a boundary condition, and the other rows are the integrated
coefficient equations. Only four vectors are stored, rather than a dense
$M\times M$ array:

| Field | Before `ul!(Q)` | After `ul!(Q)` |
| --- | --- | --- |
| `b`, length $M$ | Dense first row $b_i$ | Reciprocal $1/\beta_1$ at index 1; entries $\beta_i$ for $i\ge2$ |
| `l`, length $M-1$ | Subdiagonal $c_i$ | Multipliers $\ell_i$ of the unit lower factor |
| `dᵢ`, length $M-1$ | Interior diagonal $a_{i+1}$ | Reciprocal pivots $1/p_{i+1}$ |
| `u`, length $M-2$ | Superdiagonal $e_{i+1}$ | Unchanged |

The size constructor allocates zero-filled assembly buffers. Alternatively,
`QuasiTridiagonal(b, l, dᵢ, u)` wraps existing vectors without copying.
The name `dᵢ` describes its contents **after factorisation**; ordinary diagonal
entries must be supplied when assembling a new matrix.

### Why UL rather than LU?

`ul!(Q)` factors the assembled matrix as $Q=UL$, where

$$
U=
\begin{pmatrix}
\beta_1 & \beta_2 & \beta_3 & \beta_4 & \beta_5 \\
0 & p_2 & e_2 & 0 & 0 \\
0 & 0 & p_3 & e_3 & 0 \\
0 & 0 & 0 & p_4 & e_4 \\
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

Eliminating from the bottom upwards preserves this compact structure:
$U$ is upper bidiagonal below its dense first row, and $L$ is unit lower
bidiagonal. A conventional top-down elimination would spread the dense
boundary row into the interior.

Start with $p_M=a_M$. For $i=M-1,\ldots,2$, compute

$$
\ell_i=\frac{c_i}{p_{i+1}},\qquad
p_i=a_i-e_i\ell_i,
$$

then set $\ell_1=c_1/p_2$. The dense row satisfies

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

### Solving with the factors

`ldiv!(Q, rhs)` solves $Qx=r$ by first solving $Uz=r$ backwards:

$$
z_M=\frac{r_M}{p_M},\qquad
z_i=\frac{r_i-e_i z_{i+1}}{p_i},\qquad i=M-1,\ldots,2,
$$

$$
z_1=\frac{r_1-\sum_{j=2}^{M}\beta_j z_j}{\beta_1}.
$$

It then solves $Lx=z$ forwards:

$$
x_1=z_1,\qquad x_i=z_i-\ell_{i-1}x_{i-1},\qquad i=2,\ldots,M.
$$

Both passes overwrite `rhs`; the first entry accumulates the dense-row
residual as the other entries become available. No temporary solution vector
is needed, and the factors are preserved for the next right-hand side.
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

## Cost and numerical requirements

Storage, factorisation and scalar solves scale as $\mathcal{O}(P)$. Substitutions
multiply by the stored reciprocals instead of dividing, without allocating
additional reciprocal arrays. The coupled solver also scales as $\mathcal{O}(P)$,
with its homogeneous problems paid for at
`update!` time. The coupled solver reuses a mutable workspace; use a separate instance for
each concurrent task. Scalar and batched factors can be shared by solves
with disjoint destination arrays, provided no concurrent `update!` occurs.

UL factorisation has no pivoting. Batched Helmholtz `update!` rejects zero or
nonfinite pivots and nonfinite stored reciprocals. Construction only allocates
storage; call `update!` before solving. The scalar factorisation has no
singularity check. The chosen scalar operators must have nonzero pivots with
finite reciprocals, and the coupled influence system must be invertible.
Multiplication by a stored reciprocal can round differently from direct
division; a finite nonzero pivot can also have an overflowing reciprocal.
Floating-point conditioning and range still limit accuracy as the degree or
operator parameters increase.

## Tests

From a checkout:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The deterministic suite checks one-based coefficient vectors and views, in-place
differentiation, wall derivatives, UL reconstruction and repeated solves,
reciprocal-pivot refresh and preservation, analytic scalar and coupled solutions,
real and complex coefficients,
single and double precision, odd/even degrees, operator updates and
preservation of cached homogeneous responses. Tests need only Julia's
`Test` standard library in addition to the package dependencies.

The logo adapts the illustration and visual identity of FDHelmoltzSolver.jl.
