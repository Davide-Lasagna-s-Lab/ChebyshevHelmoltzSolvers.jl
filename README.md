<p align="center">
  <img src="assets/logo.svg" alt="ChebyshevHelmoltzSolvers.jl logo" width="900">
</p>

# ChebyshevHelmoltzSolvers.jl

[![CI](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml)

In-place Chebyshev tau solvers for one-dimensional Helmholtz boundary-value
problems on **[-1, 1]**. Scalar operators use even/odd quasi-tridiagonal UL
factorisations and cache the reciprocal pivots for repeated substitutions.
The coupled solver caches its homogeneous responses and
influence matrix, leaving only two scalar solves per right-hand side.

The scalar discretisation follows the even/odd construction in
[Channelflow's Helmholtz solver](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/helmholtz.cpp).
The package is a Chebyshev counterpart to
[FDHelmoltzSolver.jl](https://github.com/Davide-Lasagna-s-Lab/FDHelmoltzSolver.jl).

## Installation

Requires Julia 1.10 or later. Install this batched-development branch from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl",
        rev="batched-cpu-solves")
```

## Scalar Helmholtz solver

```text
θ₀ u''(y) - θ₁ u(y) = r(y)
u(+1) = u₊,  u(-1) = u₋
```

`HelmoltzSolver(P, T=Float64)` allocates factors and integration weights for degree
`P ≥ 3`, with `P + 1` coefficients. Call `update!` before solving and whenever
the operator coefficients change. Right-hand sides and boundary values can
change without another update.

For example, solve `u'' - 4u = -6 + 4y²`, whose solution is `u = 1 - y²`:

```julia
using ChebyshevHelmoltzSolvers

P = 16
h = HelmoltzSolver(P)
update!(h, 1.0, 4.0)

# Sample the right-hand side at Lobatto points, ordered from +1 to -1.
y = [cospi(j/P) for j = 0:P]
rhs = chebcoeffs(-6 .+ 4 .* y.^2)
solution = similar(rhs)
solve!(h, solution, rhs, 0.0, 0.0)  # preserves rhs

u = chebvalues(solution)  # u ≈ 1 .- y.^2

# u = (T₀ - T₂)/2: solution[1] ≈ 0.5, solution[3] ≈ -0.5
```

**Boundary arguments are upper then lower:** `solve!(h, u, f, u₊, u₋)`.
Scalar factors and coefficient vectors are real-valued (`Float32` or `Float64`).
The batched solver additionally supports complex coefficient matrices.
Use `HelmoltzSolver(P; neum=true)` to prescribe positive-y derivatives at both
walls. Pure Neumann Poisson (`θ₁=0`) is singular and is rejected; handle its
compatibility condition and pressure gauge separately.
The scalar tau equations are imposed through degree `P - 2`; the two
highest right-hand-side coefficients do not enter the solve.

## Coupled Helmholtz solver

```text
θ₀ u''(y) - θ₁ u(y) = r(y)
θ₂ v''(y) - θ₃ v(y) = u(y)
v(±1) = v'(±1) = 0
```

`CoupledHelmoltzSolver` solves this factored fourth-order problem and returns
`v`. Each `update!` factorises both operators, computes the two homogeneous
responses and stores their 2×2 influence matrix. Each `solve!` computes the
particular solution with two scalar Helmholtz solves, then adds the cached
responses to cancel the two wall derivatives.

For `v = (1 - y²)²`, the problem `v'''' = 24` has the required wall conditions:

```julia
P = 16
solver = CoupledHelmoltzSolver(P)
update!(solver, (1.0, 0.0, 1.0, 0.0))

y = [cospi(j/P) for j = 0:P]
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

## API changes on this branch

Coefficients are plain one-based vectors: `a[n+1]` is the coefficient of `T_n`.
The former coefficient wrapper is removed. Scalar and batched Helmholtz solves
use `solve!(solver, u, f, ...)`, preserving `f`; the coupled solver retains its
in-place `solve!(solver, f)` interface. Batched fields are matrices, reshaped
by the caller, and both boundary vectors are required. Downstream callers must
be migrated before using this branch; ChannelFlow is not changed here.

## Coefficient convention and utilities

A one-based `AbstractVector` stores the **ordinary**, unweighted expansion

```text
f(y) = a[1] T₀(y) + a[2] T₁(y) + ... + a[P+1] Tₚ(y).
```

Index `n+1` contains the coefficient of degree `n`. Use `zeros(T, P+1)`
to allocate an expansion or pass a one-based vector view directly. Both odd
and even polynomial degrees are supported.

Use `chebcoeffs(values)` to convert samples at descending Lobatto
points `cospi(j/P)` into a vector. This uses FFTW's DCT-I, divides the
transform by `P`, and halves the coefficients of degrees `0` and `P`. The input is preserved.

| Operation | Purpose |
| --- | --- |
| `chebcoeffs(values)` | Convert descending Lobatto samples into ordinary Chebyshev coefficients. |
| `chebvalues(a)` | Evaluate coefficients at descending Lobatto points. |
| `diff!(a)` | Differentiate Chebyshev coefficients in place on [-1, 1]. |
| `diff!(out, a)` | Differentiate into distinct, non-aliasing output storage. |
| `diff(a, :left)` | Evaluate the derivative at -1 without a derivative workspace. |
| `diff(a, :right)` | Evaluate the derivative at +1. |
| `QuasiTridiagonal(M, T)` | Allocate a matrix of size `M ≥ 2` with a dense first row and tridiagonal interior. |
| `ul!(Q)` | Factorise that matrix in place without pivoting. |
| `ldiv!(Q, rhs)` | Solve using existing UL factors; available through `LinearAlgebra`. |

For a mapped interval `[a, b]`, multiply the second-derivative coefficient
by `(2/(b-a))²`. Endpoint derivatives from this package must be multiplied
by `2/(b-a)` to obtain physical derivatives.

## Batched CPU and CUDA Helmholtz solves

`BatchedHelmoltzSolver(P, B, T=Float64; neum=false)` allocates storage for
`B > 0` independent operators of degree `P ≥ 3`, with real factors of type `T`
and real or complex right-hand sides. Call `update!(h, θ₀, θ₁)` before solving
and whenever the operator coefficients change. `θ₀` and `θ₁` are vectors of
length `B` and element type `T`, with one coefficient per system.
Use `fill(ν, B)` when all `B` systems share the same viscosity.
The batch supports Dirichlet or Neumann conditions, with derivatives taken
in the positive y direction at **both** walls. Pure Neumann Poisson problems
need a separate mean-mode solve.

Store the Chebyshev coefficient index **last**: `(system, P+1)` or
`(Nxh, Nz, P+1)`. Leading dimensions enumerate systems in Julia's column-major
order. Pass matrices of size `(B, P+1)` to `solve!`, reshaping fields at the
call site without copying. The solver assembles the integrated equations
directly into the output and solves even/odd coefficients
in place. Source and output must be distinct; no transposition, parity packing
or additional RHS array is needed.

```julia
using ChebyshevHelmoltzSolvers

P, Nxh, Nz = 32, 17, 32
ν = 1 / 400
λ = [1 + ν*((ix-1)^2 + (iz-1)^2) for ix = 1:Nxh, iz = 1:Nz]
h = BatchedHelmoltzSolver(P, Nxh*Nz)
update!(h, fill(ν, Nxh*Nz), vec(λ))

# Manufactured solution u = 1-y² = (T₀-T₂)/2 in every system.
# ν*u'' - λ*u = (-2ν-λ/2)*T₀ + (λ/2)*T₂.
rhs = zeros(ComplexF64, Nxh, Nz, P+1)
rhs[:, :, 1] .= -2ν .- λ/2
rhs[:, :, 3] .= λ/2
u = similar(rhs)
bc = zeros(Nxh*Nz)
solve!(h, reshape(u, Nxh*Nz, P+1), reshape(rhs, Nxh*Nz, P+1), bc, bc)
# Zero wall values; rhs is preserved and reshape does not copy data.
```

Required `u₊, u₋` arguments are one-based vectors of length `B`, one entry
per system. They can be ordinary vectors or externally defined constant-valued
vectors with no backing storage. CUDA boundary vectors must support device
indexing. Both may be complex. Reuse the factors
for changing right-hand sides and wall data. After initialisation, call
`update!(h, θ₀, θ₁)` only when the operator changes. Updates assemble and factor
directly in the existing arrays; they do not construct another solver or copy
a new factor bank. On the CPU, coefficient vectors may be views of larger arrays.

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
solve!(h_gpu, reshape(u_gpu, Nxh*Nz, P+1), reshape(rhs_gpu, Nxh*Nz, P+1), bc_gpu, bc_gpu)
CUDA.synchronize()  # needed for timings or explicit host-side completion
```

The CUDA kernel assigns one thread to each system. Adjacent threads operate
on adjacent Fourier modes, using real factors for complex Fourier data.
A solve performs one kernel launch, with no host transfers or temporary
RHS arrays. Keep stored fields, factors and boundary vectors on the GPU;
a custom storage-free boundary vector must be compatible with CUDA kernels.

A GPU `update!` also assembles and factors on the device. Device coefficient
vectors are used directly; CPU coefficient vectors are transferred first.
Only these two vectors need uploading, not the much larger factor arrays.
Updates reject zero or nonfinite UL pivots and nonfinite reciprocal pivots;
construction only allocates storage.

The CPU path uses SIMD across all systems for each coefficient row.
The CUDA path chooses its own thread-block size.

GPU support covers `BatchedHelmoltzSolver` and `BatchedQuasiTridiagonal`.
The existing scalar and coupled solvers and FFTW profile transforms retain
their CPU interfaces. This package does not change ChannelFlow's field layout.

For lower-level work, `BatchedQuasiTridiagonal(B, M, T)` allocates zero
storage for `B` matrices of size `M ≥ 2`. Assemble their entries and call
`ul!(batch)` before solving. The four-array constructor
`BatchedQuasiTridiagonal(b, l, dᵢ, u)` shares the supplied storage, whether
assembled or already factorised; no extra arrays are allocated.
`ldiv!(batch, rhs)` uses the factors to solve **assembled** sources,
which include the boundary equation. Raw Chebyshev forcing should instead go
through `BatchedHelmoltzSolver`. The batch size is encoded in both batched types. `BatchedQuasiTridiagonal`
also encodes the matrix size `M`; both sizes stay fixed for its lifetime.

## CPU benchmarks

Measured on an **Apple M5 MacBook Air**, Julia **1.12.6** (`apple-m1` LLVM
target), Float64, one Julia thread and one BLAS thread, on 2026-09-23.
These are **complete Helmholtz solves**, including RHS assembly, boundary
conditions, both parity substitutions and public API checks. Factors are
reused. Each point is the median of 21 warmed samples; timings are per batch.

Across `N_y = 33, 65, 129` and `B = 64, 256, 4096`, native batched solves
were **4.14–4.96× faster** than independent scalar solves with contiguous
coefficient vectors. Including both layout conversions gives
**1.79–3.20×**. The strided scalar curve shows the cost of applying
scalar solves directly to the same system-first layout.

![Full Helmholtz solve timings and speedups](perf/results/helmoltz-solves.png)

| Coefficients | Systems | Scalar (µs) | Batched (µs) | Speedup |
|---:|---:|---:|---:|---:|
| 33 | 4096 | 1371.0 | 327.1 | 4.19× |
| 65 | 4096 | 2795.2 | 674.9 | 4.14× |
| 129 | 4096 | 6617.3 | 1377.7 | 4.80× |

The measured batched solve allocates **176 bytes per batch**, independent of
batch size; the scalar loop allocates 176 bytes per system. Neither timing
path allocates an additional full RHS buffer. These figures include the
current wrapper and validation overhead; they are not zero-allocation claims.

**Updates are a trade-off:** batched `update!` was 2.4–9.6× slower than
updating the scalar solvers in this sweep. It also allocates temporary Julia
objects, despite reusing all factor arrays. Row-view assembly, strided
factorisation and batch-wide validation remain optimization targets. Reusing
factors over many solves is therefore important; batching is not a universal
speedup when operators change at every call.

![Helmholtz operator update costs](perf/results/helmoltz-updates.png)

See [benchmark scripts and reproduction instructions](perf/README.md),
[raw solve/update measurements](perf/results/helmoltz-cpu.csv), and
[environment details](perf/results/environment.txt). These are local CPU
measurements, not complete DNS timings or A100 predictions. The CUDA extension
loads on this host, but GPU execution and speed remain **unverified**: no
functional NVIDIA device is available here.

## Cost and numerical requirements

`QuasiTridiagonal` uses just four arrays: `b` (length `M`), `l` and `dᵢ`
(length `M-1`), and `u` (length `M-2`). Before `ul!`, these hold ordinary
matrix entries, with the interior diagonal assembled into `dᵢ`. After UL
factorisation, `b[1]` stores **the inverse boundary pivot** `1/U[1,1]`,
`b[2:M]` stores the remaining first-row entries, and `dᵢ[k]` stores
`1/U[k+1,k+1]`. The subdiagonal of unit-diagonal `L` is in `l`; `u` is unchanged.
Batched arrays follow the same convention with a leading system index.

After factorisation, indexing and `Matrix(Q)` recover the actual compact UL
entries by inverting the stored reciprocal pivots. Before factorisation,
access the assembly buffers directly instead. Constructors wrap storage as
supplied; copying a factorised matrix or adapting it to a GPU preserves its
representation. Reassemble all original entries before calling `ul!` again;
Helmholtz `update!` does this in place.

Storage, factorisation and scalar solves scale as **O(P)**. Substitutions
multiply by the stored reciprocals instead of dividing, without allocating
additional reciprocal arrays. The coupled solver also scales as O(P),
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
The logo adapts the illustration and visual identity of FDHelmoltzSolver.jl.
