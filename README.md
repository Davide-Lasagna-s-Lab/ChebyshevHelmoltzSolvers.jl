<p align="center">
  <img src="assets/logo.svg" alt="ChebyshevHelmoltzSolvers.jl logo" width="900">
</p>

# ChebyshevHelmoltzSolvers.jl

[![CI](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/actions/workflows/CI.yml)

In-place Chebyshev tau solvers for one-dimensional Helmholtz boundary-value
problems on **[-1, 1]**. Scalar operators use even/odd quasi-tridiagonal UL
factorisations. The coupled solver caches its homogeneous responses and
influence matrix, leaving only two scalar solves per right-hand side.

The scalar discretisation follows the even/odd construction in
[Channelflow's Helmholtz solver](https://github.com/epfl-ecps/channelflow/blob/ad37ef3022351d4e4a7a6c274c59a88605ad19e8/channelflow/helmholtz.cpp).
The package is a Chebyshev counterpart to
[FDHelmoltzSolver.jl](https://github.com/Davide-Lasagna-s-Lab/FDHelmoltzSolver.jl).

## Installation

Requires Julia 1.10 or later. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl")
```

## Scalar Helmholtz solver

```text
θ₀ u''(y) - θ₁ u(y) = r(y)
u(+1) = u₊,  u(-1) = u₋
```

`HelmoltzSolver(P, T=Float64)` allocates factors and workspaces for degree
`P ≥ 2`, with `P + 1` coefficients. Call `update!` before solving and whenever
the operator coefficients change. Right-hand sides and boundary values can
change without another update.

For example, solve `u'' - 4u = -6 + 4y²`, whose solution is `u = 1 - y²`:

```julia
using ChebyshevHelmoltzSolvers

P = 16
h = HelmoltzSolver(P)
update!(h, 1.0, 4.0)

rhs = ChebCoeffs(P)
rhs[0] = -4.0
rhs[2] = 2.0
solve!(h, rhs, 0.0, 0.0)  # overwrites rhs with the solution coefficients

# u = (T₀ - T₂)/2: rhs[0] ≈ 0.5, rhs[2] ≈ -0.5
```

**Boundary arguments are upper then lower:** `solve!(h, rhs, u₊, u₋)`.
They are real amplitudes, including when the coefficient type is complex.
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

rhs = ChebCoeffs(P)
rhs[0] = 24.0
solve!(solver, rhs)

# v = 3T₀/8 - T₂/2 + T₄/8
# rhs[0] ≈ 0.375, rhs[2] ≈ -0.5, rhs[4] ≈ 0.125
```

The source storage must be distinct from the solver's internal workspaces.
The intermediate `u` is not retained. The influence matrix must be nonsingular;
use degree at least four to represent a nonzero solution with four homogeneous
wall conditions.

## Coefficient convention and utilities

`ChebCoeffs` stores the **ordinary**, unweighted expansion

```text
f(y) = a[0] T₀(y) + a[1] T₁(y) + ... + a[P] Tₚ(y).
```

Indices are polynomial degrees `0:P`; `parent(a)` is the underlying one-based
vector. `ChebCoeffs(P, T)` allocates zeros, whereas `ChebCoeffs(vector)` wraps
existing storage without copying. Both odd and even polynomial degrees are
supported.

When converting values at descending Lobatto points `cospi(j/P)` with an
unnormalised DCT-I, divide the transform by `P` and halve coefficients
`0` and `P`. The package operates on coefficients; it does not depend on FFTW.

| Operation | Purpose |
| --- | --- |
| `diff!(out, a)` | Chebyshev differentiation on [-1, 1]; `out === a` is supported. |
| `endpoint_derivative(a, :left)` | Evaluate the derivative at -1 without a derivative workspace. |
| `endpoint_derivative(a, :right)` | Evaluate the derivative at +1. |
| `QuasiTridiagonal(M, T)` | Allocate a matrix with a dense first row and tridiagonal interior. |
| `ul!(Q)` | Factorise that matrix in place without pivoting. |
| `ldiv!(Q, rhs)` | Solve using existing UL factors; available through `LinearAlgebra`. |

For a mapped interval `[a, b]`, multiply the second-derivative coefficient
by `(2/(b-a))²`. Endpoint derivatives from this package must be multiplied
by `2/(b-a)` to obtain physical derivatives.

## Cost and numerical requirements

Storage, factorisation and scalar solves scale as **O(P)**. The coupled
solver also scales as O(P), with its homogeneous problems paid for at
`update!` time. Solvers reuse mutable workspaces; use a separate solver
instance for each concurrent task.

UL factorisation has no pivoting or singularity check. The chosen scalar
operators must have nonzero pivots, and the coupled influence system must
be invertible. Floating-point conditioning still limits accuracy as the
degree or operator parameters increase.

## Tests

From a checkout:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The deterministic suite checks zero-based indexing and broadcasting, in-place
differentiation, wall derivatives, UL reconstruction and repeated solves,
analytic scalar and coupled solutions, real and complex coefficients,
single and double precision, odd/even degrees, operator updates and
preservation of cached homogeneous responses. Tests need only Julia's
`Test` standard library in addition to the package dependencies.

GitHub Actions runs the suite on Julia 1.10 and the current stable Julia.
The logo adapts the illustration and visual identity of FDHelmoltzSolver.jl.
