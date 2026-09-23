export HelmoltzSolver, solve!, update!

#//////////////////////////////////////////////////////////////////////////////#
#///                          INTEGRATION WEIGHTS                           ///#
#//////////////////////////////////////////////////////////////////////////////#

# Shared degree-zero correction and tau cutoff for integration weights.
_c(p) = p == 0 ? 2 : 1
_β(p, P) = p > P-2 ? 0 : 1

#//////////////////////////////////////////////////////////////////////////////#
#///                    HELMHOLTZ SOLVER AND CONSTRUCTOR                    ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    HelmoltzSolver(P, T=Float64; neum=false, a=-1, b=1)

Construct a degree-`P` Chebyshev tau solver for

```math
θ₀ u''(y) - θ₁ u(y) = f(y), \\qquad a ≤ y ≤ b.
```

Use Dirichlet wall values by default, or prescribe `u′` at both walls with
`neum=true`. Neumann data are derivatives in the positive y direction, not
outward-normal derivatives. `P ≥ 3`; odd and even degrees are supported.
Call [`update!`](@ref) to specify the operator before calling [`solve!`](@ref),
and again whenever `θ₀` or `θ₁` changes.

# Numerical method

Set `ξ = (2y-a-b)/(b-a)` and `scale = 2/(b-a)`. Operator coefficients
and Neumann data use physical derivatives. The reference-coordinate
formulas below use `θ₀*scale²` in place of `θ₀` and multiply Neumann
boundary rows by `scale`.

Represent the solution and forcing by ordinary Chebyshev expansions,
`u(y) = sum(u[p+1]*T_p(ξ), p=0:P)`, with no half-weight on the zeroth coefficient.
The tau formulation sets the residual coefficients of degrees `0:P-2` to
zero. The last two differential-equation conditions are replaced by the two
wall conditions; residuals in degrees `P-1` and `P` are not constrained.
Consequently, forcing coefficients at these last two degrees do not enter
the solve.

Instead of forming a dense second-derivative matrix, integrate the retained
coefficient equations twice. For degrees `p = 2:P`, the resulting equations
couple only `u[p-1]`, `u[p+1]`, and `u[p+3]`:

```text
-θ₁ Lₚ u[p-1] + (θ₀ + θ₁ Dₚ) u[p+1] - θ₁ Hₚ u[p+3]
    = Lₚ f[p-1] - Dₚ f[p+1] + Hₚ f[p+3].
```

Here `Lₚ = cₚ₋₂/(4p(p-1))`, `Dₚ = βₚ/(2(p²-1))`, and
`Hₚ = βₚ₊₂/(4p(p+1))`, where `c₀ = 2`, `cₙ = 1` for `n > 0`,
and `βₙ = 1` for `n ≤ P-2`, otherwise zero. Terms beyond degree `P`
are absent. The β factors retain the tau truncation in the terminal rows;
they are essential to equivalence with the residual equations above.
The two integration constants affect only degrees zero and one, whose
conditions are supplied by the walls.

Because these equations couple degrees differing by two, the even and odd
coefficients form independent systems. Each has a tridiagonal interior and
one dense boundary row. For wall data `g₊` at `y=+1` and `g₋` at `y=-1`,
Dirichlet conditions give

```text
sum(u[p+1], p even) = (g₊ + g₋)/2,
sum(u[p+1], p odd)  = (g₊ - g₋)/2.
```

For Neumann conditions, `Tₚ′(+1) = p²` and
`Tₚ′(-1) = (-1)^(p+1) p²`, so the boundary rows become

```text
sum(p²*u[p+1], p even) = (g₊ - g₋)/2,
sum(p²*u[p+1], p odd)  = (g₊ + g₋)/2.
```

The two quasi-tridiagonal systems are solved by UL factorisation without
pivoting. Factorisation and each subsequent solve require `O(P)` work;
changing the forcing or wall data does not require new factors. The operator
must have nonzero pivots with finite reciprocals. Pure Neumann Poisson
(`neum=true`, `θ₁=0`) requires a compatibility condition and a pressure/solution
gauge and is not handled by this solver.

Concurrent solves may share the factors with disjoint destination storage,
provided no concurrent `update!` modifies the solver.
"""
struct HelmoltzSolver{T, P, QE<:QuasiTridiagonal, QO<:QuasiTridiagonal, V<:Vector{T}}
    scale::T    # d/dy = scale*d/dξ for the affine reference coordinate
     neum::Bool # prescribe derivatives instead of values at both walls
       Be::QE   # factorisation for even Chebyshev coefficients
       Bo::QO   # factorisation for odd Chebyshev coefficients
    cache::NTuple{3, V} # integration weights (l, d, u)

    function HelmoltzSolver(   P::Int,
                                ::Type{T}=Float64;
                            neum::Bool=false, a=-1, b=1) where {T}
        #/////////////////////////////// CHECKS ///////////////////////////////#
        # Both parity blocks must contain at least two coefficients.
        P ≥ 3 || throw(ArgumentError("P must be at least 3: got $P"))
        #//////////////////////////////////////////////////////////////////////#

        scale = _intervalscale(a, b, T)

        # Include coefficient zero in the even block. With even P this
        # block has one more entry, as in Gibson's HelmholtzSolver.
        Me, Mo = div(P, 2) + 1, div(P+1, 2)

        # Allocate the two factor banks. update! will assemble the operator
        # for the chosen θ₀ and θ₁ and factorise these arrays in place.
        Be = QuasiTridiagonal(Me, T)
        Bo = QuasiTridiagonal(Mo, T)

        # Cache integration weights once; index 1 is unused. The helpers
        # include the degree-zero correction and the terminal tau cutoff.
        l = T[p == 1 ? 0 : _c(p-2)/(4p*(p-1))    for p in 1:P]
        d = T[p == 1 ? 0 : _β(p, P)/(2*(p^2-1))  for p in 1:P]
        u = T[p == 1 ? 0 : _β(p+2, P)/(4p*(p+1)) for p in 1:P]
        cache = (l, d, u)

        return new{T, P, typeof(Be), typeof(Bo), Vector{T}}(scale, neum, Be, Bo, cache)
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                  OPERATOR ASSEMBLY AND FACTORISATION                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    update!(h::HelmoltzSolver, θ₀, θ₁)

Assemble and UL-factorise the even and odd systems for `θ₀*u'' - θ₁*u = f`.
Coefficients and boundary data refer to the physical interval selected at
construction. The solver applies the affine derivative scaling internally;
do not rescale `θ₀` or Neumann data before passing them.

Reassemble all matrix entries before factorisation, replacing any previous
factors and storing their reciprocal pivots. Subsequent solves reuse
these factors until the next update.
Return `nothing`.
"""
function update!( h::HelmoltzSolver{T, P},
                 θ₀::Real,
                 θ₁::Real) where {T, P}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Pure Neumann Poisson needs a compatibility condition and a pressure gauge.
    h.neum && iszero(θ₁) &&
        throw(ArgumentError("the pure Neumann Poisson operator requires a separate mean-mode solve"))
    #//////////////////////////////////////////////////////////////////////////#

    _assemble_helmoltz!(h.Be, h.cache, θ₀*h.scale^2, θ₁, 2, h.neum, h.scale); ul!(h.Be)
    _assemble_helmoltz!(h.Bo, h.cache, θ₀*h.scale^2, θ₁, 3, h.neum, h.scale); ul!(h.Bo)

    return nothing
end

# Assemble the scalar parity block in its existing storage.
# It replaces every matrix entry before UL modifies the stored diagonals.
function _assemble_helmoltz!(B::QuasiTridiagonal{T, M}, cache, θ₀, θ₁, p₀, neum, scale) where {T, M}
    l, d, u = cache

    # The dense first row imposes the wall value or positive-y derivative.
    for i in 1:M
        n = p₀ - 2 + 2*(i-1)
        B.b[i] = neum ? scale*n^2 : 1
    end

    # Interior rows contain integrated equations for even or odd degrees.
    @simd for i in 1:M-1
        p = p₀ + 2*(i-1)
        B.l[i] = -θ₁*l[p]
        B.dᵢ[i] = θ₀ + θ₁*d[p]
        i < M-1 && (B.u[i] = -θ₁*u[p])
    end
    return B
end

#//////////////////////////////////////////////////////////////////////////////#
#///                          DIRECT FIELD SOLVES                           ///#
#//////////////////////////////////////////////////////////////////////////////#

raw"""
    solve!(h::HelmoltzSolver, u, f, u₊=0, u₋=0)

Compute the Chebyshev coefficients of the solution to

```math
\theta_0\,u''(y) - \theta_1\,u(y) = f(y), \qquad a \le y \le b,
```
using the operator coefficients supplied to the most recent
`update!(h, θ₀, θ₁)`. Write the result into `u` and return that vector;
`f` and the solver are preserved. The initial contents of `u`
are ignored.

Both arguments are one-based coefficient vectors of length `P+1`, representing
ordinary Chebyshev expansions

```math
u_P(y) = \sum_{n=0}^{P} \widehat{u}_n T_n(ξ), \qquad
f_P(y) = \sum_{n=0}^{P} \widehat{f}_n T_n(ξ).
```

On entry, `f[n+1]` contains ``\widehat{f}_n``; on return,
`u[n+1]` contains ``\widehat{u}_n``. These are coefficients, not values
at collocation points. Both vectors must have the solver's element type and
must not alias. Solution storage must also be distinct from the solver's
factors and cached integration weights.

For a Dirichlet solver (`neum=false`), the boundary arguments prescribe

```math
u_P(b) = u_+, \qquad
u_P(a) = u_-.
```

For a Neumann solver (`neum=true`), they instead prescribe

```math
u_P'(b) = u_+, \qquad
u_P'(a) = u_-.
```

Both derivatives are taken in the positive `y` direction, not along the
outward normal. Boundary data default to zero and may change between solves
without another `update!`. The assembled boundary-value problem must be
nonsingular; pure Neumann Poisson problems require a separate pressure gauge.

The tau solution satisfies the two boundary conditions and sets the Chebyshev
coefficients of ``\theta_0 u_P'' - \theta_1 u_P - f_P`` to zero for degrees
`0:P-2`. Residual coefficients of degrees `P-1` and `P` are unconstrained;
consequently, `f[P]` and `f[P+1]` do not affect the solution.
"""
function solve!( h::HelmoltzSolver{T, P},
                 u::AbstractVector{T},
                 f::AbstractVector{T},
                u₊=0,
                u₋=0) where {T, P}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Degree fixes the number of coefficients, regardless of vector storage.
    Base.require_one_based_indexing(u, f)
    length(u) == length(f) == P+1 ||
        throw(DimensionMismatch("f and u must have P+1 coefficients"))
    # Assembly must preserve the source and reusable operator data.
    Base.mightalias(u, f) &&
        throw(ArgumentError("f and u must not alias"))
    for Q in (h.Be, h.Bo), a in (Q.b, Q.l, Q.dᵢ, Q.u)
        Base.mightalias(u, a) &&
            throw(ArgumentError("u must not alias the factors"))
    end
    any(a -> Base.mightalias(u, a), h.cache) &&
        throw(ArgumentError("u must not alias the integration weights"))
    #//////////////////////////////////////////////////////////////////////////#

    # Assemble into the output: neighbouring source coefficients remain intact.
    # Differentiation exchanges parity, swapping the Neumann wall combinations.
    u[1] = (h.neum ? u₊ - u₋ : u₊ + u₋)*0.5
    u[2] = (h.neum ? u₊ + u₋ : u₊ - u₋)*0.5
    l, d, upper_weight = h.cache
    @inbounds @simd for p in 2:P
        high = p+2 ≤ P-2 ? f[p+3] : zero(T)
        u[p+1] = l[p]*f[p-1] - d[p]*f[p+1] + upper_weight[p]*high
    end

    # Solve the two parities directly in output storage, without packing.
    ldiv!(h.Be, view(u, 1:2:P+1))
    ldiv!(h.Bo, view(u, 2:2:P+1))
    return u
end
