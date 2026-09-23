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
    HelmoltzSolver(P, T=Float64; neum=false)

Construct a degree-`P` Chebyshev tau solver for

```math
θ₀ u''(y) - θ₁ u(y) = f(y), \\qquad -1 ≤ y ≤ 1.
```

Use Dirichlet wall values by default, or prescribe `u′` at both walls with
`neum=true`. Neumann data are derivatives in the positive y direction, not
outward-normal derivatives. `P ≥ 3`; odd and even degrees are supported.
Call [`update!`](@ref) to specify the operator before calling [`solve!`](@ref),
and again whenever `θ₀` or `θ₁` changes.

# Numerical method

Represent the solution and forcing by ordinary Chebyshev expansions,
`u(y) = sum(u[p+1]*T_p(y), p=0:P)`, with no half-weight on the zeroth coefficient.
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
(`neum=true`, `θ₁=0`) is supported for compatible forcing and wall data;
its solution has zero integral over `[-1, 1]`.

Concurrent solves may share the factors with disjoint destination storage,
provided no concurrent `update!` modifies the solver.
"""
struct HelmoltzSolver{T, P, QE<:QuasiTridiagonal, QO<:QuasiTridiagonal, V<:Vector{T}}
       neum::Bool # prescribe derivatives instead of values at both walls
         Be::QE   # factorisation for even Chebyshev coefficients
         Bo::QO   # factorisation for odd Chebyshev coefficients
      cache::NTuple{3, V} # integration weights (l, d, u)
    poisson::Base.RefValue{T} # θ₀ for singular Neumann Poisson; zero otherwise

    function HelmoltzSolver(   P::Int,
                                ::Type{T}=Float64;
                              neum::Bool=false) where {T}
        #/////////////////////////////// CHECKS ///////////////////////////////#
        # Both parity blocks must contain at least two coefficients.
        P ≥ 3 || throw(ArgumentError("P must be at least 3: got $P"))
        #//////////////////////////////////////////////////////////////////////#

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

        return new{T, P, typeof(Be), typeof(Bo), Vector{T}}(neum, Be, Bo, cache, Ref(zero(T)))
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                  OPERATOR ASSEMBLY AND FACTORISATION                   ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    update!(h::HelmoltzSolver, θ₀, θ₁)

Assemble and UL-factorise the even and odd systems for `θ₀*u'' - θ₁*u = f`.
The interval is `[-1, 1]`. Coefficients must be finite and `θ₀` nonzero.
Singular Neumann Poisson operators use a zero-mean gauge at solve time.

Reassemble all matrix entries before factorisation, replacing any previous
factors and storing their reciprocal pivots. Subsequent solves reuse
these factors until the next update.
Return `h`.
"""
function update!( h::HelmoltzSolver{T, P},
                 θ₀::Real,
                 θ₁::Real) where {T, P}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Reject invalid coefficients before changing reusable factors.
    isfinite(θ₀) && isfinite(θ₁) && !iszero(θ₀) ||
        throw(ArgumentError("operator coefficients must be finite and θ₀ must be nonzero"))
    #//////////////////////////////////////////////////////////////////////////#

    _assemble_helmoltz!(h.Be, h.cache, θ₀, θ₁, 2, h.neum); ul!(h.Be)
    _assemble_helmoltz!(h.Bo, h.cache, θ₀, θ₁, 3, h.neum); ul!(h.Bo)

    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Unpivoted UL may break down even for a nonsingular negative-shift matrix.
    # A failed update must not be followed by a solve until valid factors exist.
    _check_factors(h.Be)
    _check_factors(h.Bo)
    #//////////////////////////////////////////////////////////////////////////#

    h.poisson[] = h.neum && iszero(θ₁) ? θ₀ : zero(T)
    return h
end

# Assemble the scalar parity block in its existing storage.
# It replaces every matrix entry before UL modifies the stored diagonals.
function _assemble_helmoltz!(B::QuasiTridiagonal{T, M}, cache, θ₀, θ₁, p₀, neum) where {T, M}
    l, d, u = cache

    # The dense first row imposes the wall value or positive-y derivative.
    # For singular Poisson, the even derivative condition follows from
    # compatibility; replace it by u₀=0 and select the mean after the solve.
    for i in 1:M
        n = p₀ - 2 + 2*(i-1)
        B.b[i] = neum && iszero(θ₁) && p₀ == 2 ? (i == 1) : (neum ? n^2 : 1)
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
\theta_0\,u''(y) - \theta_1\,u(y) = f(y), \qquad -1 \le y \le 1,
```
using the operator coefficients supplied to the most recent
`update!(h, θ₀, θ₁)`. Write the result into `u` and return that vector;
`f` and the solver are preserved. The initial contents of `u`
are ignored.

Both arguments are one-based coefficient vectors of length `P+1`, representing
ordinary Chebyshev expansions

```math
u_P(y) = \sum_{n=0}^{P} \widehat{u}_n T_n(y), \qquad
f_P(y) = \sum_{n=0}^{P} \widehat{f}_n T_n(y).
```

On entry, `f[n+1]` contains ``\widehat{f}_n``; on return,
`u[n+1]` contains ``\widehat{u}_n``. These are coefficients, not values
at collocation points. Both vectors must have the same element type, either
`T` or `Complex{T}` for factors of type `T`, and must not alias. Solution storage must also be distinct from the solver's
factors and cached integration weights.

For a Dirichlet solver (`neum=false`), the boundary arguments prescribe

```math
u_P(1) = u_+, \qquad
u_P(-1) = u_-.
```

For a Neumann solver (`neum=true`), they instead prescribe

```math
u_P'(1) = u_+, \qquad
u_P'(-1) = u_-.
```

Both derivatives are taken in the positive `y` direction, not along the
outward normal. Boundary data default to zero and may change between solves
without another `update!`. For pure Neumann Poisson (`θ₁=0`), compatibility
requires `integral(f, -1, 1) = θ₀*(u₊-u₋)` and the returned solution has zero
integral. Compatibility uses forcing degrees `0:P-2`, consistently with the
tau equations; incompatible data raise `ArgumentError` before changing `u`.
Uninitialised or invalid factors also raise `ArgumentError`.

The tau solution satisfies the two boundary conditions and sets the Chebyshev
coefficients of ``\theta_0 u_P'' - \theta_1 u_P - f_P`` to zero for degrees
`0:P-2`. Residual coefficients of degrees `P-1` and `P` are unconstrained;
consequently, `f[P]` and `f[P+1]` do not affect the solution.
"""
function solve!( h::HelmoltzSolver{T, P},
                 u::AbstractVector,
                 f::AbstractVector,
                u₊=0,
                u₋=0) where {T, P}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Real factors support real or complex fields at the same precision.
    _check_precision(T, u, f)
    # Detect uninitialised, failed or externally damaged factors before writes.
    _check_factors(h.Be)
    _check_factors(h.Bo)
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
    # Compatibility uses the retained tau forcing, not its last two entries.
    iszero(h.poisson[]) || _check_neumann(f, h.poisson[], u₊, u₋)
    #//////////////////////////////////////////////////////////////////////////#

    # Assemble into the output: neighbouring source coefficients remain intact.
    # Differentiation exchanges parity, swapping the Neumann wall combinations.
    u[1] = iszero(h.poisson[]) ? (h.neum ? u₊ - u₋ : u₊ + u₋)*0.5 : zero(eltype(u))
    u[2] = (h.neum ? u₊ + u₋ : u₊ - u₋)*0.5
    l, d, upper_weight = h.cache
    @inbounds @simd for p in 2:P
        high = p+2 ≤ P-2 ? f[p+3] : zero(T)
        u[p+1] = l[p]*f[p-1] - d[p]*f[p+1] + upper_weight[p]*high
    end

    # Solve the two parities directly in output storage, without packing.
    ldiv!(h.Be, view(u, 1:2:P+1))
    ldiv!(h.Bo, view(u, 2:2:P+1))
    # The temporary u₀=0 gauge is shifted to zero interval mean.
    iszero(h.poisson[]) || (u[1] = -sum(u[n+1]/(1-n^2) for n in 2:2:P))
    return u
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       SHARED SOLVER VALIDATION                         ///#
#//////////////////////////////////////////////////////////////////////////////#

# Public solves share one precision contract, checked before mutating output.
function _check_precision(::Type{T}, u, f) where {T}
    eltype(u) == eltype(f) && eltype(u) <: Union{T, Complex{T}} ||
        throw(ArgumentError("u and f must share element type $T or $(Complex{T})"))
    return nothing
end

# Called at update and scalar solve boundaries, never inside substitutions.
function _check_factors(Q::QuasiTridiagonal)
    all(a -> all(isfinite, a), (Q.b, Q.l, Q.dᵢ, Q.u)) &&
        !iszero(Q.b[1]) && all(!iszero, Q.dᵢ) ||
        throw(ArgumentError("uninitialised or invalid UL factors; call update! with a valid operator"))
    return nothing
end

# ∫f dy = θ₀(u′(1)-u′(-1)). For the tau problem f is truncated at P-2.
# The tolerance follows the sum's scale, allowing floating-point cancellation
# without silently projecting incompatible forcing onto a different problem.
function _check_neumann(f, θ₀, u₊, u₋)
    T = typeof(θ₀)
    integral = zero(eltype(f))
    magnitude = zero(T)
    for n in 0:2:length(f)-3
        term = f[n+1]/(1-n^2)
        integral += term
        magnitude += abs(term)
    end
    boundary = θ₀*(u₊-u₋)/2
    magnitude += abs(θ₀*u₊/2) + abs(θ₀*u₋/2)
    abs(integral-boundary) <= 64eps(T)*max(magnitude, floatmin(T)) ||
        throw(ArgumentError("incompatible Neumann Poisson data: integral(f) must equal θ₀*(u₊-u₋)"))
    return nothing
end
