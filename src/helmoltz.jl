export HelmoltzSolver, solve!, update!

"""
    HelmoltzSolver(P, T=Float64; neumann=false)

Cache the Chebyshev tau solve of `θ₀*u'' - θ₁*u = f` on `[-1, 1]`, with
Dirichlet wall values, or wall-normal derivatives when `neumann=true`.
Neumann data are derivatives in the positive y direction at both walls,
not outward-normal derivatives. The pure Neumann Poisson operator is singular
and must be treated separately. `P ≥ 2` is the polynomial degree, so each expansion
contains `P + 1` ordinary coefficients in `u(y) = sum(u[p]*T_p(y), p=0:P)`.
Call `update!` before solving, and again whenever `θ₀` or `θ₁` changes.

The even/odd systems correspond to `Ae_`/`Ao_` in Channelflow's
`channelflow/helmholtz.cpp`. Channelflow uses even `P` (an odd number of
coefficients); this implementation also supports odd `P`. Factors and
right-hand-side workspaces are reused, so one solver must not be used
concurrently. The scalar factorisation uses UL elimination without pivoting;
the supplied operator must have nonzero pivots.
"""
mutable struct HelmoltzSolver{T, P, QE<:QuasiTridiagonal, QO<:QuasiTridiagonal, V<:Vector{T}}
    neumann::Bool # prescribe derivatives instead of values at both walls
    Be::QE  # factorisation for even Chebyshev coefficients
    Bo::QO  # factorisation for odd Chebyshev coefficients
    ge::V   # even right-hand side and solution
    go::V   # odd right-hand side and solution
     l::V   # lower coefficient of the integrated tau equations
     d::V   # diagonal coefficient, stored with the opposite RHS sign
     u::V   # upper coefficient of the integrated tau equations

    function HelmoltzSolver(P::Int, ::Type{T}=Float64; neumann::Bool=false) where {T}
        P ≥ 2 || throw(ArgumentError("P must be at least 2: got $P"))

        # Include coefficient zero in the even block. With even P this
        # block has one more entry, as in Gibson's HelmholtzSolver.
        Me, Mo = div(P, 2) + 1, div(P+1, 2)

        Be = QuasiTridiagonal(Me, T)
        Bo = QuasiTridiagonal(Mo, T)

        ge = zeros(T, Me)
        go = zeros(T, Mo)

        _c(p)    = p == 0  ? 2 : 1
        _β(p, P) = p > P-2 ? 0 : 1

        # Cache the integration coefficients by polynomial degree p.
        # Entry 1 is unused; each row couples degrees p-2, p and p+2.
        l, d, u = zeros(T, P), zeros(T, P), zeros(T, P)
        for p ∈ 2:P
            l[p] = _c(p-2)/(4p*(p-1))
            d[p] = _β(p, P)/2/(p^2 - 1)
            u[p] = _β(p+2, P)/(4*p*(p+1))
        end

        return new{T, P, typeof(Be), typeof(Bo), Vector{T}}(neumann, Be, Bo, ge, go, l, d, u)
    end
end

"""
    update!(h::HelmoltzSolver, θ₀, θ₁)

Assemble and UL-factorise the even and odd systems for `θ₀*u'' - θ₁*u = f`.
For a physical interval `[a, b]`, use `θ₀ = ν*(2/(b-a))^2` and `θ₁ = λ`
to represent Gibson's operator `ν*d²/dy² - λ`. The right-hand side and wall
values or derivatives are not rescaled.

Reassemble all matrix entries before factorisation, replacing any previous
factors. Subsequent solves reuse these factors until the next update.
Return `nothing`.
"""
function update!( h::HelmoltzSolver{T, P},
                 θ₀::Real,
                 θ₁::Real) where {T, P}
    h.neumann && iszero(θ₁) &&
        throw(ArgumentError("the pure Neumann Poisson operator requires a separate mean-mode solve"))
    # Row one imposes the wall value or derivative for each parity. The
    # remaining rows contain the integrated equations for p = 2, 4, ...
    # or p = 3, 5, ..., respectively.
    for (B, p₀) in ((h.Be, 2), (h.Bo, 3))
        M = size(B, 1)
        for i ∈ 1:M
            n = p₀ - 2 + 2*(i-1)
            B.b[i] = h.neumann ? n^2 : 1
        end
        @simd for i ∈ 1:M-1
            p = p₀ + 2*(i-1)
            B.l[i] =     -θ₁*h.l[p]
            B.d[i] = θ₀ + θ₁*h.d[p]
            i < M-1 && (B.u[i] = -θ₁*h.u[p])
        end
        ul!(B)
    end

    return nothing
end

"""
    solve!(h::HelmoltzSolver, f::ChebCoeffs, u₊, u₋)

Overwrite the Chebyshev coefficients `f` with the solution, using the factors
from `update!`. Boundary arguments are `u(+1)` then `u(-1)` (or `u′(+1)` then
`u′(-1)` with `neumann=true`); Channelflow's
`solve(u, f, ua, ub)` uses the reverse order. The two highest residual
coefficients are tau terms, so the equation is imposed only through degree
`P - 2`.

`f` must have the solver's degree and element type, with storage distinct
from its internal workspaces. Real boundary amplitudes `u₊` and `u₋` may
change between solves without rebuilding the factors. Return the overwritten
`f`; the factors are preserved and the right-hand-side workspaces are reused.
"""
function solve!( h::HelmoltzSolver{T, P},
                 f::ChebCoeffs{T, P},
                u₊::Real,
                u₋::Real) where {T, P}
    # Differentiation swaps parity: even polynomials have odd derivatives.
    h.ge[1] = (h.neumann ? u₊-u₋ : u₊+u₋)*0.5
    h.go[1] = (h.neumann ? u₊+u₋ : u₊-u₋)*0.5

    # Each parity depends only on its own RHS coefficients, so its solution
    # can overwrite f before processing the other parity.
    _solve_parity!(h, f, h.Be, h.ge, 0)
    _solve_parity!(h, f, h.Bo, h.go, 1)
    return f
end

# Separate calls specialize on each parity's matrix size. Iterating over a
# heterogeneous tuple of even/odd factors otherwise boxes these hot solves.
function _solve_parity!(h::HelmoltzSolver{T, P}, f::ChebCoeffs{T, P},
                        B::QuasiTridiagonal, g::Vector{T}, p₀::Int) where {T, P}
    M = length(g)
    @inbounds @simd for i ∈ 2:M
        p = p₀ + 2*(i-1)
        fₚ₊₂ = p+2 ≤ P-2 ? f[p+2] : zero(T)
        g[i] = h.l[p] * f[p-2] - h.d[p] * f[p] + h.u[p] * fₚ₊₂
    end

    ldiv!(B, g)

    @inbounds @simd for i ∈ 1:M
        f[p₀ + 2*(i-1)] = g[i]
    end
    return f
end
