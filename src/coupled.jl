using StaticArrays

export CoupledHelmoltzSolver

#//////////////////////////////////////////////////////////////////////////////#
#///                     COUPLED SOLVER AND CONSTRUCTOR                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    CoupledHelmoltzSolver(P, T=Float64)

Cache the Chebyshev tau solve of the factored fourth-order problem
```text
θ₀*u'' - θ₁*u = r,
θ₂*v'' - θ₃*v = u,
v(-1) = v(1) = v'(-1) = v'(1) = 0,
```
on `[-1, 1]`, using expansions of degree `P ≥ 3` and coefficient type `T`.

Store two scalar Helmholtz solvers, a particular-solution workspace, two
homogeneous influence responses and their 2×2 influence matrix. Call
[`update!`](@ref) with the four operator coefficients before [`solve!`](@ref).
Use a complex `T` to allocate complex workspaces for Fourier coefficients;
the scalar factors always use the corresponding real precision. This solver
returns the fourth-order solution; the intermediate field is not retained.
"""
struct CoupledHelmoltzSolver{T, P, H<:HelmoltzSolver, C<:AbstractVector{T}}
       hu::H                    # factors for θ₀*D² - θ₁
       hv::H                    # factors for θ₂*D² - θ₃
       vₛ::NTuple{3, C}         # particular workspace and two cached responses
        A::MMatrix{2, 2, T, 4}  # influence matrix, filled by update!

    function CoupledHelmoltzSolver(P::Int, ::Type{T}=Float64) where {T}
        R = typeof(real(zero(T)))
        hu = HelmoltzSolver(P, R)
        hv = HelmoltzSolver(P, R)
        vₛ = ntuple(_ -> zeros(T, P+1), 3)
        A = MMatrix{2, 2, T}(undef)
        return new{T, P, typeof(hu), typeof(vₛ[1])}(hu, hv, vₛ, A)
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///               OPERATOR UPDATES AND HOMOGENEOUS RESPONSES               ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    update!(solver::CoupledHelmoltzSolver, θs)

Assemble and factorise the two scalar operators for
`θs = (θ₀, θ₁, θ₂, θ₃)`. Call again whenever an operator coefficient changes.

Compute the two homogeneous responses and their 2×2 influence matrix once
for these coefficients. They depend only on the operators and are preserved
by subsequent `solve!` calls. Return `solver`.
"""
function update!(solver::CoupledHelmoltzSolver,
                     θs::NTuple{4, Real})
    θ₀, θ₁, θ₂, θ₃ = θs
    update!(solver.hu, θ₀, θ₁)
    update!(solver.hv, θ₂, θ₃)

    # Reset the forcing when rebuilding the cached homogeneous responses.
    work, v₊, v₋ = solver.vₛ
    fill!(v₊, 0)
    fill!(v₋, 0)

    # Unit u at the upper/lower wall, respectively, and zero v at both walls.
    # The scalar backend takes upper then lower boundary values.
    solve!(solver.hu, work, v₊, 1, 0)
    solve!(solver.hv, v₊, work, 0, 0)
    solve!(solver.hu, work, v₋, 0, 1)
    solve!(solver.hv, v₋, work, 0, 0)

    # Rows select the upper/lower wall; columns select the two responses.
    solver.A[1, 1] = diff(v₊, :right)
    solver.A[2, 1] = diff(v₊, :left)
    solver.A[1, 2] = diff(v₋, :right)
    solver.A[2, 2] = diff(v₋, :left)
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Reject singular or nonfinite influence corrections during setup.
    all(isfinite, solver.A) && !iszero(det(solver.A)) ||
        throw(ArgumentError("the coupled influence matrix is singular or nonfinite"))
    #//////////////////////////////////////////////////////////////////////////#

    return solver
end

#//////////////////////////////////////////////////////////////////////////////#
#///                 COUPLED SOLVE AND INFLUENCE CORRECTION                 ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    solve!(solver::CoupledHelmoltzSolver, u, f)

Compute the Chebyshev coefficients of the clamped solution
`(θ₀ D² - θ₁)(θ₂ D² - θ₃)u = f`, with `u(±1) = u′(±1) = 0`.
Write the solution into `u`, preserve `f`, and return `u`.

Both one-based vectors have length `P+1` and the workspace element type
selected at construction (`Float32`, `Float64`, or their complex counterparts).
Their storage must be disjoint and must not alias solver workspaces or factors.
Call `update!` before the first solve and when operator coefficients change.
The intermediate second-order solution is not returned.

Cached homogeneous responses and the influence matrix are preserved. The
particular workspace is reused: do not share one solver between concurrent solves.
"""
function solve!(solver::CoupledHelmoltzSolver{T, P},
                     u::AbstractVector{T},
                     f::AbstractVector{T}) where {T, P}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Preserve the forcing and cached influence responses.
    Base.require_one_based_indexing(u, f)
    length(u) == length(f) == P+1 || throw(DimensionMismatch("u and f must have P+1 coefficients"))
    Base.mightalias(u, f) && throw(ArgumentError("u and f must not alias"))
    any(a -> Base.mightalias(u, a) || Base.mightalias(f, a), solver.vₛ) &&
        throw(ArgumentError("fields must not alias solver workspaces"))
    #//////////////////////////////////////////////////////////////////////////#

    vₚ, v₊, v₋ = solver.vₛ

    # Particular solution: choose zero wall values for the intermediate field.
    solve!(solver.hu, vₚ, f, 0, 0)
    solve!(solver.hv, u, vₚ, 0, 0)

    # Cancel its wall derivatives with the cached homogeneous responses.
    b = SVector{2}(-diff(u, :right), -diff(u, :left))
    δ₊, δ₋ = SMatrix(solver.A)\b
    u .+= δ₊ .* v₊ .+ δ₋ .* v₋
    return u
end
