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
v(±1) = v'(±1) = 0,
```
on `[-1, 1]`, using expansions of degree `P ≥ 3` and coefficient type `T`.

Store two scalar Helmholtz solvers, a particular-solution workspace, two
homogeneous influence responses and their 2×2 influence matrix. Call
[`update!`](@ref) with the four operator coefficients before [`solve!`](@ref).
This solver returns `v`; the intermediate field `u` is not retained.
"""
struct CoupledHelmoltzSolver{T, P, H<:HelmoltzSolver{T, P}, C<:AbstractVector{T}}
       hu::H                    # factors for θ₀*D² - θ₁
       hv::H                    # factors for θ₂*D² - θ₃
       vₛ::NTuple{3, C}         # particular workspace and two cached responses
    A_inf::MMatrix{2, 2, T, 4}  # influence matrix, filled by update!

    function CoupledHelmoltzSolver(P::Int, ::Type{T}=Float64) where {T}
        hu = HelmoltzSolver(P, T)
        hv = HelmoltzSolver(P, T)
        vₛ = ntuple(_ -> zeros(T, P+1), 3)
        A_inf = MMatrix{2, 2, T}(undef)
        return new{T, P, typeof(hu), typeof(vₛ[1])}(hu, hv, vₛ, A_inf)
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
by subsequent `solve!` calls. Return `nothing`.
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
    solver.A_inf[1, 1] = diff(v₊, :right)
    solver.A_inf[2, 1] = diff(v₊, :left)
    solver.A_inf[1, 2] = diff(v₋, :right)
    solver.A_inf[2, 2] = diff(v₋, :left)
    return nothing
end

#//////////////////////////////////////////////////////////////////////////////#
#///                 COUPLED SOLVE AND INFLUENCE CORRECTION                 ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    solve!(solver::CoupledHelmoltzSolver, r::AbstractVector)

Overwrite the source coefficients `r` with the solution `v` and return `r`.
The source must have the solver's degree and element type, with storage
distinct from the solver's workspaces.

Solve the two scalar tau equations successively for a particular solution.
Then use the homogeneous responses and influence matrix cached by `update!`
to choose their amplitudes and impose `v'(±1) = 0`. All three velocity
responses already satisfy `v(±1) = 0`. Each call requires only two scalar
Helmholtz solves and one 2×2 solve.

Call `update!` before the first solve and whenever the operator coefficients
change. The influence matrix must be nonsingular. Only the particular solution
workspace and `r` are overwritten; the homogeneous responses and
influence matrix are preserved. One solver instance must not be used concurrently.
"""
function solve!(solver::CoupledHelmoltzSolver{T, P},
                     r::AbstractVector{T}) where {T, P}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Preserve all reusable responses while overwriting the input coefficients.
    Base.require_one_based_indexing(r)
    length(r) == P+1 || throw(DimensionMismatch("source must have P+1 coefficients"))
    any(a -> Base.mightalias(r, a), solver.vₛ) &&
        throw(ArgumentError("source must not alias solver workspaces"))
    #//////////////////////////////////////////////////////////////////////////#

    vₚ, v₊, v₋ = solver.vₛ

    # Particular solution: choose zero wall values for the intermediate u.
    solve!(solver.hu, vₚ, r, 0, 0)
    solve!(solver.hv, r, vₚ, 0, 0)

    # Cancel the particular solution's wall derivatives using the cached matrix.
    b = SVector{2}(-diff(r, :right),
                   -diff(r, :left))
    δ₊, δ₋ = SMatrix(solver.A_inf)\b

    r .+= δ₊ .* v₊ .+ δ₋ .* v₋
    return r
end
