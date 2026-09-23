export BatchedCoupledHelmoltzSolver

#//////////////////////////////////////////////////////////////////////////////#
#///                      BATCHED COUPLED SOLVER STORAGE                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    BatchedCoupledHelmoltzSolver(P, B, T=Float64)

Allocate `B` clamped fourth-order tau problems on `[-1,1]`:
`(θ₀[s]D² - θ₁[s])(θ₂[s]D² - θ₃[s])uₛ = fₛ`, with
`uₛ(±1) = uₛ′(±1) = 0`. Fields have size `(B, P+1)` and element type `T`,
which may be real or complex. Call `update!(h, (θ₀, θ₁, θ₂, θ₃))` with
four real coefficient vectors before solving. `solve!(h, u, f)` preserves `f`.

`Adapt.adapt(CuArray, h)` transfers the solver and its workspaces to CUDA.
Each solver owns mutable workspaces and must not be shared by concurrent solves.
"""
struct BatchedCoupledHelmoltzSolver{T, H, C<:AbstractMatrix{T}}
    hu::H                # first second-order operator
    hv::H                # second second-order operator
    vₛ::NTuple{3, C}     # particular workspace and upper/lower influence responses
    A::C                 # B rows of the inverse 2×2 influence matrix, column-major
end

function BatchedCoupledHelmoltzSolver(P::Int, B::Int, ::Type{T}=Float64) where {T}
    R = typeof(real(zero(T)))
    hu, hv = BatchedHelmoltzSolver(P, B, R), BatchedHelmoltzSolver(P, B, R)
    return BatchedCoupledHelmoltzSolver(hu, hv, ntuple(_ -> zeros(T, B, P+1), 3),
                                       zeros(T, B, 4))
end

function Adapt.adapt_structure(to, h::BatchedCoupledHelmoltzSolver)
    return BatchedCoupledHelmoltzSolver(Adapt.adapt(to, h.hu), Adapt.adapt(to, h.hv),
                                       Adapt.adapt(to, h.vₛ), Adapt.adapt(to, h.A))
end

# Constant wall data need no storage or host-to-device transfer.
struct _ConstantBoundary{T} <: AbstractVector{T}
    value::T
    n::Int
end
Base.size(b::_ConstantBoundary) = (b.n,)
Base.getindex(b::_ConstantBoundary, ::Int) = b.value
Base.IndexStyle(::Type{<:_ConstantBoundary}) = IndexLinear()
Base.dataids(::_ConstantBoundary) = ()

#//////////////////////////////////////////////////////////////////////////////#
#///                    CACHED HOMOGENEOUS RESPONSES                         ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    update!(h::BatchedCoupledHelmoltzSolver, θs::NTuple{4, AbstractVector})

Update the two operators and the influence responses for every system.
`θs = (θ₀, θ₁, θ₂, θ₃)` contains vectors of length `B`, in the real
precision of the solver. Return `h`. Recompute only when coefficients change.
"""
function update!(h::BatchedCoupledHelmoltzSolver{T}, θs::NTuple{4, AbstractVector}) where {T}
    update!(h.hu, θs[1], θs[2])
    update!(h.hv, θs[3], θs[4])
    work, v₊, v₋ = h.vₛ
    B = size(work, 1)
    z, o = _ConstantBoundary(zero(T), B), _ConstantBoundary(one(T), B)
    fill!(v₊, 0)
    fill!(v₋, 0)

    # Unit intermediate velocity at either wall spans the missing two
    # boundary conditions. Cache the resulting fourth-order responses.
    solve!(h.hu, work, v₊, o, z)
    solve!(h.hv, v₊, work, z, z)
    solve!(h.hu, work, v₋, z, o)
    solve!(h.hv, v₋, work, z, z)
    _influence!(h)

    #///////////////////////////////// CHECKS /////////////////////////////////#
    # The influence correction must have two independent finite responses.
    all(isfinite, h.A) || throw(ArgumentError("singular or nonfinite influence matrix"))
    #//////////////////////////////////////////////////////////////////////////#

    return h
end

# This scalar recurrence is also compiled into the CUDA setup kernel.
function _influence_system!(h, s)
    _, v₊, v₋ = h.vₛ
    a = b = c = d = zero(eltype(h.A))
    @inbounds for n in 1:size(v₊, 2)-1
        sign = isodd(n) ? 1 : -1
        a += n^2*v₊[s, n+1]
        b += sign*n^2*v₊[s, n+1]
        c += n^2*v₋[s, n+1]
        d += sign*n^2*v₋[s, n+1]
    end
    # Store the matrix inverse: the tiny solve is then four multiplies.
    determinant = a*d-b*c
    @inbounds begin
        h.A[s, 1] = d/determinant
        h.A[s, 2] = -b/determinant
        h.A[s, 3] = -c/determinant
        h.A[s, 4] = a/determinant
    end
    return nothing
end

function _influence!(h::BatchedCoupledHelmoltzSolver{T, H, <:Matrix}) where {T, H}
    _, v₊, v₋ = h.vₛ
    B, Ny = size(v₊)
    fill!(h.A, 0)
    # Accumulate wall slopes across contiguous systems. A system-first
    # traversal would stride through whole coefficient columns on the CPU.
    @inbounds for n in 1:Ny-1
        sign = isodd(n) ? 1 : -1
        @simd for s in 1:B
            h.A[s, 1] += n^2*v₊[s, n+1]
            h.A[s, 2] += sign*n^2*v₊[s, n+1]
            h.A[s, 3] += n^2*v₋[s, n+1]
            h.A[s, 4] += sign*n^2*v₋[s, n+1]
        end
    end
    @inbounds @simd for s in 1:B
        a, b, c, d = h.A[s, 1], h.A[s, 2], h.A[s, 3], h.A[s, 4]
        determinant = a*d-b*c
        h.A[s, 1] = d/determinant
        h.A[s, 2] = -b/determinant
        h.A[s, 3] = -c/determinant
        h.A[s, 4] = a/determinant
    end
    return h
end

#//////////////////////////////////////////////////////////////////////////////#
#///                        SOLVE AND WALL CORRECTION                       ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    solve!(h::BatchedCoupledHelmoltzSolver, u, f)

Solve all clamped fourth-order problems into `u`, preserving `f` and returning
`u`. The disjoint matrices must have size `(B, P+1)`, the solver's element
type, and contiguous system rows. Neither field may alias solver storage.
The cached influence responses enforce both zero wall values and zero wall
slopes; they are preserved by the solve.
"""
function solve!(h::BatchedCoupledHelmoltzSolver{T},
               u::AbstractMatrix{T}, f::AbstractMatrix{T}) where {T}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Check both fields before the first solve changes the particular workspace.
    Base.require_one_based_indexing(u, f)
    size(u) == size(f) == size(h.vₛ[1]) || throw(DimensionMismatch("expected (B, P+1) fields"))
    Base.mightalias(u, f) && throw(ArgumentError("u and f must not alias"))
    for a in (h.vₛ..., h.A)
        (Base.mightalias(u, a) || Base.mightalias(f, a)) &&
            throw(ArgumentError("fields must not alias solver workspaces"))
    end
    for op in (h.hu, h.hv)
        for a in (op.Be.b, op.Be.l, op.Be.dᵢ, op.Be.u,
                  op.Bo.b, op.Bo.l, op.Bo.dᵢ, op.Bo.u)
            (Base.mightalias(u, a) || Base.mightalias(f, a)) &&
                throw(ArgumentError("fields must not alias solver factors"))
        end
        for a in (op.cache..., op.poisson)
            (Base.mightalias(u, a) || Base.mightalias(f, a)) &&
                throw(ArgumentError("fields must not alias solver coefficients"))
        end
    end
    #//////////////////////////////////////////////////////////////////////////#

    z = _ConstantBoundary(zero(T), size(u, 1))
    solve!(h.hu, h.vₛ[1], f, z, z)
    solve!(h.hv, u, h.vₛ[1], z, z)
    _correct!(h, u)
    return u
end

function _correct!(h::BatchedCoupledHelmoltzSolver{T, H, <:Matrix}, u) where {T, H}
    work, v₊, v₋ = h.vₛ
    B, Ny = size(u)
    # The particular workspace is no longer needed. Reuse its first two
    # columns for the wall slopes and then the influence amplitudes.
    @inbounds @simd for s in 1:B
        work[s, 1] = work[s, 2] = zero(T)
    end
    @inbounds for n in 1:Ny-1
        sign = isodd(n) ? 1 : -1
        @simd for s in 1:B
            work[s, 1] -= n^2*u[s, n+1]
            work[s, 2] -= sign*n^2*u[s, n+1]
        end
    end
    @inbounds @simd for s in 1:B
        up, lo = work[s, 1], work[s, 2]
        work[s, 1] = h.A[s, 1]*up + h.A[s, 3]*lo
        work[s, 2] = h.A[s, 2]*up + h.A[s, 4]*lo
    end
    @inbounds for n in 1:Ny
        @simd for s in 1:B
            u[s, n] += work[s, 1]*v₊[s, n] + work[s, 2]*v₋[s, n]
        end
    end
    return u
end
