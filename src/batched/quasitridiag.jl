_host_storage(::AbstractArray) = true

export BatchedQuasiTridiagonal

#//////////////////////////////////////////////////////////////////////////////#
#///                BATCHED MATRIX STORAGE AND CONSTRUCTORS                 ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    BatchedQuasiTridiagonal(B, M, T=Float64)
    BatchedQuasiTridiagonal(b, l, dᵢ, u)

Store `B` quasi-tridiagonal systems of size `M ≥ 2`, with layout
`(system, coefficient)` and sizes `B` and `M` as type parameters. The array
constructor shares its four inputs; the size constructor allocates zeros.

The storage convention matches [`QuasiTridiagonal`](@ref): assemble ordinary
entries before `ul!`; afterwards `b[s,1]` is the reciprocal boundary pivot,
`dᵢ[s,k]` is the reciprocal interior pivot, and the other entries are ordinary
UL factors. `dᵢ` has shape `(B, M-1)`. Already-factorised inputs can be copied
or wrapped without refactorisation. Solves preserve the factors.
"""
struct BatchedQuasiTridiagonal{T<:AbstractFloat, B, M, A<:AbstractMatrix{T}}
     b::A  # first rows of U; b[:,1] stores reciprocal boundary pivots
     l::A  # subdiagonals of unit-diagonal L, shape (B, M-1)
    dᵢ::A  # reciprocal interior U pivots, shape (B, M-1)
     u::A  # upper diagonals, unchanged by UL factorisation

    function BatchedQuasiTridiagonal( b::A,
                                      l::A,
                                     dᵢ::A,
                                      u::A) where {T<:AbstractFloat, A<:AbstractMatrix{T}}
        #/////////////////////////////// CHECKS ///////////////////////////////#
        # Factor arrays share one-based system indices and compatible widths.
        Base.require_one_based_indexing(b, l, dᵢ, u)
        B, M = size(b)
        B > 0 && M ≥ 2 || throw(ArgumentError("batch size must be positive and matrix size at least 2"))
        size(l) == size(dᵢ) == (B, M-1) && size(u) == (B, M-2) ||
            throw(DimensionMismatch("inconsistent quasi-tridiagonal factor sizes"))
        #//////////////////////////////////////////////////////////////////////#

        return new{T, B, M, A}(b, l, dᵢ, u)
    end

    function BatchedQuasiTridiagonal(B::Int, M::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
        #/////////////////////////////// CHECKS ///////////////////////////////#
        # Check sizes before allocating the four factor arrays.
        B > 0 && M ≥ 2 || throw(ArgumentError("batch size must be positive and matrix size at least 2"))
        #//////////////////////////////////////////////////////////////////////#

        return BatchedQuasiTridiagonal(zeros(T, B, M), zeros(T, B, M-1),
                                       zeros(T, B, M-1), zeros(T, B, M-2))
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                          STORAGE ADAPTATION                           ///#
#//////////////////////////////////////////////////////////////////////////////#

# Adapt transfers factors once, preserving precision with adapt(CuArray, Q).
function Adapt.adapt_structure(to, Q::BatchedQuasiTridiagonal)
    return BatchedQuasiTridiagonal(Adapt.adapt(to, Q.b), Adapt.adapt(to, Q.l),
                                   Adapt.adapt(to, Q.dᵢ), Adapt.adapt(to, Q.u))
end

#//////////////////////////////////////////////////////////////////////////////#
#///                        BATCHED UL FACTORISATION                        ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ul!(Q::BatchedQuasiTridiagonal)

Factorise all stored matrices in place and return `Q`, preserving the storage
arrays. Reassemble the original matrix entries before calling again. Pivots
must have finite, nonzero reciprocals; this routine does not validate them.
"""
function ul!(Q::BatchedQuasiTridiagonal{T, B, M, Matrix{T}}) where {T, B, M}
    # The recurrence is sequential in coefficient index, but systems are
    # independent. Sweep contiguous columns rather than strided system rows.
    @inbounds begin
        @simd for s in 1:B
            Q.l[s, M-1] /= Q.dᵢ[s, M-1]
            Q.b[s, M-1] -= Q.b[s, M] * Q.l[s, M-1]
        end
        for i in M-2:-1:1
            @simd for s in 1:B
                Q.dᵢ[s, i] -= Q.u[s, i] * Q.l[s, i+1]
                Q.l[s, i] /= Q.dᵢ[s, i]
                Q.b[s, i] -= Q.b[s, i+1] * Q.l[s, i]
            end
        end
        # Preserve the scalar storage convention: invert pivots only after
        # elimination, leaving all other dense-row entries unchanged.
        @simd for s in 1:B
            Q.b[s, 1] = inv(Q.b[s, 1])
        end
        for k in 1:M-1
            @simd for s in 1:B
                Q.dᵢ[s, k] = inv(Q.dᵢ[s, k])
            end
        end
    end
    return Q
end

# Factor one system directly in the batch buffers, on either CPU or GPU.
# The CUDA caller selects s with one thread per system.
# No views or per-system arrays are needed; s must be a valid system index.
@inline function _ul_system!(Q::BatchedQuasiTridiagonal{T, B, M}, s) where {T, B, M}
    @inbounds begin
        # Eliminate upwards, updating the dense boundary row at each step.
        Q.l[s, M-1] /= Q.dᵢ[s, M-1]
        Q.b[s, M-1] -= Q.b[s, M] * Q.l[s, M-1]
        for i in M-2:-1:1
            Q.dᵢ[s, i] -= Q.u[s, i] * Q.l[s, i+1]
            Q.l[s, i] /= Q.dᵢ[s, i]
            Q.b[s, i] -= Q.b[s, i+1] * Q.l[s, i]
        end
        # Store reciprocal pivots only after elimination is complete.
        Q.b[s, 1] = inv(Q.b[s, 1])
        for k in 1:M-1
            Q.dᵢ[s, k] = inv(Q.dᵢ[s, k])
        end
    end
    return nothing
end


#//////////////////////////////////////////////////////////////////////////////#
#///                        BATCHED UL SUBSTITUTIONS                        ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    LinearAlgebra.ldiv!(batch::BatchedQuasiTridiagonal, rhs)

Solve the systems previously factorised by `ul!`, overwriting and returning
`rhs`. The factors are unchanged. The RHS must have shape `(B, M)`, layout
`(system, coefficient)`, contiguous systems, and storage distinct from the
factors. Real factors of type `T` support `T` or `Complex{T}` right-hand sides.
"""
function LinearAlgebra.ldiv!(Q::BatchedQuasiTridiagonal{T, B, M, Matrix{T}},
                             rhs::StridedMatrix{S}) where {T, B, M, S<:Union{T, Complex{T}}}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # CUDA arrays may satisfy Julia's StridedMatrix alias, but CPU loops
    # must reject them before scalar indexing can access device memory.
    _host_storage(rhs) || throw(ArgumentError("CPU factors require CPU storage"))
    # Shape and stride establish valid, contiguous accesses across systems.
    Base.require_one_based_indexing(rhs)
    size(rhs) == size(Q.b) || throw(DimensionMismatch("RHS must have size $(size(Q.b))"))
    stride(rhs, 1) == 1 || throw(ArgumentError("RHS systems must be contiguous"))

    # The solve overwrites only rhs; its storage must not contain the factors.
    any(a -> Base.mightalias(rhs, a), (Q.b, Q.l, Q.dᵢ, Q.u)) &&
        throw(ArgumentError("RHS must not alias the factors"))
    #//////////////////////////////////////////////////////////////////////////#

    # Apply the same U*z = rhs and L*x = z passes as in the scalar solve.
    # Coefficients depend on their neighbours, so their loop is sequential;
    # the inner SIMD loop spans independent, contiguous systems.
    # dᵢ[s,k-1] and b[s,1] are inverse pivots. rhs[s,1] accumulates the
    # dense-row residual directly: O(B*M) work, no divisions or RHS workspace.
    l, u, dᵢ, b = Q.l, Q.u, Q.dᵢ, Q.b
    @inbounds begin
        # Solve the last U equation for every system, then remove its
        # contribution from the corresponding dense first-row equation.
        @simd for s in 1:B
            rhs[s, M] = rhs[s, M]*dᵢ[s, M-1]
            rhs[s, 1] -= rhs[s, M]*b[s, M]
        end
        # Each new z[k] overwrites rhs[s,k]; the next coefficient is known.
        for k = reverse(2:M-1)
            @simd for s in 1:B
                rhs[s, k] = (rhs[s, k] - u[s, k-1]*rhs[s, k+1])*dᵢ[s, k-1]
                rhs[s, 1] -= rhs[s, k]*b[s, k]
            end
        end
        # Scale each remaining first-row residual by its inverse pivot.
        @simd for s in 1:B
            rhs[s, 1] = rhs[s, 1]*b[s, 1]
        end

        # Solve L*x = z. The previous coefficient already contains x[k-1].
        # L has unit diagonal, so only its subdiagonal contribution is removed.
        for k = 2:M
            @simd for s in 1:B
                rhs[s, k] = rhs[s, k] - rhs[s, k-1]*l[s, k-1]
            end
        end
    end
    return rhs
end
