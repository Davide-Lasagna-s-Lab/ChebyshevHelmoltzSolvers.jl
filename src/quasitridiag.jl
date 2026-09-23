export QuasiTridiagonal, ul!

#//////////////////////////////////////////////////////////////////////////////#
#///                    MATRIX STORAGE AND CONSTRUCTORS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    QuasiTridiagonal(b, l, dᵢ, u)
    QuasiTridiagonal(M, T)

Compact storage for an `M × M` matrix with a dense first row and a
tridiagonal interior (`M ≥ 2`). The vector constructor shares its inputs;
the size constructor allocates zeros.

Before [`ul!`](@ref), assemble ordinary matrix entries into `b`, `l`,
`dᵢ`, and `u`. Factorisation overwrites these same four arrays: `b[1]`
stores `1/U[1,1]`, `b[2:M]` stores the rest of the first row of `U`, and
`dᵢ[k]` stores `1/U[k+1,k+1]`. `l` stores the subdiagonal of unit-diagonal
`L`; `u` is unchanged. No additional reciprocal storage is allocated.

Indexing is defined for the factorised representation: it recovers the
actual diagonal entries by inverting the stored reciprocals and exposes
the compact UL factors, not the original matrix. Before factorisation,
the arrays are assembly buffers; access them directly.
"""
struct QuasiTridiagonal{T<:AbstractFloat, M, V<:AbstractVector{T}} <: AbstractMatrix{T}
     b::V  # first row of U; b[1] stores the reciprocal boundary pivot
     l::V  # subdiagonal of unit-diagonal L, length M-1
    dᵢ::V  # reciprocal interior U pivots, length M-1
     u::V  # upper diagonal below the first row, length M-2

    function QuasiTridiagonal( b::V,
                               l::V,
                              dᵢ::V,
                               u::V) where {T<:AbstractFloat, V<:AbstractVector{T}}
        #/////////////////////////////// CHECKS ///////////////////////////////#
        # Views are allowed, with one-based indices and compatible lengths.
        Base.require_one_based_indexing(b, l, dᵢ, u)
        M = length(b)
        M ≥ 2 || throw(ArgumentError("matrix size must be at least 2"))
        length(l) == length(dᵢ) == M-1 && length(u) == M-2 ||
            throw(ArgumentError("incompatible lengths"))
        #//////////////////////////////////////////////////////////////////////#

        return new{T, M, V}(b, l, dᵢ, u)
    end

    function QuasiTridiagonal(M::Int, ::Type{T}) where {T}
        #/////////////////////////////// CHECKS ///////////////////////////////#
        # Check sizes before allocating the four factor arrays.
        M ≥ 2 || throw(ArgumentError("matrix size must be at least 2"))
        #//////////////////////////////////////////////////////////////////////#

        return QuasiTridiagonal(zeros(T, M), zeros(T, M-1),
                                zeros(T, M-1), zeros(T, M-2))
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            MATRIX INTERFACE                            ///#
#//////////////////////////////////////////////////////////////////////////////#

Base.size(::QuasiTridiagonal{T, M}) where {T, M} = (M, M)

Base.@propagate_inbounds function Base.getindex(Q::QuasiTridiagonal{T},
                                                i::Int,
                                                j::Int) where {T}
    @boundscheck checkbounds(Q, i, j)
    i == 1   && return j == 1 ? inv(Q.b[1]) : Q.b[j]
    i == j   && return inv(Q.dᵢ[i-1])
    i == j-1 && return Q.u[i-1]
    i == j+1 && return Q.l[i-1]
    return zero(T)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            UL FACTORISATION                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ul!(Q::QuasiTridiagonal)

Factorise `Q = U*L` in place without pivoting, and return `Q`. Eliminate from
the last row upwards to retain the dense first row and tridiagonal interior.

Overwrite `Q.l` with the subdiagonal of unit-diagonal `L`. After elimination,
replace the interior U pivots in `Q.dᵢ` and the boundary pivot in `Q.b[1]`
with their reciprocals; the remaining first-row entries and `Q.u` are ordinary
U entries. Pivots must have finite, nonzero reciprocals; this low-level routine
does not validate them. Multiplication by reciprocals can round differently
from division.

Call once after assembling the matrix. Reassemble its original entries before
factorising again; use [`LinearAlgebra.ldiv!`](@ref) to reuse existing factors.
"""
function ul!(Q::QuasiTridiagonal{T, M}) where {T, M}
    l, dᵢ, u, b = Q.l, Q.dᵢ, Q.u, Q.b
    @inbounds begin
        l[M-1] = l[M-1]/dᵢ[M-1]
        b[M-1] = b[M-1] - b[M]*l[M-1]
        for i = reverse(1:M-2)
            dᵢ[i] = dᵢ[i] - u[i]*l[i+1]
            l[i] = l[i]/dᵢ[i]
            b[i] = b[i] - b[i+1]*l[i]
        end

        # Replace the pivots in place; subsequent solves only multiply.
        b[1] = inv(b[1])
        for k = 1:M-1
            dᵢ[k] = inv(dᵢ[k])
        end
    end
    return Q
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            UL SUBSTITUTIONS                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    LinearAlgebra.ldiv!(Q::QuasiTridiagonal, rhs::AbstractVector)

Solve `Q*x = rhs` from the previously computed factorisation `Q = U*L`.
Overwrite `rhs` with `x` and return the same vector; preserve all factors.
Call `ul!` before solving. The one-based RHS must have length `M`, the same
precision as `Q` (real or complex), and storage distinct from the factors.

"""
function LinearAlgebra.ldiv!(Q::QuasiTridiagonal{T, M},
                             rhs::AbstractVector{S}) where {T, M, S<:Union{T, Complex{T}}}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # The substitutions assume one-based indexing and exactly M coefficients.
    Base.require_one_based_indexing(rhs)
    length(rhs) == M || throw(DimensionMismatch("right-hand side must have length $M"))
    #//////////////////////////////////////////////////////////////////////////#

    # Q = U*L: first solve U*z = rhs backwards, then L*x = z forwards.
    # U is upper bidiagonal below its dense first row. The stored reciprocals
    # dᵢ[k-1] = 1/U[k,k] and b[1] = 1/U[1,1] replace pivot divisions.
    # rhs[1] accumulates the dense-row residual as each z[k] becomes known.
    # Both passes reuse rhs and take O(M) work without a temporary vector.
    l, u, dᵢ, b = Q.l, Q.u, Q.dᵢ, Q.b
    @inbounds begin
        # Solve the last U equation, which has no upper-diagonal contribution.
        # Subtract its contribution from the dense first-row equation.
        rhs[M] = rhs[M]*dᵢ[M-1]
        rhs[1] -= rhs[M]*b[M]
        # Each new z[k] overwrites rhs[k]; rhs[k+1] already contains z[k+1].
        for k = reverse(2:M-1)
            rhs[k] = (rhs[k] - u[k-1]*rhs[k+1])*dᵢ[k-1]
            rhs[1] -= rhs[k]*b[k]
        end
        # All off-diagonal first-row terms are now removed; b[1] is inverse.
        rhs[1] = rhs[1]*b[1]

        # Solve L*x = z. rhs[k-1] already contains x[k-1], while rhs[k] is z[k].
        # The diagonal of L is one, so no pivot scaling is required.
        for k = 2:M
            rhs[k] = rhs[k] - rhs[k-1]*l[k-1]
        end
    end
    return rhs
end
