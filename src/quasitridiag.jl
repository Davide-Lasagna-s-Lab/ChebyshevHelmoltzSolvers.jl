export QuasiTridiagonal, ul!

"""
    QuasiTridiagonal(b, l, d, u)
    QuasiTridiagonal(M, T)

Store an `M × M` matrix with a dense first row and a tridiagonal interior:
```text
b₁  b₂  b₃  b₄  ⋯
l₁  d₁  u₁
    l₂  d₂  u₂
        l₃  d₃  ⋱
```

The vector constructor shares its inputs: `b` has length `M ≥ 1`, `l` and
`d` have length `M - 1`, and `u` has length `max(M - 2, 0)`. The size/type
constructor allocates zero storage to be filled before factorisation.

[`ul!`](@ref) overwrites the matrix with compact UL factors without pivoting.
Factorisation and subsequent [`LinearAlgebra.ldiv!`](@ref) solves take `O(M)`
work. After factorisation, indexing exposes the stored factors rather than
the original matrix entries.
"""
struct QuasiTridiagonal{T, M} <: AbstractMatrix{T}
    b::Vector{T}  # dense first row, length M
    l::Vector{T}  # lower diagonal, length M-1
    d::Vector{T}  # diagonal below the first row, length M-1
    u::Vector{T}  # upper diagonal below the first row, length max(M-2, 0)

    function QuasiTridiagonal(b::Vector{T},
                              l::Vector{T},
                              d::Vector{T},
                              u::Vector{T}) where {T}
        M = length(b)
        M ≥ 1 || throw(ArgumentError("matrix size must be positive"))
        (length(l) == M-1) && (length(d) == M-1) && (length(u) == max(M-2, 0)) ||
            throw(ArgumentError("incompatible lengths"))
        return new{T, M}(b, l, d, u)
    end

    QuasiTridiagonal(M::Int, ::Type{T}) where {T} =
        QuasiTridiagonal(zeros(T, M), zeros(T, M-1), zeros(T, M-1), zeros(T, max(M-2, 0)))
end

Base.size(Q::QuasiTridiagonal{T, M}) where {T, M} = (M, M)

Base.@propagate_inbounds function Base.getindex(Q::QuasiTridiagonal{T},
                                                i::Int,
                                                j::Int) where {T}
    @boundscheck checkbounds(Q, i, j)
    i == 1   && return Q.b[j]
    i == j   && return Q.d[i-1]
    i == j-1 && return Q.u[i-1]
    i == j+1 && return Q.l[i-1]
    return zero(T)
end

"""
    ul!(Q::QuasiTridiagonal)

Factorise `Q = U*L` in place without pivoting, and return `Q`. Eliminate from
the last row upwards to retain the dense first row and tridiagonal interior.

Overwrite `Q.l` with the subdiagonal of unit-diagonal `L`, and overwrite
`Q.d` and `Q.b` with the diagonal and first row of `U`; `Q.u` is unchanged.
All pivots must be nonzero. The routine does not check for singularity.

Call once after assembling the matrix. Reassemble its original entries before
factorising again; use [`LinearAlgebra.ldiv!`](@ref) to reuse existing factors.
"""
function ul!(Q::QuasiTridiagonal{T, M}) where {T, M}
    # With one coefficient, only the boundary row remains.
    M == 1 && return Q
    l, d, u, b = Q.l, Q.d, Q.u, Q.b
    @inbounds begin
        l[M-1] = l[M-1]/d[M-1]
        b[M-1] = b[M-1] - b[M]*l[M-1]
        for i = reverse(1:M-2)
            d[i] = d[i] - u[i]*l[i+1]
            l[i] = l[i]/d[i]
            b[i] = b[i] - b[i+1]*l[i]
        end
    end
    return Q
end

"""
    LinearAlgebra.ldiv!(Q::QuasiTridiagonal, c::AbstractVector)

Solve the system using the UL factors already stored in `Q`, overwriting
the right-hand side `c` with the solution and returning `c`.

`c` must be a one-based vector of length `size(Q, 1)`, with the same element
type as `Q` and storage distinct from its factors. Solve `U*z = c` backwards,
then `L*x = z` forwards. The factors are preserved for subsequent solves.
"""
function LinearAlgebra.ldiv!(Q::QuasiTridiagonal{T, M},
                             c::AbstractVector{T}) where {T, M}
    Base.require_one_based_indexing(c)
    length(c) == M || throw(DimensionMismatch("right-hand side must have length $M"))
    if M == 1
        c[1] /= Q.b[1]
        return c
    end

    l, u, d, b = Q.l, Q.u, Q.d, Q.b
    @inbounds begin
        # Backward substitution through U, accumulating its dense first row.
        c[M] = c[M]/d[M-1]
        Σ = c[M]*b[M]
        for k = reverse(2:M-1)
            c[k] = (c[k] - u[k-1]*c[k+1])/d[k-1]
            Σ += c[k]*b[k]
        end
        c[1] = (c[1] - Σ)/b[1]

        # Forward substitution through the unit lower-bidiagonal factor.
        for k = 2:M
            c[k] = c[k] - c[k-1]*l[k-1]
        end
    end
    return c
end
