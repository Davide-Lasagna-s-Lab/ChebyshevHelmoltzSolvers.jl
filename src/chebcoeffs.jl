export ChebCoeffs

"""
    ChebCoeffs(P, T=Float64)
    ChebCoeffs(v::AbstractVector)
    ChebCoeffs{T, P}(v::AbstractVector{T})

Store the ordinary coefficients of `f(y) = sum(a[n]*T_n(y), n=0:P)` on
`[-1, 1]`. Polynomial degree `n` is also the index: valid indices are `0:P`,
including the constant coefficient `a[0]` without an implicit factor of two.

The first constructor allocates `P + 1` zero coefficients of type `T`.
The second wraps the one-based vector `v` without copying and infers
`P = length(v) - 1`. The third preserves a degree already known from a
solver type, avoiding runtime type construction when wrapping modal views.
It checks that the storage contains exactly `P + 1` entries.
`parent(a)` returns this underlying storage.
`P` must be nonnegative and `v` must be nonempty; invalid inputs throw
`ArgumentError`.
"""
struct ChebCoeffs{T, P, V<:AbstractVector{T}} <: AbstractVector{T}
    data::V

    function ChebCoeffs(P::Int, ::Type{T}=Float64) where {T}
        P ≥ 0 || throw(ArgumentError("polynomial degree must be nonnegative"))
        return new{T, P, Vector{T}}(zeros(T, P + 1))
    end

    function ChebCoeffs{T, P}(v::V) where {T, P, V<:AbstractVector{T}}
        Base.require_one_based_indexing(v)
        P ≥ 0 || throw(ArgumentError("polynomial degree must be nonnegative"))
        length(v) == P + 1 || throw(DimensionMismatch("expected $(P + 1) coefficients"))
        return new{T, P, V}(v)
    end

    function ChebCoeffs(v::V) where {T, V<:AbstractVector{T}}
        Base.require_one_based_indexing(v)
        isempty(v) && throw(ArgumentError("a Chebyshev expansion needs at least one coefficient"))
        return new{T, length(v) - 1, V}(v)
    end
end

# Expose degree-based indexing while retaining ordinary one-based storage.
@inline Base.LinearIndices(s::ChebCoeffs) = axes(s, 1)
@inline Base.axes(s::ChebCoeffs{T, P}) where {T, P} = (Base.IdentityUnitRange(0:P),)
@inline Base.size(s::ChebCoeffs{T, P}) where {T, P} = (P + 1,)

Base.similar(s::ChebCoeffs) = ChebCoeffs(similar(s.data))
Base.copy(s::ChebCoeffs) = ChebCoeffs(copy(s.data))
Base.parent(s::ChebCoeffs) = s.data

Base.@propagate_inbounds function Base.getindex(s::ChebCoeffs, i::Int)
    @boundscheck checkbounds(s.data, i + 1)
    @inbounds ret = s.data[i + 1]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(s::ChebCoeffs, v, i::Int)
    @boundscheck checkbounds(s.data, i + 1)
    @inbounds s.data[i + 1] = v
    return v
end

# Let broadcasting recognise shared storage through the coefficient wrapper.
Base.dataids(s::ChebCoeffs) = Base.dataids(parent(s))
Broadcast.broadcast_unalias(dest::ChebCoeffs, src::ChebCoeffs) =
    parent(dest) === parent(src) ? src : Broadcast.unalias(dest, src)
