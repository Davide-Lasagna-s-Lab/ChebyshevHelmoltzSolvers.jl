export ChebCoeffs, diff!, endpoint_derivative

"""
    ChebCoeffs(P, T=Float64)
    ChebCoeffs(v::AbstractVector)

Store the ordinary coefficients of `f(y) = sum(a[n]*T_n(y), n=0:P)` on
`[-1, 1]`. Polynomial degree `n` is also the index: valid indices are `0:P`,
including the constant coefficient `a[0]` without an implicit factor of two.

The first constructor allocates `P + 1` zero coefficients of type `T`.
The second wraps the one-based vector `v` without copying and infers
`P = length(v) - 1`. `parent(a)` returns this underlying storage.
`P` must be nonnegative and `v` must be nonempty; invalid inputs throw
`ArgumentError`.
"""
struct ChebCoeffs{T, P, V<:AbstractVector{T}} <: AbstractVector{T}
    data::V

    function ChebCoeffs(P::Int, ::Type{T}=Float64) where {T}
        P ≥ 0 || throw(ArgumentError("polynomial degree must be nonnegative"))
        return new{T, P, Vector{T}}(zeros(T, P + 1))
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

"""
    diff!(out::ChebCoeffs, a::ChebCoeffs)

Write the ordinary Chebyshev coefficients of `da/dy` on `[-1, 1]` into
`out`, and return `out`. Both expansions must have the same element type and
degree `P`; the derivative's coefficient at degree `P` is zero.

Use the backward recurrence `b[n] = b[n+2] + 2*(n+1)*a[n+1]`, with a final
factor of one half for `b[0]`. The operation takes `O(P)` work and constant
extra storage. `out === a` is supported; otherwise their storage must not
overlap. `a` is preserved when the storage is distinct.
"""
function diff!(out::ChebCoeffs{T, P},
                 a::ChebCoeffs{T, P}) where {T, P}
    dnext = dnext2 = zero(T)
    anext = zero(T)
    @inbounds for n = P:-1:0
        # Save a[n] before writing out[n], allowing differentiation in place.
        coefficient = a[n]
        value = dnext2 + 2*(n + 1)*anext
        out[n] = n == 0 ? value/2 : value
        anext = coefficient
        dnext2, dnext = dnext, value
    end
    return out
end

"""
    endpoint_derivative(a::ChebCoeffs, side::Symbol)

Return `da/dy` at `y = -1` for `side = :left`, or at `y = +1` for
`side = :right`, without modifying `a` or forming its derivative expansion.

Use `T_n'(+1) = n²` and `T_n'(-1) = (-1)^(n+1)*n²`. Derivatives refer to
the reference interval `[-1, 1]`; multiply by `2/(b-a)` when mapping to a
physical interval `[a, b]`. Throw `ArgumentError` for any other `side`.
"""
function endpoint_derivative(   a::ChebCoeffs{T, P},
                             side::Symbol) where {T, P}
    side === :left || side === :right ||
        throw(ArgumentError("side must be :left or :right"))
    value = zero(T)
    @inbounds for n = 1:P
        value += (side === :right || isodd(n) ? 1 : -1) * n^2 * a[n]
    end
    return value
end
