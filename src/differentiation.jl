export diff, diff!, diff2!

#//////////////////////////////////////////////////////////////////////////////#
#///                      COEFFICIENT DIFFERENTIATION                       ///#
#//////////////////////////////////////////////////////////////////////////////#

function _diff!(out::AbstractVector{T},
                  a::AbstractVector{T}) where {T}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # The recurrence uses one-based coefficient indices and equal lengths.
    Base.require_one_based_indexing(out, a)
    length(out) == length(a) || throw(DimensionMismatch("coefficient lengths must match"))
    isempty(a) && throw(ArgumentError("at least one coefficient is required"))
    #//////////////////////////////////////////////////////////////////////////#

    # Backward Chebyshev recurrence, with the constant coefficient halved.
    # Retain the next input coefficient so the same loop works in place.
    dnext = dnext2 = zero(T)
    anext = zero(T)
    @inbounds for j = length(a):-1:1
        # Save a[j] before writing out[j], allowing differentiation in place.
        coefficient = a[j]
        value = dnext2 + 2*j*anext
        out[j] = j == 1 ? value/2 : value
        anext = coefficient
        dnext2, dnext = dnext, value
    end
    return out
end

"""
    diff!(a::AbstractVector)
    diff!(out::AbstractVector, a::AbstractVector)

Differentiate ordinary Chebyshev coefficients on `[-1, 1]`. The one-argument
method overwrites `a`. The two-argument method writes into `out`, preserves
`a`, and requires non-aliasing storage. Both expansions must have the same
element type and degree `P`; the derivative's coefficient at degree `P` is
zero.

Vectors must be one-based; `a[n+1]` is the coefficient of degree `n`.
"""
diff!(a::AbstractVector) = _diff!(a, a)

function diff!(out::AbstractVector{T},
                 a::AbstractVector{T}) where {T}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Partial overlap would corrupt coefficients that have not been read yet.
    Base.mightalias(out, a) &&
        throw(ArgumentError("input and output coefficients must not alias; use diff!(a) in place"))
    #//////////////////////////////////////////////////////////////////////////#

    return _diff!(out, a)
end

"""
    diff2!(a::AbstractVector)
    diff2!(out::AbstractVector, a::AbstractVector)

Differentiate twice. The one-argument method overwrites `a`; the two-argument
method preserves `a` and requires distinct storage.
"""
diff2!(a::AbstractVector) = diff!(diff!(a))

diff2!(out::AbstractVector{T}, a::AbstractVector{T}) where {T} = diff!(diff!(out, a))

#//////////////////////////////////////////////////////////////////////////////#
#///                          ENDPOINT DERIVATIVES                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    diff(a::AbstractVector, :left)
    diff(a::AbstractVector, :right)

Return `da/dy` at `y = -1` for `side = :left`, or at `y = +1` for
`side = :right`, without modifying `a` or forming its derivative expansion.

Use `T_n'(+1) = n²` and `T_n'(-1) = (-1)^(n+1)*n²`. Derivatives refer to
the reference interval `[-1, 1]`; multiply by `2/(b-a)` when mapping to a
physical interval `[a, b]`. Throw `ArgumentError` for any other `side`.
"""
function Base.diff(   a::AbstractVector{T},
                  side::Symbol) where {T}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Endpoint evaluation assumes an ordinary, nonempty coefficient vector.
    Base.require_one_based_indexing(a)
    isempty(a) && throw(ArgumentError("at least one coefficient is required"))
    side === :left || side === :right ||
        throw(ArgumentError("side must be :left or :right"))
    #//////////////////////////////////////////////////////////////////////////#

    value = zero(T)
    @inbounds for n = 1:length(a)-1
        value += (side === :right || isodd(n) ? 1 : -1) * n^2 * a[n+1]
    end
    return value
end
