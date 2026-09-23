export diff, diff!, diff2!

"""
    diff!(a::ChebCoeffs)
    diff!(out::ChebCoeffs, a::ChebCoeffs)

Differentiate ordinary Chebyshev coefficients on `[-1, 1]`. The one-argument
method overwrites `a`. The two-argument method writes into `out`, preserves
`a`, and requires non-aliasing storage. Both expansions must have the same
element type and degree `P`; the derivative's coefficient at degree `P` is
zero.

Use the backward recurrence `b[n] = b[n+2] + 2*(n+1)*a[n+1]`, with a final
factor of one half for `b[0]`. Both methods take `O(P)` work and constant
extra storage.
"""
function _diff!(out::ChebCoeffs{T, P},
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

diff!(a::ChebCoeffs) = _diff!(a, a)

function diff!(out::ChebCoeffs{T, P},
                 a::ChebCoeffs{T, P}) where {T, P}
    Base.mightalias(parent(out), parent(a)) &&
        throw(ArgumentError("input and output coefficients must not alias; use diff!(a) in place"))
    return _diff!(out, a)
end

"""
    diff2!(a::ChebCoeffs)
    diff2!(out::ChebCoeffs, a::ChebCoeffs)

Differentiate twice. The one-argument method overwrites `a`; the two-argument
method preserves `a` and requires distinct storage.
"""
diff2!(a::ChebCoeffs) = diff!(diff!(a))

diff2!(out::ChebCoeffs{T, P}, a::ChebCoeffs{T, P}) where {T, P} = diff!(diff!(out, a))

"""
    diff(a::ChebCoeffs, :left)
    diff(a::ChebCoeffs, :right)

Return `da/dy` at `y = -1` for `side = :left`, or at `y = +1` for
`side = :right`, without modifying `a` or forming its derivative expansion.

Use `T_n'(+1) = n²` and `T_n'(-1) = (-1)^(n+1)*n²`. Derivatives refer to
the reference interval `[-1, 1]`; multiply by `2/(b-a)` when mapping to a
physical interval `[a, b]`. Throw `ArgumentError` for any other `side`.
"""
function Base.diff(   a::ChebCoeffs{T, P},
                  side::Symbol) where {T, P}
    side === :left || side === :right ||
        throw(ArgumentError("side must be :left or :right"))
    value = zero(T)
    @inbounds for n = 1:P
        value += (side === :right || isodd(n) ? 1 : -1) * n^2 * a[n]
    end
    return value
end
