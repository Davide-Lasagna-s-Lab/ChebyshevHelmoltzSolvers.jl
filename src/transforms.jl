export chebpoints, chebcoeffs, chebvalues

#//////////////////////////////////////////////////////////////////////////////#
#///                         CHEBYSHEV LOBATTO POINTS                        ///#
#//////////////////////////////////////////////////////////////////////////////#

# Reference-to-physical derivative scale. Solvers keep this scalar rather
# than mapped coefficient arrays; all systems in a batch share one interval.
function _intervalscale(a, b, ::Type{T}=Float64) where {T}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # An affine map needs finite, ordered endpoints and a usable scale.
    isfinite(a) && isfinite(b) && a < b ||
        throw(ArgumentError("interval endpoints must be finite with a < b"))
    scale = T(2 / (b-a))
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("interval scale must be finite and positive"))
    #//////////////////////////////////////////////////////////////////////////#

    return scale
end

"""
    chebpoints(P::Integer; a=-1, b=1)

Return the `P+1` Chebyshev–Lobatto points on `[a, b]`, ordered from `b`
to `a`. The polynomial degree `P` must be positive. Default to `[-1, 1]`.
"""
function chebpoints(P::Integer; a=-1, b=1)
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Both endpoints require a positive polynomial degree.
    P >= 1 || throw(ArgumentError("P must be positive"))
    #//////////////////////////////////////////////////////////////////////////#

    _intervalscale(a, b)
    return [a + (b-a)*(1 + cospi(j/P))/2 for j = 0:P]
end

#//////////////////////////////////////////////////////////////////////////////#
#///                     CHEBYSHEV PROFILE COEFFICIENTS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    chebcoeffs(values::AbstractVector)

Return ordinary Chebyshev coefficients from values at the Lobatto points
`y[j+1] = cospi(j/(N-1))`, ordered from +1 to -1. Require at least two
values. Preserve the input and return a newly allocated one-based vector.
Index `n+1` stores the coefficient of degree `n`, with
no implicit factor of two on the constant coefficient.
"""
function chebcoeffs(values::AbstractVector)
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # DCT-I needs at least two values on a one-based grid.
    Base.require_one_based_indexing(values)
    length(values) >= 2 ||
        throw(ArgumentError("at least two Lobatto values are required"))
    #//////////////////////////////////////////////////////////////////////////#

    coefficients = FFTW.r2r(float.(values), FFTW.REDFT00)
    coefficients ./= length(values) - 1
    coefficients[1] /= 2
    coefficients[end] /= 2
    return coefficients
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      VALUES AT COLLOCATION POINTS                      ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    chebvalues(a::AbstractVector)

Evaluate ordinary Chebyshev coefficients at Lobatto points
`y[j+1] = cospi(j/P)`, ordered from +1 to -1, where `P = length(a)-1`.
Return a newly allocated, one-based vector and preserve `a`. A constant
expansion returns its single value. This is the inverse of [`chebcoeffs`](@ref)
for expansions with at least two coefficients.
"""
function chebvalues(a::AbstractVector)
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # A constant expansion is valid; an empty expansion is not.
    Base.require_one_based_indexing(a)
    isempty(a) && throw(ArgumentError("at least one coefficient is required"))
    #//////////////////////////////////////////////////////////////////////////#

    values = float.(a)
    length(values) == 1 && return values
    # DCT-I doubles interior contributions, whereas endpoint coefficients
    # enter once. Halve only the interior to evaluate the ordinary expansion.
    values[2:end-1] ./= 2
    return FFTW.r2r(values, FFTW.REDFT00)
end
