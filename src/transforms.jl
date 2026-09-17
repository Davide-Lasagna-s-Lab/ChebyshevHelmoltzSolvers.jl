export chebcoeffs, chebvalues

#//////////////////////////////////////////////////////////////////////////////#
#///                     CHEBYSHEV PROFILE COEFFICIENTS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    chebcoeffs(values::AbstractVector)

Return ordinary Chebyshev coefficients from values at the Lobatto points
`y[j+1] = cospi(j/(N-1))`, ordered from +1 to -1. Require at least two
values. Preserve the input and return a newly allocated `ChebCoeffs`, indexed
by polynomial degree from zero.
The DCT-I normalization matches the expansion used by `ChebCoeffs`, with
no implicit factor of two on the constant coefficient.
"""
function chebcoeffs(values::AbstractVector)
    length(values) >= 2 ||
        throw(ArgumentError("at least two Lobatto values are required"))
    coefficients = FFTW.r2r(float.(values), FFTW.REDFT00)
    coefficients ./= length(values) - 1
    coefficients[1] /= 2
    coefficients[end] /= 2
    return ChebCoeffs(coefficients)
end

"""
    chebvalues(a::ChebCoeffs)

Evaluate ordinary Chebyshev coefficients at Lobatto points
`y[j+1] = cospi(j/P)`, ordered from +1 to -1, where `P = length(a)-1`.
Return a newly allocated, one-based vector and preserve `a`. A constant
expansion returns its single value. This is the inverse of [`chebcoeffs`](@ref)
for expansions with at least two coefficients.
"""
function chebvalues(a::ChebCoeffs)
    values = float.(parent(a))
    length(values) == 1 && return values
    # DCT-I doubles interior contributions, whereas endpoint coefficients
    # enter once. Halve only the interior to evaluate the ordinary expansion.
    values[2:end-1] ./= 2
    return FFTW.r2r(values, FFTW.REDFT00)
end
