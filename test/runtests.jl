using ChebyshevHelmoltzSolvers
using LinearAlgebra
using Test

# Independent cosine interpolation/evaluation for the analytic test problems.
# No FFT, plotting, debugger or benchmark packages are needed by the suite.
function coefficients(f, P, ::Type{T}=Float64) where {T}
    P == 0 && return ChebCoeffs(T[f(1.0)])
    values = [f(cospi(j/P)) for j = 0:P]
    a = ChebCoeffs(P, T)
    for n = 0:P
        value = (values[1] + (-1)^n*values[end])/2
        value += sum(values[j+1]*cospi(n*j/P) for j = 1:P-1; init=zero(value))
        a[n] = 2value/P
    end
    a[0] /= 2
    a[P] /= 2
    return a
end

evaluate(a, y) = sum(a[n]*cos(n*acos(y)) for n = 0:length(a)-1)
tolerance(::Type{T}) where {T} = real(zero(T)) isa Float32 ? 2e-4 : 2e-11

@testset "ChebyshevHelmoltzSolvers" begin
    include("test_chebcoeffs.jl")
    include("test_quasitridiag.jl")
    include("test_helmoltz.jl")
    include("test_coupled.jl")
end
