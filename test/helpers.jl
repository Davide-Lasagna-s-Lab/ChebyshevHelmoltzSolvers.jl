# Independent cosine interpolation/evaluation for the analytic test problems.
# The analytic reference below does not use the FFT-based transforms.
function coefficients(f, P, ::Type{T}=Float64) where {T}
    P == 0 && return T[f(1.0)]
    values = [f(cospi(j/P)) for j = 0:P]
    a = zeros(T, P+1)
    for n = 0:P
        value = (values[1] + (-1)^n*values[end])/2
        value += sum(values[j+1]*cospi(n*j/P) for j = 1:P-1; init=zero(value))
        a[n+1] = 2value/P
    end
    a[1] /= 2
    a[P+1] /= 2
    return a
end

evaluate(a, y) = sum(a[n+1]*cos(n*acos(y)) for n = 0:length(a)-1)
tolerance(::Type{T}) where {T} = real(zero(T)) isa Float32 ? 2e-4 : 2e-11

