@testset "Chebyshev coefficients" begin
    data = [9.0, 8, 7, 6]
    a = ChebCoeffs(data)
    @test parent(a) === data
    @test size(a) == (4,)
    @test axes(a) == (0:3,)
    @test axes(a, 1) == 0:3
    @test axes(a, 2) == Base.OneTo(1)
    @test firstindex(a) == 0
    @test lastindex(a) == 3
    @test collect(eachindex(a)) == collect(0:3)
    @test a[0] == 9
    @test a[3] == 6
    a[0] = 5
    @test data[1] == 5
    @test_throws BoundsError a[-1]
    @test_throws BoundsError a[4]
    @test_throws BoundsError a[4] = 0

    b = copy(a)
    b[0] = -1
    @test a[0] == 5
    @test parent(b) !== parent(a)
    c = similar(a)
    @test axes(c) == axes(a)
    @test eltype(c) == eltype(a)
    c .= a
    @test parent(c) == parent(a)
    c .= 2 .* c .+ a
    @test parent(c) == 3 .* parent(a)
    wrapped = ChebCoeffs(parent(c))
    c .= wrapped .+ 1
    @test parent(c) == 3 .* parent(a) .+ 1
    @test_throws ArgumentError ChebCoeffs(-1)
    @test_throws ArgumentError ChebCoeffs(Float64[])
    @test_throws ArgumentError ChebCoeffs(a)
    storage = zeros(8)
    viewcoeffs = ChebCoeffs(view(storage, 1:2:7))
    viewcoeffs[2] = 3
    @test storage[5] == 3

    # Modal solves already know the polynomial degree. The typed wrapper
    # must infer a concrete return type and retain the original view, while
    # rejecting storage that cannot represent that degree.
    column = view(storage, 1:2:7)
    typed = @inferred ChebCoeffs{Float64, 3}(column)
    @test parent(typed) === column
    @test typed[2] == storage[5]
    @test_throws DimensionMismatch ChebCoeffs{Float64, 2}(column)
    @test_throws ArgumentError ChebCoeffs{Float64, -1}(Float64[])
    @test_throws ArgumentError ChebCoeffs{Float64, 3}(a)

    for T in (Float32, Float64, ComplexF64), P in (0, 1, 2, 7, 8)
        @testset "$T, degree $P" begin
            amplitude = T <: Complex ? T(1 + 0.25im) : T(1)
            a = ChebCoeffs(P, T)
            a[P] = amplitude
            original = copy(parent(a))
            derivative = similar(a)
            @test diff!(derivative, a) === derivative
            @test parent(a) == original
            @test derivative[P] == 0
            for y in (-0.75, -0.2, 0.4, 0.8)
                exact = P == 0 ? zero(T) : amplitude*P*sin(P*acos(y))/sqrt(1-y^2)
                @test evaluate(derivative, y) ≈ exact atol=tolerance(T) rtol=tolerance(T)
            end
            @test endpoint_derivative(a, :right) ≈ amplitude*P^2
            @test endpoint_derivative(a, :left) ≈ amplitude*(-1)^(P+1)*P^2
            @test_throws ArgumentError diff!(a, a)
            @test diff!(a) === a
            @test parent(a) == parent(derivative)
        end
    end
    @test_throws ArgumentError endpoint_derivative(a, :upper)
end
