#//////////////////////////////////////////////////////////////////////////////#
#///             COEFFICIENT STORAGE AND DIFFERENTIATION TESTS              ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Coefficient differentiation" begin
    for T in (Float32, Float64, ComplexF64), P in (0, 1, 2, 7, 8)
        @testset "$T, degree $P" begin
            amplitude = T <: Complex ? T(1 + 0.25im) : T(1)
            a = zeros(T, P+1)
            a[P+1] = amplitude
            original = copy(a)
            derivative = similar(a)
            @test diff!(derivative, a) === derivative
            @test a == original
            @test derivative[P+1] == 0
            for y in (-0.75, -0.2, 0.4, 0.8)
                exact = P == 0 ? zero(T) : amplitude*P*sin(P*acos(y))/sqrt(1-y^2)
                @test evaluate(derivative, y) ≈ exact atol=tolerance(T) rtol=tolerance(T)
            end
            @test diff(a, :right) ≈ amplitude*P^2
            @test diff(a, :left) ≈ amplitude*(-1)^(P+1)*P^2
            @test_throws ArgumentError diff!(a, a)
            @test diff!(a) === a
            @test a == derivative
        end
    end
    @test_throws ArgumentError diff([1.0], :upper)
end

@testset "Differentiation of vector views" begin
    # Differentiate (1-y²)² twice: -4+12y² = 2T₀+6T₂. Strided storage
    # exercises coefficient indexing independently of contiguous vectors.
    storage = zeros(10)
    a = view(storage, 1:2:9)
    a .= [3/8, 0, -1/2, 0, 1/8]
    out = view(zeros(10), 2:2:10)
    @test diff2!(out, a) === out
    @test out ≈ [2, 0, 6, 0, 0]
    @test a == [3/8, 0, -1/2, 0, 1/8]
    @test diff2!(a) ≈ out

    # Overlapping views cannot be used as distinct source and destination.
    @test_throws ArgumentError diff!(view(storage, 1:5), view(storage, 2:6))
    @test_throws DimensionMismatch diff!(zeros(4), zeros(5))
    @test_throws ArgumentError diff!(Float64[])
end
