@testset "Quasi-tridiagonal UL" begin
    for T in (Float32, Float64, ComplexF64), M in (1, 2, 3, 8)
        @testset "$T, size $M" begin
            Q = QuasiTridiagonal(M, T)
            Q.b .= T(0.2)
            Q.b[1] = T(3)
            Q.l .= T(-0.3)
            Q.d .= T(2.5)
            Q.u .= T(0.4)
            A = diagm(0 => vcat(T(3), fill(T(2.5), M-1)))
            for i = 2:M
                A[i, i-1] = T(-0.3)
                i < M && (A[i, i+1] = T(0.4))
            end
            A[1, :] .= Q.b
            @test Matrix(Q) == A
            @test_throws BoundsError Q[0, 1]
            @test_throws BoundsError Q[1, M+1]
            @test ul!(Q) === Q
            U = triu(Matrix(Q))
            L = tril(Matrix(Q), -1) + I
            @test U*L ≈ A rtol=tolerance(T)
            factors = map(copy, (Q.b, Q.l, Q.d, Q.u))
            @test_throws DimensionMismatch ldiv!(Q, zeros(T, M+1))
            @test_throws ArgumentError ldiv!(Q, ChebCoeffs(M-1, T))

            for scale in (1, -0.5)
                c = T[scale*(j + (T <: Complex ? 0.2im*j : 0)) for j = 1:M]
                expected = A\c
                @test ldiv!(Q, c) === c
                @test c ≈ expected rtol=tolerance(T)
                @test (Q.b, Q.l, Q.d, Q.u) == factors
            end
        end
    end
    @test_throws ArgumentError QuasiTridiagonal(0, Float64)
    @test_throws ArgumentError QuasiTridiagonal(zeros(3), zeros(1), zeros(2), zeros(1))
end
