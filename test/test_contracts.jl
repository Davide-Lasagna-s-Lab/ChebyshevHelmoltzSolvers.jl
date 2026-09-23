#//////////////////////////////////////////////////////////////////////////////#
#///                       PUBLIC API AND FAILURE SAFETY                    ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Public solver contracts" begin
    @testset "Scalar setup and pivot failures" begin
        h = HelmoltzSolver(4)
        u, f = fill(123.0, 5), ones(5)
        # Construction allocates storage only. A solve must not turn an
        # uninitialised operator into plausible-looking zero coefficients.
        @test_throws ArgumentError solve!(h, u, f)
        @test all(==(123), u)
        for θ in ((0.0, 0.0), (Inf, 1.0), (1.0, NaN), (1.0, -6.0))
            # The last case is nonsingular but breaks the unpivoted UL path.
            @test_throws ArgumentError update!(h, θ...)
        end
        @test_throws ArgumentError solve!(h, u, f)
        @test all(==(123), u)
        # A successful update recovers the same storage after breakdown.
        @test update!(h, 1.0, 2.0) === h
        @test all(isfinite, solve!(h, u, f))
        h.Be.dᵢ[1] = NaN
        @test_throws ArgumentError solve!(h, u, f)
    end

    @testset "Precision rejected before output mutation" begin
        scalar = HelmoltzSolver(4)
        batch = BatchedHelmoltzSolver(4, 2)
        update!(scalar, 1.0, 1.0)
        update!(batch, ones(2), ones(2))
        for S in (Float32, ComplexF32, Int)
            u, f = fill(S(123), 5), ones(S, 5)
            @test_throws ArgumentError solve!(scalar, u, f)
            @test all(==(123), u)
            ub, fb = fill(S(123), 2, 5), ones(S, 2, 5)
            @test_throws ArgumentError solve!(batch, ub, fb, zeros(2), zeros(2))
            @test all(==(123), ub)
        end
        @test_throws ArgumentError solve!(scalar, zeros(ComplexF64, 5), ones(5))
    end

    @testset "Real factors with complex Fourier coefficients" begin
        for T in (Float32, Float64), P in (4, 5, 16, 17)
            h = HelmoltzSolver(P, T)
            update!(h, 0.7, 2.0)
            amplitude = Complex{T}(0.5, -0.3)
            f = coefficients(y -> amplitude*(-1.4-2*(1-y^2)), P, Complex{T})
            saved = copy(f)
            u = similar(f)
            solve!(h, u, f)
            @test u ≈ coefficients(y -> amplitude*(1-y^2), P, Complex{T}) atol=tolerance(T)
            @test f == saved
        end
    end

    @testset "Reference interval only" begin
        @test chebpoints(4) ≈ [1, sqrt(0.5), 0, -sqrt(0.5), -1]
        @test_throws ArgumentError chebpoints(0)
        @test_throws MethodError chebpoints(4; a=0, b=1)
        @test_throws MethodError HelmoltzSolver(4; a=0, b=1)
        @test_throws MethodError BatchedHelmoltzSolver(4, 2; a=0, b=1)
        @test_throws MethodError CoupledHelmoltzSolver(4; a=0, b=1)
    end
end
