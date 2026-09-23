#//////////////////////////////////////////////////////////////////////////////#
#///                          MAPPED PHYSICAL DOMAINS                        ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Physical intervals" begin
    # Shifted intervals of different lengths test translation and derivative
    # scaling independently. Nonzero Neumann data detect missing wall scaling.
    for (a, b) in ((2.0, 5.0), (-3.0, -2.0)), T in (Float32, Float64)
        P = 12
        y = chebpoints(P; a, b)
        @test first(y) == b
        @test last(y) == a
        @test all(diff(y) .< 0)
        exact = chebcoeffs(T.(1 .+ y .+ y.^3))
        θ₀, θ₁ = T(0.7), T(2)
        f = chebcoeffs(T.(θ₀ .* (6 .* y) .- θ₁ .* (1 .+ y .+ y.^3)))
        for neum in (false, true)
            h = HelmoltzSolver(P, T; a, b, neum)
            update!(h, θ₀, θ₁)
            walls = neum ? (1+3b^2, 1+3a^2) : (1+b+b^3, 1+a+a^3)
            u = similar(f)
            solve!(h, u, f, walls...)
            @test u ≈ exact rtol=tolerance(T) atol=tolerance(T)
        end

        # A quartic with double zeros at both physical endpoints satisfies
        # the coupled boundary conditions; its fourth derivative is 24.
        h = CoupledHelmoltzSolver(P, T; a, b)
        update!(h, (one(T), zero(T), one(T), zero(T)))
        f4 = chebcoeffs(fill(T(24), P+1))
        solve!(h, f4)
        @test chebvalues(f4) ≈ (y .- a).^2 .* (y .- b).^2 rtol=tolerance(T) atol=tolerance(T)
    end

    # Reject maps that collapse, reverse, or have nonfinite endpoints.
    for (a, b) in ((1, 1), (2, -1), (NaN, 1), (-1, Inf))
        @test_throws ArgumentError chebpoints(8; a, b)
        @test_throws ArgumentError HelmoltzSolver(8; a, b)
        @test_throws ArgumentError BatchedHelmoltzSolver(8, 4; a, b)
        @test_throws ArgumentError CoupledHelmoltzSolver(8; a, b)
    end
end
