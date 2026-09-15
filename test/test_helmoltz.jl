@testset "Scalar Helmholtz" begin
    @test_throws ArgumentError HelmoltzSolver(1)
    ys = range(-1, 1; length=41)
    for T in (Float32, Float64, ComplexF64), P in (2, 3, 4, 5, 16, 17)
        @testset "$T, degree $P" begin
            h = HelmoltzSolver(P, T)
            γ = T <: Complex ? 0.25im : 0.0
            u(y) = 1 + 2y + 3y^2 + γ*(1-y^2)
            d2u(y) = 6 - 2γ
            for (θ₀, θ₁) in ((1.0, 0.0), (1.0, 2.0), (2.5, 0.1))
                update!(h, θ₀, θ₁)
                for scale in (1, -0.75)
                    rhs = coefficients(y -> scale*(θ₀*d2u(y)-θ₁*u(y)), P, T)
                    saved = copy(parent(rhs))
                    result = copy(rhs)
                    @test solve!(h, result, scale*real(u(1)), scale*real(u(-1))) === result
                    @test maximum(abs(evaluate(result, y)-scale*u(y)) for y in ys) < tolerance(T)
                    @test parent(rhs) == saved
                    # The two highest RHS coefficients are momentum tau terms.
                    rhs[P-1] += 10
                    rhs[P] -= 20
                    solve!(h, rhs, scale*real(u(1)), scale*real(u(-1)))
                    @test parent(rhs) ≈ parent(result) atol=tolerance(T) rtol=tolerance(T)
                end
            end
        end
    end

    # Smooth nonpolynomial solutions exercise the full spectral expansion,
    # including upper/lower wall ordering and a negative Helmholtz shift.
    for P in (24, 25)
        h = HelmoltzSolver(P)
        for (u, d2u, θ₀, θ₁) in ((exp, exp, 3.0, 2.0),
                                 (y -> sinpi(y), y -> -π^2*sinpi(y), 2.0, 2.0),
                                 (sin, y -> -sin(y), 2.0, -1.0))
            update!(h, θ₀, θ₁)
            rhs = coefficients(y -> θ₀*d2u(y)-θ₁*u(y), P)
            solve!(h, rhs, u(1.0), u(-1.0))
            @test maximum(abs(evaluate(rhs, y)-u(y)) for y in ys) < 2e-11
        end
    end
end
