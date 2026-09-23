#//////////////////////////////////////////////////////////////////////////////#
#///                     ANALYTIC HELMHOLTZ SOLVE TESTS                     ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Scalar Helmholtz" begin
    @test_throws ArgumentError HelmoltzSolver(1)
    @test_throws ArgumentError HelmoltzSolver(2)
    ys = range(-1, 1; length=41)
    for T in (Float32, Float64), P in (3, 4, 5, 16, 17)
        @testset "$T, degree $P" begin
            h = HelmoltzSolver(P, T)
            u(y) = 1 + 2y + 3y^2
            d2u(y) = 6
            for (θ₀, θ₁) in ((1.0, 0.0), (1.0, 2.0), (2.5, 0.1))
                update!(h, θ₀, θ₁)
                for scale in (1, -0.75)
                    rhs = coefficients(y -> scale*(θ₀*d2u(y)-θ₁*u(y)), P, T)
                    saved = copy(rhs)
                    result = similar(rhs)
                    @test_throws ArgumentError solve!(h, rhs, rhs)
                    @test solve!(h, result, rhs, scale*real(u(1)), scale*real(u(-1))) === result
                    @test maximum(abs(evaluate(result, y)-scale*u(y)) for y in ys) < tolerance(T)
                    @test rhs == saved
                    # The two highest RHS coefficients are momentum tau terms.
                    rhs[P] += 10
                    rhs[P+1] -= 20
                    solve!(h, rhs, copy(rhs), scale*real(u(1)), scale*real(u(-1)))
                    @test rhs ≈ result atol=tolerance(T) rtol=tolerance(T)
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
            solve!(h, rhs, copy(rhs), u(1.0), u(-1.0))
            @test maximum(abs(evaluate(rhs, y)-u(y)) for y in ys) < 2e-11
        end
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                           FACTOR REUSE TESTS                           ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Scalar Helmholtz reciprocal pivot updates" begin
    P = 8
    h = HelmoltzSolver(P, Float64)
    caches = (h.Be.dᵢ, h.Bo.dᵢ)
    previous = map(copy, caches)
    for (theta0, theta1) in ((0.5, 1.0), (1.5, 3.0))
        update!(h, theta0, theta1)
        @test h.Be.dᵢ === caches[1] && h.Bo.dᵢ === caches[2]
        for (Q, old) in zip((h.Be, h.Bo), previous)
            @test Q.dᵢ != old
            old .= Q.dᵢ
        end
        rhs = coefficients(y -> -2theta0 - theta1*(1-y^2), P)
        solve!(h, rhs, copy(rhs), 0.0, 0.0)
        @test maximum(abs(evaluate(rhs, y) - (1-y^2)) for y in range(-1, 1; length=21)) < 2e-11
        @test (h.Be.dᵢ, h.Bo.dᵢ) == previous
    end
end

@testset "Helmholtz coefficient views" begin
    # u=1-y² solves u''-u=y²-3 with homogeneous walls. Source and
    # destination use separate strided storage, without coefficient wrappers.
    h = HelmoltzSolver(6)
    update!(h, 1.0, 1.0)
    src = view(zeros(14), 1:2:13)
    dest = view(zeros(14), 2:2:14)
    src[1], src[3] = -2.5, 0.5
    saved = copy(src)
    @test solve!(h, dest, src) === dest
    @test dest ≈ [0.5, 0, -0.5, 0, 0, 0, 0]
    @test src == saved
    @test_throws DimensionMismatch solve!(h, zeros(6), src)
end
