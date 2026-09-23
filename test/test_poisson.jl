#//////////////////////////////////////////////////////////////////////////////#
#///                  SINGULAR NEUMANN POISSON AND GAUGE                     ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Compatible Neumann Poisson" begin
    # u=y³+y²-1/3 has zero interval mean and nonzero derivatives at both
    # walls. It checks the derivative sign convention as well as the gauge.
    for T in (Float32, Float64), S in (T, Complex{T}), P in (3, 4, 15, 16, 63, 64)
        @testset "$S, degree $P" begin
            θ₀ = T(0.7)
            amplitude = S <: Complex ? S(0.5, 0.3) : S(0.5)
            h = HelmoltzSolver(P, T; neum=true)
            @test update!(h, θ₀, zero(T)) === h
            f = zeros(S, P+1)
            f[1], f[2] = 2θ₀*amplitude, 6θ₀*amplitude
            u = similar(f)
            solve!(h, u, f, 5amplitude, amplitude)
            exact = zeros(S, P+1)
            exact[1:4] .= amplitude .* [1/6, 3/4, 1/2, 1/4]
            @test u ≈ exact atol=tolerance(T) rtol=tolerance(T)
            @test abs(sum(u[n+1]/(1-n^2) for n in 0:2:P)) < tolerance(T)
            @test diff(u, :right) ≈ 5amplitude atol=tolerance(T)
            @test diff(u, :left) ≈ amplitude atol=tolerance(T)

            # Tau forcing is deliberately excluded from compatibility too.
            f[end-1:end] .= S(100)
            solve!(h, u, f, 5amplitude, amplitude)
            @test u ≈ exact atol=tolerance(T) rtol=tolerance(T)
            # Incompatible retained forcing fails before changing the result.
            saved = copy(u)
            f[1] += one(S)
            @test_throws ArgumentError solve!(h, u, f, 5amplitude, amplitude)
            @test u == saved
            solve!(h, u, zeros(S, P+1))
            @test all(iszero, u)
        end
    end

    @testset "Mixed shifted and singular systems" begin
        for T in (Float32, Float64), S in (T, Complex{T}), P in (4, 5, 16, 17)
            B = 3
            h = BatchedHelmoltzSolver(P, B, T; neum=true)
            θ₀, θ₁ = T[0.5, 1, 2], T[0, 2, 0]
            update!(h, θ₀, θ₁)
            exact = zeros(S, B, P+1)
            f = similar(exact)
            amplitude = S <: Complex ? S(0.7, -0.2) : S(0.7)
            for s in 1:B
                exact[s, 1:4] .= amplitude .* [1/6, 3/4, 1/2, 1/4]
                f[s, :] .= -θ₁[s] .* exact[s, :]
                f[s, 1] += 2θ₀[s]*amplitude
                f[s, 2] += 6θ₀[s]*amplitude
            end
            saved = copy(f)
            u = similar(f)
            solve!(h, u, f, fill(5amplitude, B), fill(amplitude, B))
            @test u ≈ exact atol=tolerance(T) rtol=tolerance(T)
            @test f == saved
            f[3, 1] += one(S)
            before = copy(u)
            @test_throws ArgumentError solve!(h, u, f, fill(5amplitude, B), fill(amplitude, B))
            @test u == before
            # Updating back to shifted operators must clear the singular flags.
            update!(h, θ₀, ones(T, B))
            @test all(iszero, h.poisson)
            @test all(isfinite, solve!(h, u, f, zeros(S, B), zeros(S, B)))
        end
    end
end
