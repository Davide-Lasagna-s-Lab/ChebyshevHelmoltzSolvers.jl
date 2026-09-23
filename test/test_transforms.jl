#//////////////////////////////////////////////////////////////////////////////#
#///                       CHEBYSHEV TRANSFORM TESTS                        ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Chebyshev transforms" begin
    # Check every basis polynomial, including both endpoint coefficients.
    # Independent cosine samples expose normalization errors that could
    # cancel between a forward and inverse round trip.
    for P in (1, 7, 16), n = 0:P
        values = [cospi(n*j/P) for j = 0:P]
        saved = copy(values)
        a = chebcoeffs(values)
        expected = zeros(P+1)
        expected[n+1] = 1
        @test a ≈ expected atol=2e-13
        @test values == saved
        coefficients_before = copy(a)
        @test chebvalues(a) ≈ values atol=2e-13
        @test a == coefficients_before
    end
    @test chebvalues([2.0]) == [2.0]

    # Reproduce the README solves end to end: sampled source -> coefficients
    # -> tau solution -> physical values, compared with analytic polynomials.
    P = 16
    y = chebpoints(P)
    h = HelmoltzSolver(P)
    update!(h, 1.0, 4.0)
    rhs = chebcoeffs(-6 .+ 4 .* y.^2)
    solve!(h, rhs, copy(rhs), 0.0, 0.0)
    @test chebvalues(rhs) ≈ 1 .- y.^2 atol=2e-12

    h = CoupledHelmoltzSolver(P)
    update!(h, (1.0, 0.0, 1.0, 0.0))
    rhs = chebcoeffs(fill(24.0, P+1))
    solve!(h, rhs, copy(rhs))
    @test chebvalues(rhs) ≈ (1 .- y.^2).^2 atol=2e-12
end
