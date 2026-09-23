#//////////////////////////////////////////////////////////////////////////////#
#///                 INDEPENDENT TAU REFERENCE AND ACCURACY                 ///#
#//////////////////////////////////////////////////////////////////////////////#

# An independent dense coefficient differentiation matrix follows directly
# from T_m′: entries are 2m at opposite-parity degrees below m, except m at
# degree zero. Squaring it avoids the solver's integrated recurrence entirely.
function dense_tau(P, θ₀, θ₁; neum=false)
    D = [n < m && isodd(m-n) ? (n == 0 ? m : 2m) : 0.0 for n in 0:P, m in 0:P]
    A = θ₀*(D*D) - θ₁*I
    A[P, :] = neum ? [n^2 for n in 0:P] : ones(P+1)
    A[P+1, :] = neum ? [(-1)^(n+1)*n^2 for n in 0:P] : [(-1)^n for n in 0:P]
    return A
end

@testset "Numerical accuracy" begin
    @testset "Independent dense tau reference" begin
        for P in (4, 5, 16, 17, 32, 64), neum in (false, true), θ₁ in (0.001, 1.0, 1e6, -0.1)
            # All degrees of the forcing are populated, including tau entries.
            # Complex data reproduce Fourier columns without splitting real/imag.
            f = ComplexF64[sin(n+1) + im*cos(2n+1) for n in 0:P]
            up, lo = 0.3+0.2im, -0.1+0.4im
            rhs = copy(f)
            rhs[P:P+1] .= (up, lo)
            A = dense_tau(P, 0.7, θ₁; neum)
            reference = A \ rhs
            h = HelmoltzSolver(P; neum)
            update!(h, 0.7, θ₁)
            u = similar(f)
            solve!(h, u, f, up, lo)
            @test u ≈ reference rtol=2e-8 atol=2e-9
            # A normalized backward residual distinguishes conditioning from
            # a wrong discretisation, especially for small Neumann shifts.
            @test norm(A*u-rhs, Inf)/(norm(A, Inf)*norm(u, Inf)+norm(rhs, Inf)) < 1e-12
        end
    end

    @testset "Spectral convergence and high degree" begin
        errors = Float64[]
        for P in (8, 16, 32, 64, 128, 256, 512)
            # exp(y) excites every Chebyshev degree. Compare at independent
            # off-grid points; increasing P should converge then reach roundoff.
            h = HelmoltzSolver(P)
            update!(h, 1.0, 2.0)
            f = coefficients(y -> -exp(y), P)
            u = similar(f)
            solve!(h, u, f, exp(1), exp(-1))
            push!(errors, maximum(abs(evaluate(u, y)-exp(y)) for y in range(-1, 1; length=137)))
        end
        @test errors[2] < errors[1]/1000
        @test maximum(errors[2:end]) < 2e-11
    end

    @testset "Nearly singular shifted Neumann operator" begin
        # A constant solution isolates sensitivity of the mean coefficient.
        # For θ₁>0 its value is determined by the equation, not a zero-mean gauge.
        for θ₁ in (1e-2, 1e-6, 1e-10)
            h = HelmoltzSolver(32; neum=true)
            update!(h, 1.0, θ₁)
            f = zeros(33); f[1] = -θ₁
            u = similar(f)
            solve!(h, u, f)
            @test u[1] ≈ 1 rtol=1e-12
            @test maximum(abs, u[2:end]) < 1e-12
        end
    end

    @testset "Transforms preserve precision and complex data" begin
        for T in (Float32, Float64), S in (T, Complex{T}), P in (3, 16, 31)
            values = S[sin(y)+ (S <: Complex ? im*cos(y) : 0) for y in chebpoints(P)]
            saved = copy(values)
            a = chebcoeffs(values)
            @test eltype(a) == S
            @test chebvalues(a) ≈ values atol=tolerance(T) rtol=tolerance(T)
            @test values == saved
        end
    end
end
