#//////////////////////////////////////////////////////////////////////////////#
#///                   COUPLED SOLVE AND INFLUENCE TESTS                    ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Coupled Helmholtz and cached influence" begin
    ys = range(-1, 1; length=41)
    for T in (Float32, Float64), P in (4, 5, 8, 9, 16, 17)
        @testset "$T, degree $P" begin
            solver = CoupledHelmoltzSolver(P, T)
            α = P >= 5 ? 0.2 : 0.0
            v(y) = (1-y^2)^2*(1+α*y)
            d2v(y) = -4 - 12α*y + 12y^2 + 20α*y^3
            d4v(y) = 24 + 120α*y
            for θs in ((1.0, 2.0, 3.0, 4.0), (0.75, 0.0, 1.25, 0.0),
                       (2.0, 0.4, 0.8, 1.5))
                θ₀, θ₁, θ₂, θ₃ = θs
                update!(solver, θs)
                plus = copy(solver.vₛ[2])
                minus = copy(solver.vₛ[3])
                influence = copy(solver.A)
                for scale in (T(1), T <: Complex ? T(-0.5 + 0.3im) : T(-0.5))
                    r(y) = scale*(θ₀*θ₂*d4v(y) -
                                   (θ₀*θ₃+θ₁*θ₂)*d2v(y) + θ₁*θ₃*v(y))
                    rhs = coefficients(r, P, T)
                    @test solve!(solver, rhs) === rhs
                    @test maximum(abs(evaluate(rhs, y)-scale*v(y)) for y in ys) < tolerance(T)
                    @test abs(evaluate(rhs, 1.0)) < tolerance(T)
                    @test abs(evaluate(rhs, -1.0)) < tolerance(T)
                    @test abs(diff(rhs, :left)) < tolerance(T)
                    @test abs(diff(rhs, :right)) < tolerance(T)
                    @test solver.vₛ[2] == plus
                    @test solver.vₛ[3] == minus
                    @test solver.A == influence
                end
            end
        end
    end

    for P in (24, 25)
        solver = CoupledHelmoltzSolver(P)
        θs = (0.7, 1.1, 1.3, 0.8)
        θ₀, θ₁, θ₂, θ₃ = θs
        update!(solver, θs)
        rhs = coefficients(y -> (θ₀*θ₂*π^4 + (θ₀*θ₃+θ₁*θ₂)*π^2)*cospi(y) +
                                θ₁*θ₃*(cospi(y)+1), P)
        solve!(solver, rhs)
        @test maximum(abs(evaluate(rhs, y)-(cospi(y)+1)) for y in ys) < 2e-11
    end
end
