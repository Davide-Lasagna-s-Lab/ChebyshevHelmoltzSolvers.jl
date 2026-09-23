#//////////////////////////////////////////////////////////////////////////////#
#///                   MANUFACTURED CLAMPED BATCHED SOLVES                   ///#
#//////////////////////////////////////////////////////////////////////////////#

function test_batched_coupled(to_backend=identity; backend="CPU")
    @testset "Batched coupled: $backend" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64), P in (8, 9, 24), B in (1, 67)
            R = typeof(real(zero(T)))
            h = to_backend(BatchedCoupledHelmoltzSolver(P, B, T))
            y = chebpoints(P)
            exact = zeros(T, B, P+1)
            f = similar(exact)
            for shift in (0, 1)
                θs = (fill(R(0.7), B), R[shift*(0.2+s/B) for s in 1:B],
                      fill(R(1.3), B), fill(R(shift*0.8), B))
                # u=(1-y²)² is clamped at both walls. Its derivatives are
                # independent polynomial references, not solver operations.
                for s in 1:B
                    scale = T <: Complex ? T(s/B + 0.2im) : T(s/B)
                    a,b,c,d = (θ[s] for θ in θs)
                    values = [scale*(24a*c - (a*d+b*c)*(-4+12t^2) + b*d*(1-t^2)^2) for t in y]
                    f[s, :] .= chebcoeffs(values)
                    exact[s, 1] = scale*3/8
                    exact[s, 3] = -scale/2
                    exact[s, 5] = scale/8
                end
                @test update!(h, map(to_backend, θs)) === h
                responses = map(Array, h.vₛ[2:3])
                A = Array(h.A)
                src = to_backend(f)
                dest = similar(src)
                @test solve!(h, dest, src) === dest
                result = Array(dest)
                tol = R === Float32 ? 2e-4 : 2e-11
                @test result ≈ exact rtol=tol atol=tol
                @test Array(src) == f
                @test map(Array, h.vₛ[2:3]) == responses
                @test Array(h.A) == A
                # Check both wall values and derivatives, including the lower
                # wall parity sign. These are the influence method's purpose.
                for s in 1:B
                    @test abs(sum(result[s, :])) < tol
                    @test abs(sum((-1)^n*result[s, n+1] for n in 0:P)) < tol
                    @test abs(sum(n^2*result[s, n+1] for n in 1:P)) < 10tol
                    @test abs(sum((-1)^(n+1)*n^2*result[s, n+1] for n in 1:P)) < 10tol
                end
                @test_throws ArgumentError solve!(h, src, src)
                @test_throws ArgumentError solve!(h, h.vₛ[1], src)
                @test_throws DimensionMismatch solve!(h, dest[:, 1:end-1], src)
            end
        end
    end
end

test_batched_coupled()
