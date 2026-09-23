#//////////////////////////////////////////////////////////////////////////////#
#///                  SHARED CPU AND CUDA HELMHOLTZ TESTS                   ///#
#//////////////////////////////////////////////////////////////////////////////#

# External storage-free boundary vector exercises the AbstractVector contract.
struct ZeroBoundary{T} <: AbstractVector{T}
    n::Int
end
Base.size(z::ZeroBoundary) = (z.n,)
Base.getindex(::ZeroBoundary{T}, i::Int) where {T} = zero(T)
Base.IndexStyle(::Type{<:ZeroBoundary}) = IndexLinear()

# The numerical cases are shared by the CPU suite and the optional CUDA suite.
# `to_backend` transfers both factors and data; it is the identity on the CPU.
# Every reference is constructed on the host, so the GPU checks compare against
# known polynomial solutions rather than against another GPU implementation.
function test_batched_helmholtz(to_backend=identity; backend="CPU")
    @testset "Batched Helmholtz: $backend" begin
        for T in (Float32, Float64), S in (T, Complex{T}), P in (3, 4, 16, 17),
            neum in (false, true)
            @testset "$S, degree $P, neum=$neum" begin
                # Both odd and even degrees are needed because the parity
                # blocks have different lengths when P is even.
                B, Ny = 6, P + 1
                leading = iseven(P) ? (2, 3) : (B,)
                theta0 = T[0.2 + 0.1s for s = 1:B]
                theta1 = T[0.5 + 0.2s^2 for s = 1:B]
                exact = zeros(S, B, Ny)
                source = similar(exact)

                for s = 1:B
                    # Retain every Chebyshev degree, with independent real and
                    # imaginary parts. A decaying spectrum keeps Float32 wall
                    # derivatives well scaled while still exercising all rows.
                    a = S[(-1)^n*(1 + 0.1s)/(n+1)^4 for n = 0:P]
                    if S <: Complex
                        for n = 0:P
                            a[n+1] += im*T((s + n)/(10*(n+1)^4))
                        end
                    end
                    # Enforce homogeneous walls by correcting only the lowest
                    # degrees. Higher degrees still exercise the full operator;
                    # complex coefficients are allowed, but wall data are real.
                    if neum
                        right = sum(n^2*a[n+1] for n = 1:P)
                        left = sum((-1)^(n+1)*n^2*a[n+1] for n = 1:P)
                        a[2] -= (right + left)/2
                        a[3] -= (right - left)/8
                    else
                        right = sum(a)
                        left = sum((-1)^n*a[n+1] for n = 0:P)
                        a[1] -= (right + left)/2
                        a[2] -= (right - left)/2
                    end
                    d2a = diff2!(copy(a))
                    exact[s, :] .= a
                    source[s, :] .= theta0[s].*d2a .- theta1[s].*a
                end

                h = BatchedHelmoltzSolver(P, B, T; neum)
                update!(h, theta0, theta1)
                cached = (copy(h.Be.dᵢ), copy(h.Bo.dᵢ))
                h = to_backend(h)
                @test (Array(h.Be.dᵢ), Array(h.Bo.dᵢ)) == cached
                rhs = to_backend(reshape(source, leading..., Ny))
                dest = similar(rhs)
                u, f = reshape(dest, B, Ny), reshape(rhs, B, Ny)
                tol = T === Float32 ? 5e-4 : 3e-11

                # A distinct output array lets the solver assemble directly in
                # caller-supplied matrix views while preserving the physical RHS.
                @test solve!(h, u, f, ZeroBoundary{S}(B), ZeroBoundary{S}(B)) === u
                result = reshape(Array(dest), B, Ny)
                @test result ≈ exact atol=tol rtol=tol
                @test reshape(Array(rhs), B, Ny) == source

                # Compare independently against the existing scalar solver too.
                # Real and imaginary parts share homogeneous boundary data;
                # the scalar solver uses real coefficient storage.
                reference = similar(exact)
                for s = 1:B
                    scalar = HelmoltzSolver(P, T; neum)
                    update!(scalar, theta0[s], theta1[s])
                    realpart = T.(real.(source[s, :]))
                    solve!(scalar, realpart, copy(realpart), 0, 0)
                    reference[s, :] .= realpart
                    if S <: Complex
                        imagpart = T.(imag.(source[s, :]))
                        solve!(scalar, imagpart, copy(imagpart), 0, 0)
                        reference[s, :] .+= im.*imagpart
                    end
                end
                @test result ≈ reference atol=tol rtol=tol

                # The last two momentum equations are replaced by wall data.
                # Changing only those RHS coefficients must therefore leave the
                # solution unchanged, including for unequal parity block sizes.
                tau_source = copy(source)
                tau_source[:, end-1] .+= S(13)
                tau_source[:, end] .-= S(7)
                tau_rhs = to_backend(reshape(tau_source, leading..., Ny))
                solve!(h, u, reshape(tau_rhs, B, Ny), ZeroBoundary{S}(B), ZeroBoundary{S}(B))
                @test reshape(Array(dest), B, Ny) ≈ result atol=tol rtol=tol
                @test (Array(h.Be.dᵢ), Array(h.Bo.dᵢ)) == cached
            end
        end

        @testset "Per-system wall data" begin
            # Manufacture u_s(y)=a_s+b_s*y+c_s*y². Its values and derivatives
            # at the walls are analytic and differ between systems, including
            # their imaginary parts. Pass one boundary entry per system.
            P, B = 6, 6
            theta0, theta1 = fill(0.7, B), collect(1.0:B)
            for neum in (false, true), shared_upper in (false, true)
                h = BatchedHelmoltzSolver(P, B; neum)
                update!(h, theta0, theta1)
                a = ComplexF64[0.1s + 0.2im for s = 1:B]
                b = ComplexF64[0.2s - 0.1im*s for s = 1:B]
                c = ComplexF64[0.3 + 0.05im*s for s = 1:B]
                if shared_upper
                    # Keep the upper datum fixed while the lower one varies.
                    neum ? (b .= 1 .- 2c) : (a .= 1 .- b .- c)
                end
                upper = neum ? b .+ 2c : a .+ b .+ c
                lower = neum ? b .- 2c : a .- b .+ c
                exact = zeros(ComplexF64, B, P+1)
                exact[:, 1] .= a .+ c/2
                exact[:, 2] .= b
                exact[:, 3] .= c/2
                source = -theta1 .* exact
                source[:, 1] .+= 2theta0 .* c
                rhs = to_backend(reshape(source, 2, 3, P+1))
                dest = similar(rhs)
                u₊ = to_backend(upper)
                u₋ = to_backend(lower)
                solve!(to_backend(h), reshape(dest, B, P+1), reshape(rhs, B, P+1), u₊, u₋)
                @test reshape(Array(dest), B, P+1) ≈ exact atol=3e-11
                @test reshape(Array(rhs), B, P+1) == source
                @test vec(Array(u₋)) == lower
                shared_upper || @test vec(Array(u₊)) == upper
            end
        end

        @testset "Uniform viscosity and uniform wall values" begin
            # Uniform viscosity and uniform boundary data are useful when every
            # Fourier mode shares the same wall specification. The operators
            # still differ through theta1, so broadcasting a single solution
            # across systems would not suffice to solve these distinct RHSs.
            P, B = 6, 5
            theta0, theta1 = fill(0.7, B), collect(1.0:B)
            for neum in (false, true)
                h = BatchedHelmoltzSolver(P, B; neum)
                update!(h, theta0, theta1)
                h = to_backend(h)
                a = zeros(P+1)
                a[1], a[2], a[3] = 1.5, 1.0, 0.5 # u = 1 + y + y^2
                d2a = diff2!(copy(a))
                source = [theta0[s]*d2a[n+1] - theta1[s]*a[n+1] for s = 1:B, n = 0:P]
                rhs = to_backend(source)
                dest = similar(rhs)
                upper, lower = neum ? (3.0, -1.0) : (3.0, 1.0)
                solve!(h, dest, rhs, to_backend(fill(upper, B)), to_backend(fill(lower, B)))
                @test Array(dest) ≈ repeat(reshape(a, 1, :), B, 1)
                @test Array(rhs) == source
            end

            # Zero-valued boundary vectors impose homogeneous Dirichlet data. The
            # polynomial 1-y^2 satisfies those walls and includes the constant
            # Fourier component, so this also checks the lazy boundary vectors.
            h = BatchedHelmoltzSolver(P, B)
            update!(h, theta0, theta1)
            h = to_backend(h)
            a = zeros(P+1)
            a[1], a[3] = 0.5, -0.5
            d2a = diff2!(copy(a))
            source = [theta0[s]*d2a[n+1] - theta1[s]*a[n+1] for s = 1:B, n = 0:P]
            rhs = to_backend(source)
            dest = similar(rhs)
            solve!(h, dest, rhs, ZeroBoundary{Float64}(B), ZeroBoundary{Float64}(B))
            @test Array(dest) ≈ repeat(reshape(a, 1, :), B, 1)
        end
    end
end

test_batched_helmholtz()

#//////////////////////////////////////////////////////////////////////////////#
#///                     CPU UPDATE AND ARGUMENT TESTS                      ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Batched Helmholtz CPU updates and argument checks" begin
    # Updating an existing solver must overwrite both parity blocks in place
    # and all mode-dependent coefficients. The same destination is reused,
    # confirming that the solve never depends on its previous contents.
    P, B = 8, 3
    h = BatchedHelmoltzSolver(P, B)
    a = zeros(P+1)
    a[1], a[3], a[5] = 3/8, -1/2, 1/8 # (1-y^2)^2
    d2a = diff2!(copy(a))
    dest = zeros(B, P+1)
    factor_storage = (h.Be.b, h.Be.l, h.Be.dᵢ, h.Be.u,
                      h.Bo.b, h.Bo.l, h.Bo.dᵢ, h.Bo.u)
    # Fresh storage has no assembled operators or UL factors before update!.
    @test all(a -> all(iszero, a), factor_storage)
    for (theta0, theta1) in ((fill(0.2, B), [0.3, 0.7, 1.1]),
                            ([0.5, 1.0, 1.5], [2.0, 3.0, 4.0]))
        # Views may select coefficients from a larger table. Updating must
        # write into the original factor storage, without replacing the batch.
        coefficients = hcat(theta0, theta1)
        @test update!(h, view(coefficients, :, 1), view(coefficients, :, 2)) === h
        @test all(a === b for (a, b) in zip(factor_storage,
            (h.Be.b, h.Be.l, h.Be.dᵢ, h.Be.u,
             h.Bo.b, h.Bo.l, h.Bo.dᵢ, h.Bo.u)))
        cached = (copy(h.Be.dᵢ), copy(h.Bo.dᵢ))
        rhs = [theta0[s]*d2a[n+1] - theta1[s]*a[n+1] for s = 1:B, n = 0:P]
        fill!(dest, NaN)
        solve!(h, dest, rhs, ZeroBoundary{Float64}(B), ZeroBoundary{Float64}(B))
        @test dest ≈ repeat(reshape(a, 1, :), B, 1) atol=3e-11
        @test (h.Be.dᵢ, h.Bo.dᵢ) == cached
    end

    # Both array shape and non-aliasing are part of the public contract: the
    # assembly uses neighboring forcing coefficients before the substitutions
    # overwrite the destination. Overlapping views are as unsafe as f === u.
    rhs = zeros(B, P+1)
    @test_throws ArgumentError solve!(h, rhs, rhs, zeros(B), zeros(B))
    overlapping = zeros(B, P+2)
    @test_throws ArgumentError solve!(h, view(overlapping, :, 1:P+1),
                                     view(overlapping, :, 2:P+2), zeros(B), zeros(B))
    @test_throws DimensionMismatch solve!(h, zeros(B, P), rhs, zeros(B), zeros(B))
    @test_throws DimensionMismatch solve!(h, zeros(B+1, P+1), zeros(B+1, P+1), zeros(B), zeros(B))
    @test_throws DimensionMismatch solve!(h, dest, rhs, zeros(B+1), zeros(B))
    @test_throws DimensionMismatch solve!(h, dest, rhs, zeros(B), zeros(B+1))
    # The matrix and boundary-vector API is explicit: no internal reshape,
    # no scalar wall broadcasting, and no omitted boundary defaults.
    @test_throws MethodError solve!(h, dest, rhs)
    @test_throws MethodError solve!(h, dest, rhs, 0, 0)
    @test_throws MethodError solve!(h, reshape(dest, 1, B, P+1),
                                      reshape(rhs, 1, B, P+1), zeros(B), zeros(B))
    @test_throws ArgumentError solve!(h, dest, rhs, view(dest, :, 1), zeros(B))


    # Construction only validates dimensions and allocates storage; coefficient
    # checks and factorisation happen when update! first initialises the solver.
    @test_throws ArgumentError BatchedHelmoltzSolver(1, B)
    @test_throws ArgumentError BatchedHelmoltzSolver(2, B)
    @test_throws ArgumentError BatchedHelmoltzSolver(P, 0)
    @test_throws DimensionMismatch update!(h, ones(B-1), ones(B))
    @test_throws DimensionMismatch update!(h, ones(B), ones(B-1))

    # Pure Neumann Poisson problems need a separate compatibility condition and
    # pressure gauge, so updating must reject their singular factors.
    neum = BatchedHelmoltzSolver(P, 2; neum=true)
    @test_throws ArgumentError update!(neum, ones(2), [1.0, 0.0])
    # A scalar viscosity is deliberately outside the batched update! API.
    @test_throws MethodError update!(h, 1.0, ones(B))
    # Factor storage is being updated in place, so it cannot also supply the
    # coefficients needed by the still-unassembled parity systems.
    @test_throws ArgumentError update!(h, view(h.Be.b, :, 1), ones(B))
    @test_throws ArgumentError update!(h, ones(B), view(h.Bo.b, :, 1))
    @test_throws ArgumentError update!(h, view(h.Be.dᵢ, :, 1), ones(B))
    @test_throws ArgumentError update!(h, ones(B), view(h.Bo.dᵢ, :, 1))
    @test_throws ArgumentError update!(h, zeros(B), zeros(B))

    # A finite, nonzero pivot can still have an unrepresentable reciprocal.
    tiny = BatchedHelmoltzSolver(3, 1)
    @test_throws ArgumentError update!(tiny, [nextfloat(0.0)], [0.0])
    for Q in (tiny.Be, tiny.Bo)
        @test any(x -> !isfinite(x), Q.dᵢ)
    end
end
