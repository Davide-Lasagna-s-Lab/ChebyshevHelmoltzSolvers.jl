#//////////////////////////////////////////////////////////////////////////////#
#///                     INDEPENDENT REFERENCE HELPERS                      ///#
#//////////////////////////////////////////////////////////////////////////////#

# Build the dense reference independently of QuasiTridiagonal indexing and
# packing. Every system and coefficient differs, exposing transposed storage
# and accidental reuse of one Fourier mode's factors.
function batched_manufactured_system(::Type{T}, M, system) where {T}
    A = zeros(T, M, M)
    for j = 1:M
        A[1, j] = T((-1)^j * (0.08 + 0.001system + 0.003j) / M)
    end
    A[1, 1] = T(4 + 0.005system)
    for j = 2:M
        A[j, j-1] = T(-0.2 - 0.001system - 0.002j)
        A[j, j] = T(2 + 0.003system + 0.01j)
        j < M && (A[j, j+1] = T(0.13 + 0.002system + 0.004j))
    end
    Q = QuasiTridiagonal(collect(A[1, :]),
                        T[A[j, j-1] for j = 2:M],
                        T[A[j, j] for j = 2:M],
                        T[A[j, j+1] for j = 2:M-1])
    return A, ul!(Q)
end

function batched_known_solution(::Type{S}, nsystems, M, scale) where {S}
    return S[scale * (sin(0.2j) + 0.03system +
                      (S <: Complex ? (0.2cos(0.3j) - 0.01system)*im : 0))
             for system = 1:nsystems, j = 1:M]
end

# Rows are separate systems, so form each dense product explicitly rather
# than relying on the same layout assumptions as the batched implementation.
function batched_dense_rhs(matrices, solution)
    rhs = similar(solution)
    for system = axes(solution, 1)
        rhs[system, :] = matrices[system] * solution[system, :]
    end
    return rhs
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            BATCHED UL TESTS                            ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "Batched quasi-tridiagonal UL" begin
    for T in (Float32, Float64), M in (2, 3, 17, 33), nsystems in (1, 7, 67)
        @testset "$T, size $M, systems $nsystems" begin
            systems = [batched_manufactured_system(T, M, s) for s = 1:nsystems]
            matrices = first.(systems)
            factors = last.(systems)
            original = [map(copy, (Q.b, Q.l, Q.dᵢ, Q.u)) for Q in factors]
            batch = BatchedQuasiTridiagonal(length(factors), size(first(factors), 1), eltype(first(factors)))
            for (s, Q) in enumerate(factors), field in (:b, :l, :dᵢ, :u)
                @views getproperty(batch, field)[s, :] .= getproperty(Q, field)
            end
            packed = map(copy, (batch.b, batch.l, batch.dᵢ, batch.u))

            for (field, width) in ((:b, M), (:l, M-1), (:dᵢ, M-1), (:u, M-2))
                @test size(getproperty(batch, field)) == (nsystems, width)
                for s = 1:nsystems
                    @test getproperty(batch, field)[s, :] == getproperty(factors[s], field)
                end
            end
            @test [map(copy, (Q.b, Q.l, Q.dᵢ, Q.u)) for Q in factors] == original

            # Reuse each pack for both real and Fourier-complex RHS, with
            # different data on every solve.
            for S in (T, Complex{T}), scale in (1, -0.75)
                expected = batched_known_solution(S, nsystems, M, scale)
                rhs = batched_dense_rhs(matrices, expected)
                saved = copy(rhs)
                @test ldiv!(batch, rhs) === rhs
                @test rhs ≈ expected rtol=tolerance(T) atol=tolerance(T)
                for s = 1:nsystems
                    @test matrices[s] * rhs[s, :] ≈ saved[s, :] rtol=tolerance(T) atol=tolerance(T)
                end
                @test (batch.b, batch.l, batch.dᵢ, batch.u) == packed
            end
            @test [map(copy, (Q.b, Q.l, Q.dᵢ, Q.u)) for Q in factors] == original

            # Copying into independent storage keeps subsequent scalar-factor
            # updates from changing the batch.
            for Q in factors, field in (:b, :l, :dᵢ, :u)
                getproperty(Q, field) .+= one(T)
            end
            @test (batch.b, batch.l, batch.dᵢ, batch.u) == packed
        end
    end

    @testset "Factor original matrices in place" begin
        # Build unfactorised batches directly from the independent dense
        # matrices. ul! must operate on the wrapped arrays and leave a batch
        # that solves these original systems.
        for T in (Float32, Float64), M in (2, 17)
            nsystems = 7
            matrices = [first(batched_manufactured_system(T, M, s)) for s = 1:nsystems]
            b = T[A[1, j] for A in matrices, j = 1:M]
            l = T[A[j, j-1] for A in matrices, j = 2:M]
            d = T[A[j, j] for A in matrices, j = 2:M]
            u = T[A[j, j+1] for A in matrices, j = 2:M-1]
            batch = BatchedQuasiTridiagonal(b, l, d, u)
            cache = batch.dᵢ
            @test cache === d
            @test ul!(batch) === batch
            @test batch.dᵢ === cache
            wrapped = BatchedQuasiTridiagonal(b, l, d, u)
            @test all(a === b for (a, b) in zip((b, l, d, u),
                (wrapped.b, wrapped.l, wrapped.dᵢ, wrapped.u)))
            @test batch.b === b && batch.l === l && batch.dᵢ === d && batch.u === u
            exact = batched_known_solution(Complex{T}, nsystems, M, 1)
            rhs = batched_dense_rhs(matrices, exact)
            ldiv!(wrapped, rhs)
            @test rhs ≈ exact rtol=tolerance(T) atol=tolerance(T)
        end
    end

    @testset "Views and input validation" begin
        # Array adaptation uses the four-array constructor. Invalid factor
        # array shapes must be rejected before unchecked substitutions.
        @test_throws ArgumentError BatchedQuasiTridiagonal(zeros(0, 2), zeros(0, 1), zeros(0, 1), zeros(0, 0))
        @test_throws ArgumentError BatchedQuasiTridiagonal(zeros(1, 0), zeros(1, 0), zeros(1, 0), zeros(1, 0))
        @test_throws ArgumentError BatchedQuasiTridiagonal(zeros(1, 1), zeros(1, 0), zeros(1, 0), zeros(1, 0))
        @test_throws DimensionMismatch BatchedQuasiTridiagonal(zeros(2, 3), zeros(2, 1), zeros(2, 2), zeros(2, 1))
        @test_throws DimensionMismatch BatchedQuasiTridiagonal(zeros(2, 3), zeros(2, 2), zeros(2, 2), zeros(2, 2))

        for T in (Float32, Float64), S in (T, Complex{T})
            nsystems, M = 7, 3
            systems = [batched_manufactured_system(T, M, s) for s = 1:nsystems]
            matrices = first.(systems)
            factors = last.(systems)
            batch = BatchedQuasiTridiagonal(length(factors), size(first(factors), 1), eltype(first(factors)))
            for (s, Q) in enumerate(factors), field in (:b, :l, :dᵢ, :u)
                @views getproperty(batch, field)[s, :] .= getproperty(Q, field)
            end
            expected = batched_known_solution(S, nsystems, M, 1)
            rhs = batched_dense_rhs(matrices, expected)

            # A gap between coefficient columns is legal; only the first
            # dimension must be contiguous. Unselected columns are sentinels.
            storage = fill(S(123), nsystems, 2M)
            view_rhs = @view storage[:, 1:2:2M]
            view_rhs .= rhs
            @test ldiv!(batch, view_rhs) === view_rhs
            @test view_rhs ≈ expected rtol=tolerance(T) atol=tolerance(T)
            @test all(==(S(123)), @view storage[:, 2:2:2M])

            @test_throws DimensionMismatch ldiv!(batch, zeros(S, nsystems+1, M))
            @test_throws DimensionMismatch ldiv!(batch, zeros(S, nsystems, M+1))
            strided_storage = zeros(S, 2nsystems, M)
            @test_throws ArgumentError ldiv!(batch, @view strided_storage[1:2:2nsystems, :])
            @test rhs == batched_dense_rhs(matrices, expected)

            saved = map(copy, (batch.b, batch.l, batch.dᵢ, batch.u))
            @test_throws ArgumentError ldiv!(batch, batch.b)
            @test_throws ArgumentError ldiv!(batch, @view batch.b[:, :])
            @test (batch.b, batch.l, batch.dᵢ, batch.u) == saved
        end
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                 HELMHOLTZ PARITY REFERENCES AND TESTS                  ///#
#//////////////////////////////////////////////////////////////////////////////#

# Independent integrated Helmholtz matrices, including the special terminal
# tau rows and degree-zero coefficient. This exercises actual Be/Bo factors
# for different Fourier wavenumbers without using scalar ldiv! as the oracle.
function batched_dense_helmholtz(::Type{T}, P, parity, theta0, theta1, neum) where {T}
    degrees = collect(parity:2:P)
    M = length(degrees)
    A = zeros(T, M, M)
    A[1, :] .= neum ? degrees.^2 : ones(T, M)
    for row = 2:M
        p = degrees[row]
        A[row, row-1] = -theta1 * (p == 2 ? 2 : 1) / (4p*(p-1))
        A[row, row] = theta0 + theta1 * (p <= P-2 ? 1 : 0) / (2*(p^2-1))
        row < M && (A[row, row+1] = -theta1 * (p+2 <= P-2 ? 1 : 0) / (4p*(p+1)))
    end
    return A
end

@testset "Batched Helmholtz parity factors" begin
    for T in (Float32, Float64), P in (3, 4, 16, 17), neum in (false, true)
        nsystems = 7
        theta0 = T(0.7)
        # Dirichlet includes the zero Fourier mode; positive shift keeps the
        # Neumann systems nonsingular, as required by the scalar solver.
        shifts = T[(neum ? 0.8 : 0.0) + ((s-1)/3)^2 for s = 1:nsystems]
        solvers = [HelmoltzSolver(P, T; neum) for _ = 1:nsystems]
        for s = 1:nsystems
            update!(solvers[s], theta0, shifts[s])
        end
        for (field, parity) in ((:Be, 0), (:Bo, 1))
            factors = [getproperty(h, field) for h in solvers]
            batch = BatchedQuasiTridiagonal(length(factors), size(first(factors), 1), eltype(first(factors)))
            for (s, Q) in enumerate(factors), field in (:b, :l, :dᵢ, :u)
                @views getproperty(batch, field)[s, :] .= getproperty(Q, field)
            end
            matrices = [batched_dense_helmholtz(T, P, parity, theta0, shift, neum)
                        for shift in shifts]
            M = size(first(matrices), 1)
            for scale in (1, -0.5)
                expected = batched_known_solution(Complex{T}, nsystems, M, scale)
                # Smooth spectra avoid making the Neumann wall derivative a
                # cancellation-dominated sum of growing high-degree terms.
                expected ./= reshape(T[j^4 for j = 1:M], 1, M)
                rhs = batched_dense_rhs(matrices, expected)
                saved = copy(rhs)
                ldiv!(batch, rhs)
                @test rhs ≈ expected rtol=tolerance(T) atol=tolerance(T)
                for s = 1:nsystems
                    @test matrices[s]*rhs[s, :] ≈ saved[s, :] rtol=tolerance(T) atol=tolerance(T)
                end
            end
        end
    end
end
