#//////////////////////////////////////////////////////////////////////////////#
#///                            CUDA TEST SETUP                             ///#
#//////////////////////////////////////////////////////////////////////////////#

using Adapt, CUDA, ChebyshevHelmoltzSolvers, LinearAlgebra, Test

# This is an explicit GPU validation target: missing hardware must fail rather
# than turn the entire suite into skipped tests that appear to have passed.
CUDA.functional() || error("CUDA is not functional; run this suite on the NVIDIA GPU host")
CUDA.versioninfo()
CUDA.allowscalar(false)
to_gpu(x) = Adapt.adapt(CuArray, x) # preserve Float64, unlike convenience conversion

#//////////////////////////////////////////////////////////////////////////////#
#///                         SHARED HELMHOLTZ TESTS                         ///#
#//////////////////////////////////////////////////////////////////////////////#

# The shared cases include Float32/64, complex walls, both boundary types and
# 2D/3D y-last fields. Their independent references are built on the CPU.
include(joinpath(@__DIR__, "..", "batched", "helmoltz.jl"))
test_batched_helmholtz(to_gpu; backend="CUDA")

#//////////////////////////////////////////////////////////////////////////////#
#///                             CUDA UL TESTS                              ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "CUDA batched UL" begin
    for T in (Float32, Float64), M in (2, 17), B in (1, 67, 257)
        # Each system has different, diagonally dominant entries. Save its
        # dense matrix BEFORE UL overwrites the compact representation.
        systems = map(1:B) do s
            b = T[0.02j for j = 1:M]
            b[1] = T(4 + 0.001s)
            Q = QuasiTridiagonal(b, fill(T(-0.2), M-1),
                                fill(T(2 + 0.002s), M-1), fill(T(0.1), M-2))
            A = diagm(0 => vcat(b[1], fill(T(2 + 0.002s), M-1)))
            A[1, :] .= b
            for i = 2:M
                A[i, i-1] = T(-0.2)
                i < M && (A[i, i+1] = T(0.1))
            end
            (A, Q)
        end
        factors = last.(systems)
        cpu = BatchedQuasiTridiagonal(length(factors), size(first(factors), 1), eltype(first(factors)))
        for (s, Q) in enumerate(factors), field in (:b, :l, :dᵢ, :u)
            @views getproperty(cpu, field)[s, :] .= getproperty(Q, field)
        end
        gpu = to_gpu(cpu)
        # Factor the original matrices on each backend. Device UL must modify
        # its own arrays and produce factors suitable for repeated solves.
        @test ul!(cpu) === cpu
        @test ul!(gpu) === gpu
        tol = T === Float32 ? 2e-4 : 2e-11
        for (actual, expected) in zip((gpu.b, gpu.l, gpu.dᵢ, gpu.u),
                                      (cpu.b, cpu.l, cpu.dᵢ, cpu.u))
            @test Array(actual) ≈ expected atol=tol rtol=tol
        end
        saved = map(Array, (gpu.b, gpu.l, gpu.dᵢ, gpu.u))
        @test eltype(gpu.b) === T
        for S in (T, Complex{T})
            exact = S[(s + j)/100 + (S <: Complex ? im*(s-j)/200 : 0)
                      for s = 1:B, j = 1:M]
            rhs = similar(exact)
            for s = 1:B
                rhs[s, :] = systems[s][1]*exact[s, :]
            end

            # The view has contiguous systems and every-other coefficient
            # columns, like the even/odd portions of a spectral field. Untouched
            # sentinel columns reveal accidental writes outside the view.
            storage = fill(S(123), B, 2M)
            storage[:, 1:2:end] .= rhs
            device_storage = to_gpu(storage)
            device_rhs = @view device_storage[:, 1:2:2M]
            @test ldiv!(gpu, device_rhs) === device_rhs
            result = Array(device_storage)
            tol = T === Float32 ? 2e-4 : 2e-11
            @test result[:, 1:2:end] ≈ exact atol=tol rtol=tol
            @test all(==(S(123)), result[:, 2:2:end])
            @test map(Array, (gpu.b, gpu.l, gpu.dᵢ, gpu.u)) == saved
            @test_throws ArgumentError ldiv!(gpu, rhs)
            @test_throws ArgumentError ldiv!(cpu, to_gpu(rhs))
        end
        # B=257 reaches beyond one full CUDA block.
        @test_throws ArgumentError ldiv!(gpu, gpu.b)
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                     CUDA UPDATE AND BACKEND TESTS                      ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "CUDA updates and backend contracts" begin
    P, B = 8, 257
    cpu = BatchedHelmoltzSolver(P, B)
    gpu = to_gpu(cpu)
    allocations = (gpu.Be.b, gpu.Be.l, gpu.Be.dᵢ, gpu.Be.u,
                   gpu.Bo.b, gpu.Bo.l, gpu.Bo.dᵢ, gpu.Bo.u)
    source = ComplexF64[sin(s+n)/(n+1)^4 + im*cos(s-n)/(n+1)^4
                        for s = 1:B, n = 0:P]
    rhs = to_gpu(reshape(source, 1, B, P+1))
    dest = similar(rhs)
    u, f = reshape(dest, B, P+1), reshape(rhs, B, P+1)
    expected = similar(source)
    for (theta0, device_coefficients) in ((fill(0.5, B), false),
        (collect(range(0.1, 0.6; length=B)), true))
        shifts = collect(range(0.3, 1.8; length=B))
        # Accept CPU vectors or coefficients already on the GPU. Both paths
        # factor directly in the same device arrays; only host coefficients
        # require an upload. The subsequent solve must preserve the forcing.
        theta0_input = device_coefficients ? to_gpu(theta0) : theta0
        theta1_input = device_coefficients ? to_gpu(shifts) : shifts
        @test update!(gpu, theta0_input, theta1_input) === gpu
        update!(cpu, theta0, shifts)
        @test all(a === b for (a, b) in zip(allocations,
            (gpu.Be.b, gpu.Be.l, gpu.Be.dᵢ, gpu.Be.u,
             gpu.Bo.b, gpu.Bo.l, gpu.Bo.dᵢ, gpu.Bo.u)))
        for (actual, reference) in zip((gpu.Be, gpu.Bo), (cpu.Be, cpu.Bo))
            @test Array(actual.dᵢ) ≈ reference.dᵢ atol=3e-11 rtol=3e-11
        end
        solve!(cpu, expected, source, ZeroBoundary{Float64}(B), ZeroBoundary{Float64}(B))
        solve!(gpu, u, f, ZeroBoundary{Float64}(B), ZeroBoundary{Float64}(B))
        @test reshape(Array(dest), B, P+1) ≈ expected atol=3e-11 rtol=3e-11
        @test reshape(Array(rhs), B, P+1) == source
    end
    @test_throws ArgumentError solve!(gpu, expected, source, ZeroBoundary{Float64}(B), ZeroBoundary{Float64}(B))
    @test_throws ArgumentError solve!(cpu, u, f, ZeroBoundary{Float64}(B), ZeroBoundary{Float64}(B))
    @test_throws ArgumentError solve!(gpu, f, f, ZeroBoundary{Float64}(B), ZeroBoundary{Float64}(B))
end

# Surface errors from asynchronous launches before declaring validation done.
CUDA.synchronize()
