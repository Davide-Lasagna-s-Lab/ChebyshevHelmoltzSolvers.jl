# Run: julia --startup-file=no --threads=1 --project perf/benchmark_helmoltz.jl
using ChebyshevHelmoltzSolvers, LinearAlgebra, Random, Printf

#//////////////////////////////////////////////////////////////////////////////#
#///                          COMPLETE SOLVE PATHS                          ///#
#//////////////////////////////////////////////////////////////////////////////#

# Baseline: each independent scalar solve gets a contiguous coefficient column.
function scalar_solve!(solvers, u, f)
    for s in eachindex(solvers)
        solve!(solvers[s], view(u, :, s), view(f, :, s))
    end
end

# Same scalar solver, operating directly on the batch's system-first layout.
function strided_solve!(solvers, u, f)
    for s in eachindex(solvers)
        solve!(solvers[s], view(u, s, :), view(f, s, :))
    end
end

function scalar_update!(solvers, theta0, theta1)
    for s in eachindex(solvers)
        update!(solvers[s], theta0[s], theta1[s])
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                         MEASUREMENT AND OUTPUT                         ///#
#//////////////////////////////////////////////////////////////////////////////#

# Timings include public API validation, RHS assembly and both parity solves.
# Compilation, initial factorisation and GC are outside each timed solve sample.
function measure(f::F) where {F}
    for _ in 1:5; f(); end
    elapsed = @elapsed f()
    repeats = clamp(ceil(Int, 0.003/max(elapsed, eps())), 1, 2000)
    bytes = @allocated f()
    GC.gc()
    samples = zeros(100)
    for i in eachindex(samples)
        samples[i] = (@elapsed begin
            for _ in 1:repeats; f(); end
        end)/repeats
    end
    return minimum(samples)*1e6, bytes
end

function benchmark_case(io, Ny, B)
    P, nu = Ny-1, 1/400
    theta0 = fill(nu, B)
    theta1 = [1 + nu*((s-1)%64)^2 for s in 1:B]
    solvers = [HelmoltzSolver(P) for _ in 1:B]
    scalar_update!(solvers, theta0, theta1)
    batch = BatchedHelmoltzSolver(P, B)
    update!(batch, theta0, theta1)
    f = randn(MersenneTwister(1234), Ny, B)
    u = similar(f)
    packed_f, packed_u = permutedims(f), zeros(B, Ny)
    bc = zeros(B)

    # Check the full batch against scalar solves before collecting any timings.
    scalar_solve!(solvers, u, f)
    reference = copy(u)
    solve!(batch, packed_u, packed_f, bc, bc)
    @assert isapprox(reference, transpose(packed_u); rtol=3e-12, atol=3e-12)
    strided_solve!(solvers, packed_u, packed_f)
    @assert isapprox(reference, transpose(packed_u); rtol=3e-12, atol=3e-12)

    baseline = measure(() -> scalar_solve!(solvers, u, f))
    cases = (
        ("scalar_contiguous", baseline),
        ("scalar_strided", measure(() -> strided_solve!(solvers, packed_u, packed_f))),
        ("batched", measure(() -> solve!(batch, packed_u, packed_f, bc, bc))),
        ("scalar_update", measure(() -> scalar_update!(solvers, theta0, theta1))),
        ("batched_update", measure(() -> update!(batch, theta0, theta1))),
    )
    for (mode, (us, bytes)) in cases
        @printf(io, "%d,%d,%s,%.6f,%d,%.6f\n", Ny, B, mode, us, bytes,
                endswith(mode, "update") ? NaN : baseline[1]/us)
    end
    flush(io)
    @printf(stderr, "Ny=%d B=%d: scalar %.1f us; batched %.1f us (%.2fx)\n",
            Ny, B, baseline[1], cases[3][2][1], baseline[1]/cases[3][2][1])
end

function main()
    Threads.nthreads() == 1 || error("Run with --threads=1")
    BLAS.set_num_threads(1)
    output = isempty(ARGS) ? joinpath(@__DIR__, "results", "helmoltz-cpu.csv") : only(ARGS)
    mkpath(dirname(output))
    open(joinpath(dirname(output), "environment.txt"), "w") do io
        println(io, "Julia: ", VERSION)
        println(io, "Hardware: ", Sys.cpu_info()[1].model)
        println(io, "LLVM CPU target: ", Sys.CPU_NAME)
        println(io, "OS: ", Sys.KERNEL, "; architecture: ", Sys.ARCH)
        println(io, "Julia threads: ", Threads.nthreads(), "; BLAS threads: ", BLAS.get_num_threads())
        println(io, "Precision: Float64; 100 warmed samples; minimum per complete batch")
        println(io, "Solve timings exclude factor setup; update timings include assembly, UL and validation")
    end
    open(output, "w") do io
        println(io, "Ny,systems,mode,minimum_us,allocated_bytes,speedup_vs_scalar")
        for Ny in 2 .^ (3:9), B in (64, 256, 1024, 4096, 16384)
            benchmark_case(io, Ny, B)
        end
    end
end
main()
