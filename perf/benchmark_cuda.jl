# Run inside a Slurm GPU allocation, with one Julia thread:
# julia --project=test/cuda perf/benchmark_cuda.jl results.csv
using Adapt, CUDA, ChebyshevHelmoltzSolvers, LinearAlgebra, Printf, Dates
using SHA: sha256

#//////////////////////////////////////////////////////////////////////////////#
#///                     CONFIGURATION AND ENVIRONMENT                      ///#
#//////////////////////////////////////////////////////////////////////////////#

CUDA.functional() || error("Run inside an NVIDIA GPU allocation")
CUDA.allowscalar(false)
Threads.nthreads() == 1 || error("Use one Julia thread for the SIMD CPU baseline")
BLAS.set_num_threads(1)

const NSAMPLES = parse(Int, get(ENV, "CHEB_BENCH_SAMPLES", "100"))
const SIZES = parse.(Int, split(get(ENV, "CHEB_BENCH_NY", "8,16,32,64,128,256,512"), ','))
const BATCHES = parse.(Int, split(get(ENV, "CHEB_BENCH_BATCHES", "64,256,1024,4096,16384"), ','))
NSAMPLES > 0 || error("CHEB_BENCH_SAMPLES must be positive")

#//////////////////////////////////////////////////////////////////////////////#
#///                      SYNCHRONIZED WARMED MEASUREMENTS                   ///#
#//////////////////////////////////////////////////////////////////////////////#

# Synchronise outside the start timer and after the final launch. Timing just
# an asynchronous launch would measure submission, not completion of the solve.
# Repetition reduces timer noise; report minima consistently for both backends.
function measure(f::F, synchronize::S) where {F, S}
    for _ in 1:5
        f()
    end
    synchronize()
    elapsed = @elapsed begin
        f()
        synchronize()
    end
    repeats = clamp(ceil(Int, 0.003/max(elapsed, eps())), 1, 1000)
    samples = zeros(NSAMPLES)
    GC.gc()
    for i in eachindex(samples)
        synchronize()
        samples[i] = (@elapsed begin
            for _ in 1:repeats
                f()
            end
            synchronize()
        end)/repeats
    end
    return minimum(samples)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                 SOLVES, UPDATES AND TRANSFER COMPARISON                 ///#
#//////////////////////////////////////////////////////////////////////////////#

function benchmark_case(io, Ny, B)
    θ₀ = fill(1/400, B)
    θ₁ = [1 + θ₀[s]*((s-1)%64)^2 for s in 1:B]
    cpu = BatchedHelmoltzSolver(Ny-1, B)
    update!(cpu, θ₀, θ₁)
    # Fourier fields have complex coefficients, contiguous across systems.
    f = ComplexF64[sin(s+n)/(n+1)^4 + im*cos(s-n)/(n+1)^4 for s in 1:B, n in 0:Ny-1]
    u = similar(f)
    bc = zeros(B)
    gpu = adapt(CuArray, cpu)
    fg, ug, bcg = CuArray(f), CuArray(u), CuArray(bc)
    θ₀g, θ₁g = CuArray(θ₀), CuArray(θ₁)

    # Exercise device factor assembly, rather than only uploading CPU factors.
    update!(gpu, θ₀g, θ₁g)
    solve!(cpu, u, f, bc, bc)
    solve!(gpu, ug, fg, bcg, bcg)
    CUDA.synchronize()
    @assert isapprox(Array(ug), u; rtol=3e-11, atol=3e-11)

    cpu_solve = measure(() -> solve!(cpu, u, f, bc, bc), () -> nothing)
    gpu_solve = measure(() -> solve!(gpu, ug, fg, bcg, bcg), CUDA.synchronize)
    cpu_update = measure(() -> update!(cpu, θ₀, θ₁), () -> nothing)
    gpu_update = measure(() -> update!(gpu, θ₀g, θ₁g), CUDA.synchronize)

    # Separate end-to-end measurement: one RHS upload, solve and result download.
    # Factors and all buffers are already allocated; resident timings above are
    # the relevant case when the surrounding DNS also runs on the GPU.
    gpu_roundtrip = measure(CUDA.synchronize) do
        copyto!(fg, f)
        solve!(gpu, ug, fg, bcg, bcg)
        copyto!(u, ug)
    end
    @printf(io, "%d,%d,ComplexF64,%.9g,%.9g,%.6f,%.9g,%.9g,%.6f,%.9g,%.6f\n",
            Ny, B, cpu_solve, gpu_solve, cpu_solve/gpu_solve,
            cpu_update, gpu_update, cpu_update/gpu_update,
            gpu_roundtrip, cpu_solve/gpu_roundtrip)
    flush(io)
    @printf("Ny=%d B=%d: solve %.2fx; update %.2fx; with transfers %.2fx\n",
            Ny, B, cpu_solve/gpu_solve, cpu_update/gpu_update, cpu_solve/gpu_roundtrip)
    flush(stdout)
    return nothing
end

#//////////////////////////////////////////////////////////////////////////////#
#///                         RESULTS AND SOURCE PROVENANCE                  ///#
#//////////////////////////////////////////////////////////////////////////////#

function main()
    length(ARGS) == 1 || error("Pass a fresh output CSV path")
    output = only(ARGS)
    isfile(output) && error("Refusing to overwrite existing benchmark results")
    mkpath(dirname(abspath(output)))
    open(output * ".environment.txt", "w") do io
        println(io, "UTC: ", Dates.now(Dates.UTC))
        println(io, "Source commit: ", get(ENV, "CHEB_SOURCE_COMMIT", "unrecorded"))
        println(io, "Source patch SHA256: ", get(ENV, "CHEB_PATCH_SHA256", "none"))
        println(io, "Benchmark SHA256: ", bytes2hex(sha256(read(@__FILE__))))
        println(io, "Slurm job: ", get(ENV, "SLURM_JOB_ID", "none"))
        println(io, "Host: ", gethostname(), "; CPU: ", Sys.cpu_info()[1].model)
        println(io, "Julia: ", VERSION, "; threads: ", Threads.nthreads())
        println(io, "BLAS threads: ", BLAS.get_num_threads())
        println(io, "Samples: ", NSAMPLES, "; statistic: minimum; buffers allocated before timing")
        println(io, "Ny: ", SIZES, "; batches: ", BATCHES)
        CUDA.versioninfo(io)
    end
    open(output, "w") do io
        println(io, "Ny,systems,precision,cpu_solve_seconds,gpu_solve_seconds,solve_speedup,cpu_update_seconds,gpu_update_seconds,update_speedup,gpu_roundtrip_seconds,roundtrip_speedup")
        for Ny in SIZES, B in BATCHES
            benchmark_case(io, Ny, B)
            # Release unreachable device buffers between configurations.
            GC.gc()
            CUDA.reclaim()
        end
    end
end
main()
