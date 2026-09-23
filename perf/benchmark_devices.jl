# Usage: julia --threads=1 --project=. perf/benchmark_devices.jl cpu results.csv
# CUDA:  julia --threads=1 --project=test/cuda perf/benchmark_devices.jl cuda results.csv
using ChebyshevHelmoltzSolvers, LinearAlgebra, Printf, Dates
using SHA: sha256
const GPU = length(ARGS) == 2 && ARGS[1] == "cuda"
if GPU
    @eval using CUDA, Adapt
    CUDA.functional() || error("CUDA is unavailable")
    CUDA.allowscalar(false)
end
const SAMPLES = parse(Int, get(ENV, "CHEB_BENCH_SAMPLES", "500"))
const SIZES = parse.(Int, split(get(ENV, "CHEB_BENCH_NY", "8,16,32,64,128,256,512,1024"), ','))
const BATCHES = parse.(Int, split(get(ENV, "CHEB_BENCH_BATCHES", "64,256,1024,4096,16384,65536"), ','))
const KINDS = split(get(ENV, "CHEB_BENCH_KINDS", "helmholtz,coupled"), ',')

#//////////////////////////////////////////////////////////////////////////////#
#///                      WARMED MINIMUM-TIME ESTIMATES                      ///#
#//////////////////////////////////////////////////////////////////////////////#

function measure(f::F, synchronize::S=()->nothing) where {F, S}
    for _ in 1:5; f(); end
    synchronize()
    elapsed = @elapsed begin f(); synchronize(); end
    repeats = clamp(ceil(Int, 0.001/max(elapsed, eps())), 1, 1000)
    times = zeros(SAMPLES)
    GC.gc()
    for i in eachindex(times)
        synchronize()
        times[i] = (@elapsed begin
            for _ in 1:repeats; f(); end
            synchronize()
        end)/repeats
    end
    return minimum(times)
end

function case!(io, kind, Ny, B)
    θ₀ = fill(1/400, B)
    θ₁ = [1 + θ₀[s]*((s-1)%64)^2 for s in 1:B]
    coupled = kind == "coupled"
    h = coupled ? BatchedCoupledHelmoltzSolver(Ny-1, B, ComplexF64) : BatchedHelmoltzSolver(Ny-1, B)
    θs = (θ₀, θ₁, ones(B), fill(0.5, B))
    update_cpu = coupled ? ()->update!(h, θs) : ()->update!(h, θ₀, θ₁)
    update_cpu()
    f = ComplexF64[sin(s+n)/(n+1)^4 + im*cos(s-n)/(n+1)^4 for s in 1:B, n in 0:Ny-1]
    u = similar(f)
    bc = zeros(B)
    solve_cpu = coupled ? ()->solve!(h, u, f) : ()->solve!(h, u, f, bc, bc)
    solve_cpu()
    cpu_solve = measure(solve_cpu)
    cpu_update = measure(update_cpu)
    gpu_solve = gpu_update = NaN
    if GPU
        hg = adapt(CuArray, h)
        fg, ug, bcg = CuArray(f), CuArray(u), CuArray(bc)
        θg = map(CuArray, θs)
        update_gpu = coupled ? ()->update!(hg, θg) : ()->update!(hg, θg[1], θg[2])
        solve_gpu = coupled ? ()->solve!(hg, ug, fg) : ()->solve!(hg, ug, fg, bcg, bcg)
        # Check device assembly AND solve before timing. Chunked comparison
        # avoids allocating a second full host result for the largest cases.
        update_gpu()
        solve_gpu()
        CUDA.synchronize()
        for columns in Iterators.partition(1:Ny, 16)
            @assert isapprox(Array(view(ug, :, columns)), view(u, :, columns); rtol=3e-10, atol=3e-11)
        end
        gpu_solve = measure(solve_gpu, CUDA.synchronize)
        gpu_update = measure(update_gpu, CUDA.synchronize)
    end
    @printf(io, "%s,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.6f,%.6f\n",
        kind, Ny, B, SAMPLES, cpu_solve, cpu_update, gpu_solve, gpu_update,
        cpu_solve/gpu_solve, cpu_update/gpu_update)
    flush(io)
    @printf("%s Ny=%d B=%d: CPU %.3f ms; GPU %.3f ms; solve %.2fx\n",
        kind, Ny, B, 1000cpu_solve, 1000gpu_solve, cpu_solve/gpu_solve)
    flush(stdout)
    return nothing
end

#//////////////////////////////////////////////////////////////////////////////#
#///                       IMMUTABLE RESULT PROVENANCE                       ///#
#//////////////////////////////////////////////////////////////////////////////#

function main()
    length(ARGS) == 2 && ARGS[1] in ("cpu", "cuda") || error("Pass cpu|cuda and a fresh CSV path")
    Threads.nthreads() == 1 || error("Use --threads=1 for the SIMD CPU baseline")
    SAMPLES > 0 || error("Sample count must be positive")
    BLAS.set_num_threads(1)
    output = ARGS[2]
    isfile(output) && error("Refusing to overwrite recorded results")
    mkpath(dirname(abspath(output)))
    open(output*".environment.txt", "w") do io
        println(io, "UTC: ", Dates.now(Dates.UTC))
        println(io, "Source commit: ", get(ENV, "CHEB_SOURCE_COMMIT", "unrecorded"))
        println(io, "Julia: ", VERSION, "; CPU: ", Sys.cpu_info()[1].model)
        println(io, "Host: ", gethostname(), "; Slurm: ", get(ENV, "SLURM_JOB_ID", "none"))
        println(io, "Julia/BLAS threads: 1/1; ComplexF64; minimum of ", SAMPLES, " warmed samples")
        println(io, "A sample averages repeated calls for >= approximately 1ms; includes public API checks.")
        println(io, "GPU timings include synchronization; exclude transfers and allocation.")
        println(io, "Ny: ", SIZES, "; B: ", BATCHES, "; solvers: ", KINDS)
        println(io, "Script SHA256: ", bytes2hex(sha256(read(@__FILE__))))
        GPU && CUDA.versioninfo(io)
    end
    open(output, "w") do io
        println(io, "solver,Ny,systems,samples,cpu_solve_seconds,cpu_update_seconds,gpu_solve_seconds,gpu_update_seconds,solve_speedup,update_speedup")
        for kind in KINDS, Ny in SIZES, B in BATCHES
            case!(io, kind, Ny, B)
            GC.gc()
            GPU && CUDA.reclaim()
        end
    end
end
main()
