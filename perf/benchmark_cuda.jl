#//////////////////////////////////////////////////////////////////////////////#
#///                          CUDA BENCHMARK SETUP                          ///#
#//////////////////////////////////////////////////////////////////////////////#

using Adapt, CUDA, ChebyshevHelmoltzSolvers, Printf

CUDA.functional() || error("CUDA is not functional; run this benchmark on the NVIDIA GPU host")
CUDA.versioninfo(stderr)
CUDA.allowscalar(false)

#//////////////////////////////////////////////////////////////////////////////#
#///                              SOLVE TIMING                              ///#
#//////////////////////////////////////////////////////////////////////////////#

# Measure complete solves with native y-last storage: assembly, wall conditions
# and both parity substitutions. Factor setup, host/device transfers and JIT
# compilation are outside timings. Each GPU sample synchronizes before and
# after its repeated launches, so timings include the actual device work.
function seconds_per_solve(solver, dest, rhs, bc, synchronize)
    for _ = 1:5
        solve!(solver, dest, rhs, bc, bc)
    end
    synchronize()
    estimate = @elapsed begin
        solve!(solver, dest, rhs, bc, bc)
        synchronize()
    end
    repetitions = clamp(ceil(Int, 0.05/max(estimate, eps())), 1, 1000)
    samples = map(1:7) do _
        synchronize()
        elapsed = @elapsed begin
            for _ = 1:repetitions
                solve!(solver, dest, rhs, bc, bc)
            end
            synchronize()
        end
        elapsed/repetitions
    end
    return sort!(samples)[4]
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      CPU AND CUDA BENCHMARK SWEEP                      ///#
#//////////////////////////////////////////////////////////////////////////////#

function benchmark(io)
    println(io, "degree,Ny,systems,precision,cpu_seconds,gpu_seconds,speedup")
    for P in (32, 64, 128), B in (256, 4096)
        # Changing each mode's shift represents Fourier-dependent Helmholtz
        # operators. Small smooth complex RHSs keep repeated solves comparable.
        theta0 = 1/400
        shifts = [10 + theta0*((s-1)%64)^2 for s = 1:B]
        cpu = BatchedHelmoltzSolver(P, B)
        update!(cpu, fill(theta0, B), shifts)
        rhs = reshape(ComplexF64[sin(s+n)/(n+1)^4 + im*cos(s-n)/(n+1)^4
                                 for s = 1:B, n = 0:P], 16, B÷16, P+1)
        dest = similar(rhs)
        gpu = Adapt.adapt(CuArray, cpu)
        gpu_rhs = CuArray(rhs)
        gpu_dest = similar(gpu_rhs)

        # Expose y-last fields as matrices once, without copying their data.
        dest, rhs = reshape(dest, B, P+1), reshape(rhs, B, P+1)
        gpu_dest, gpu_rhs = reshape(gpu_dest, B, P+1), reshape(gpu_rhs, B, P+1)

        bc = zeros(B)
        gpu_bc = CuArray(bc)

        # Validate each exact benchmark configuration before measuring it.
        solve!(cpu, dest, rhs, bc, bc)
        solve!(gpu, gpu_dest, gpu_rhs, gpu_bc, gpu_bc)
        @assert isapprox(Array(gpu_dest), dest; atol=3e-11, rtol=3e-11)
        cpu_seconds = seconds_per_solve(cpu, dest, rhs, bc, () -> nothing)
        gpu_seconds = seconds_per_solve(gpu, gpu_dest, gpu_rhs, gpu_bc, CUDA.synchronize)
        @printf(io, "%d,%d,%d,ComplexF64,%.9g,%.9g,%.4f\n",
                P, P+1, B, cpu_seconds, gpu_seconds, cpu_seconds/gpu_seconds)
        flush(io)
    end
end

#//////////////////////////////////////////////////////////////////////////////#
#///                             RESULT OUTPUT                              ///#
#//////////////////////////////////////////////////////////////////////////////#

# Pass an output path to save CSV; otherwise emit the table to standard output.
# CUDA.versioninfo above records the device/runtime environment separately.
if isempty(ARGS)
    benchmark(stdout)
else
    open(benchmark, only(ARGS), "w")
end
