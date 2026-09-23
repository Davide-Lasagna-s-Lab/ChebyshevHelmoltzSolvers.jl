# Run: julia --threads=1 --project=. perf/benchmark_batched.jl [results.csv]
# CPU Float64, EVEN parity only (M = (Ny+1)/2); factors are built once, outside timing.
# Every timed iteration resets the RHS; batched solves use native storage.
using ChebyshevHelmoltzSolvers, LinearAlgebra, Random, Printf

#//////////////////////////////////////////////////////////////////////////////#
#///                   BENCHMARK DATA AND SOLVE VARIANTS                    ///#
#//////////////////////////////////////////////////////////////////////////////#

function factor_bank(Ny, nsystems)
    # Channel-like Fourier shifts: Re=400, lambda=1, Lx=4pi, Lz=2pi.
    nu, lambda = 1 / 400, 1.0
    nx = ceil(Int, sqrt(nsystems))
    nz = cld(nsystems, nx)
    return map(1:nsystems) do s
        kx = mod(s - 1, nx) - nx ÷ 2
        kz = div(s - 1, nx) - nz ÷ 2
        h = HelmoltzSolver(Ny - 1)
        update!(h, nu, lambda + nu * ((0.5kx)^2 + kz^2))
        h.Be                         # update! already performs ul! on both parities
    end
end

function scalar_step!(factors, rhs, source)
    copyto!(rhs, source)
    @inbounds for s in eachindex(factors)
        ldiv!(factors[s], view(rhs, :, s))
    end
    return nothing
end

function packed_step!(batch, rhs, source)
    copyto!(rhs, source)
    ldiv!(batch, rhs)
    return nothing
end

#//////////////////////////////////////////////////////////////////////////////#
#///                      TIMING AND RESULT REPORTING                       ///#
#//////////////////////////////////////////////////////////////////////////////#

# Specialize the harness on the callable and concrete argument tuple; warm it
# before both @allocated and @elapsed. Report per-complete-batch time, not per RHS.
function measure!(f::F, args::A) where {F,A<:Tuple}
    for _ in 1:5
        f(args...)
    end
    elapsed = @elapsed f(args...)
    repeats = clamp(ceil(Int, 0.005 / max(elapsed, eps())), 1, 1000)
    bytes = @allocated f(args...)
    times = Vector{Float64}(undef, 31)
    GC.gc()
    for sample in eachindex(times)
        times[sample] = (@elapsed for _ in 1:repeats
            f(args...)
        end) / repeats
    end
    return sort!(times)[(length(times)+1)÷2] * 1e6, bytes
end

function row(io, Ny, M, nsystems, mode, result, baseline)
    us, bytes = result
    @printf(io, "%d,%d,%d,%s,%.3f,%d,%.3f\n",
            Ny, M, nsystems, mode, us, bytes, baseline / us)
    flush(io)
end

function benchmark_case(io, Ny, nsystems, factors)
    M = size(first(factors), 1)
    batch = BatchedQuasiTridiagonal(length(factors), size(first(factors), 1), eltype(first(factors)))
    for (s, Q) in enumerate(factors), field in (:b, :l, :dᵢ, :u)
        @views getproperty(batch, field)[s, :] .= getproperty(Q, field)
    end
    source = randn(MersenneTwister(1234), M, nsystems)
    packed_source = permutedims(source)
    rhs, packed, reference = similar(source), similar(packed_source), similar(source)
    scalar_step!(factors, reference, source)
    scalar = measure!(scalar_step!, (factors, rhs, source))
    row(io, Ny, M, nsystems, "scalar", scalar, first(scalar))
    packed_step!(batch, packed, packed_source)
    @assert isapprox(reference, transpose(packed); rtol=2e-12, atol=1e-12)
    native = measure!(packed_step!, (batch, packed, packed_source))
    row(io, Ny, M, nsystems, "packed", native, first(scalar))

end

#//////////////////////////////////////////////////////////////////////////////#
#///                            BENCHMARK SWEEP                             ///#
#//////////////////////////////////////////////////////////////////////////////#

function main()
    Threads.nthreads() == 1 || error("Run with julia --threads=1")
    BLAS.set_num_threads(1)
    io = isempty(ARGS) ? stdout : open(only(ARGS), "w")
    println(stderr, "Julia $VERSION; CPU=$(Sys.CPU_NAME); Float64; even parity; one Julia thread")
    try
        println(io, "Ny,M,nsystems,mode,median_us,allocated_bytes,speedup_vs_scalar")
        for Ny in (33, 65, 129), nsystems in (256, 4096)
            benchmark_case(io, Ny, nsystems, factor_bank(Ny, nsystems))
        end
    finally
        io === stdout || close(io)
    end
end
main()
