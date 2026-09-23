# Compatibility entry point; the shared harness also runs on a CPU-only host.
# Usage: julia --threads=1 --project=test/cuda perf/benchmark_cuda.jl output.csv
length(ARGS) == 1 || error("Pass a fresh output CSV path")
pushfirst!(ARGS, "cuda")
include("benchmark_devices.jl")
