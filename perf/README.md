# Benchmarks

Run from the package root with an otherwise idle machine. Scripts warm up
specialized code before timing, validate results against a scalar reference,
and record medians rather than compilation latency. Stored results were
collected on an Apple M5 MacBook Air with Julia 1.12.6, using one Julia and one
BLAS thread. They do not describe an A100 or a complete DNS time step.

## Complete Helmholtz solves

```sh
julia --startup-file=no --threads=1 --project perf/benchmark_helmoltz.jl
python3 perf/plot_benchmarks.py
```

The plot script requires Matplotlib (`python3 -m pip install matplotlib` in a
suitable Python environment). It only reads the CSV; it does not rerun solves.
Figures are saved as PNG and SVG in `results/`.

`benchmark_helmoltz.jl [output.csv]` measures Float64 problems at coefficient
counts 33, 65 and 129, with 64, 256 and 4096 different operators. For each case:

- **scalar_contiguous**: independent scalar solvers with contiguous coefficient vectors.
- **scalar_strided**: the same scalar solvers applied to rows of system-first fields.
- **batched**: full batched solves in native `(system, coefficient)` storage.
- **roundtrip**: batched solves plus both layout conversions, using preallocated buffers.
- **scalar_update / batched_update**: assembly and UL factorisation, including each API's validation.

Solve measurements include checks, source assembly, wall conditions and both
parity substitutions. They exclude factor setup and preallocation. Each sample
repeats calls for approximately 3 ms; the CSV reports the median of 21 samples
in microseconds **per complete batch**, and allocated bytes per call.
`speedup_vs_scalar` compares solve paths only; update rows use `NaN` there.
No BLAS work occurs in the solver kernels; one BLAS thread is set for reproducibility.

Results: [CSV](results/helmoltz-cpu.csv), [environment](results/environment.txt),
[solve figure](results/helmoltz-solves.png), [update figure](results/helmoltz-updates.png).
The main README discusses both the solve benefit and the slower, allocating
batched update. Factor buffers are reused, but temporary Julia view/wrapper
objects still allocate; reuse of arrays alone does not imply zero allocations.

## Substitution-only microbenchmark

```sh
julia --startup-file=no --threads=1 --project perf/benchmark_batched.jl perf/results/quasitridiag-cpu.csv
```

This measures only even-parity UL substitutions, with RHS reset included.
It excludes assembly, the odd block and all physical-space transforms.
`packed` uses native system-first storage; `roundtrip` additionally packs and
unpacks. Its speedup is not interchangeable with the full Helmholtz benchmark.
Current results are in [quasitridiag-cpu.csv](results/quasitridiag-cpu.csv).
The older `batched-cpu.csv` is retained as historical data from the removed
blocked implementation and is not used in the current figures.

## CUDA execution on the GPU host

```sh
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
julia --project=test/cuda perf/benchmark_cuda.jl gpu-results.csv
```

The suite fails explicitly when no functional NVIDIA device exists. It covers
real/complex fields, both wall conditions, per-system boundary vectors and a
storage-free zero vector, factor updates, parity views and backend contracts.
The benchmark compares full CPU and GPU batched ComplexF64 solves, with
synchronization around each timed batch. Matrices are reshaped at the call
site; factors, fields and stored wall vectors are transferred before timing.

The CUDA extension was successfully loaded locally, but `CUDA.functional()`
was false. **No GPU execution, numerical validation or A100 timings are claimed.**
Run these commands on the A100 before relying on the GPU path.
