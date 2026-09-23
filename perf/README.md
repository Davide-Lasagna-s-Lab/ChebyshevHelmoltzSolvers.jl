# Benchmarks

Run from the package root with an otherwise idle machine. Scripts warm up
specialized code before timing and exclude compilation latency. The scalar
comparison checks its scalar reference; the device benchmark checks CUDA
results against the batched CPU result. The current sweep measures an Apple
M5 MacBook Air and an IRIDIS X A100 80 GB PCIe compute node. CPU measurements
use one Julia and one BLAS thread. These are solver timings, not complete DNS
time steps.

## Batched CPU and GPU benchmarks

[Recorded results at `92072b1`](results/92072b1/README.md) include raw timings,
environment records, test logs, and figures for the current implementation.

```sh
CHEB_SOURCE_COMMIT=$(git rev-parse HEAD) CHEB_BENCH_SAMPLES=500 \
  julia --startup-file=no --threads=1 --project=. \
  perf/benchmark_devices.jl cpu local-cpu.csv

CHEB_SOURCE_COMMIT=$(git rev-parse HEAD) CHEB_BENCH_SAMPLES=500 \
  julia --startup-file=no --threads=1 --project=test/cuda \
  perf/benchmark_devices.jl cuda a100.csv

python3 perf/plot_devices.py local-cpu.csv
python3 perf/plot_devices.py a100.csv
```

The shared harness measures **both batched Helmholtz and batched coupled
Helmholtz solvers**, using native `(system, coefficient)` ComplexF64 fields.
It defaults to `Ny = 8,16,32,64,128,256,512,1024` and
`B = 64,256,1024,4096,16384,65536`. Override comma-separated lists through
`CHEB_BENCH_NY`, `CHEB_BENCH_BATCHES`, and `CHEB_BENCH_KINDS`
(`helmholtz,coupled`). `CHEB_BENCH_SAMPLES` controls the sample count.

Each reported time is the **minimum of 500 warmed per-call sample averages**.
Small operations repeat within a sample for approximately 1 ms to reduce
timer noise. Large operations use one call per sample. This estimates the
least-interrupted execution time, not typical latency or a confidence interval.
More samples give more opportunities to observe a low-interference run;
they do not remove thermal, clock, or memory-bandwidth differences.

GPU runs also measure the batched CPU implementation **on the same compute
node**. Their speedup is that CPU minimum divided by the GPU minimum, not a
ratio against a different machine. Local Mac measurements are reported
separately. CPU runs use one Julia and one BLAS thread; CPU batching uses
SIMD across systems. GPU timings synchronize before and after the sample,
include public API validation and kernel launches, and exclude solver/field
construction, compilation and host/device transfers. Any internal allocation
performed by the measured call is included. Keep data on the GPU for this use case.
Coupled updates include both factorisations and rebuilding the cached influence
responses. GPU results are checked against CPU results before timing.

CSV times are seconds **per whole batch**; plots divide by `B`. Each raw run has
an environment sidecar recording the source commit, hardware, Julia/CUDA
versions, sample count, and benchmark script hash. The combined A100 CSV
identifies the raw source file for each row. Output files are never
overwritten. Large coupled cases require several GiB of host and device
memory; the largest coupled case uses roughly 9 GiB for solver/field arrays
alone. Use a host with at least 32 GB RAM for the full sweep, or select smaller
cases. Avoid interpreting swap-limited measurements as solver throughput.

The recorded local sweep omits only the coupled `Ny=1024, B=65536` case:
it exceeded comfortable memory capacity on the 16 GB Mac and was stopped.
Its [memory-limit record](results/92072b1/local-memory-limit.txt) explains the
missing point. The compute-node sweep includes the full size range.

## Historical scalar-versus-batched comparison

```sh
julia --startup-file=no --threads=1 --project perf/benchmark_helmoltz.jl
python3 perf/plot_benchmarks.py perf/results/<commit>/helmoltz-cpu.csv
```

The plot script requires Matplotlib (`python3 -m pip install matplotlib` in a
suitable Python environment). It only reads the CSV; it does not rerun solves.
Figures are saved as PNG and SVG in `results/`.

`benchmark_helmoltz.jl [output.csv]` measures Float64 problems at coefficient
counts `8, 16, 32, 64, 128, 256, 512`, with batch sizes
`64, 256, 1024, 4096, 16384`. For each case:

- **scalar_contiguous**: independent scalar solvers with contiguous coefficient vectors.
- **scalar_strided**: the same scalar solvers applied to rows of system-first fields.
- **batched**: full batched solves in native `(system, coefficient)` storage.
- **scalar_update / batched_update**: assembly and UL factorisation, including each API's validation.

Solve measurements include checks, source assembly, wall conditions and both
parity substitutions. They exclude factor setup and preallocation. Each sample
repeats calls for approximately 3 ms; the CSV reports the minimum of 100 samples
in microseconds **per complete batch**, and allocated bytes per call.
`minimum_us` is the lowest sampled per-call time after warm-up, an estimate
of execution under minimal interference rather than typical latency.
`speedup_vs_scalar` is the ratio of the scalar and batched minima; update
rows use `NaN` there.
No BLAS work occurs in the solver kernels; one BLAS thread is set for reproducibility.

Results: [CSV](results/helmoltz-cpu.csv), [environment](results/environment.txt),
[solve figure](results/helmoltz-solves.png), [update figure](results/helmoltz-updates.png).
The main README reports both solve and update speedups. CPU assembly and
factorisation sweep contiguous systems with SIMD; warmed batched updates
allocate zero bytes, including coefficient and pivot validation.

## Substitution-only microbenchmark

```sh
julia --startup-file=no --threads=1 --project perf/benchmark_batched.jl perf/results/quasitridiag-cpu.csv
```

This measures only even-parity UL substitutions, with RHS reset included.
It excludes assembly, the odd block and all physical-space transforms.
`packed` uses native system-first storage. Its speedup is not interchangeable
with the full Helmholtz benchmark.
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

The CUDA tests have passed on an A100 80 GB PCIe with scalar indexing
disabled. The target includes 7,458 CUDA checks plus CPU references. Singular
Neumann Poisson batches remain unsupported on CUDA.

## Source provenance

Commit solver and benchmark sources before recording a comparison. The device
harness takes an explicit output path and records `CHEB_SOURCE_COMMIT`; set
it to the measured revision as shown above. The historical scalar-comparison
harness defaults to `results/<full-commit>/helmoltz-cpu.csv`. Environment
sidecars record the measured commit and execution environment. Existing CSV files are not overwritten: supply
a fresh output path for repeated measurements at the same commit. Version tags
are not required; an ordinary source commit is sufficient.

The earlier results are also archived under `results/e172199923eec97924d8e9a8040277813a2b0ff2/`.
Their provenance file distinguishes the commit that stored the results from an
unrecorded measurement revision. They must not be presented as measurements of
newer solver changes. The historical results do not describe the new coupled solver.
