# Batched CPU and A100 benchmarks

Solver and benchmark source: [`92072b170312604ed4f60cfcfcca4d5a08084af5`](https://github.com/Davide-Lasagna-s-Lab/ChebyshevHelmoltzSolvers.jl/commit/92072b170312604ed4f60cfcfcca4d5a08084af5).
The subsequent documentation/results commit does not change the measured solver.

- `local-cpu.csv`: Apple M5 MacBook Air, Julia 1.12.6.
- `a100.csv`: A100 80 GB PCIe and the batched CPU baseline measured on its
  compute node, Intel Xeon Gold 6336Y. Julia 1.12.4, CUDA.jl 6.4.0.
- Environment sidecars record execution hosts, Slurm allocations, full CUDA
  toolchain information, and the benchmark script SHA256.
- [CPU tests](local-tests.log): 15,724 passed.
- [A100 tests](a100-tests.log): 7,458 CUDA checks plus 7,241 CPU references passed;
  CUDA scalar indexing is disabled.

Both sweeps use ComplexF64 fields, native `(system, coefficient)` storage,
one Julia/BLAS thread, and the minimum of 500 warmed sample averages.
The GPU speedup uses its **same-node CPU baseline**, never the local Mac.
CPU batching uses SIMD across systems. CPU and GPU do the same solves.

`Ny` is the number of Chebyshev coefficients (`P+1`). The sweep spans
8, 16, 32, 64, 128, 256, 512, 1024 coefficients and 64, 256, 1024,
4096, 16384, 65536 systems for each solver. CSV times are seconds per batch;
figures divide time by the number of systems. Solve and update timings are
separate. Coupled updates include rebuilding the cached influence responses.

Solver/field construction, compilation and host/device transfers are outside
the timed region; work performed inside the public calls remains included.
The measured revision's environment message says "exclude allocation"; this
means setup allocation, not allocations inside the measured call. Later
metadata wording clarifies this without changing the timing procedure.
GPU samples include synchronization. Each GPU configuration is compared with
the CPU result before its timings are recorded. These timings estimate minimum
execution cost, not typical latency, statistical confidence intervals, or
end-to-end DNS performance.

Terminal colour codes and trailing spaces are removed from the test logs;
test results and diagnostics are retained.

Reproduce using [the benchmark instructions](../../README.md).
Figures are generated only from the saved CSV files by `perf/plot_devices.py`.

## A100 run provenance

`a100.csv` combines completed rows from three allocations on the same `rose07`
node. The `source_file` column points to the unmodified raw CSV in `runs/`;
each raw CSV has its own environment sidecar:

- `a100-92072b1.csv`: allocation 1634058, 89 configurations.
- `a100-92072b1-cont1634168-coupled-ny512.csv`: allocation 1634168,
  coupled `Ny=512, B=65536`.
- `a100-92072b1-last1636271.csv`: allocation 1636271, six coupled
  `Ny=1024` configurations.

All use solver revision `92072b1` and the identical benchmark script SHA256
`0bc4d44a153258e9200e9d85530e0005f5b9ce697bdc1db46c1e8ceeb7701773`.
The combined file contains 96 distinct configurations, with 500 samples per
operation, and no repeated or extrapolated rows. The CPU baseline and GPU
measurement for each row were taken together in its originating run.

## Local memory limit

The local CSV contains 95 complete configurations. The largest coupled case
(`Ny=1024, B=65536`) was stopped before writing a row because it triggered
substantial memory pressure on the 16 GB Mac. See [the observation record](local-memory-limit.txt).
No missing point is interpolated or silently reported as a valid timing.
