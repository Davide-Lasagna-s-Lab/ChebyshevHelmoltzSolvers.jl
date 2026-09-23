# Test suite

Run the CPU suite from the package root:

```sh
julia --startup-file=no --project -e 'using Pkg; Pkg.test()'
```

Tests are deterministic and organized by numerical responsibility. Test helpers
live in `helpers.jl`; they are not part of the package API.

| Files | What is established |
| --- | --- |
| `test_differentiation.jl`, `test_transforms.jl` | Coefficient conventions, endpoint derivatives, views, transforms and examples. |
| `test_quasitridiag.jl`, `batched/quasitridiag.jl` | Factor reconstruction, substitution, precision, repeated solves and batched layout. |
| `test_helmoltz.jl`, `batched/helmoltz.jl` | Manufactured solutions, both parities and boundary types, tau truncation, factor reuse and allocations. |
| `test_coupled.jl` | Clamped fourth-order solutions, real/complex workspaces and cached influence responses. |
| `test_contracts.jl` | Precision and storage contracts, uninitialised/broken factors, recovery after failed updates and fixed-domain API. |
| `test_poisson.jl` | Neumann compatibility, zero-mean gauge, incompatible data and mixed singular/shifted CPU batches. |
| `test_accuracy.jl` | Independent dense tau reference, normalized residuals, spectral convergence, high degree and near-singular shifts. |

Manufactured polynomial and smooth solutions check the differential equations
independently of the implementation. A dense differentiation matrix supplies a
second reference that does not reuse the integration-weight recurrence.
Convergence tests evaluate at off-grid points and allow a floating-point error
plateau. Tolerances distinguish Float32 and Float64; near-singular problems are
not expected to have uniformly small forward errors for arbitrary data.

Allocation tests warm methods before measurement. Timing comparisons belong in
`perf/`, not in correctness tests, and do not impose machine-dependent CI limits.

## CUDA

GPU tests remain separate in `cuda/` and require a functional NVIDIA device.
They are not part of the CPU CI claim. GPU execution and singular Neumann support
will be validated separately on the A100; no device results are inferred from
CPU tests.
