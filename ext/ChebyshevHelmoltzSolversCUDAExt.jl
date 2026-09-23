module ChebyshevHelmoltzSolversCUDAExt

#//////////////////////////////////////////////////////////////////////////////#
#///                      CUDA DEPENDENCIES AND TYPES                       ///#
#//////////////////////////////////////////////////////////////////////////////#

using CUDA
using ChebyshevHelmoltzSolvers: BatchedHelmoltzSolver, BatchedQuasiTridiagonal, BatchedCoupledHelmoltzSolver
import ChebyshevHelmoltzSolvers: _ul_system!, solve!,
                                update!, ul!, _check_precision, _influence!, _influence_system!, _correct!, _host_storage
import LinearAlgebra: ldiv!

_host_storage(::CUDA.AnyCuArray) = false

const CuBatchedQuasiTridiagonal{T, B, M} = BatchedQuasiTridiagonal{T, B, M, A} where {A<:CuArray{T, 2}}

#//////////////////////////////////////////////////////////////////////////////#
#///                              CUDA METHODS                              ///#
#//////////////////////////////////////////////////////////////////////////////#

include("batched/quasitridiag.jl")
include("batched/helmoltz.jl")
include("batched/coupled.jl")

end
