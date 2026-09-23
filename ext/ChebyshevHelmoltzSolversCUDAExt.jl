module ChebyshevHelmoltzSolversCUDAExt

#//////////////////////////////////////////////////////////////////////////////#
#///                      CUDA DEPENDENCIES AND TYPES                       ///#
#//////////////////////////////////////////////////////////////////////////////#

using CUDA
using ChebyshevHelmoltzSolvers: BatchedHelmoltzSolver, BatchedQuasiTridiagonal
import ChebyshevHelmoltzSolvers: _ul_system!, solve!,
                                update!, ul!, _check_precision
import LinearAlgebra: ldiv!

const CuBatchedQuasiTridiagonal{T, B, M} = BatchedQuasiTridiagonal{T, B, M, A} where {A<:CuArray{T, 2}}

#//////////////////////////////////////////////////////////////////////////////#
#///                              CUDA METHODS                              ///#
#//////////////////////////////////////////////////////////////////////////////#

include("batched/quasitridiag.jl")
include("batched/helmoltz.jl")

end
