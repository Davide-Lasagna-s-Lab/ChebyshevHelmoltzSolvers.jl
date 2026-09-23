"""
    ChebyshevHelmoltzSolvers

Chebyshev coefficient operations and tau solvers on `[-1, 1]`. Scalar
Helmholtz systems use even/odd quasi-tridiagonal UL factors; the coupled
solver adds an influence correction to impose velocity and derivative
boundary conditions.
"""
module ChebyshevHelmoltzSolvers

#//////////////////////////////////////////////////////////////////////////////#
#///                              DEPENDENCIES                              ///#
#//////////////////////////////////////////////////////////////////////////////#

using LinearAlgebra
import FFTW
import Adapt

#//////////////////////////////////////////////////////////////////////////////#
#///                           PACKAGE COMPONENTS                           ///#
#//////////////////////////////////////////////////////////////////////////////#

include("differentiation.jl")
include("transforms.jl")
include("quasitridiag.jl")
include("batched/quasitridiag.jl")
include("helmoltz.jl")
include("batched/helmoltz.jl")
include("coupled.jl")

end
