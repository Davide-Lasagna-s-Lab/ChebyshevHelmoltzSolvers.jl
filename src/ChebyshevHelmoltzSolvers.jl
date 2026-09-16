"""
    ChebyshevHelmoltzSolvers

Chebyshev coefficient operations and tau solvers on `[-1, 1]`. Scalar
Helmholtz systems use even/odd quasi-tridiagonal UL factors; the coupled
solver adds an influence correction to impose velocity and derivative
boundary conditions.
"""
module ChebyshevHelmoltzSolvers

using LinearAlgebra
import FFTW

include("chebcoeffs.jl")
include("quasitridiag.jl")
include("helmoltz.jl")
include("coupled.jl")

end
