#//////////////////////////////////////////////////////////////////////////////#
#///                           TEST DEPENDENCIES                            ///#
#//////////////////////////////////////////////////////////////////////////////#

using ChebyshevHelmoltzSolvers
using LinearAlgebra
using Test

#//////////////////////////////////////////////////////////////////////////////#
#///                       ANALYTIC REFERENCE HELPERS                       ///#
#//////////////////////////////////////////////////////////////////////////////#

include("helpers.jl")

#//////////////////////////////////////////////////////////////////////////////#
#///                             CPU TEST SUITE                             ///#
#//////////////////////////////////////////////////////////////////////////////#

@testset "ChebyshevHelmoltzSolvers" begin
    @testset "Coefficient operations" begin
        include("test_differentiation.jl")
        include("test_transforms.jl")
    end
    @testset "Factorisation and substitution" begin
        include("test_quasitridiag.jl")
        include("batched/quasitridiag.jl")
    end
    @testset "Boundary-value solvers" begin
        include("test_helmoltz.jl")
        include("batched/helmoltz.jl")
        include("test_coupled.jl")
        include("test_poisson.jl")
    end
    @testset "Contracts and numerical accuracy" begin
        include("test_contracts.jl")
        include("test_accuracy.jl")
    end
end
