export BatchedHelmoltzSolver

#//////////////////////////////////////////////////////////////////////////////#
#///                       BATCHED HELMHOLTZ FACTORS                        ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    BatchedHelmoltzSolver(P, B, T=Float64; neum=false, a=-1, b=1)

Allocate storage for `B` independent problems `θ₀[s]*uₛ'' - θ₁[s]*uₛ = fₛ`
on `[a, b]`, using precision `T`. All systems share this interval; operator
coefficients and Neumann data refer to physical derivatives. `P ≥ 3` is the polynomial degree, so
there are `P + 1` coefficients per solution. The positive batch size `B` is
stored as a type parameter; factors use `(system, coefficient)` storage.

Call `update!(h, θ₀, θ₁)` before the first solve and whenever the operator
coefficients change. Both coefficients must be vectors of length `B` and
eltype `T`. Construction only allocates storage;
`update!` assembles and factorises the operators and checks their UL pivots.

Boundary data are Dirichlet values by default, or positive-y derivatives at
both walls when `neum=true`. Pure Neumann Poisson problems need a separate
mean-mode treatment.

The batch owns no RHS workspace. With CUDA loaded, `Adapt.adapt(CuArray, h)`
transfers its factors without changing precision. Updates subsequently
assemble and factor in the existing CPU or GPU arrays.
"""
struct BatchedHelmoltzSolver{T, B, Q<:BatchedQuasiTridiagonal{T, B}, R<:BatchedQuasiTridiagonal{T, B}, V<:AbstractVector{T}}
    scale::T            # physical derivative scale 2/(b-a), shared by the batch
     neum::Bool         # prescribe positive-y derivatives instead of wall values
       Be::Q            # UL factors for the even Chebyshev coefficients, system first
       Bo::R            # UL factors for the odd Chebyshev coefficients, system first
    cache::NTuple{3, V} # integration weights (l, d, u), shared by all systems
end

function BatchedHelmoltzSolver(   P::Int,
                                  B::Int,
                                   ::Type{T}=Float64;
                               neum::Bool=false, a=-1, b=1) where {T<:AbstractFloat}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Both parity blocks must contain at least two coefficients.
    P ≥ 3 || throw(ArgumentError("P must be at least 3"))
    #//////////////////////////////////////////////////////////////////////////#

    scale = _intervalscale(a, b, T)

    # Allocate B independent factor banks for each parity. Even degrees
    # include zero, giving one extra coefficient when P is even. update!
    # will assemble and factorise the operators in this storage.
    Be = BatchedQuasiTridiagonal(B, div(P, 2) + 1, T)
    Bo = BatchedQuasiTridiagonal(B, div(P+1, 2), T)

    # Cache integration weights once; index 1 is unused. The helpers
    # include the degree-zero correction and the terminal tau cutoff.
    l = T[p == 1 ? 0 : _c(p-2)/(4p*(p-1))    for p in 1:P]
    d = T[p == 1 ? 0 : _β(p, P)/(2*(p^2-1))  for p in 1:P]
    u = T[p == 1 ? 0 : _β(p+2, P)/(4p*(p+1)) for p in 1:P]
    cache = (l, d, u)

    return BatchedHelmoltzSolver(scale, neum, Be, Bo, cache)
end

#//////////////////////////////////////////////////////////////////////////////#
#///                           STORAGE ADAPTATION                           ///#
#//////////////////////////////////////////////////////////////////////////////#

# Transfer factor banks and cached integration weights to the same device.
function Adapt.adapt_structure(to, h::BatchedHelmoltzSolver)
    return BatchedHelmoltzSolver(h.scale, h.neum,
                                Adapt.adapt(to, h.Be),
                                Adapt.adapt(to, h.Bo),
                                Adapt.adapt(to, h.cache))
end

#//////////////////////////////////////////////////////////////////////////////#
#///                            OPERATOR UPDATES                            ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    update!(h::BatchedHelmoltzSolver, θ₀::AbstractVector, θ₁::AbstractVector)

Reassemble and UL-factorise the operators in the existing factor arrays.
Both coefficient vectors must match the batch size and precision. CPU
updates operate directly on the batched storage; GPU updates assemble and
factor on device. No replacement solver or factor
bank is constructed. Coefficients must not alias the factor storage. CPU
coefficient vectors passed to a GPU batch are uploaded; device vectors avoid
those transfers.

Check coefficient validity before modifying storage, then check every pivot
after factorisation. A failed pivot check leaves invalid factors: supply a
valid operator with another `update!` before solving. Return `h`.
"""
function update!( h::BatchedHelmoltzSolver{T, B, Q},
                 θ₀::AbstractVector{T},
                 θ₁::AbstractVector{T}) where {T, B, M, Q<:BatchedQuasiTridiagonal{T, B, M, Matrix{T}}}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # CPU factors require host-accessible coefficient vectors.
    θ₀ isa Union{StridedVector, AbstractRange} && θ₁ isa Union{StridedVector, AbstractRange} ||
        throw(ArgumentError("CPU factors require CPU coefficient vectors"))
    # Coefficients must enumerate this fixed-size batch without offset indices.
    Base.require_one_based_indexing(θ₀, θ₁)
    length(θ₀) == length(θ₁) == B || throw(DimensionMismatch("expected $B operator coefficients"))

    # Views of external coefficient tables are allowed; views into factors
    # would be overwritten before all parity systems have read their values.
    for a in (h.Be.b, h.Be.l, h.Be.dᵢ, h.Be.u,
              h.Bo.b, h.Bo.l, h.Bo.dᵢ, h.Bo.u)
        (Base.mightalias(θ₀, a) || Base.mightalias(θ₁, a)) &&
            throw(ArgumentError("operator coefficients must not alias the factors"))
    end

    # Nonfinite coefficients and the singular Neumann mean mode are rejected
    # before assembly. CUDA implements these predicates as device reductions.
    all(isfinite, θ₀) && all(isfinite, θ₁) ||
        throw(ArgumentError("operator coefficients must be finite"))
    h.neum && any(iszero, θ₁) &&
        throw(ArgumentError("the pure Neumann Poisson operator requires a separate mean-mode solve"))
    #//////////////////////////////////////////////////////////////////////////#

    # Assemble and factor each parity across contiguous systems. No scalar
    # wrappers or row views are constructed for individual operators.
    _assemble_helmoltz!(h.Be, h.cache, θ₀, θ₁, 2, h.neum, h.scale); ul!(h.Be)
    _assemble_helmoltz!(h.Bo, h.cache, θ₀, θ₁, 3, h.neum, h.scale); ul!(h.Bo)

    #///////////////////////////////// CHECKS /////////////////////////////////#
    # UL is unpivoted: validate the new factors after every update before
    # they can be used by a solve.
    # Also reject overflowing reciprocals of tiny pivots. These predicates
    # use CPU loops or CUDA reductions, without copying the factor arrays.
    # The first column of b contains each inverse boundary pivot.
    # Iterate arrays rather than differently sized parity wrappers, keeping
    # every check concretely typed even when the two blocks have unequal sizes.
    for a in (h.Be.b, h.Be.l, h.Be.dᵢ, h.Be.u,
              h.Bo.b, h.Bo.l, h.Bo.dᵢ, h.Bo.u)
        all(isfinite, a) ||
            throw(ArgumentError("the batch has a zero or nonfinite reciprocal UL pivot"))
    end
    all(!iszero, view(h.Be.b, :, 1)) && all(!iszero, h.Be.dᵢ) &&
        all(!iszero, view(h.Bo.b, :, 1)) && all(!iszero, h.Bo.dᵢ) ||
        throw(ArgumentError("the batch has a zero or nonfinite reciprocal UL pivot"))
    #//////////////////////////////////////////////////////////////////////////#

    return h
end

# Assemble the same entries as the scalar method, with systems in the
# inner loop so each pass writes contiguous memory and can use SIMD.
function _assemble_helmoltz!(Q::BatchedQuasiTridiagonal{T, B, M, Matrix{T}},
                            cache, θ₀, θ₁, p₀, neum, scale) where {T, B, M}
    l, d, u = cache
    @inbounds for i in 1:M
        n = p₀ - 2 + 2*(i-1)
        boundary = neum ? scale*n^2 : 1
        @simd for s in 1:B
            Q.b[s, i] = boundary
        end
    end
    @inbounds for i in 1:M-1
        p = p₀ + 2*(i-1)
        @simd for s in 1:B
            Q.l[s, i] = -θ₁[s]*l[p]
            Q.dᵢ[s, i] = θ₀[s]*scale^2 + θ₁[s]*d[p]
        end
        if i < M-1
            @simd for s in 1:B
                Q.u[s, i] = -θ₁[s]*u[p]
            end
        end
    end
    return Q
end

#//////////////////////////////////////////////////////////////////////////////#
#///                          DIRECT FIELD SOLVES                           ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    solve!(h::BatchedHelmoltzSolver, u, f, u₊::AbstractVector, u₋::AbstractVector)

Solve into `u`, preserving `f`. Both arguments are matrices of size
`(B, P+1)`: row `s` holds system `s`, and column `n+1` holds Chebyshev
coefficient `n`. Rows must be contiguous in storage. Reshape higher-dimensional
fields at the call site. Real factors support real or complex data of matching
precision.

`u₊` and `u₋` are one-based vectors of length `B`. Entry `s` prescribes
system `s`'s values at `b` and `a`, or positive-y derivatives when
`neum=true`. They must not alias `u`. A custom constant-valued vector may
supply homogeneous conditions without allocating boundary storage. Vectors
must support indexing on the selected backend; for CUDA, any stored data
must be device-accessible and custom vectors must be kernel-compatible.

The source and destination must not alias. The integrated RHS is written
directly into `u`, then its even/odd coefficients are solved in place,
without packing or an additional RHS array. The two highest forcing
coefficients are tau terms and do not enter the equations.

The CPU solve processes the full batch with SIMD across systems. CUDA
launches one thread per system and returns asynchronously on the current
stream. Return `u`.
"""
function solve!( h::BatchedHelmoltzSolver{T, B, Q},
                 u::AbstractMatrix,
                 f::AbstractMatrix,
                u₊::AbstractVector,
                u₋::AbstractVector) where {T, B, M, Q<:BatchedQuasiTridiagonal{T, B, M, Matrix{T}}}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # CPU factors require host fields with strided storage.
    u isa StridedMatrix && f isa StridedMatrix ||
        throw(ArgumentError("CPU factors require strided CPU input and output matrices"))
    # Each row is a system and each column is a Chebyshev coefficient.
    Base.require_one_based_indexing(u, f)
    Ny = size(h.Be.b, 2) + size(h.Bo.b, 2)
    size(u) == size(f) == (B, Ny) ||
        throw(DimensionMismatch("expected matrices of size ($B, $Ny)"))

    # SIMD and GPU kernels use contiguous systems and disjoint columns.
    for A in (u, f)
        stride(A, 1) == 1 || throw(ArgumentError("system rows must be contiguous"))
        stride(A, 2) ≥ B || throw(ArgumentError("coefficient columns must not overlap"))
    end

    # Array wall data follow the same linear system ordering as the fields.
    for bc in (u₊, u₋)
        Base.require_one_based_indexing(bc)
        length(bc) == B ||
            throw(DimensionMismatch("boundary data must have $B entries"))
        Base.mightalias(u, bc) && throw(ArgumentError("u must not alias boundary data"))
    end

    # Only u is overwritten. Protect the forcing, factors and cache.
    Base.mightalias(u, f) && throw(ArgumentError("source and destination must not alias"))
    for factors in (h.Be, h.Bo), a in (factors.b, factors.l, factors.dᵢ, factors.u)
        Base.mightalias(u, a) && throw(ArgumentError("destination must not alias the factors"))
    end
    any(a -> Base.mightalias(u, a), h.cache) &&
        throw(ArgumentError("destination must not alias the integration weights"))
    #//////////////////////////////////////////////////////////////////////////#

    P = size(f, 2) - 1
    l, d, upper_weight = h.cache
    @inbounds @simd for s in 1:B
        # Differentiation exchanges parity, swapping Neumann wall combinations.
        up, lo = u₊[s], u₋[s]
        u[s, 1] = 0.5 * (h.neum ? up-lo : up+lo)
        u[s, 2] = 0.5 * (h.neum ? up+lo : up-lo)
    end
    @inbounds for p in 2:P
        @simd for s in 1:B
            high = p+2 ≤ P-2 ? f[s, p+3] : zero(eltype(f))
            u[s, p+1] = l[p]*f[s, p-1] - d[p]*f[s, p+1] + upper_weight[p]*high
        end
    end

    # Parity views double the column stride while retaining contiguous systems.
    ldiv!(h.Be, view(u, :, 1:2:P+1))
    ldiv!(h.Bo, view(u, :, 2:2:P+1))
    return u
end
