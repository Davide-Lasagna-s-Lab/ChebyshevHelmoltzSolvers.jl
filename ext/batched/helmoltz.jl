#//////////////////////////////////////////////////////////////////////////////#
#///                         CUDA OPERATOR UPDATES                          ///#
#//////////////////////////////////////////////////////////////////////////////#

# Reuse device coefficient vectors directly. Host vectors need only this
# small upload; updating never replaces or copies the full factor arrays.
_device_coefficients(values::CUDA.AnyCuVector) = values
_device_coefficients(values::AbstractVector) = CuArray(values)

function update!( h::BatchedHelmoltzSolver{T, B, Q},
                 θ₀::AbstractVector{T},
                 θ₁::AbstractVector{T}) where {T, B, Q<:CuBatchedQuasiTridiagonal{T, B}}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Coefficients must enumerate this fixed-size batch without offset indices.
    Base.require_one_based_indexing(θ₀, θ₁)
    length(θ₀) == length(θ₁) == B || throw(DimensionMismatch("expected $B operator coefficients"))

    # Views of external coefficient tables are allowed; views into factors
    # would be overwritten before all parity systems have read their values.
    for factors in (h.Be, h.Bo), a in (factors.b, factors.l, factors.dᵢ, factors.u)
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

    θ₀_device = _device_coefficients(θ₀)
    θ₁_device = _device_coefficients(θ₁)
    threads = min(256, B)
    @cuda threads=threads blocks=cld(B, threads) _update_kernel!(h, θ₀_device, θ₁_device)

    #///////////////////////////////// CHECKS /////////////////////////////////#
    # UL is unpivoted: validate the new factors after every update before
    # they can be used by a solve.
    # Also reject overflowing reciprocals of tiny pivots. These predicates
    # use CPU loops or CUDA reductions, without copying the factor arrays.
    # The first column of b contains each inverse boundary pivot.
    for factors in (h.Be, h.Bo)
        all(a -> all(isfinite, a), (factors.b, factors.l, factors.dᵢ, factors.u)) &&
            all(!iszero, view(factors.b, :, 1)) && all(!iszero, factors.dᵢ) ||
            throw(ArgumentError("the batch has a zero or nonfinite reciprocal UL pivot"))
    end
    #//////////////////////////////////////////////////////////////////////////#

    return h
end

# Assembly and factorisation share one launch. Every thread overwrites the
# old entries of its two parity systems before applying the UL recurrence.
function _update_kernel!(h::BatchedHelmoltzSolver{T, B}, θ₀, θ₁) where {T, B}
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if s <= B
        @inbounds begin
            _assemble_operator!(h.Be, h, s, 2, θ₀[s], θ₁[s])
            _assemble_operator!(h.Bo, h, s, 3, θ₀[s], θ₁[s])
        end
        _ul_system!(h.Be, s)
        _ul_system!(h.Bo, s)
    end
    return nothing
end

# The boundary row contains T_n(+1), or T_n'(+1) for Neumann data. Interior
# rows contain the same integrated equations as in the scalar solver.
@inline function _assemble_operator!(Q::BatchedQuasiTridiagonal{T, B, M}, h, s, p₀, θ₀, θ₁) where {T, B, M}
    l, d, u = h.cache
    @inbounds begin
        for i in 1:M
            n = p₀ - 2 + 2*(i-1)
            Q.b[s, i] = h.neum ? n^2 : 1
        end
        for i in 1:M-1
            p = p₀ + 2*(i-1)
            Q.l[s, i] = -θ₁*l[p]
            Q.dᵢ[s, i] = θ₀ + θ₁*d[p]
            i < M-1 && (Q.u[s, i] = -θ₁*u[p])
        end
    end
    return nothing
end

#//////////////////////////////////////////////////////////////////////////////#
#///                         CUDA HELMHOLTZ SOLVES                          ///#
#//////////////////////////////////////////////////////////////////////////////#

function solve!( h::BatchedHelmoltzSolver{T, B, Q},
                 u::AbstractMatrix,
                 f::AbstractMatrix,
                u₊::AbstractVector,
                u₋::AbstractVector) where {T, B, Q<:CuBatchedQuasiTridiagonal{T, B}}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # CUDA factors require device input and output fields.
    u isa CUDA.StridedCuMatrix && f isa CUDA.StridedCuMatrix ||
        throw(ArgumentError("CUDA factors require CUDA input and output matrices"))
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

    threads = min(256, B)
    @cuda threads=threads blocks=cld(B, threads) _solve_kernel!(h, u, f, u₊, u₋)
    return u
end

# Launches use CUDA.jl's current stream. Each thread assembles and solves
# one full expansion in the destination, preserving the source coefficients.
function _solve_kernel!(h::BatchedHelmoltzSolver{T, B},
                        u, f, u₊, u₋) where {T, B}
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if s <= B
        _assemble_system!(h, u, f, u₊, u₋, s)
        _ldiv_system!(h.Be, u, s, 1, 2)
        _ldiv_system!(h.Bo, u, s, 2, 2)
    end
    return nothing
end

# Ordinary Chebyshev coefficients occupy the last array dimension. The
# first two entries impose the even/odd combinations of wall conditions.
@inline function _assemble_system!(h::BatchedHelmoltzSolver{T},
                                   u, f, u₊, u₋, s) where {T}
    S = eltype(u)
    P = size(f, 2) - 1
    l, d, upper_weight = h.cache
    up, lo = u₊[s], u₋[s]
    @inbounds begin
        u[s, 1] = (h.neum ? up - lo : up + lo) * T(0.5)
        u[s, 2] = (h.neum ? up + lo : up - lo) * T(0.5)
        for p in 2:P
            # The two highest residual coefficients are tau terms, excluded
            # from the integrated equations exactly as in the scalar solve.
            high = p + 2 <= P - 2 ? f[s, p + 3] : zero(S)
            u[s, p + 1] = l[p]*f[s, p - 1] - d[p]*f[s, p + 1] + upper_weight[p]*high
        end
    end
    return nothing
end
