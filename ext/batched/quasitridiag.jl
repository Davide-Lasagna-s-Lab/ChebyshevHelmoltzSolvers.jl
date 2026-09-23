#//////////////////////////////////////////////////////////////////////////////#
#///                         CUDA UL FACTORISATION                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ul!(Q::CuBatchedQuasiTridiagonal)

Overwrite the batch with UL factors on the device. One CUDA thread owns one
system, and neighbouring threads access neighbouring systems. Launch on the
current stream and return `Q` without synchronising. Reassemble the original
matrix entries before calling again. Store reciprocal pivots in `Q.dᵢ` and `Q.b[:,1]`
after completing the UL recurrence. Pivot validation belongs to `update!`.
"""
function ul!(Q::CuBatchedQuasiTridiagonal{T, B}) where {T, B}
    threads = min(256, B)
    @cuda threads=threads blocks=cld(B, threads) _ul_kernel!(Q)
    return Q
end

function _ul_kernel!(Q::BatchedQuasiTridiagonal{T, B}) where {T, B}
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if s <= B
        _ul_system!(Q, s)
    end
    return nothing
end

#//////////////////////////////////////////////////////////////////////////////#
#///                         CUDA UL SUBSTITUTIONS                          ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    ldiv!(Q::CuBatchedQuasiTridiagonal, rhs)

Solve all factorised systems on the GPU, overwriting and returning `rhs`.
Factors and RHS must reside on the GPU; the RHS has shape `(B, M)` and must
not alias the factors. Real factors support real or complex right-hand sides.

Call `ul!` before solving. The factors remain unchanged. The operation is
asynchronous on the current CUDA stream.
"""
function ldiv!(  Q::CuBatchedQuasiTridiagonal{T, B, M},
               rhs::CUDA.StridedCuMatrix) where {T, B, M}
    #///////////////////////////////// CHECKS /////////////////////////////////#
    # Shape and stride establish valid, contiguous accesses across systems.
    Base.require_one_based_indexing(rhs)
    size(rhs) == size(Q.b) || throw(DimensionMismatch("RHS must have size $(size(Q.b))"))
    stride(rhs, 1) == 1 || throw(ArgumentError("RHS systems must be contiguous"))

    # The solve overwrites only rhs; its storage must not contain the factors.
    any(a -> Base.mightalias(rhs, a), (Q.b, Q.l, Q.dᵢ, Q.u)) &&
        throw(ArgumentError("RHS must not alias the factors"))
    #//////////////////////////////////////////////////////////////////////////#

    # One thread performs both substitution passes for one system;
    # neighbouring threads access neighbouring systems in the factor arrays.
    threads = min(256, B)
    @cuda threads=threads blocks=cld(B, threads) _ldiv_kernel!(Q, rhs)
    return rhs
end

function _ldiv_kernel!(Q::BatchedQuasiTridiagonal{T, B}, rhs) where {T, B}
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if s <= B
        _ldiv_system!(Q, rhs, s, 1, 1)
    end
    return nothing
end

# `first` and `step` select a complete RHS or either parity of a Chebyshev
# expansion. The dense boundary row accumulates its residual in rhs[s, first];
# neither parity needs packing or a temporary array. (first, step) = (1, 1)
# selects a full RHS; (1, 2) and (2, 2) select even and odd Chebyshev degrees.
@inline function _ldiv_system!(Q::BatchedQuasiTridiagonal{T, B, M}, rhs, s, first, step) where {T, B, M}
    @inbounds begin
        # Solve U*z = rhs backwards. The last equation has no upper neighbour.
        # k indexes the factor, while j indexes the selected RHS coefficients.
        last = first + step * (M - 1)
        rhs[s, last] *= Q.dᵢ[s, M-1]
        rhs[s, first] -= rhs[s, last] * Q.b[s, M]
        # Reuse each computed z[k] to reduce the dense first-row residual.
        for k in M-1:-1:2
            j = first + step * (k - 1)
            rhs[s, j] = (rhs[s, j] - Q.u[s, k - 1] * rhs[s, j + step]) * Q.dᵢ[s, k-1]
            rhs[s, first] -= rhs[s, j] * Q.b[s, k]
        end
        # b[s,1] stores the reciprocal first pivot, not the pivot itself.
        rhs[s, first] = rhs[s, first] * Q.b[s, 1]

        # Solve L*x = z forwards: the previous selected entry already holds x.
        # Unit diagonal means there is no division or reciprocal scaling.
        for k in 2:M
            j = first + step * (k - 1)
            rhs[s, j] -= Q.l[s, k - 1] * rhs[s, j - step]
        end
    end
    return nothing
end
