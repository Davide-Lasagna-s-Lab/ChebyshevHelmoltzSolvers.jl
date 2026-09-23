#//////////////////////////////////////////////////////////////////////////////#
#///                    CUDA COUPLED INFLUENCE CORRECTION                    ///#
#//////////////////////////////////////////////////////////////////////////////#

function _influence!(h::BatchedCoupledHelmoltzSolver{T, H, <:CuArray}) where {T, H}
    B = size(h.A, 1)
    threads = min(256, B)
    @cuda threads=threads blocks=cld(B, threads) _influence_kernel!(h)
    return h
end

function _influence_kernel!(h)
    s = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    if s <= size(h.A, 1)
        _influence_system!(h, s)
    end
    return nothing
end

function _correct!(h::BatchedCoupledHelmoltzSolver{T, H, <:CuArray}, u) where {T, H}
    B = size(u, 1)
    threads = min(256, B)
    @cuda threads=threads blocks=cld(B, threads) _correct_kernel!(h, u)
    return u
end

# One thread computes both wall derivatives for its system, solves the 2x2
# influence problem and applies its two cached responses in place.
function _correct_kernel!(h, u)
    s = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    if s <= size(u, 1)
        _, v₊, v₋ = h.vₛ
        up = lo = zero(eltype(u))
        @inbounds for n in 1:size(u, 2)-1
            up -= n^2*u[s, n+1]
            lo -= (isodd(n) ? 1 : -1)*n^2*u[s, n+1]
        end
        @inbounds begin
            δ₊ = h.A[s, 1]*up + h.A[s, 3]*lo
            δ₋ = h.A[s, 2]*up + h.A[s, 4]*lo
            for n in 1:size(u, 2)
                u[s, n] += δ₊*v₊[s, n] + δ₋*v₋[s, n]
            end
        end
    end
    return nothing
end
