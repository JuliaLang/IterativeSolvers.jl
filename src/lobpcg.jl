#=
The code below was derived from the scipy implementation of the LOBPCG algorithm in https://github.com/scipy/scipy/blob/v1.1.0/scipy/sparse/linalg/eigen/lobpcg/lobpcg.py#L109-L568.

Since the link above mentions the license is BSD license, the notice for the BSD license 2.0 is hereby given below giving credit to the authors of the Python implementation.

Copyright (c) 2018, Robert Cimrman, Andrew Knyazev
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

export lobpcg, lobpcg!, LOBPCGIterator

using LinearAlgebra
using Random

struct LOBPCGState{TR,TL}
    iteration::Int
    residual_norms::TR
    ritz_values::TL
end
function Base.show(io::IO, t::LOBPCGState)
    @printf io "%8d    %14e\n" t.iteration maximum(t.residual_norms)
    return
end

const LOBPCGTrace{TR,TL} = Vector{LOBPCGState{TR,TL}}
function Base.show(io::IO, tr::LOBPCGTrace)
    @printf io "Iteration    Maximum residual norm \n"
    @printf io "---------    ---------------------\n"
    for state in tr
        show(io, state)
    end
    return
end

struct LOBPCGResults{TL, TX, T, TR, TI, TM, TB, TTrace <: Union{LOBPCGTrace, AbstractVector{<:LOBPCGTrace}}}
    λ::TL
    X::TX
    tolerance::T
    residual_norms::TR
    iterations::TI
    maxiter::TM
    converged::TB
    trace::TTrace
end
function EmptyLOBPCGResults(X::TX, k::Integer, tolerance, maxiter) where {T, TX<:AbstractMatrix{T}}
    blocksize = size(X,2)
    λ = Vector{T}(undef, k)
    X = TX(undef, size(X, 1), k)

    iterations = zeros(Int, ceil(Int, k/blocksize))
    residual_norms = copy(λ)
    converged = falses(k)
    trace = fill(LOBPCGTrace{Vector{real(T)},Vector{T}}(), k÷blocksize+1)

    return LOBPCGResults(λ, X, tolerance, residual_norms, iterations, maxiter, converged, trace)
end

function Base.append!(r1::LOBPCGResults, r2::LOBPCGResults, n1, n2=length(r2.λ))
    n = n1 + n2
    r1.λ[n1+1:n] .= @view r2.λ[end-n2+1:end]
    r1.residual_norms[n1+1:n] .= @view r2.residual_norms[end-n2+1:end]
    r1.X[:, n1+1:n] .= @view r2.X[:, end-n2+1:end]

    ind = n1 ÷ length(r2.λ) + 1
    r1.iterations[ind] = r2.iterations
    r1.converged[n1+1:n] .= r2.converged
    r1.trace[ind] = r2.trace

    return r1
end
function Base.show(io::IO, r::LOBPCGResults)
    first_two(fr) = [x for (i, x) in enumerate(fr)][1:2]

    @printf io "Results of LOBPCG Algorithm\n"
    @printf io " * Algorithm: LOBPCG - CholQR\n"

    if length(join(r.λ, ",")) < 40 || length(r.λ) <= 2
        @printf io " * λ: [%s]\n" join(r.λ, ",")
    else
        @printf io " * λ: [%s, ...]\n" join(first_two(r.λ), ",")
    end

    if length(join(r.residual_norms, ",")) < 40 || length(r.residual_norms) <= 2
        @printf io " * Residual norm(s): [%s]\n" join(r.residual_norms, ",")
    else
        @printf io " * Residual norm(s): [%s, ...]\n" join(first_two(r.residual_norms), ",")
    end
    @printf io " * Convergence\n"
    @printf io "   * Iterations: %s\n" r.iterations
    @printf io "   * Converged: %s\n" all(r.converged)
    @printf io "   * Iterations limit: %s\n" r.maxiter

    return
end

struct Blocks{Generalized, T, TA<:AbstractArray{T}}
    block::TA # X, R or P
    A_block::TA # AX, AR or AP
    B_block::TA # BX, BR or BP
end
Blocks(X, AX) = Blocks{false, eltype(X), typeof(X)}(X, AX, X)
Blocks(X, AX, BX) = Blocks{true, eltype(X), typeof(X)}(X, AX, BX)
function A_mul_X!(b::Blocks, A)
    mul!(b.A_block, A, b.block)
    return
end
function A_mul_X!(b::Blocks, A, n)
    @views mul!(b.A_block[:, 1:n], A, b.block[:, 1:n])
    return
end
function B_mul_X!(b::Blocks{true}, B)
    mul!(b.B_block, B, b.block)
    return
end
function B_mul_X!(b::Blocks{true}, B, n)
    @views mul!(b.B_block[:, 1:n], B, b.block[:, 1:n])
    return
end
function B_mul_X!(b::Blocks{false}, B, n = 0)
    return
end

mutable struct Constraint{T, TVorM<:Union{AbstractVector{T}, AbstractMatrix{T}}, TM<:AbstractMatrix{T}, TC}
    Y::TVorM
    BY::TVorM
    gram_chol::TC
    gramYBV::TM # to be used in view
    tmp::TM # to be used in view
end
struct BWrapper end
struct NotBWrapper end

Constraint(::Nothing, B, X) = Constraint(nothing, B, X, BWrapper())
Constraint(::Nothing, B, X, ::NotBWrapper) = Constraint(nothing, B, X, BWrapper())
function Constraint(::Nothing, B, X, ::BWrapper)
    return Constraint{Nothing, Matrix{Nothing}, Matrix{Nothing}, Nothing}(Matrix{Nothing}(undef,0,0), Matrix{Nothing}(undef,0,0), nothing, Matrix{Nothing}(undef,0,0), Matrix{Nothing}(undef,0,0))
end

Constraint(Y, B, X) = Constraint(Y, B, X, BWrapper())
function Constraint(Y, B, X, ::BWrapper)
    T = eltype(X)
    if B isa Nothing
        BY = Y
    else
        BY = similar(Y)
        mul!(BY, B, Y)
    end
    return Constraint(Y, BY, X, NotBWrapper())
end
function Constraint(Y, BY, X, ::NotBWrapper)
    T = eltype(X)
    if Y isa SubArray
        gramYBY = @view Matrix{T}(I, size(Y.parent, 2), size(Y.parent, 2))[1:size(Y, 2), 1:size(Y, 2)]
        mul!(gramYBY, adjoint(Y), BY)
        gramYBV = @view zeros(T, size(Y.parent, 2), size(X, 2))[1:size(Y, 2), :]
    else
        gramYBY = adjoint(Y)*BY
        gramYBV = zeros(T, size(Y, 2), size(X, 2))
    end
    realdiag!(gramYBY)
    gramYBY_chol = cholesky!(Hermitian(gramYBY))
    tmp = deepcopy(gramYBV)

    return Constraint{eltype(Y), typeof(Y), typeof(gramYBV), typeof(gramYBY_chol)}(Y, BY, gramYBY_chol, gramYBV, tmp)
end

function update!(c::Constraint, X, BX)
    sizeY = size(c.Y, 2)
    sizeX = size(X, 2)
    c.Y.parent[:, sizeY+1:sizeY+sizeX] .= X
    if X !== BX
        c.BY.parent[:, sizeY+1:sizeY+sizeX] .= BX
    end
    sizeY += sizeX
    Y = @view c.Y.parent[:, 1:sizeY]
    BY = @view c.BY.parent[:, 1:sizeY]
    c.Y = Y
    c.BY = BY
    gram_chol = c.gram_chol
    new_factors = @view gram_chol.factors.parent[1:sizeY, 1:sizeY]
    c.gram_chol = typeof(gram_chol)(new_factors, gram_chol.uplo, convert(LinearAlgebra.BlasInt, 0))
    c.gramYBV = @view c.gramYBV.parent[1:sizeY, :]
    c.tmp = @view c.tmp.parent[1:sizeY, :]
    return c
end

function (constr!::Constraint{Nothing})(X, X_temp)
    nothing
end

function (constr!::Constraint)(X, X_temp)
    @views if size(constr!.Y, 2) > 0
        sizeX = size(X, 2)
        sizeY = size(constr!.Y, 2)
        gramYBV_view = constr!.gramYBV[1:sizeY, 1:sizeX]
        mul!(gramYBV_view, adjoint(constr!.BY), X)
        tmp_view = constr!.tmp[1:sizeY, 1:sizeX]
        ldiv!(tmp_view, constr!.gram_chol, gramYBV_view)
        mul!(X_temp, constr!.Y, tmp_view)
        @inbounds X .= X .- X_temp
    end
    nothing
end

struct RPreconditioner{TM, T, TA<:AbstractArray{T}}
    M::TM
    buffer::TA
    RPreconditioner{TM, T, TA}(M, X) where {TM, T, TA<:AbstractArray{T}} = new(M, similar(X))
end
RPreconditioner(M, X) = RPreconditioner{typeof(M), eltype(X), typeof(X)}(M, X)

function (precond!::RPreconditioner{Nothing})(X)
    nothing
end
function (precond!::RPreconditioner)(X)
    bs = size(X, 2)
    @views ldiv!(precond!.buffer[:, 1:bs], precond!.M, X)
    # Just returning buffer would be cheaper but struct at call site must be mutable
    @inbounds @views X .= precond!.buffer[:, 1:bs]
    nothing
end

struct BlockGram{Generalized, TA}
    XAX::TA
    XAR::TA
    XAP::TA
    RAR::TA
    RAP::TA
    PAP::TA
end
function BlockGram(XBlocks::Blocks{Generalized, T}) where {Generalized, T}
    sizeX = size(XBlocks.block, 2)
    XAX = zeros(T, sizeX, sizeX)
    XAP = zeros(T, sizeX, sizeX)
    XAR = zeros(T, sizeX, sizeX)
    RAR = zeros(T, sizeX, sizeX)
    RAP = zeros(T, sizeX, sizeX)
    PAP = zeros(T, sizeX, sizeX)
    return BlockGram{Generalized, Matrix{T}}(XAX, XAR, XAP, RAR, RAP, PAP)
end
XAX!(BlockGram, XBlocks) = mul!(BlockGram.XAX, adjoint(XBlocks.block), XBlocks.A_block)
XAP!(BlockGram, XBlocks, PBlocks, n) = @views mul!(BlockGram.XAP[:, 1:n], adjoint(XBlocks.block), PBlocks.A_block[:, 1:n])
XAR!(BlockGram, XBlocks, RBlocks, n) = @views mul!(BlockGram.XAR[:, 1:n], adjoint(XBlocks.block), RBlocks.A_block[:, 1:n])
RAR!(BlockGram, RBlocks, n) = @views mul!(BlockGram.RAR[1:n, 1:n], adjoint(RBlocks.block[:, 1:n]), RBlocks.A_block[:, 1:n])
RAP!(BlockGram, RBlocks, PBlocks, n) = @views mul!(BlockGram.RAP[1:n, 1:n], adjoint(RBlocks.A_block[:, 1:n]), PBlocks.block[:, 1:n])
PAP!(BlockGram, PBlocks, n) = @views mul!(BlockGram.PAP[1:n, 1:n], adjoint(PBlocks.block[:, 1:n]), PBlocks.A_block[:, 1:n])
XBP!(BlockGram, XBlocks, PBlocks, n) = @views mul!(BlockGram.XAP[:, 1:n], adjoint(XBlocks.block), PBlocks.B_block[:, 1:n])
XBR!(BlockGram, XBlocks, RBlocks, n) = @views mul!(BlockGram.XAR[:, 1:n], adjoint(XBlocks.block), RBlocks.B_block[:, 1:n])
RBP!(BlockGram, RBlocks, PBlocks, n) = @views mul!(BlockGram.RAP[1:n, 1:n], adjoint(RBlocks.B_block[:, 1:n]), PBlocks.block[:, 1:n])
#XBX!(BlockGram, XBlocks) = mul!(BlockGram.XAX, adjoint(XBlocks.block), XBlocks.B_block)
#RBR!(BlockGram, RBlocks, n) = @views mul!(BlockGram.RAR[1:n, 1:n], adjoint(RBlocks.block[:, 1:n]), RBlocks.B_block[:, 1:n])
#PBP!(BlockGram, PBlocks, n) = @views mul!(BlockGram.PAP[1:n, 1:n], adjoint(PBlocks.block[:, 1:n]), PBlocks.B_block[:, 1:n])

function I!(G, xr)
    @inbounds for j in xr, i in xr
        G[i, j] = ifelse(i==j, 1, 0)
    end
    return
end

function (g::BlockGram)(gram, lambda, n1::Int, n2::Int, n3::Int)
    xr = 1:n1
    rr = n1+1:n1+n2
    pr = n1+n2+1:n1+n2+n3
    @inbounds @views begin
        if n1 > 0
            #gram[xr, xr] .= g.XAX[1:n1, 1:n1]
            gram[xr, xr] .= Diagonal(lambda[1:n1])
        end
        if n2 > 0
            gram[rr, rr] .= g.RAR[1:n2, 1:n2]
            gram[xr, rr] .= g.XAR[1:n1, 1:n2]
            conj!(transpose!(gram[rr, xr], g.XAR[1:n1, 1:n2]))
        end
        if n3 > 0
            gram[pr, pr] .= g.PAP[1:n3, 1:n3]
            gram[rr, pr] .= g.RAP[1:n2, 1:n3]
            gram[xr, pr] .= g.XAP[1:n1, 1:n3]
            conj!(transpose!(gram[pr, rr], g.RAP[1:n2, 1:n3]))
            conj!(transpose!(gram[pr, xr], g.XAP[1:n1, 1:n3]))
        end
    end
    return
end
function (g::BlockGram)(gram, n1::Int, n2::Int, n3::Int, normalized::Bool=true)
    xr = 1:n1
    rr = n1+1:n1+n2
    pr = n1+n2+1:n1+n2+n3
    @views if n1 > 0
        if normalized
            I!(gram, xr)
        #else
        #    @inbounds gram[xr, xr] .= g.XAX[1:n1, 1:n1]
        end
    end
    @views if n2 > 0
        if normalized
            I!(gram, rr)
        #else
        #    @inbounds gram[rr, rr] .= g.RAR[1:n2, 1:n2]
        end
        @inbounds gram[xr, rr] .= g.XAR[1:n1, 1:n2]
        @inbounds conj!(transpose!(gram[rr, xr], g.XAR[1:n1, 1:n2]))
    end
    @views if n3 > 0
        if normalized
            I!(gram, pr)
        #else
        #    @inbounds gram[pr, pr] .= g.PAP[1:n3, 1:n3]
        end
        @inbounds gram[rr, pr] .= g.RAP[1:n2, 1:n3]
        @inbounds gram[xr, pr] .= g.XAP[1:n1, 1:n3]
        @inbounds conj!(transpose!(gram[pr, rr], g.RAP[1:n2, 1:n3]))
        @inbounds conj!(transpose!(gram[pr, xr], g.XAP[1:n1, 1:n3]))
    end
    return
end

# Any orthogonalization algorithm should guarantee that X' B X is the identity
# A * X and B * X should be updated accordingly
abstract type AbstractOrtho end

struct CholQROrtho{TA, TB, TM} <: AbstractOrtho
    A::TA
    B::TB
    gramVBV::TM # to be used in view
end

function rdiv!(A, B::UpperTriangular)
    s = size(A, 2)
    @inbounds @views A[:,1] .= A[:, 1] ./ B[1,1]
    @inbounds @views for i in 2:s
        for j in 1:i-1
            A[:,i] .= A[:,i] .- A[:,j] .* B[j,i]
        end
        A[:,i] .= A[:,i] ./ B[i,i]
    end
    return A
end

realdiag!(M) = M
function realdiag!(M::AbstractMatrix{TC}) where TC <: Complex
    @inbounds for i in 1:minimum(size(M))
        M[i,i] = real(M[i,i])
    end
    return M
end

# Orthogonalizes X s.t. X' B X = I
# New orthogonal X must span at least the same span as the old X
# If update_AX is true, A * X is updated
# If update_BX is true, B * X is updated
function (ortho!::CholQROrtho)(XBlocks::Blocks{Generalized}, sizeX = -1; update_AX=false, update_BX=false) where Generalized
    useview = sizeX != -1
    if sizeX == -1
        sizeX = size(XBlocks.block, 2)
    end
    X = XBlocks.block
    BX = XBlocks.B_block # Assumes it is premultiplied
    AX = XBlocks.A_block
    A = ortho!.A
    B = ortho!.B
    T = real(eltype(X))
    @views gram_view = ortho!.gramVBV[1:sizeX, 1:sizeX]

    # Compute the gram matrix
    @views if useview
        mul!(gram_view, adjoint(X[:, 1:sizeX]), BX[:, 1:sizeX])
    else
        mul!(gram_view, adjoint(X), BX)
    end
    # Ensure the diagonal is real, may not be due to computational error
    realdiag!(gram_view)

    # Compute the Cholesky factors R' * R = X' * B * X
    # inv(R') * X' * B * X * inv(R) = I
    cholf = cholesky!(Hermitian(gram_view), check=false)
    if issuccess(cholf)
        # Update X to X * inv(R)
        # Optionally update A * X to A * X * inv(R)
        # Optionally update B * X to B * X * inv(R)
        R = cholf.factors
        @views if useview
            rdiv!(X[:, 1:sizeX], UpperTriangular(R))
            update_AX && rdiv!(AX[:, 1:sizeX], UpperTriangular(R))
            Generalized && update_BX && rdiv!(BX[:, 1:sizeX], UpperTriangular(R))
        else
            rdiv!(X, UpperTriangular(R))
            update_AX && rdiv!(AX, UpperTriangular(R))
            Generalized && update_BX && rdiv!(BX, UpperTriangular(R))
        end
    else
        # X is nearly not full-column rank
        # Find the QR decomposition of X
        # New X is the B-orthonormlized compact Q from the QR decomposition
        qrf = qr!(X)
        @views if useview
            # Extract compact Q into X
            X[:, 1:sizeX] .= 0
            I!(X, sizeX)
            lmul!(qrf.Q, X[:, 1:sizeX])
            if Generalized
                # Find X' B X
                mul!(BX[:, 1:sizeX], B, X[:, 1:sizeX])
                mul!(gram_view, adjoint(X[:, 1:sizeX]), BX[:, 1:sizeX])
                realdiag!(gram_view)
                # Find R' * R = X' * B * X
                cholf = cholesky!(Hermitian(gram_view), check=true)
                R = cholf.factors
                # Update X to X * inv(R)
                # New X is full column rank and B-orthonormal
                rdiv!(X[:, 1:sizeX], UpperTriangular(R))
                # Optionally update B * X
                update_BX && rdiv!(BX[:, 1:sizeX], UpperTriangular(R))
            end
            # Optionally update A * X
            update_AX && mul!(AX[:,1:sizeX], A, X[:,1:sizeX])
        else
            # Extract compact Q into X
            X .= 0
            I!(X, size(X, 2))
            lmul!(qrf.Q, X)
            if Generalized
                # Find X' B X
                mul!(BX, B, X)
                mul!(gram_view, adjoint(X), BX)
                realdiag!(gram_view)
                # Find R' * R = X' * B * X
                cholf = cholesky!(Hermitian(gram_view), check=true)
                R = cholf.factors
                # Update X to X * inv(R)
                # New X is full column rank and B-orthonormal
                rdiv!(X, UpperTriangular(R))
                # Optionally update B * X
                update_BX && rdiv!(BX, UpperTriangular(R))
            end
            # Optionally update A * X
            update_AX && mul!(AX, A, X)
        end
    end

    return
end

struct LOBPCGIterator{Generalized, T, TA, TB, TL<:AbstractVector{T}, TR<:AbstractVector, TPerm<:AbstractVector{Int}, TV<:AbstractArray{T}, TBlocks<:Blocks{Generalized, T}, TO<:AbstractOrtho, TP, TC, TG, TM, TTrace}
    A::TA
    B::TB
    ritz_values::TL
    λperm::TPerm
    λ::TL
    V::TV
    residuals::TR
    largest::Bool
    XBlocks::TBlocks
    tempXBlocks::TBlocks
    PBlocks::TBlocks
    activePBlocks::TBlocks # to be used in view
    RBlocks::TBlocks
    activeRBlocks::TBlocks # to be used in view
    iteration::Base.RefValue{Int}
    currentBlockSize::Base.RefValue{Int}
    ortho!::TO
    precond!::TP
    constr!::TC
    gramABlock::TG
    gramBBlock::TG
    gramA::TV
    gramB::TV
    activeMask::TM
    trace::TTrace
end

"""
    LOBPCGIterator(A, largest::Bool, X, P=nothing, C=nothing) -> iterator

# Arguments

- `A`: linear operator;
- `X`: initial guess of the Ritz vectors; to be overwritten by the eigenvectors;
- `largest`: `true` if largest eigenvalues are desired and false if smallest;
- `P`: preconditioner of residual vectors, must overload `ldiv!`;
- `C`: constraint to deflate the residual and solution vectors orthogonal
    to a subspace; must overload `mul!`.
"""
LOBPCGIterator(A, largest::Bool, X, P=nothing, C=nothing) = LOBPCGIterator(A, nothing, largest, X, P, C)

"""
    LOBPCGIterator(A, B, largest::Bool, X, P=nothing, C=nothing) -> iterator

# Arguments

- `A`: linear operator;
- `B`: linear operator;
- `X`: initial guess of the Ritz vectors; to be overwritten by the eigenvectors;
- `largest`: `true` if largest eigenvalues are desired and false if smallest;
- `P`: preconditioner of residual vectors, must overload `ldiv!`;
- `C`: constraint to deflate the residual and solution vectors orthogonal
    to a subspace; must overload `mul!`;
"""
function LOBPCGIterator(A, B, largest::Bool, X, P=nothing, C=nothing)
    precond! = RPreconditioner(P, X)
    constr! = Constraint(C, B, X)
    return LOBPCGIterator(A, B, largest, X, precond!, constr!)
end
function LOBPCGIterator(A, B, largest::Bool, X, precond!::RPreconditioner, constr!::Constraint)
    T = eltype(X)
    nev = size(X, 2)
    if B isa Nothing
        XBlocks = Blocks(X, similar(X))
        tempXBlocks = Blocks(copy(X), similar(X))
        RBlocks = Blocks(similar(X), similar(X))
        activeRBlocks = Blocks(similar(X), similar(X))
        PBlocks = Blocks(similar(X), similar(X))
        activePBlocks = Blocks(similar(X), similar(X))
    else
        XBlocks = Blocks(X, similar(X), similar(X))
        tempXBlocks = Blocks(copy(X), similar(X), similar(X))
        RBlocks = Blocks(similar(X), similar(X), similar(X))
        activeRBlocks = Blocks(similar(X), similar(X), similar(X))
        PBlocks = Blocks(similar(X), similar(X), similar(X))
        activePBlocks = Blocks(similar(X), similar(X), similar(X))
    end
    ritz_values = zeros(T, nev*3)
    λ = zeros(T, nev)
    λperm = zeros(Int, nev*3)
    V = zeros(T, nev*3, nev*3)
    residuals = fill(real(T)(NaN), nev)
    iteration = Ref(1)
    currentBlockSize = Ref(nev)
    generalized = !(B isa Nothing)
    ortho! = CholQROrtho(A, B, zeros(T, nev, nev))

    gramABlock = BlockGram(XBlocks)
    gramBBlock = BlockGram(XBlocks)

    gramA = zeros(T, 3*nev, 3*nev)
    gramB = zeros(T, 3*nev, 3*nev)

    activeMask = ones(Bool, nev)
    trace = LOBPCGTrace{Vector{real(T)},Vector{T}}()

    return LOBPCGIterator{generalized, T, typeof(A), typeof(B), typeof(λ), typeof(residuals), typeof(λperm), typeof(V), typeof(XBlocks), typeof(ortho!), typeof(precond!), typeof(constr!), typeof(gramABlock), typeof(activeMask), typeof(trace)}(A, B, ritz_values, λperm, λ, V, residuals, largest, XBlocks, tempXBlocks, PBlocks, activePBlocks, RBlocks, activeRBlocks, iteration, currentBlockSize, ortho!, precond!, constr!, gramABlock, gramBBlock, gramA, gramB, activeMask, trace)
end
function LOBPCGIterator(A, largest::Bool, X, nev::Int, P=nothing, C=nothing)
    LOBPCGIterator(A, nothing, largest, X, nev, P, C)
end
function LOBPCGIterator(A, B, largest::Bool, X, nev::Int, P=nothing, C=nothing)
    T = eltype(X)
    n = size(X, 1)
    sizeX = size(X, 2)
    if C isa Nothing
        sizeC = 0
        new_C = typeof(X)(undef, n, (nev÷sizeX)*sizeX)
    else
        sizeC = size(C,2)
        new_C = typeof(C)(undef, n, sizeC+(nev÷sizeX)*sizeX)
        new_C[:,1:sizeC] .= C
    end
    if B isa Nothing
        new_BC = new_C
    else
        new_BC = similar(new_C)
    end
    Y = @view new_C[:, 1:sizeC]
    BY = @view new_BC[:, 1:sizeC]
    if !(B isa Nothing)
        mul!(BY, B, Y)
    end
    constr! = Constraint(Y, BY, X, NotBWrapper())
    precond! = RPreconditioner(P, X)
    return LOBPCGIterator(A, B, largest, X, precond!, constr!)
end

function ortho_AB_mul_X!(blocks::Blocks, ortho!, A, B, bs=-1)
    # Finds BX
    bs == -1 ? B_mul_X!(blocks, B) : B_mul_X!(blocks, B, bs)
    # Orthonormalizes X and updates BX
    bs == -1 ? ortho!(blocks, update_BX=true) : ortho!(blocks, bs, update_BX=true)
    # Updates AX
    bs == -1 ? A_mul_X!(blocks, A) : A_mul_X!(blocks, A, bs)
    return
end
function residuals!(iterator)
    sizeX = size(iterator.XBlocks.block, 2)
    @views mul!(iterator.RBlocks.block, iterator.XBlocks.B_block, Diagonal(iterator.ritz_values[1:sizeX]))
    @inbounds iterator.RBlocks.block .= iterator.XBlocks.A_block .- iterator.RBlocks.block
    # Finds residual norms
    @inbounds for j in 1:size(iterator.RBlocks.block, 2)
        iterator.residuals[j] = 0
        for i in 1:size(iterator.RBlocks.block, 1)
            x = iterator.RBlocks.block[i,j]
            iterator.residuals[j] += real(x*conj(x))
        end
        iterator.residuals[j] = sqrt(iterator.residuals[j])
    end
    return
end

function update_mask!(iterator, residualTolerance)
    sizeX = size(iterator.XBlocks.block, 2)
    # Update active vectors mask
    @inbounds @views iterator.activeMask .= iterator.residuals[1:sizeX] .> residualTolerance
    iterator.currentBlockSize[] = sum(iterator.activeMask)
    return
end

function update_active!(mask, bs::Int, blockPairs...)
    @inbounds @views for (activeblock, block) in blockPairs
        activeblock[:, 1:bs] .= block[:, mask]
    end
    return
end

function precond_constr!(block, temp_block, bs, precond!, constr!)
    @views precond!(block[:, 1:bs])
    # Constrain the active residual vectors to be B-orthogonal to Y
    @views constr!(block[:, 1:bs], temp_block[:, 1:bs])
    return
end
function block_grams_1x1!(iterator)
    # Finds gram matrix X'AX
    XAX!(iterator.gramABlock, iterator.XBlocks)
    return
end
function block_grams_2x2!(iterator, bs)
    sizeX = size(iterator.XBlocks.block, 2)
    #XAX!(iterator.gramABlock, iterator.XBlocks)
    XAR!(iterator.gramABlock, iterator.XBlocks, iterator.activeRBlocks, bs)
    RAR!(iterator.gramABlock, iterator.activeRBlocks, bs)
    XBR!(iterator.gramBBlock, iterator.XBlocks, iterator.activeRBlocks, bs)
    @views iterator.gramABlock(iterator.gramA, iterator.ritz_values[1:sizeX], sizeX, bs, 0)
    iterator.gramBBlock(iterator.gramB, sizeX, bs, 0, true)

    return
end
function block_grams_3x3!(iterator, bs)
    # Find R'AR, P'AP, X'AR, X'AP and R'AP
    sizeX = size(iterator.XBlocks.block, 2)
    #XAX!(iterator.gramABlock, iterator.XBlocks)
    XAR!(iterator.gramABlock, iterator.XBlocks, iterator.activeRBlocks, bs)
    XAP!(iterator.gramABlock, iterator.XBlocks, iterator.activePBlocks, bs)
    RAR!(iterator.gramABlock, iterator.activeRBlocks, bs)
    RAP!(iterator.gramABlock, iterator.activeRBlocks, iterator.activePBlocks, bs)
    PAP!(iterator.gramABlock, iterator.activePBlocks, bs)
    # Find X'BR, X'BP and P'BR
    XBR!(iterator.gramBBlock, iterator.XBlocks, iterator.activeRBlocks, bs)
    XBP!(iterator.gramBBlock, iterator.XBlocks, iterator.activePBlocks, bs)
    RBP!(iterator.gramBBlock, iterator.activeRBlocks, iterator.activePBlocks, bs)
    # Update the gram matrix [X R P]' A [X R P]
    @views iterator.gramABlock(iterator.gramA, iterator.ritz_values[1:sizeX], sizeX, bs, bs)
    # Update the gram matrix [X R P]' B [X R P]
    iterator.gramBBlock(iterator.gramB, sizeX, bs, bs, true)

    return
end

function sub_problem!(iterator, sizeX, bs1, bs2)
    subdim = sizeX+bs1+bs2
    @views if bs1 == 0
        gramAview = iterator.gramABlock.XAX[1:subdim, 1:subdim]
        # Source of type instability
        realdiag!(gramAview)
        eigf = eigen!(Hermitian(gramAview))
    else
        gramAview = iterator.gramA[1:subdim, 1:subdim]
        gramBview = iterator.gramB[1:subdim, 1:subdim]
        # Source of type instability
        realdiag!(gramAview)
        realdiag!(gramBview)
        eigf = eigen!(Hermitian(gramAview), Hermitian(gramBview))
    end
    # Selects extremal eigenvalues and corresponding vectors
    @views partialsortperm!(iterator.λperm[1:subdim], eigf.values, 1:subdim; rev=iterator.largest)
    @inbounds @views iterator.ritz_values[1:sizeX] .= eigf.values[iterator.λperm[1:sizeX]]
    @inbounds @views iterator.V[1:subdim, 1:sizeX] .= eigf.vectors[:, iterator.λperm[1:sizeX]]
    return
end

function update_X_P!(iterator::LOBPCGIterator{Generalized}, bs1, bs2) where Generalized
    sizeX = size(iterator.XBlocks.block, 2)
    @views begin
        x_eigview = iterator.V[1:sizeX, 1:sizeX]
        r_eigview = iterator.V[sizeX+1:sizeX+bs1, 1:sizeX]
        p_eigview = iterator.V[sizeX+bs1+1:sizeX+bs1+bs2, 1:sizeX]
        r_blockview = iterator.activeRBlocks.block[:, 1:bs1]
        ra_blockview = iterator.activeRBlocks.A_block[:, 1:bs1]
        p_blockview = iterator.activePBlocks.block[:, 1:bs2]
        pa_blockview = iterator.activePBlocks.A_block[:, 1:bs2]
        if Generalized
            rb_blockview = iterator.activeRBlocks.B_block[:, 1:bs1]
            pb_blockview = iterator.activePBlocks.B_block[:, 1:bs2]
        end
    end
    if bs1 > 0
        mul!(iterator.PBlocks.block, r_blockview, r_eigview)
        mul!(iterator.PBlocks.A_block, ra_blockview, r_eigview)
        if Generalized
            mul!(iterator.PBlocks.B_block, rb_blockview, r_eigview)
        end
    end
    if bs2 > 0
        mul!(iterator.tempXBlocks.block, p_blockview, p_eigview)
        mul!(iterator.tempXBlocks.A_block, pa_blockview, p_eigview)
        if Generalized
            mul!(iterator.tempXBlocks.B_block, pb_blockview, p_eigview)
        end
        @inbounds iterator.PBlocks.block .= iterator.PBlocks.block .+ iterator.tempXBlocks.block
        @inbounds iterator.PBlocks.A_block .= iterator.PBlocks.A_block .+ iterator.tempXBlocks.A_block
        if Generalized
            @inbounds iterator.PBlocks.B_block .= iterator.PBlocks.B_block .+ iterator.tempXBlocks.B_block
        end
    end
    block = iterator.XBlocks.block
    tempblock = iterator.tempXBlocks.block
    mul!(tempblock, block, x_eigview)
    block = iterator.XBlocks.A_block
    tempblock = iterator.tempXBlocks.A_block
    mul!(tempblock, block, x_eigview)
    if Generalized
        block = iterator.XBlocks.B_block
        tempblock = iterator.tempXBlocks.B_block
        mul!(tempblock, block, x_eigview)
    end
    @inbounds begin
        if bs1 > 0
            iterator.XBlocks.block .= iterator.tempXBlocks.block .+ iterator.PBlocks.block
            iterator.XBlocks.A_block .= iterator.tempXBlocks.A_block .+ iterator.PBlocks.A_block
            if Generalized
                iterator.XBlocks.B_block .= iterator.tempXBlocks.B_block .+ iterator.PBlocks.B_block
            end
        else
            iterator.XBlocks.block .= iterator.tempXBlocks.block
            iterator.XBlocks.A_block .= iterator.tempXBlocks.A_block
            if Generalized
                iterator.XBlocks.B_block .= iterator.tempXBlocks.B_block
            end
        end
    end
    return
end

function (iterator::LOBPCGIterator{Generalized})(residualTolerance, log) where {Generalized}
    sizeX = size(iterator.XBlocks.block, 2)
    iteration = iterator.iteration[]
    if iteration == 1
        ortho_AB_mul_X!(iterator.XBlocks, iterator.ortho!, iterator.A, iterator.B)
        # Finds gram matrix X'AX
        block_grams_1x1!(iterator)
        sub_problem!(iterator, sizeX, 0, 0)
        # Updates Ritz vectors X and updates AX and BX accordingly
        update_X_P!(iterator, 0, 0)
        residuals!(iterator)
        update_mask!(iterator, residualTolerance)
    elseif iteration == 2
        bs = iterator.currentBlockSize[]
        # Update active R blocks
        update_active!(iterator.activeMask, bs, (iterator.activeRBlocks.block, iterator.RBlocks.block))
        # Precondition and constrain the active residual vectors
        precond_constr!(iterator.activeRBlocks.block, iterator.tempXBlocks.block, bs, iterator.precond!, iterator.constr!)
        # Orthonormalizes R[:,1:bs] and finds AR[:,1:bs] and BR[:,1:bs]
        ortho_AB_mul_X!(iterator.activeRBlocks, iterator.ortho!, iterator.A, iterator.B, bs)
        # Find [X R] A [X R] and [X R]' B [X R]
        block_grams_2x2!(iterator, bs)
        # Solve the Rayleigh-Ritz sub-problem
        sub_problem!(iterator, sizeX, bs, 0)
        update_X_P!(iterator, bs, 0)
        residuals!(iterator)
        update_mask!(iterator, residualTolerance)
    else
        # Update active blocks
        bs = iterator.currentBlockSize[]
        # Update active R and P blocks
        update_active!(iterator.activeMask, bs,
            (iterator.activeRBlocks.block, iterator.RBlocks.block),
            (iterator.activePBlocks.block, iterator.PBlocks.block),
            (iterator.activePBlocks.A_block, iterator.PBlocks.A_block),
            (iterator.activePBlocks.B_block, iterator.PBlocks.B_block))
        # Precondition and constrain the active residual vectors
        precond_constr!(iterator.activeRBlocks.block, iterator.tempXBlocks.block, bs, iterator.precond!, iterator.constr!)
        # Orthonormalizes R[:,1:bs] and finds AR[:,1:bs] and BR[:,1:bs]
        ortho_AB_mul_X!(iterator.activeRBlocks, iterator.ortho!, iterator.A, iterator.B, bs)
        # Orthonormalizes P and updates AP
        iterator.ortho!(iterator.activePBlocks, bs, update_AX=true, update_BX=true)
        # Update the gram matrix [X R P]' A [X R P] and [X R P]' B [X R P]
        block_grams_3x3!(iterator, bs)
        # Solve the Rayleigh-Ritz sub-problem
        sub_problem!(iterator, sizeX, bs, bs)
        # Updates Ritz vectors X and updates AX and BX accordingly
        # And updates P, AP and BP
        update_X_P!(iterator, bs, bs)
        residuals!(iterator)
        update_mask!(iterator, residualTolerance)
    end
    if log
        return LOBPCGState(iteration, iterator.residuals[1:sizeX], iterator.ritz_values[1:sizeX])
    else
        return LOBPCGState(iteration, nothing, nothing)
    end
end

default_tolerance(::Type{T}) where {T<:Number} = eps(real(T))^(real(T)(3)/10)

"""
The Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

Finds the `nev` extremal eigenvalues and their corresponding eigenvectors satisfying `AX = λBX`.

`A` and `B` may be generic types but `Base.mul!(C, AorB, X)` must be defined for vectors and strided matrices `X` and `C`. `size(A, i::Int)` and `eltype(A)` must also be defined for `A`.

    lobpcg(A, [B,] largest, nev; kwargs...) -> results

# Arguments

- `A`: linear operator;
- `B`: linear operator;
- `largest`: `true` if largest eigenvalues are desired and false if smallest;
- `nev`: number of eigenvalues desired.

## Keywords

- `log::Bool`: default is `false`; if `true`, `results.trace` will store iterations
    states; if `false` only `results.trace` will be empty;

- `P`: preconditioner of residual vectors, must overload `ldiv!`;

- `C`: constraint to deflate the residual and solution vectors orthogonal
    to a subspace; must overload `mul!`;

- `maxiter`: maximum number of iterations; default is 200;

- `tol::Real`: tolerance to which residual vector norms must be under.

# Output

- `results`: a `LOBPCGResults` struct. `r.λ` and `r.X` store the eigenvalues and eigenvectors.
"""
function lobpcg(A, largest::Bool, nev::Int; kwargs...)
    lobpcg(A, nothing, largest, nev; kwargs...)
end
function lobpcg(A, B, largest::Bool, nev::Int; kwargs...)
    lobpcg(A, B, largest, rand(eltype(A), size(A, 1), nev); not_zeros=true, kwargs...)
end

"""
    lobpcg(A, [B,] largest, X0; kwargs...) -> results

# Arguments

- `A`: linear operator;
- `B`: linear operator;
- `largest`: `true` if largest eigenvalues are desired and false if smallest;
- `X0`: Initial guess, will not be modified. The number of columns is the number of eigenvectors desired.

## Keywords

- `not_zeros`: default is `false`. If `true`, `X0` will be assumed to not have any all-zeros column.

- `log::Bool`: default is `false`; if `true`, `results.trace` will store iterations
    states; if `false` only `results.trace` will be empty;

- `P`: preconditioner of residual vectors, must overload `ldiv!`;

- `C`: constraint to deflate the residual and solution vectors orthogonal
    to a subspace; must overload `mul!`;

- `maxiter`: maximum number of iterations; default is 200;

- `tol::Real`: tolerance to which residual vector norms must be under.

# Output

- `results`: a `LOBPCGResults` struct. `r.λ` and `r.X` store the eigenvalues and eigenvectors.
"""
function lobpcg(A, largest::Bool, X0; kwargs...)
    lobpcg(A, nothing, largest, X0; kwargs...)
end
function lobpcg(A, B, largest, X0;
                not_zeros=false, log=false, P=nothing, maxiter=200,
                C=nothing, tol::Real=default_tolerance(eltype(X0)))
    X = copy(X0)
    n = size(X, 1)
    sizeX = size(X, 2)
    sizeX > n && throw("X column dimension exceeds the row dimension")
    3*sizeX > n && throw("The LOBPCG algorithms is not stable to use when the matrix size is less than 3 times the block size. Please use a dense solver instead.")

    iterator = LOBPCGIterator(A, B, largest, X, P, C)

    return lobpcg!(iterator, log=log, tol=tol, maxiter=maxiter, not_zeros=not_zeros)
end

"""
    lobpcg!(iterator::LOBPCGIterator; kwargs...) -> results

# Arguments

- `iterator::LOBPCGIterator`: a struct having all the variables required
    for the LOBPCG algorithm.

## Keywords

- `not_zeros`: default is `false`. If `true`, the initial Ritz vectors will be assumed to not have any all-zeros column.

- `log::Bool`: default is `false`; if `true`, `results.trace` will store iterations
    states; if `false` only `results.trace` will be empty;

- `maxiter`: maximum number of iterations; default is 200;

- `tol::Real`: tolerance to which residual vector norms must be under.

# Output

- `results`: a `LOBPCGResults` struct. `r.λ` and `r.X` store the eigenvalues and eigenvectors.

"""
function lobpcg!(iterator::LOBPCGIterator; log=false, maxiter=200, not_zeros=false,
                 tol::Real=default_tolerance(eltype(iterator.XBlocks.block)))
    X = iterator.XBlocks.block
    iterator.constr!(iterator.XBlocks.block, iterator.tempXBlocks.block)
    if !not_zeros
        @views for j in 1:size(X,2)
            if all(x -> x==0, X[:, j])
                @inbounds X[:,j] .= rand.()
            end
        end
        iterator.constr!(iterator.XBlocks.block, iterator.tempXBlocks.block)
    end
    n = size(X, 1)
    sizeX = size(X, 2)
    iterator.iteration[] = 1
    while iterator.iteration[] <= maxiter 
        state = iterator(tol, log)
        if log
            push!(iterator.trace, state)
        end
        iterator.currentBlockSize[] == 0 && break
        iterator.iteration[] += 1
    end
    @inbounds @views iterator.λ .= iterator.ritz_values[1:sizeX]

    @views results = LOBPCGResults(iterator.λ, X, tol, iterator.residuals, iterator.iteration[], maxiter, all((x)->(norm(x)<=tol), iterator.residuals[1:sizeX]), iterator.trace)

    return results
end

"""
lobpcg(A, [B,] largest, X0, nev; kwargs...) -> results

# Arguments

- `A`: linear operator;
- `B`: linear operator;
- `largest`: `true` if largest eigenvalues are desired and false if smallest;
- `X0`: block vectors such that the eigenvalues will be found size(X0, 2) at a time;
    the columns are also used to initialize the first batch of Ritz vectors;
- `nev`: number of eigenvalues desired.

## Keywords

- `log::Bool`: default is `false`; if `true`, `results.trace` will store iterations
    states; if `false` only `results.trace` will be empty;

- `P`: preconditioner of residual vectors, must overload `ldiv!`;

- `C`: constraint to deflate the residual and solution vectors orthogonal
    to a subspace; must overload `mul!`;

- `maxiter`: maximum number of iterations; default is 200;

- `tol::Real`: tolerance to which residual vector norms must be under.

# Output

- `results`: a `LOBPCGResults` struct. `r.λ` and `r.X` store the eigenvalues and eigenvectors.
"""
function lobpcg(A, largest::Bool, X0, nev::Int; kwargs...)
    lobpcg(A, nothing, largest, X0, nev; kwargs...)
end
function lobpcg(A, B, largest::Bool, X0, nev::Int;
                not_zeros=false, log=false, P=nothing, maxiter=200,
                C=nothing, tol::Real=default_tolerance(eltype(X0)))
    n = size(X0, 1)
    sizeX = size(X0, 2)
    nev > n && throw("Number of eigenvectors desired exceeds the row dimension.")
    3*sizeX > n && throw("The LOBPCG algorithms is not stable to use when the matrix size is less than 3 times the block size. Please use a dense solver instead.")

    sizeX = min(nev, sizeX)
    X = X0[:, 1:sizeX]
    iterator = LOBPCGIterator(A, B, largest, X, nev, P, C)

    r = EmptyLOBPCGResults(X, nev, tol, maxiter)
    rnext = lobpcg!(iterator, log=log, tol=tol, maxiter=maxiter, not_zeros=not_zeros)
    append!(r, rnext, 0)
    converged_x = sizeX
    while converged_x < nev
        @views if nev-converged_x < sizeX
            cutoff = sizeX-(nev-converged_x)
            update!(iterator.constr!, iterator.XBlocks.block[:, 1:cutoff], iterator.XBlocks.B_block[:, 1:cutoff])
            X[:, 1:sizeX-cutoff] .= X[:, cutoff+1:sizeX]
            rand!(X[:, cutoff+1:sizeX])
            rnext = lobpcg!(iterator, log=log, tol=tol, maxiter=maxiter, not_zeros=true)
            append!(r, rnext, converged_x, sizeX-cutoff)
            converged_x += sizeX-cutoff
        else
            update!(iterator.constr!, iterator.XBlocks.block, iterator.XBlocks.B_block)
            rand!(X)
            rnext = lobpcg!(iterator, log=log, tol=tol, maxiter=maxiter, not_zeros=true)
            append!(r, rnext, converged_x)
            converged_x += sizeX
        end
    end
    return r
end
