export jacobi, jacobi!, gauss_seidel, gauss_seidel!, sor, sor!, ssor, ssor!

import Base.LinAlg.SingularException
import Base: start, next, done, getindex

function check_diag(A::AbstractMatrix)
    for i = 1 : size(A, 1)
        if iszero(A[i,i])
            throw(SingularException(i))
        end
    end
end

"""
    jacobi(A, b) -> x

Same as [`jacobi!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
jacobi(A::AbstractMatrix, b; kwargs...) = jacobi!(zerox(A, b), A, b; kwargs...)

"""
    jacobi!(x, A::AbstractMatrix, b; maxiter=10) -> x

Performs exactly `maxiter` Jacobi iterations.

Allocates a single temporary vector and traverses `A` columnwise.

Throws `Base.LinAlg.SingularException` when the diagonal has a zero. This check
is performed once beforehand.
"""
function jacobi!(x, A::AbstractMatrix, b; maxiter::Int=10)
    check_diag(A)
    iterable = DenseJacobiIterable(A, x, similar(x), b, maxiter)
    for _ = iterable end
    x
end

mutable struct DenseJacobiIterable{matT,vecT}
    A::matT
    x::vecT
    next::vecT
    b::vecT
    maxiter::Int
end

start(::DenseJacobiIterable) = 1
done(it::DenseJacobiIterable, iteration::Int) = iteration > it.maxiter
function next(j::DenseJacobiIterable, iteration::Int)
    n = size(j.A, 1)

    copy!(j.next, j.b)
    
    # Computes next = b - (A - D)x
    for col = 1 : n
        @simd for row = 1 : col - 1
            @inbounds j.next[row] -= j.A[row, col] * j.x[col]
        end

        @simd for row = col + 1 : n
            @inbounds j.next[row] -= j.A[row, col] * j.x[col]
        end
    end

    # Computes x = D \ next
    for col = 1 : n
        @inbounds j.x[col] = j.next[col] / j.A[col, col]
    end

    nothing, iteration + 1
end

"""
    gauss_seidel(A, b) -> x

Same as [`gauss_seidel!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
gauss_seidel(A::AbstractMatrix, b; kwargs...) = gauss_seidel!(zerox(A, b), A, b; kwargs...)

"""
    gauss_seidel!(x, A::AbstractMatrix, b; maxiter=10) -> x

Performs exactly `maxiter` Gauss-Seidel iterations.

Works fully in-place and traverses `A` columnwise.

Throws `Base.LinAlg.SingularException` when the diagonal has a zero. This check
is performed once beforehand.
"""
function gauss_seidel!(x, A::AbstractMatrix, b; maxiter::Int=10)
    check_diag(A)
    iterable = DenseGaussSeidelIterable(A, x, b, maxiter)
    for _ = iterable end
    x
end

mutable struct DenseGaussSeidelIterable{matT,vecT}
    A::matT
    x::vecT
    b::vecT
    maxiter::Int
end

start(::DenseGaussSeidelIterable) = 1
done(it::DenseGaussSeidelIterable, iteration::Int) = iteration > it.maxiter

function next(s::DenseGaussSeidelIterable, iteration::Int)
    n = size(s.A, 1)
    
    for col = 1 : n
        @simd for row = 1 : col - 1
            @inbounds s.x[row] -= s.A[row, col] * s.x[col]
        end

        s.x[col] = s.b[col]
    end

    for col = 1 : n
        @inbounds s.x[col] /= s.A[col, col]
        @simd for row = col + 1 : n
            @inbounds s.x[row] -= s.A[row, col] * s.x[col]
        end
    end

    nothing, iteration + 1
end

"""
    sor(A, b, ω::Real) -> x

Same as [`sor!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
sor(A::AbstractMatrix, b, ω::Real; kwargs...) =
    sor!(zerox(A, b), A, b, ω; kwargs...)

"""
    sor!(x, A::AbstractMatrix, b, ω::Real; maxiter=10) -> x

Performs exactly `maxiter` SOR iterations with relaxation parameter `ω`.

Allocates a single temporary vector and traverses `A` columnwise.

Throws `Base.LinAlg.SingularException` when the diagonal has a zero. This check
is performed once beforehand.
"""
function sor!(x, A::AbstractMatrix, b, ω::Real; maxiter::Int=10)
    check_diag(A)
    iterable = DenseSORIterable(A, x, similar(x), b, ω, maxiter)
    for _ = iterable end
    x
end

mutable struct DenseSORIterable{matT,vecT,numT}
    A::matT
    x::vecT
    tmp::vecT
    b::vecT
    ω::numT
    maxiter::Int
end

start(::DenseSORIterable) = 1
done(it::DenseSORIterable, iteration::Int) = iteration > it.maxiter
function next(s::DenseSORIterable, iteration::Int)
    n = size(s.A, 1)

    for col = 1 : n
        @simd for row = 1 : col - 1
            @inbounds s.tmp[row] -= s.A[row, col] * s.x[col]
        end

        s.tmp[col] = s.b[col]
    end

    for col = 1 : n
        @inbounds s.x[col] += s.ω * (s.tmp[col] / s.A[col, col] - s.x[col])
        @simd for row = col + 1 : n
            @inbounds s.tmp[row] -= s.A[row, col] * s.x[col]
        end
    end

    nothing, iteration + 1
end

"""
    ssor(A, b, ω::Real) -> x

Same as [`ssor!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
ssor(A::AbstractMatrix, b, ω::Real; kwargs...) =
    ssor!(zerox(A, b), A, b, ω; kwargs...)

"""
    ssor!(x, A::AbstractMatrix, b, ω::Real; maxiter=10) -> x

Performs exactly `maxiter` SSOR iterations with relaxation parameter `ω`. Each iteration 
is basically a forward *and* backward sweep of SOR.

Allocates a single temporary vector and traverses `A` columnwise.

Throws `Base.LinAlg.SingularException` when the diagonal has a zero. This check
is performed once beforehand.
"""
function ssor!(x, A::AbstractMatrix, b, ω::Real; maxiter::Int=10)
    check_diag(A)
    iterable = DenseSSORIterable(A, x, similar(x), b, ω, maxiter)
    for _ = iterable end
    x
end

mutable struct DenseSSORIterable{matT,vecT,numT}
    A::matT
    x::vecT
    tmp::vecT
    b::vecT
    ω::numT
    maxiter::Int
end

start(::DenseSSORIterable) = 1
done(it::DenseSSORIterable, iteration::Int) = iteration > it.maxiter
function next(s::DenseSSORIterable, iteration::Int)
    n = size(s.A, 1)

    for col = 1 : n
        @simd for row = 1 : col - 1
            @inbounds s.tmp[row] -= s.A[row, col] * s.x[col]
        end

        s.tmp[col] = s.b[col]
    end

    for col = 1 : n
        @inbounds s.x[col] += s.ω * (s.tmp[col] / s.A[col, col] - s.x[col])
        @simd for row = col + 1 : n
            @inbounds s.tmp[row] -= s.A[row, col] * s.x[col]
        end
    end

    for col = n : -1 : 1
        s.tmp[col] = s.b[col]
        @simd for row = col + 1 : n
            @inbounds s.tmp[row] -= s.A[row, col] * s.x[col]
        end
    end

    for col = n : -1 : 1
        @simd for row = 1 : col - 1
            @inbounds s.tmp[row] -= s.A[row, col] * s.x[col]
        end

        @inbounds s.x[col] += s.ω * (s.tmp[col] / s.A[col, col] - s.x[col])
    end

    nothing, iteration + 1
end
