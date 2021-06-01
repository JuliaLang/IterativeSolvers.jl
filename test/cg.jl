module TestCG

using IterativeSolvers
using LinearMaps
using Test
using LinearAlgebra
using SparseArrays
using Random

import LinearAlgebra.ldiv!

include("laplace_matrix.jl")

struct JacobiPrec
    diagonal
end

ldiv!(y, P::JacobiPrec, x) = y .= x ./ P.diagonal

@testset "Conjugate Gradients" begin

Random.seed!(1234321)

@testset "Small full system" begin
    n = 10

    @testset "Matrix{$T}, conjugated dot product" for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, n, n)
        A = A' * A + I
        b = rand(T, n)
        reltol = √eps(real(T))

        x,ch = cg(A, b; reltol=reltol, maxiter=2n, log=true)
        @test isa(ch, ConvergenceHistory)
        @test A*x ≈ b rtol=reltol
        @test ch.isconverged

        # If you start from the exact solution, you should converge immediately
        x,ch = cg!(A \ b, A, b; abstol=2n*eps(real(T)), reltol=zero(real(T)), log=true)
        @test niters(ch) ≤ 1
        @test nprods(ch) ≤ 2

        # Test with cholfact as preconditioner should converge immediately
        F = cholesky(A, Val(false))
        x,ch = cg(A, b; Pl=F, log=true)
        @test niters(ch) ≤ 2
        @test nprods(ch) ≤ 2

        # All-zeros rhs should give all-zeros lhs
        x0 = cg(A, zeros(T, n))
        @test x0 == zeros(T, n)
    end

    @testset "Matrix{$T}, unconjugated dot product" for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, n, n)
        A = A + transpose(A) + 15I
        x = ones(T, n)
        b = A * x

        reltol = √eps(real(T))

        # Solve without preconditioner
        x1, his1 = cocg(A, b, reltol = reltol, maxiter = 100, log = true)
        @test isa(his1, ConvergenceHistory)
        @test A*x1 ≈ b rtol=reltol

        # With an initial guess
        x_guess = rand(T, n)
        x2, his2 = cocg!(x_guess, A, b, reltol = reltol, maxiter = 100, log = true)
        @test isa(his2, ConvergenceHistory)
        @test x2 == x_guess
        @test A*x2 ≈ b rtol=reltol

        # Do an exact LU decomp of a nearby matrix
        F = lu(A + rand(T, n, n))
        x3, his3 = cocg(A, b, Pl = F, maxiter = 100, reltol = reltol, log = true)
        @test A*x3 ≈ b rtol=reltol
    end
end

@testset "Sparse Laplacian" begin
    A = laplace_matrix(Float64, 10, 2)
    P = JacobiPrec(diag(A))

    rhs = randn(size(A, 2))
    rmul!(rhs, inv(norm(rhs)))
    abstol = 1e-5
    reltol = 1e-5

    @testset "SparseMatrixCSC{$T, $Ti}" for T in (Float64, Float32), Ti in (Int64, Int32)
        xCG = cg(A, rhs; reltol=reltol, maxiter=100)
        xJAC = cg(A, rhs; Pl=P, reltol=reltol, maxiter=100)
        @test A*xCG ≈ rhs rtol=reltol
        @test A*xJAC ≈ rhs rtol=reltol
    end

    Af = LinearMap(A)
    @testset "Function" begin
        xCG = cg(Af, rhs; reltol=reltol, maxiter=100)
        xJAC = cg(Af, rhs; Pl=P, reltol=reltol, maxiter=100)
        @test A*xCG ≈ rhs rtol=reltol
        @test A*xJAC ≈ rhs rtol=reltol
    end

    @testset "Function with specified starting guess" begin
        x0 = randn(size(rhs))
        xCG, hCG = cg!(copy(x0), Af, rhs; abstol=abstol, reltol=0.0, maxiter=100, log=true)
        xJAC, hJAC = cg!(copy(x0), Af, rhs; Pl=P, abstol=abstol, reltol=0.0, maxiter=100, log=true)
        @test A*xCG ≈ rhs rtol=reltol
        @test A*xJAC ≈ rhs rtol=reltol
        @test niters(hJAC) == niters(hCG)
    end
end

@testset "CG with a view" begin
    A = rand(10, 10)
    A = A + A' + 100I
    x = view(rand(10, 2), :, 1)
    b = rand(10)
    x, hist = cg!(x, A, b, log = true)
    @test hist.isconverged
end

@testset "Termination criterion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = T[ 2 -1  0
              -1  2 -1
               0 -1  2]
        n = size(A, 2)
        b = ones(T, n)
        x0 = A \ b
        perturbation = 10 * sqrt(eps(real(T))) * T[(-1)^i for i in 1:n]

        # If the initial residual is small and a small relative tolerance is used,
        # many iterations are necessary
        x = x0 + perturbation
        initial_residual = norm(A * x - b)
        x, ch = cg!(x, A, b, log=true)
        @test 2 ≤ niters(ch) ≤ n

        # If the initial residual is small and a large absolute tolerance is used,
        # no iterations are necessary
        x = x0 + perturbation
        initial_residual = norm(A * x - b)
        x, ch = cg!(x, A, b, abstol=2*initial_residual, reltol=zero(real(T)), log=true)
        @test niters(ch) == 0
    end
end

end

end # module
