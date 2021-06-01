import Base: iterate
using Printf
export cg, cg!, cocg, cocg!, CGIterable, PCGIterable, cg_iterator!, CGStateVariables

mutable struct CGIterable{matT, solT, vecT, numT <: Real, paramT <: Number, dotT <: AbstractDot}
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    tol::numT
    residual::numT
    ρ_prev::paramT
    maxiter::Int
    mv_products::Int
    dotproduct::dotT
end

mutable struct PCGIterable{precT, matT, solT, vecT, numT <: Real, paramT <: Number, dotT <: AbstractDot}
    Pl::precT
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    tol::numT
    residual::numT
    ρ_prev::paramT
    maxiter::Int
    mv_products::Int
    dotproduct::dotT
end

@inline converged(it::Union{CGIterable, PCGIterable}) = it.residual ≤ it.tol

@inline start(it::Union{CGIterable, PCGIterable}) = 0

@inline done(it::Union{CGIterable, PCGIterable}, iteration::Int) = iteration ≥ it.maxiter || converged(it)


###############
# Ordinary CG #
###############

function iterate(it::CGIterable, iteration::Int=start(it))
    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # u := r + βu (almost an axpy)
    ρ = isa(it.dotproduct, ConjugatedDot) ?  it.residual^2 : _norm(it.r, it.dotproduct)^2
    β = ρ / it.ρ_prev

    it.u .= it.r .+ β .* it.u

    # c = A * u
    mul!(it.c, it.A, it.u)
    α = ρ / _dot(it.u, it.c, it.dotproduct)

    # Improve solution and residual
    it.ρ_prev = ρ
    it.x .+= α .* it.u
    it.r .-= α .* it.c

    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

#####################
# Preconditioned CG #
#####################

function iterate(it::PCGIterable, iteration::Int=start(it))
    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # Apply left preconditioner
    ldiv!(it.c, it.Pl, it.r)

    # u := c + βu (almost an axpy)
    ρ = _dot(it.r, it.c, it.dotproduct)
    β = ρ / it.ρ_prev
    it.u .= it.c .+ β .* it.u

    # c = A * u
    mul!(it.c, it.A, it.u)
    α = ρ / _dot(it.u, it.c, it.dotproduct)

    # Improve solution and residual
    it.ρ_prev = ρ
    it.x .+= α .* it.u
    it.r .-= α .* it.c

    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

# Utility functions

"""
Intermediate CG state variables to be used inside cg and cg!. `u`, `r` and `c` should be of the same type as the solution of `cg` or `cg!`.
```
struct CGStateVariables{T,Tx<:AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end
```
"""
struct CGStateVariables{T,Tx<:AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end

function cg_iterator!(x, A, b, Pl = Identity();
                      abstol::Real = zero(real(eltype(b))),
                      reltol::Real = sqrt(eps(real(eltype(b)))),
                      maxiter::Int = size(A, 2),
                      statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
                      initially_zero::Bool = false,
                      dotproduct::AbstractDot = ConjugatedDot())
    u = statevars.u
    r = statevars.r
    c = statevars.c
    u .= zero(eltype(x))
    copyto!(r, b)

    # Compute r with an MV-product or not.
    if initially_zero
        mv_products = 0
    else
        mv_products = 1
        mul!(c, A, x)
        r .-= c
    end
    residual = norm(r)
    tolerance = max(reltol * residual, abstol)

    # Return the iterable
    if isa(Pl, Identity)
        return CGIterable(A, x, r, c, u,
            tolerance, residual, one(eltype(r)),
            maxiter, mv_products, dotproduct)
    else
        return PCGIterable(Pl, A, x, r, c, u,
            tolerance, residual, one(eltype(r)),
            maxiter, mv_products, dotproduct)
    end
end

"""
    cg(A, b; kwargs...) -> x, [history]

Same as [`cg!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
cg(A, b; kwargs...) = cg!(zerox(A, b), A, b; initially_zero = true, kwargs...)

"""
    cg!(x, A, b; kwargs...) -> x, [history]

# Arguments

- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.

## Keywords

- `statevars::CGStateVariables`: Has 3 arrays similar to `x` to hold intermediate results;
- `initially_zero::Bool`: If `true` assumes that `iszero(x)` so that one
  matrix-vector product can be saved when computing the initial
  residual vector;
- `Pl = Identity()`: left preconditioner of the method. Should be symmetric,
  positive-definite like `A`;
- `abstol::Real = zero(real(eltype(b)))`,
  `reltol::Real = sqrt(eps(real(eltype(b))))`: absolute and relative
  tolerance for the stopping condition
  `|r_k| ≤ max(reltol * |r_0|, abstol)`, where `r_k ≈ A * x_k - b`
  is approximately the residual in the `k`th iteration.
  !!! note
      The true residual norm is never explicitly computed during the iterations
      for performance reasons; it may accumulate rounding errors.
- `maxiter::Int = size(A,2)`: maximum number of iterations;
- `verbose::Bool = false`: print method information;
- `log::Bool = false`: keep track of the residual norm in each iteration.

# Output

**if `log` is `false`**

- `x`: approximated solution.

**if `log` is `true`**

- `x`: approximated solution.
- `ch`: convergence history.

**ConvergenceHistory keys**

- `:tol` => `::Real`: stopping tolerance.
- `:resnom` => `::Vector`: residual norm at each iteration.
"""
function cg!(x, A, b;
             abstol::Real = zero(real(eltype(b))),
             reltol::Real = sqrt(eps(real(eltype(b)))),
             maxiter::Int = size(A, 2),
             log::Bool = false,
             statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
             verbose::Bool = false,
             Pl = Identity(),
             dotproduct::AbstractDot = ConjugatedDot(),
             kwargs...)
    history = ConvergenceHistory(partial = !log)
    history[:abstol] = abstol
    history[:reltol] = reltol
    log && reserve!(history, :resnorm, maxiter + 1)

    # Actually perform CG
    iterable = cg_iterator!(x, A, b, Pl; abstol = abstol, reltol = reltol, maxiter = maxiter,
                            statevars = statevars, dotproduct = dotproduct, kwargs...)
    if log
        history.mvps = iterable.mv_products
    end
    for (iteration, item) = enumerate(iterable)
        if log
            nextiter!(history, mvps = 1)
            push!(history, :resnorm, iterable.residual)
        end
        verbose && @printf("%3d\t%1.2e\n", iteration, iterable.residual)
    end

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end

"""
    cocg(A, b; kwargs...) -> x, [history]

Same as [`cocg!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
cocg(A, b; kwargs...) = cocg!(zerox(A, b), A, b; initially_zero = true, kwargs...)

"""
    cocg!(x, A, b; kwargs...) -> x, [history]

Same as [`cg!`](@ref), but uses the unconjugated dot product (`xᵀy`) instead of the usual,
conjugated dot product (`x'y`) in the algorithm.  It is for solving linear systems with
matrices `A` that are complex-symmetric (`Aᵀ == A`) rahter than Hermitian (`A' == A`).
"""
cocg!(x, A, b; kwargs...) = cg!(x, A, b; dotproduct = UnconjugatedDot(), kwargs...)
