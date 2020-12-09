import Base: iterate
using Printf
export cg, cg!, CGIterable, PCGIterable, cg_iterator!, CGStateVariables

mutable struct CGIterable{matT, solT, vecT, numT <: Real}
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    tol::numT
    residual::numT
    prev_residual::numT
    maxiter::Int
    mv_products::Int
end

mutable struct PCGIterable{precT, matT, solT, vecT, numT <: Real, paramT <: Number}
    Pl::precT
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    tol::numT
    residual::numT
    ρ::paramT
    maxiter::Int
    mv_products::Int
end

struct CGResult{Tx, T, Thistory}
    x::Tx
    residual::T
    tol::T
    iterations::Int
    maxiter::Int
    converged::Bool
    history::Thistory
end
function Base.show(io::IO, r::CGResult)
    first_two(fr) = [x for (i, x) in enumerate(fr)][1:2]

    @printf io "Result of CG Algorithm\n"
    @printf io " * Algorithm: CG \n"

    if length(join(r.x, ",")) < 40 || length(r.x) <= 2
        @printf io " * x: [%s]\n" join(r.x, ",")
    else
        @printf io " * x: [%s, ...]\n" join(first_two(r.x), ",")
    end

    @printf io " * Convergence\n"
    @printf io "   * Residual: %s\n" r.residual
    @printf io "   * Tolerance: %s\n" r.tol
    @printf io "   * Converged: %s\n" r.converged
    @printf io "   * Iterations: %s\n" r.iterations
    @printf io "   * Iterations limit: %s\n" r.maxiter

    return
end

@inline isconverged(it::Union{CGIterable, PCGIterable}) = it.residual ≤ it.tol

@inline start(it::Union{CGIterable, PCGIterable}) = 0

@inline done(it::Union{CGIterable, PCGIterable}, iteration::Int) = iteration ≥ it.maxiter || isconverged(it)


###############
# Ordinary CG #
###############

function iterate(it::CGIterable, iteration::Int=start(it))
    if done(it, iteration) return nothing end

    # u := r + βu (almost an axpy)
    β = it.residual^2 / it.prev_residual^2
    it.u .= it.r .+ β .* it.u

    # c = A * u
    mul!(it.c, it.A, it.u)
    α = it.residual^2 / dot(it.u, it.c)

    # Improve solution and residual
    it.x .+= α .* it.u
    it.r .-= α .* it.c

    it.prev_residual = it.residual
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

    ldiv!(it.c, it.Pl, it.r)

    ρ_prev = it.ρ
    it.ρ = dot(it.c, it.r)

    # u := c + βu (almost an axpy)
    β = it.ρ / ρ_prev
    it.u .= it.c .+ β .* it.u

    # c = A * u
    mul!(it.c, it.A, it.u)
    α = it.ρ / dot(it.u, it.c)

    # Improve solution and residual
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
    reltol = sqrt(eps(real(eltype(b)))),
    tol = zero(real(eltype(b))),
    maxiter::Int = size(A, 2),
    statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
    initially_zero::Bool = false
)
    u = statevars.u
    r = statevars.r
    c = statevars.c
    u .= zero(eltype(x))
    copyto!(r, b)

    # Compute r with an MV-product or not.
    if initially_zero
        mv_products = 0
        c = similar(x)
        residual = norm(b)
        tol = max(residual * reltol, tol) # Save one dot product
    else
        mv_products = 1
        mul!(c, A, x)
        r .-= c
        residual = norm(r)
        tol = max(norm(b) * reltol, tol)
    end

    # Return the iterable
    if isa(Pl, Identity)
        return CGIterable(A, x, r, c, u,
            tol, residual, one(residual),
            maxiter, mv_products
        )
    else
        return PCGIterable(Pl, A, x, r, c, u,
            tol, residual, one(eltype(x)),
            maxiter, mv_products
        )
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
- `reltol::Real = sqrt(eps(real(eltype(b))))`: relative tolerance for stopping condition `|r_k| / |r_0| ≤ reltol`;
- `tol` = zero(real(eltype(b))): tolerance for stopping condition `|r_k| ≤ tol`,
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
    reltol = sqrt(eps(real(eltype(b)))),
    tol = zero(real(eltype(b))),
    maxiter::Int = size(A, 2),
    log::Bool = false,
    statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
    verbose::Bool = false,
    Pl = Identity(),
    kwargs...
)
    history = ConvergenceHistory(partial = !log)
    log && reserve!(history, :resnorm, maxiter + 1)

    # Actually perform CG
    iterable = cg_iterator!(x, A, b, Pl; tol = tol, reltol = reltol, maxiter = maxiter, statevars = statevars, kwargs...)
    history[:tol] = iterable.tol
    if log
        history.mvps = iterable.mv_products
    end
    iteration = 0
    for item in iterable
        iteration += 1
        if log
            nextiter!(history, mvps = 1)
            push!(history, :resnorm, iterable.residual)
        end
        verbose && @printf("%3d\t%1.2e\n", iteration, iterable.residual)
    end

    verbose && println()
    converged = isconverged(iterable)
    log && setconv(history, converged)
    log && shrink!(history)

    return CGResult(iterable.x, iterable.residual, iterable.tol, iteration, maxiter, converged, history)
end
