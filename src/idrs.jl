export idrs, idrs!

####################
# API method calls #
####################

idrs(A, b; kwargs...) = idrs!(zerox(A,b), A, b; kwargs...)

function idrs!(x, A, b;
    s = 8, tol=sqrt(eps(typeof(real(b[1])))), maxiter=length(x)^2,
    verbose::Bool=false, plot::Bool=false, log::Bool=false
    )
    if log
        history = ConvergenceHistory()
        history[:tol] = tol
        reserve!(history,:resnorm, maxiter)
    else
        history = DummyHistory()
    end
    idrs_method!(x, linsys_op, (A,), b, s, tol, maxiter; log=history, verbose=verbose)
    if log
        shrink!(history)
        plot && showplot(history)
        x, history
    else
        x
    end
end

#########################
# Method Implementation #
#########################

@inline function omega(t, s)
    angle = sqrt(2.)/2
    ns = vecnorm(s)
    nt = vecnorm(t)
    ts = vecdot(t,s)
    rho = abs(ts/(nt*ns))
    om = ts/(nt*nt)
    if rho < angle
        om = om*convert(typeof(om),angle)/rho
    end
    om
end

@inline linsys_op(x, A) = A*x

"""
The Induced Dimension Reduction method is a family of simple and fast Krylov
subspace algorithms for solving large nonsymmetric linear systems. The idea
behind the IDR(s) variant is to generate residuals that are in the nested
subspaces of shrinking dimensions.
"""
function idrs_method!{T}(X, op, args, C::T, s, tol, maxiter;
    log::MethodLog=MethodLog(), verbose::Bool=false
    )
    verbose && @printf("=== idrs ===\n%4s\t%7s\n","iter","relres")
    R = C - op(X, args...)::T
    normR = vecnorm(R)
	iter = 0

    Z = zero(C)

    P = T[rand!(copy(C)) for k in 1:s]
    U = T[copy(Z) for k in 1:s]
    G = T[copy(Z) for k in 1:s]
    Q = copy(Z)
    V = copy(Z)

    M = eye(eltype(C),s,s)
    f = zeros(eltype(C),s)
    c = zeros(eltype(C),s)

    om::eltype(C) = 1
    iter = 0
    while normR > tol && iter < maxiter
        for i in 1:s,
            f[i] = vecdot(P[i], R)
        end
        for k in 1:s
            # Solve small system and make v orthogonal to P

            c = LowerTriangular(M[k:s,k:s])\f[k:s]
            copy!(V, G[k])
            scale!(c[1], V)

            copy!(Q, U[k])
            scale!(c[1], Q)
            for i = k+1:s
                axpy!(c[i-k+1], G[i], V)
                axpy!(c[i-k+1], U[i], Q)
            end

            # Compute new U[:,k] and G[:,k], G[:,k] is in space G_j

            #V = R - V
            scale!(-1., V)
            axpy!(1., R, V)

            copy!(U[k], Q)
            axpy!(om, V, U[k])
            G[k] = op(U[k], args...)

            # Bi-orthogonalise the new basis vectors

            for i in 1:k-1
                alpha = vecdot(P[i],G[k])/M[i,i]
                axpy!(-alpha, G[i], G[k])
                axpy!(-alpha, U[i], U[k])
            end

            # New column of M = P'*G  (first k-1 entries are zero)

            for i in k:s
                M[i,k] = vecdot(P[i],G[k])
            end

            #  Make r orthogonal to q_i, i = 1..k

            beta = f[k]/M[k,k]
            axpy!(-beta, G[k], R)
            axpy!(beta, U[k], X)

            normR = vecnorm(R)
            iter += 1
            nextiter!(log)
            push!(log, :resnorm, normR)
            verbose && @printf("%3d\t%1.2e\n",iter,normR)
            ((normR < tol) | (iter > maxiter)) && (setconv(log, 0<=normR<tol); return)
            if k < s
                f[k+1:s] = f[k+1:s] - beta*M[k+1:s,k]
            end

        end

        # Now we have sufficient vectors in G_j to compute residual in G_j+1
        # Note: r is already perpendicular to P so v = r

        copy!(V, R)
        Q = op(V, args...)::T
        om = omega(Q, R)
        axpy!(-om, Q, R)
        axpy!(om, V, X)

        normR = vecnorm(R)
        iter += 1
        nextiter!(log)
        push!(log, :resnorm, normR)
        verbose && @printf("%3d\t%1.2e\n",iter,normR)
    end
    setconv(log, 0<=normR<tol)
    setmvps(log, iter)
    verbose && @printf("\n")
end

#################
# Documentation #
#################

#Initialize parameters
doc_call = """    idrs(A, b)
"""
doc!_call = """    idrs!(x, A, b)
"""

doc_msg = "Solve A*x=b using the induced dimension reduction method."
doc!_msg = "Overwrite `x`.\n\n" * doc_msg

doc_arg = ""
doc!_arg = """* `x`: initial guess, overwrite final estimation."""

doc_version = (idrs, doc_call, doc_msg, doc_arg)
doc!_version = (idrs!, doc!_call, doc!_msg, doc!_arg)

#Build docs
for (func, call, msg, arg) in [doc_version, doc!_version]
@doc """
$call

$msg

If `log` is set to `true` is given, method will output a tuple `x, ch`. Where
`ch` is a [`ConvergenceHistory`](@ref) object. Otherwise it will only return `x`.

The `plot` attribute can only be used when `log` is set version.

**Arguments**

$arg

* `A`: linear operator.

* `b`: right hand side.

*Keywords*

* `Pl = 1`: left preconditioner of the method.

* `Pr = 1`: left preconditioner of the method.

* `tol::Real = sqrt(eps())`: stopping tolerance.

* `restart::Integer = min(20,length(b))`: maximum number of iterations per restart.

* `maxiter::Integer = min(20,length(b))`: maximum number of iterations.

* `verbose::Bool = false`: print method information.

* `log::Bool = false`: output an extra element of type `ConvergenceHistory`
containing extra information of the method execution.

* `plot::Bool = false`: plot data. (Only when `log` is set)

**Output**

*`log` is `false`:*

* `x`: approximated solution.

*`log` is `true`:*

* `x`: approximated solution.

* `ch`: convergence history.

*ConvergenceHistory keys*

* `:tol` => `::Real`: stopping tolerance.

* `:resnom` => `::Vector`: residual norm at each iteration.

""" -> func
end
