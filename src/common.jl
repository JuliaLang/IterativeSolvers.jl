import LinearAlgebra: ldiv!, \

export Identity, ConjugatedDot, UnconjugatedDot

#### Type-handling
"""
    Adivtype(A, b)
Determine type of the division of an element of `b` against an element of `A`:
`typeof(one(eltype(b))/one(eltype(A)))`
"""
Adivtype(A, b) = typeof(one(eltype(b))/one(eltype(A)))

"""
    zerox(A, b)

Build a zeros vector similar to `b` of eltype `Adivtype(A,b)`.
"""
function zerox(A, b)
    T = Adivtype(A, b)
    x = similar(b, T, size(A, 2))
    fill!(x, zero(T))
    return x
end

"""
No-op preconditioner
"""
struct Identity end

\(::Identity, x) = copy(x)
ldiv!(::Identity, x) = x
ldiv!(y, ::Identity, x) = copyto!(y, x)

"""
Conjugated and unconjugated dot products
"""
abstract type AbstractDot end
struct ConjugatedDot <: AbstractDot end
struct UnconjugatedDot <: AbstractDot end

_norm(x, ::ConjugatedDot) = norm(x)
_dot(x, y, ::ConjugatedDot) = dot(x, y)

_norm(x, ::UnconjugatedDot) = sqrt(sum(xₖ->xₖ^2, x))
_dot(x, y, ::UnconjugatedDot) = transpose(@view(x[:])) * @view(y[:])  # allocating, but faster than sum(prod, zip(x,y))
