# Internal Documentation

Documentation for `IterativeSolvers.jl`'s internals.

```@contents
Pages = ["internal.md"]
Depth = 4
```

## Index

```@index
Pages = ["internal.md"]
```

## ConvergenceHistory Internals

**`Typealiases`**

```@docs
IterativeSolvers.PlainHistory
IterativeSolvers.RestartedHistory
```

**`Functions`**

```@docs
IterativeSolvers.nextiter!
IterativeSolvers.reserve!
IterativeSolvers.shrink!
IterativeSolvers.setmvps
IterativeSolvers.setmtvps
IterativeSolvers.setconv
IterativeSolvers.showplot
```

## KrylovSubspace Internals

**`Functions`**

```@docs
IterativeSolvers.lastvec
IterativeSolvers.nextvec
IterativeSolvers.init!
IterativeSolvers.initrand!
IterativeSolvers.appendunit!
IterativeSolvers.orthogonalize
```

## Functions

```@docs
IterativeSolvers.idfact
IterativeSolvers.isconverged
IterativeSolvers.thickrestart!
IterativeSolvers.harmonicrestart!
IterativeSolvers.extend!
```