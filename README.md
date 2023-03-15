# manifold-valued approximations

`Segre.jl` implements the Segre manifold of rank 1 tensors as an `AbstractManifold` from the `Manifolds.jl` package.

`Aca.jl` implements adaptive cross approximation for approximating functions of the type $$\mathbb{R}^m \to \mathbb{R}^n.$$

`Main.jl` extends the adaptive cross approximation method to functions of the type $$\mathbb{R}^n \to M^n$$ where $M^n$ is an $n$-dimensional manifold.

`Tests.jl` contains numeric tests for some of the methods.
For example, it tests that `exp(::Segre, ...)` really maps vectors in the tangent space to unit speed curves with zero geodesic curvature.

`QOL.jl` contains some quality of life functions.


## TODO
Optimize the ACA.

implement `to_tucker`.

Go through all of the code and replace the `@assert`s with something else as assert may be turned off as an optimization.
