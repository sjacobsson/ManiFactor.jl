# manifold-valued approximations

`Main.jl` extends the approximation schemes in [`Approximations.jl`](TODO) to approximate functions of the type $$\mathbb{R}^n \to M^n$$ where $M^n$ is an $n$-dimensional manifold.

`segre manifold/Segre.jl` implements the Segre manifold of rank 1 tensors as an `AbstractManifold` from the `Manifolds.jl` package.

`segre manifold/Tests.jl` contains unit tests.
For example, it tests that `exp(::Segre, ...)` really maps vectors in the tangent space to unit speed curves with zero geodesic curvature.

`Example1.jl` TODO

`Example2.jl` TODO

`Example3.jl` defines a function $\mathbb{R}^{n} \to \mathbb{R}^{m_1 + m_2}$ that computes the best rank 1 approximation to a pencil of $m_1 \times m_2$ matrices.

`QOL.jl` contains some quality of life functions.


## TODO
implement `to_tucker`.
