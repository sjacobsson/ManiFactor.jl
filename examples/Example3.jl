# Approximate a function f: [-1, 1]^m -> Gr(n, k)
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebyshev
using TensorToolbox: sthosvd # Available tensor decomposition methods are `sthosvd`, `hosvd`, `TTsvd`, `cp_als`.
using LinearAlgebra
using Random; Random.seed!(3)
using Plots; pyplot()

m = 1
Ns = 2:1:25

n = 100
k = 4
M = Grassmann(n, k)

# Fix some stuff
include("hotfix.jl")

# Green's function for heat equation on [0, pi]
a = 1 / 2
K(x, y, t) = (1 / 2) * sum([sin.(l * x) * sin.(l * y) * exp(-t * a^2 * l^2) for l in 1:10])

# Discretize K
A(t) = K([pi * i / (n - 1) for i in 0:(n - 1)], [pi * i / (n - 1) for i in 0:(n - 1)]', t)

v = normalize(2 * rand(n) .- 1.0)
function f(x) # :: [-1, 1]^m -> Grasssmann(n, k)

     # Arnoldi iteration
     qs = [zeros(n) for _ in 1:k] # Initialize
     qs[1] = v
     for i in 2:k
         qs[i] = A(x[1] / 2 + 1.5) * qs[i - 1]
         for j in 1:(i - 1)
             qs[i] = qs[i] - dot(qs[j], qs[i]) * qs[j]
         end
         qs[i] = normalize(qs[i])
     end
     return hcat(qs...)

end

# Loop over nbr of interpolation points
es = [NaN for _ in Ns]
bs = [NaN for _ in Ns]
xs = [2 * rand(m) .- 1.0 for _ in 1:10000]
p = mean(M, f.(xs))
for (i, N) = enumerate(Ns)
    local fhat = approximate(
        m,
        M,
        f;
        p=p,
        univariate_scheme=chebyshev(N),
        decomposition_method=sthosvd,
        tolerance=1e-15,
        )

    local g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))
    local ghat = get_ghat(fhat)

    es[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])
    bs[i] = maximum([norm(g(x) - ghat(x)) for x in xs])
end

p = plot(
    xlabel="N",
    xticks=Ns,
    yaxis=:log,
    ylims=(1e-16, 2 * maximum([es..., bs...])),
    yticks=([1e0, 1e-5, 1e-10, 1e-15]),
    legend=:topright,
    )
plot!(p, Ns[1:end - 2], bs[1:end - 2]; label="error bound")
scatter!(p, Ns, es; label="measured error")
# cs = [(3 + 2 * sqrt(2))^-N for N in Ns]
# scatter!(p, Ns, cs; label="1 / (2 + sqrt(3))^N")
display(p)

# To see that k = 4 is enough for the Krylov subspace to capture the range of A,
#   using IterativeSolvers
#   b = A(2.0) * [0.5 - abs(i / (n - 1) - 0.5) for i in 0:(n - 1)] # Triangle initial condition
#   gmres!(deepcopy(v), A(1.0), b; verbose=true, maxiter=10, restart=100)
# outputs
#   === gmres ===
#   rest	iter	resnorm
#   1       1       8.18e-02
#   1       2       6.02e-02
#   1       3       1.17e-03
#   1       4       1.34e-05
#   1       5       4.85e-09

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example3.png")
# CSV.write("Example3.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))
