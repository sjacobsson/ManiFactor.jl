# Approximate a function f: [-1, 1]^m -> Gr(n, k)
# TODO: Why I have to run this example twice?
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebyshev
using TensorToolbox: sthosvd # Available tensor decomposition methods are `sthosvd`, `hosvd`, `TTsvd`, `cp_als`.
using LinearAlgebra
using Random; Random.seed!(1)
using Plots; pyplot()

m = 1
Ns = 2:1:22

n = 100
k = 3
M = Grassmann(n, k)

# Fix some stuff
include("hotfix.jl")

# Green's function for heat equation on [0, pi]
a = 1 / 2
infinity = 10
K(x, y, t) = (pi / 2) * sum([sin.(l * x) * sin.(l * y) * exp(-t * a * l^2) for l in 1:infinity])

# Discretize K
A(t) = pi / (n - 1) * K([pi * i / (n - 1) for i in 0:(n - 1)], [pi * i / (n - 1) for i in 0:(n - 1)]', t)

v = 2 * rand(n) .- 1.0
function f(x) # :: [-1, 1]^m -> Grasssmann(n, k)

     # Arnoldi iteration
     qs = [zeros(n) for _ in 1:k] # Initialize
     qs[1] = normalize(v)
     for i in 2:k
         qs[i] = A(x[1] / 2 + 1.5) * qs[i - 1]
         for j in 1:(i - 1)
             qs[i] = qs[i] - dot(qs[j], qs[i]) * qs[j]
         end
         # println(norm(qs[i])) # If these are too small we might be introducing rounding errors
         qs[i] = normalize(qs[i])
     end

     return hcat(qs...)
end

# Loop over nbr of interpolation points
es = [NaN for _ in Ns]
bs = [NaN for _ in Ns]
xs = [2 * rand(m) .- 1.0 for _ in 1:100]
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

plt = plot(
    xlabel="N",
    xticks=Ns,
    yaxis=:log,
    ylims=(1e-16, 2 * maximum([es..., bs...])),
    yticks=([1e0, 1e-5, 1e-10, 1e-15]),
    legend=:topright,
    )
plot!(plt, Ns[1:end - 3], bs[1:end - 3]; label="error bound")
scatter!(plt, Ns, es; label="measured error")
cs = [(3 + 2 * sqrt(2))^-N for N in Ns]
# scatter!(plt, Ns, cs; label="1 / (3 + 2 * sqrt(2))^N")
display(plt)

# To see that k = 4 is enough for the Krylov subspace to capture the range of A,
#   using IterativeSolvers
#   b = normalize([exp(-((i / (n - 1) - 0.5) * 10)^2) for i in 0:(n - 1)]) # Gaussian
#   gmres!(deepcopy(v), A(2.0), A(2.0) * b; verbose=true, maxiter=10, restart=100)
# outputs
#   === gmres ===
#   rest    iter    resnorm
#   1       1       1.14e+00
#   1       2       2.70e-01
#   1       3       2.77e-03
#   1       4       6.44e-05
#   1       5       4.65e-07
#   1       6       3.81e-10

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example3.png")
# CSV.write("Example3.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs, :cs => cs]))
