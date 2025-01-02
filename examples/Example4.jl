# Approximate a function f: [-1, 1]^m -> Gr(n, k)
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebyshev

using LinearAlgebra
using SparseArrays
using Preconditioners

using Random; Random.seed!(1)
using Plots; pyplot()

m = 2
Ns = 2:1:14

n = 1000
k = 5
M = Grassmann(n, k)
# M = MetricManifold(Grassmann(n, k), CanonicalMetric()) # Euclidean and canonical metric are the same

# Fix some stuff
include("hotfix.jl")

B = [(1.0 - i / (n - 1)) * (i / (n - 1)) for i in 0:(n - 1)]
function f(x) # :: [-1, 1]^m -> Grassmann(n, k)

    # Discretize d^2/dt + x1 d/dt + x2
    Delta = 1.0 / (n - 1)
    A = sparse(
        Tridiagonal(ones(n - 1), -2 * ones(n), ones(n - 1)) / Delta^2 +
        (1.5 + x[1] / 2) * Tridiagonal(-ones(n - 1), zeros(n), ones(n - 1)) / (2 * Delta) +
        (1.5 + x[2] / 2) * Diagonal(ones(n))
        )
    P = AMGPreconditioner(A)

     # Arnoldi iteration
     qs = [zeros(n) for _ in 1:k] # Initialize
     qs[1] = normalize(B)
     for i in 2:k
         qs[i] = P \ (A * qs[i - 1])
         for j in 1:(i - 1)
             qs[i] = qs[i] - dot(qs[j], qs[i]) * qs[j]
         end
         # println(norm(qs[i])) # If these are too small we might be introducing rounding errors
         qs[i] = normalize(qs[i])
     end

     return hcat(qs...)
end

# Loop over nbr of interpolation points
xs = [2 * rand(m) .- 1.0 for _ in 1:100]
p = mean(M, f.(xs))
es0 = [NaN for _ in Ns]
for (i, N) = enumerate(Ns)
    local fhat = approximate(
        m,
        M,
        f;
        p=p,
        univariate_scheme=chebyshev(N),
        )

    local g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))
    local ghat = get_ghat(fhat)

    es0[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])
end
plt = plot(
    xlabel="N",
    xticks=Ns,
    yaxis=:log,
    ylims=(1e-11, 2 * maximum([1e1, es0...])),
    yticks=([1e0, 1e-5, 1e-10]),
    legend=:topright,
    )
scatter!(plt, Ns, es0; label="error using exp")
display(plt)

# Loop over nbr of interpolation points
es1 = [NaN for _ in Ns]
for (i, N) = enumerate(Ns)
    local fhat = approximate(
        m,
        M,
        f;
        p=p,
        exp=(M, p, X) -> retract(M, p, X, QRRetraction()),
        log=(M, p, X) -> inverse_retract(M, p, X, QRInverseRetraction()),
        univariate_scheme=chebyshev(N),
        )

    local g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))
    local ghat = get_ghat(fhat)

    es1[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])
end
scatter!(plt, Ns, es1; label="error using QR retraction")
display(plt)

# Loop over nbr of interpolation points
es2 = [NaN for _ in Ns]
for (i, N) = enumerate(Ns)
    local fhat = approximate(
        m,
        M,
        f;
        p=p,
        exp=(M, p, X) -> retract(M, p, X, PolarRetraction()),
        log=(M, p, X) -> inverse_retract(M, p, X, PolarInverseRetraction()),
        univariate_scheme=chebyshev(N),
        )

    local g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))
    local ghat = get_ghat(fhat)

    es2[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])
end
scatter!(plt, Ns, es2; label="error using polar retraction")
display(plt)

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example4.png")
# CSV.write("Example4.csv", DataFrame([:Ns => Ns, :es0 => es0, :es1 => es1, :es2 => es2]))

# # To benchmark
# using BenchmarkTools
# fhat0 = approximate(
#     m,
#     M,
#     f;
#     p=p,
#     univariate_scheme=chebyshev(10),
#     reqrank=[5, 5, 5],
#     )
# fhat1 = approximate(
#     m,
#     M,
#     f;
#     p=p,
#     exp=(M, p, X) -> retract(M, p, X, QRRetraction()),
#     log=(M, p, X) -> inverse_retract(M, p, X, QRInverseRetraction()),
#     univariate_scheme=chebyshev(10),
#     reqrank=[5, 5, 5],
#     )
# fhat2 = approximate(
#     m,
#     M,
#     f;
#     p=p,
#     exp=(M, p, X) -> retract(M, p, X, PolarRetraction()),
#     log=(M, p, X) -> inverse_retract(M, p, X, PolarInverseRetraction()),
#     univariate_scheme=chebyshev(10),
#     reqrank=[5, 5, 5],
#     )
# print("f time: "); @btime f(rand(2)) # btime reports fastest time
# println("fhat0 error: ", maximum([distance(M, f(x), fhat0(x)) for x in xs]))
# print("fhat0 time: "); @btime fhat0(rand(2))
# println("fhat1 error: ", maximum([distance(M, f(x), fhat1(x)) for x in xs]))
# print("fhat1 time: "); @btime fhat1(rand(2))
# println("fhat2 error: ", maximum([distance(M, f(x), fhat2(x)) for x in xs]))
# print("fhat2 time: "); @btime fhat2(rand(2))
# "" # BODGE
