# Approximate a function f: [-1, 1]^3 -> Seg((k, k))
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebfun
using TensorToolbox: hosvd
using LinearAlgebra
using Random
using Plots; pyplot()

m=3
k=30
Ns=2:1:16

n = 1 + (k - 1) + (k - 1)
M = Segre((k, k))

# closest rank 1 approximation to
#   exp(x1) exp(V x2) diag(2^-1, 2^-2, ...) exp(W x2)
Random.seed!(420)
W1 = normalize(rand(k, k)); W1 = (W1 - W1') / 2
W2 = normalize(rand(k, k)); W2 = (W2 - W2') / 2
function f(x) # :: [-1, 1]^m -> Segre((k, k))
    return [
        [exp(x[1]) / 2.0],
        exp(-W1 * x[2]) * [1, zeros(k - 1)...],
        exp(-W2 * x[3]) * [1, zeros(k - 1)...]
        ]
end

V1(nu) = 1 / 2 # Bound for |(d/dx1)^nu f|
V2(nu) = norm(W1)^nu / 2
V3(nu) = norm(W2)^nu / 2
sigma = maximum([ # Radius of chart
    distance(M, f(zeros(m)), f(x))
    for x in [2 * rand(m) .- 1.0 for _ in 1:1000]])
H = -1.0 / (0.5)^2 # Curvature lower bound
Lambda(N) = (2 / pi) * log(N + 1) + 1 # Chebyshev interpolation operator norm
epsilon(N) = minimum([ # Bound for |g - ghat|
    4 * V1(nu1) / (pi * nu1 * (N - nu1)^nu1) +
    4 * V2(nu2) * Lambda(N) / (pi * nu2 * (N - nu2)^nu2) +
    4 * V3(nu3) * Lambda(N)^2 / (pi * nu3 * (N - nu3)^nu3)
    for nu1 in 1:(N - 1) for nu2 in 1:(N - 1) for nu3 in 1:(N - 1)])

# b is the error bound on the manifold
b(N) = epsilon(N) + 2 / sqrt(abs(H)) * asinh(epsilon(N) * sinh(sqrt(abs(H)) * sigma) / (2 * sigma))

# Loop over nbr of interpolation points
es = [NaN for _ in Ns]
bs = [NaN for _ in Ns]
for (i, N) = enumerate(Ns)
    local fhat = approximate(
        m,
        M,
        f;
        univariate_scheme=chebfun(N),
        decomposition_method=hosvd,
        eps_rel=1e-15,
        )

    # e = max(|f - fhat|)
    es[i] = maximum([
        distance(M, f(x), fhat(x))
        for x in [2 * rand(m) .- 1.0 for _ in 1:1000]])
    bs[i] = b(N)
end

p = plot(Ns, bs;
    label="error bound",
    xlabel="nbr sample points in each direction",
    xticks=Ns,
    yaxis=:log,
    ylims=(1e-16, 2 * maximum([es..., bs...])),
    yticks=([1e0, 1e-5, 1e-10, 1e-15]),
    legend=:topright,
    )
scatter!(p, Ns, es;
    label="measured error")
display(p)

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig(string(f, ".pdf"))
# CSV.write(string(f, ".csv"), DataFrame([:Ns => Ns, :es => es, :bs => bs]))
