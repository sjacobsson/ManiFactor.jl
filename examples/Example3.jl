# Approximate a function f: [-1, 1]^3 -> Seg((k, k))
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebfun
using TensorToolbox: hosvd
using LinearAlgebra
using Random
using Plots; pyplot()

m=3
Ns=2:1:16

s=30
M = Segre((s, s))

# f(x) is the closest rank 1 approximation to
#   exp(a x1) exp(V x2) diag(2^-1, 2^-2, ...) exp(W x2)
Random.seed!(420)
W1 = rand(s, s); W1 = W1 / norm(W1); W1 = (W1 - W1') / 2
W2 = rand(s, s); W2 = W2 / norm(W2); W2 = (W2 - W2') / 2
function f(x) # :: [-1, 1]^m -> Segre((k, k))
    return [
        [exp(x[1])],
        exp(-W1 * x[2]) * [1, zeros(s - 1)...],
        exp(-W2 * x[3]) * [1, zeros(s - 1)...]
        ]
end

sigma = sqrt(2) * max(opnorm(W1), opnorm(W2)) + exp(1) - 1 # Radius of chart
H = -exp(2) # Lower bound for curvature
Lambda(N) = (2 / pi) * log(N + 1) + 1 # Chebyshev interpolation operator norm
C1(rho) = exp((rho + rho^-1) / 2) - 1
V2(nu) = 2 * opnorm(W1)^(nu + 1) # Bound for variation(|(d/dx2)^nu f(x)|)
V3(nu) = 2 * opnorm(W2)^(nu + 1) # Bound for variation(|(d/dx3)^nu f(x)|)
epsilon(N) = minimum([ # Bound for |g - ghat|
    4 / ((rho - 1) * rho^N) * C1(rho) + 
    # 4 * Lambda(N) / ((rho - 1) * rho^N) * C2(rho) + 
    4 * V2(nu) * Lambda(N) / (pi * nu * (N - nu)^nu) +
    # 4 * Lambda(N)^2 / ((rho - 1) * rho^N) * C3(rho) 
    4 * V3(nu) * Lambda(N)^2 / (pi * nu * (N - nu)^nu)
    for rho in LinRange(1.01, 40.0, 100) for nu in 1:(N - 1)])

# Bound for d(f(x), fhat(x))
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

    # e = max(d(f(x), fhat(x)))
    es[i] = maximum([
        distance(M, f(x), fhat(x))
        for x in [2 * rand(m) .- 1.0 for _ in 1:1000]])
    bs[i] = b(N)
end

p = plot(Ns, bs;
    label="error bound",
    xlabel="N",
    xticks=Ns,
    yaxis=:log,
    ylims=(1e-16, 2 * maximum([es..., bs...])),
    yticks=([1e0, 1e-5, 1e-10, 1e-15]),
    legend=:topright,
    )
# plot!(p, Ns, epsilon.(Ns);
#     label="epsilon")
scatter!(p, Ns, es;
    label="measured error")
display(p)

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example3.png")
# CSV.write("Example3.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))
