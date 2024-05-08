# Approximate a function f: [-1, 1]^3 -> Seg((k, k))
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebyshev
using TensorToolbox: sthosvd # Available tensor decomposition methods are `sthosvd`, `hosvd`, `TTsvd`, `cp_als`.
using LinearAlgebra
using Random; Random.seed!(1)
using Plots; pyplot()

m=3
Ns=2:1:16

s=100
s=10 # BODGE
M = Segre((s, s))

# f(x) is the closest rank 1 approximation to
#   exp(a x1) exp(V x2) diag(2^-1, 2^-2, ...) exp(W x2)
W1 = rand(s, s); W1 = (W1 - W1') / 2; W1 = W1 / norm(W1)
W2 = rand(s, s); W2 = (W2 - W2') / 2; W2 = W2 / norm(W2)
e1 = [1, zeros(s - 1)...]
function f(x) # :: [-1, 1]^m -> Segre((k, k))
    return [
        [exp(x[1])],
        exp(W1 * x[2]) * e1,
        exp(W2 * x[3]) * e1
        ]
end

H = -exp(2) # Lower bound for curvature

# Loop over nbr of interpolation points
es = [NaN for _ in Ns]
bs = [NaN for _ in Ns]
xs = [2 * rand(m) .- 1.0 for _ in 1:1000]
p = mean(M, f.(xs))
sigma = maximum([ distance(M, p, f(x)) for x in xs])
for (i, N) = enumerate(Ns)
    local fhat = approximate(
        m,
        M,
        f;
        p=p,
        univariate_scheme=chebyshev(N),
        decomposition_method=sthosvd,
        # tolerance=1e-11,
        )

    # To see what multilinear rank is used, add the line
    #   println(mrank(G_decomposed))
    # in approximate_vector

    local ghat = get_ghat(fhat)
    local g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))

    es[i] = maximum([
        distance(M, f(x), fhat(x))
        for x in xs])
    bs[i] = let
            epsilon = maximum([norm(g(x) - ghat(x)) for x in xs])
            epsilon + 2 / sqrt(abs(H)) * asinh(epsilon * sinh(sqrt(abs(H)) * sigma) / (2 * sigma))
        end
end

plt = plot(;
    xlabel="N",
    xticks=Ns,
    yaxis=:log,
    ylims=(1e-16, 2 * maximum([es..., bs...])),
    yticks=([1e0, 1e-5, 1e-10, 1e-15]),
    legend=:topright,
    )
plot!(plt, Ns[1:end-3], bs[1:end-3]; label="error bound")
scatter!(plt, Ns, es; label="measured error")
# scatter!(plt, Ns, [minimum([4 * exp((rho + 1 / rho) / 2) * rho^-N / (rho - 1) for rho in 1:0.1:100]) for N in Ns]; label="inf_rho(M rho^-N)")
display(plt)

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example4.png")
# CSV.write("Example4.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))
