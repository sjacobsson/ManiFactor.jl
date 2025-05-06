# Approximate a function f: [-1, 1]^m -> Gr(n, k)
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebyshev

using LinearAlgebra
using SparseArrays
using Preconditioners

using Random; Random.seed!(1)
using Plots; pyplot()
using BenchmarkTools; benchmark = false # Set to true if you want to get timings aswell (sloow)

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
es0 = [NaN for _ in Ns] # Errors
ts0 = [NaN for _ in Ns] # Timings
for (i, N) = enumerate(Ns)
    fhat = approximate(
        m,
        M,
        f;
        p=p,
        univariate_scheme=chebyshev(N),
        )

    g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))
    ghat = get_ghat(fhat)

    es0[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])

    if benchmark
        bmark = @benchmark $fhat(xs[1])
        ts0[i] = minimum(bmark).time * 1e-6 # [ms]
    end
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
ts1 = [NaN for _ in Ns]
for (i, N) = enumerate(Ns)
    fhat = approximate(
        m,
        M,
        f;
        p=p,
        exp=(M, p, X) -> retract(M, p, X, QRRetraction()),
        log=(M, p, X) -> inverse_retract(M, p, X, QRInverseRetraction()),
        univariate_scheme=chebyshev(N),
        )

    g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))
    ghat = get_ghat(fhat)

    es1[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])

    if benchmark
        bmark = @benchmark $fhat(xs[1])
        ts1[i] = minimum(bmark).time * 1e-6 # [ms]
    end
end
scatter!(plt, Ns, es1; label="error using QR retraction")
display(plt)

# Loop over nbr of interpolation points
es2 = [NaN for _ in Ns]
ts2 = [NaN for _ in Ns]
for (i, N) = enumerate(Ns)
    fhat = approximate(
        m,
        M,
        f;
        p=p,
        exp=(M, p, X) -> retract(M, p, X, PolarRetraction()),
        log=(M, p, X) -> inverse_retract(M, p, X, PolarInverseRetraction()),
        univariate_scheme=chebyshev(N),
        )

    g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))
    ghat = get_ghat(fhat)

    es2[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])

    if benchmark
        bmark = @benchmark $fhat(xs[1])
        ts2[i] = minimum(bmark).time * 1e-6 # [ms]
    end
end
scatter!(plt, Ns, es2; label="error using polar retraction")
display(plt)

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example4.png")
# CSV.write("Example4.csv", DataFrame([:Ns => Ns, :es0 => es0, :es1 => es1, :es2 => es2, :ts0 => ts0, :ts1 => ts1, :ts2 => ts2]))

# # To plot the timings
# plt_ = plot(
#     xlabel="N",
#     xticks=Ns,
#     ylims=(0.0, maximum([ts0..., ts1..., ts2...])),
#     legend=:topright,
#     )
# scatter!(plt_, Ns, ts0; label="time using exp")
# scatter!(plt_, Ns, ts1; label="time using QR retraction")
# scatter!(plt_, Ns, ts2; label="time using polar retraction")
