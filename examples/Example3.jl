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
        Tridiagonal(-ones(n - 1), 2 * ones(n), -ones(n - 1)) / Delta^2 +
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
    ylims=(1e-11, 2 * maximum([1e1, es..., bs...])),
    yticks=([1e0, 1e-5, 1e-10]),
    legend=:topright,
    )
plot!(plt, Ns[1:end - 2], bs[1:end - 2]; label="error bound")
scatter!(plt, Ns, es; label="measured error")
display(plt)

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example3.png")
# CSV.write("Example3.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))

# # To see that k = 5 is enough for the Krylov subspace to capture the range of A,
# using IterativeSolvers
# Delta = 1.0 / (n - 1)
# A = sparse(
#     Tridiagonal(-ones(n - 1), 2 * ones(n), -ones(n - 1)) / Delta^2 +
#     1.5 * Tridiagonal(-ones(n - 1), zeros(n), ones(n - 1)) / (2 * Delta) +
#     1.5 * Diagonal(ones(n))
#     )
# P = AMGPreconditioner(A)
# gmres(A, B; verbose=true, maxiter=10, restart=100, Pl=P);
# # outputs
# # === gmres ===
# # rest	iter	resnorm
# #   1	  1	3.15e-02
# #   1	  2	1.95e-03
# #   1	  3	1.32e-04
# #   1	  4	9.68e-06
# #   1	  5	6.96e-07
# #   1	  6	4.60e-08
# #   1	  7	2.70e-09

# # To see that n = 1000 is enough to solve the original differential equation,
# using DifferentialEquations
# using Interpolations
# function my_ode!(du, u, p, t)
#     du[1] = u[2]
#     du[2] = 1.5 * u[2] + 1.5 * u[1] - t * (1 - t)
# end
# function bc!(residual, u, p, t)
#     residual[1] = u[1][1]
#     residual[2] = u[end][1]
# end
# u0 = [0.0, 0.0]
# tspan = (0.0, 1.0)
# solution = solve(BVProblem(my_ode!, bc!, u0, tspan), reltol=1e-15)
# Y = gmres(A, B; maxiter=5, restart=100, Pl=P)
# estimate = linear_interpolation([i / (n - 1) for i in 0:(n - 1)], Y)
# for _ in 1:10
#     t = rand()
#     println(abs(solution(t)[1] - estimate(t)))
# end
# # outputs
# # 5.323703117913678e-5
# # 5.706897863845886e-5
# # 5.1134400665084866e-5
# # 8.578179572703553e-5
# # 5.1012730846297996e-5
# # 6.367391520439511e-5
# # 6.889607805461023e-5
# # 5.161241203422741e-5
# # 5.1012019066916034e-5
# # 5.130422256646554e-5
