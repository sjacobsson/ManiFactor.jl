# Approximate a function f: [-1, 1]^m -> Gr(n, k)
using Manifolds
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebfun
using TensorToolbox: hosvd
using LinearAlgebra
using Random
using Plots; pyplot()

m = 2
Ns = 2:22

n = 100
k = 3
M = Grassmann(n, k)

# Fix some stuff
import ManifoldsBase: get_coordinates_orthonormal, get_vector_orthonormal
get_coordinates_orthonormal(::Grassmann{<:Any, <:Any, ℝ}, ::Matrix{Float64}, v::Matrix{Float64}, _) = v[:]
get_vector_orthonormal(::Grassmann{n, k, ℝ}, ::Matrix{Float64}, c::Vector{Float64}, _) where {n, k} = reshape(c, (n, k))

# TODO: Cite Higham
Random.seed!(3)
A = SymTridiagonal(2 * ones(n), -1 * ones(n - 1))
A[end] = 1.0
B = SymTridiagonal(4 * ones(n), ones(n - 1))
B[end] = 2.0
C = zeros(n, n)
C[end] = 1.0
b = rand(n)
function f(x) # :: [-1, 1]^m -> Grasssmann(n, k)
    lambda = x[1] + 3
    sigma = x[2]
    h = 1.0 / n
    R = A / h - lambda * h * B / 6 + lambda / (lambda - sigma) * C
    krylov = hcat([R^i * b for i in 0:(k - 1)]...)
    return Matrix(qr(krylov).Q)
end

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

    local p = get_p(fhat)
    local ghat = get_ghat(fhat)
    local g = (X -> get_coordinates(M, p, X, DefaultOrthonormalBasis())) ∘ (x -> log(M, p, f(x)))

    xs = [2 * rand(m) .- 1.0 for _ in 1:1000]
    es[i] = maximum([distance(M, f(x), fhat(x)) for x in xs])
    bs[i] = maximum([norm(g(x) - ghat(x)) for x in xs])
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
scatter!(p, Ns, es;
    label="measured error")
cs = [(2 + sqrt(3))^-N for N in Ns]
# scatter!(p, Ns, cs;
#     label="1 / (2 + sqrt(3))^N")
display(p)

# # To save figure and data to file:
# using CSV
# using DataFrames: DataFrame
# savefig("Example4.png")
# CSV.write("Example4.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs, :cs => cs]))
