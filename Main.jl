import Manifolds:#={{{=#
    check_point,
    check_vector,
    exp,
    log,
    norm,
    distance,
    embed,
    rand
import LinearAlgebra:
    normalize#=}}}=#
using#={{{=#
    ManifoldsBase,
    Manifolds,
    LinearAlgebra,
    Kronecker,
    SplitApplyCombine, #TODO: is this needed?
    ApproxFun,
    Plots#=}}}=#

include("QOL.jl")
include("Segre.jl")
# TODO: Go through all of the code and replace all @asserts with smthing else as assert may be turned off as an aoptimization


#################### Setup ####################
#
function stereographic_projection(#={{{=#
    xs::Vector{Float64};
    pole::Int64=1
    )::Vector{Float64}

    n = length(xs)
    @assert(pole <= n + 1)
    ys = zeros(n + 1) # Initialize

    for i in 1:(n + 1)
        if i < pole
            ys[i] = (2 * xs[i]) / (1 + norm(xs)^2)
        elseif i == pole
            ys[i] = (-1 + norm(xs)^2) / (1 + norm(xs)^2)
        elseif i > pole
            ys[i] = (2 * xs[i - 1]) / (1 + norm(xs)^2)
        end
    end
    
    return ys
end#=}}}=#

function inverse_stereographic_projection(#={{{=#
    ys::Vector{Float64};
    pole::Int64=1
    )::Vector{Float64}

    n = length(ys) - 1
    @assert(pole <= n + 1)
    xs = zeros(n) # Initialize

    for i in 1:n
        xs[i] = ys[i] / ys[n + 1]
    end
    
    return xs
end#=}}}=#

using Random
Random.seed!(666)

# f : [-1, 1]^k -> M^n is the function we wish to approximate

# k = 2
# n = 3
# M = Sphere(n)
# A = rand(n, k)
# f(x) = stereographic_projection(A * x)

m = 4
M = OrthogonalMatrices(k)
n = manifold_dimension(M)
function f(x::Vector{Float64})
    # Assert that the entries of x fit in a square matrix
    l = Int64(sqrt(length(x)))
    @assert(length(x) == l^2)
    Q, _ = qr(reshape(x, l, l))
    return Q
end

# k = 4
# # n = Int64(k^2 + (k + 1) * k / 2)
# M = ProductManifold(
#     OrthogonalMatrices(k),
#     Euclidean(Int64((k + 1) * k / 2))
#     )
# n = manifold_dimension(M)
# # TODO: Write better check_point(M::Euclidean, p)?
# function f(x::Vector{Float64})
#     # Assert that the entries of x fit in a square matrix
#     l = Int64(sqrt(length(x)))
#     @assert(length(x) == l^2)
#     Q, R = qr(reshape(x, l, l))
#     vec_R = [R[i, j] for i in 1:l for j in 1:l if i <= j] # Flatten upper triangular part of R
#     return ProductRepr(Q, vec_R)
# end

# println(check_point(M, (Q))

#################### Approximate f ####################

function cheb2(#={{{=#
    g::Function,# R^k -> R^n
    )::Function# R^k -> R^n

    grid_length = 10
    fdomain = TensorSpace(repeat([Chebyshev()], k)...) # assumes k = 2
    grid = Vector{Vector{Float64}}(points(fdomain, grid_length^k))

    gs = g.(grid)

    # Compute c_ij in f(x, y) = c_ij T_i(x) T_j(y)
    cs = [transform(fdomain, [y[i] for y in gs]) for i in 1:n]

    return v -> [Fun(fdomain, c)(v) for c in cs]
end#=}}}=#

# Adaptive Cross Approximation of scalar valued functions using products of Chebyshev polynomials as a basis
u = rand(m) # BODGE
import Optim: optimize, minimizer, Fminbox
function aca(#={{{=#
    f;# R^k -> R^n
    tol = 1e-6::Float64
    )# R^k -> R^n
    # The implementation is similar to Townsend's and Trefethen's cheb2paper fig. 2.1

    # Initialize
    es = push!([], f)
    gs = push!([], _ -> 0.0)
    error = Inf

    # Define a domain
    lower = -1.0 * ones(m)
    upper = ones(m)

    k = 1
    while abs(error) > tol
        e = deepcopy(es[k])
        g = deepcopy(gs[k])

        x_min = minimizer(optimize(
            x -> -abs(e(x)),
            lower,
            upper,
            2.0 * rand(m) .- 1.0,
            Fminbox()
            ))
        error = e(x_min)
        println(abs(error))

        # for j in 1:m
        e_restricted = [t -> e([x_min[1:j-1]..., t, x_min[j+1:end]...]) for j in 1:m]
        nbr_interpolation_points = 1000 # TODO: Make this grow
        factors = [Fun(
            Chebyshev(),
            ApproxFun.transform(
                Chebyshev(),
                e_restricted[j].(points(Chebyshev(), nbr_interpolation_points)) / error)
            ) for j in 1:m]

        push!(es, x -> e(x) - error * prod(a(t) for (a, t) in zip(factors, x)))
        push!(gs, x -> g(x) + error * prod(a(t) for (a, t) in zip(factors, x)))
        k = k + 1
    end

    return gs[end]
end#=}}}=#
# TODO: Use succesively more Chebyshev points in the chebfuns, we don't need floating-point precision from the start.
# TODO: Use several guesses when computing x_min
# TODO: Right now chebk assumes g is scalar-valued

function approximate(#={{{=#
    M::AbstractManifold,
    f::Function; # :: R^k -> M^n
    approximate_linear=aca::Function # :: (R^k -> R^n) -> (R^k -> R^n)
    # TODO: chart
    )::Function # :: R^k -> M^n

    # Evaluate f on a point cloud in [-1, 1]^k
    xs = [2.0 * rand(k) .- 1.0 for _ in 1:100]
    fs = f.(xs)
    
    # Choose a point on M and linearize from there
    p = mean(M, fs)
    B = DefaultOrthonormalBasis()
    chart = (X -> get_coordinates(M, p, X, B)) ∘ (q -> log(M, p, q)) # log : M -> T_p M, get_coordinates : T_p M -> R^n
    chart_inv = (X -> exp(M, p, X)) ∘ (X -> get_vector(M, p, X, B)) # get_vector : R^n -> T_p M, exp : T_p M -> M^n

    g = chart ∘ f
    ghat = approximate_linear(g)
    fhat = chart_inv ∘ ghat
    return fhat
end#=}}}=#


#################### Plot approximation ####################
ENV["MPLBACKEND"] = "TkAgg" ;# Solves "Warning: No working GUI backend found for matplotlib"
pyplot()

function plot_(_::Sphere{2, ℝ}, kwargs...)#={{{=#
	n = 50
	u = range(0, stop=2 * pi, length=n)
	v = range(0, stop=pi, length=n)
	X = cos.(u) * sin.(v)'
	Y = sin.(u) * sin.(v)'
	Z = ones(n) * cos.(v)'

	plot(
		X,
		Y,
		Z,
		seriestype=:surface,
		colorbar=:none,
		alpha=0.3,
		c=cgrad(:grays),
		order=-1,
        ;kwargs...
	    )
end#=}}}=#

function plot_(_::Hyperbolic{2}, kwargs...)#={{{=#
    n = 30
	a = range(0, stop=1.0, length=n)
	b = range(0, stop=2 * pi, length=n)
	X = sinh.(a) * cos.(b)'
	Y = sinh.(a) * sin.(b)'
	Z = cosh.(a) * ones(n)'
	plot(
		X,
		Y,
		Z,
		seriestype=:surface,
		colorbar=:none,
		alpha=0.3,
		c=cgrad(:grays),
		# camera=(150, 30),
		order=-1,
        ;kwargs...
	)
end#=}}}=#

function plot_image_of_unit_grid(#={{{=#
    M::AbstractManifold,
    f::Function;
    label=false,
    color="blue",
    linestyle=:solid
    )
    plot_kwargs = Dict(
        :wireframe=>false,
        :geodesic_interpolation=>10,
        :linewidth=>2,
        # :viewangle=>(120, 40)
        )

    xs = [x for x in range(-1.0, 1.0, 10)]
    ys = [y for y in range(-1.0, 1.0, 10)]
    plot!(M, [f([xs[1], ys[1]]), f([xs[1], ys[2]])];
        color=color, label=label, linestyle=linestyle, plot_kwargs...)
    for i in range(1, length(xs) - 1)
        for j in range(1, length(ys) - 1)
            x = xs[i]
            x_ = xs[i + 1]
            y = ys[j]
            y_ = ys[j + 1]
            plot!(M, [f([x, y]), f([x, y_])];
                color=color, label=false, linestyle=linestyle, plot_kwargs...)
            plot!(M, [f([x, y]), f([x_, y])];
                color=color, label=false, linestyle=linestyle, plot_kwargs...)
        end
    end
    xaxis!([-1., 1.]) # BODGE
    yaxis!([-1., 1.]) # BODGE
end#=}}}=#

using Printf
function main()
    fhat = approximate(M, f)

    nbr_samples = 100
    ds = zeros(nbr_samples)::Vector{Float64}
    for i in 1:nbr_samples
        x = ones(k) - 2.0 * rand(k)
        ds[i] = distance(M, f(x), fhat(x))
    end
    rms_error = sqrt(sum(ds.^2 / nbr_samples))
    
    print("rms error: ")
    @printf("%.1E", rms_error)
    print("\n")
    
    # plot_(M)
    # plot_image_of_unit_grid(M, f;
    #     color="cyan", label="f")
    # plot_image_of_unit_grid(M, fhat;
    #     color="magenta", label="f_hat", linestyle=:dot)
    # plot!([], label=false) # BODGE
end
