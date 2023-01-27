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

# k = 2
# n = 3
# todo = 3 * 2
# M = ProductManifold(OrthogonalMatrices(n), Euclidean(todo))
# function f(x::Vector{Float64})
#     Q, R = qr(X)
#     return vcat(, flatten(R))
# end


#################### Approximate f ####################

function approximate(#={{{=#
    M::AbstractManifold,
    # p::Vector{Float64}, # p € M^n
    f::Function; # f: R^k -> M
    # chart::Function=pa(exp, M), # chart: T_p M -> M
    )::Function # : R^k -> M

    # Evaluate f on a point cloud in R^k
    grid_length = 10
    fdomain = TensorSpace(repeat([Chebyshev()], k)...)
    grid = Vector{Vector{Float64}}(points(fdomain, grid_length^k)) # TODO: Use static arrays?
    fs = f.(grid)
    # TODO: Is there a better way to choose the domain, e.g. a disk or other shape?
    
    # Linearize M from p (for this we don't need to evaluate f specifically on grid, but why not)
    p = mean(M, fs)
    B = DefaultOrthonormalBasis()
    chart = (X -> get_coordinates(M, p, X, B)) ∘ (q -> log(M, p, q)) # log : M -> T_p M, get_coordinates : T_p M -> R^n
    chart_inv = (X -> exp(M, p, X)) ∘ (X -> get_vector(M, p, X, B)) # get_vector : R^n -> T_p M, exp : T_p M -> M

    gs = chart.(fs)

    # Compute c_ij in f(x, y) = c_ij T_i(x) T_j(y)
    cs = [transform(fdomain, [g[i] for g in gs]) for i in 1:n]
    # TODO: How is this best rewritten to allow for low-rank approximations of f_ij and/or c_ij?
    # TODO: Look into cross adaptive approximation, and the cheb2paper etc
    # TODO: Maybe look at LowRankFun(::Function, ::TensorSpace)

    # Approximate the linearized function g = log_p . f
    ghat(v) = [Fun(fdomain, c)(v) for c in cs] # project(M, p, .) projects to the tangent space
    # TODO: since ghat = c_ij T_i T_j, can we use the identity for products of Chebyshev polys?

    # Approximate f by exp_p . ghat
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
