using#={{{=#
    ManifoldsBase,
    Manifolds,
    Kronecker, # TODO: is this needed?
    SplitApplyCombine, #TODO: is this needed?
    Plots#=}}}=#

include("QOL.jl")
include("segre manifold/Segre.jl")
include("../vector-valued approximations/Approximations.jl") # TODO: Make Approximations into a package


#################### Setup ####################

using Random
Random.seed!(420)
# include("Example1.jl")
# include("Example2.jl")
include("Example3.jl")

################ Define approximation scheme ###############

"""
    function approximate(
        M::AbstractManifold,
        m::Int64,
        n::Int64,
        f::Function; # :: [-1, 1]^m -> M^n
        base_approximate=approximate_vector::Function, # :: (m::Int) -> (n::Int) -> ([-1, 1]^m -> R^n) -> ([-1, 1]^m -> R^n)
        kwargs...
        )::Function # :: [-1, 1]^m -> M^n

Approximate a manifold-valued function using Riemann normal coordinate chart.
"""
function approximate(#={{{=#
    M::AbstractManifold,
    m::Int,
    n::Int,
    f::Function; # :: [-1, 1]^m -> M^n
    base_approximate=approximate_vector::Function, # :: (m::Int) -> (n::Int) -> ([-1, 1]^m -> R^n) -> ([-1, 1]^m -> R^n)
    # TODO: chart?
    kwargs...
    )::Function # :: [-1, 1]^m -> M^n

    # Evaluate f on a point cloud in [-1, 1]^m
    xs = [2.0 * rand(m) .- 1.0 for _ in 1:100]
    fs = f.(xs)
    
    # Choose a point on M and linearize from there
    p = mean(M, fs)
    B = DefaultOrthonormalBasis()
    chart = (X -> get_coordinates(M, p, X, B)) ∘ (q -> log(M, p, q)) # log :: M -> T_p M, get_coordinates : T_p M -> R^n
    chart_inv = (X -> exp(M, p, X)) ∘ (X -> get_vector(M, p, X, B)) # get_vector :: R^n -> T_p M, exp : T_p M -> M^n

    g = chart ∘ f
    ghat = base_approximate(m, n, g; kwargs...)
    fhat = chart_inv ∘ ghat
    return fhat
end#=}}}=#


#################### Plot approximation ####################
# ENV["MPLBACKEND"] = "TkAgg" ;# Solves "Warning: No working GUI backend found for matplotlib"
# pyplot()

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
function main(;kwargs...)#={{{=#
    fhat = approximate(M, m, n, f; kwargs...)

    nbr_samples = 100
    ds = zeros(nbr_samples)
    for i in 1:nbr_samples
        x = ones(m) - 2.0 * rand(m)
        ds[i] = distance(M, f(x), fhat(x))
    end
    error = maximum(ds)
    
    print("max error: ")
    @printf("%.1E", error)
    print("\n")

    x = ones(m) - 2.0 * rand(m)

    print("evaluating f takes   ")
    @time(f(x))
    print("evaluating fhat takes")
    @time(fhat(x))
    
    # plot_(M)
    # plot_image_of_unit_grid(M, f;
    #     color="cyan", label="f")
    # plot_image_of_unit_grid(M, fhat;
    #     color="magenta", label="f_hat", linestyle=:dot)
    # plot!([], label=false) # BODGE
    
    return 0
end#=}}}=#
