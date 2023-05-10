using#={{{=#
    ManifoldsBase,
    Manifolds,
    Kronecker, # TODO: is this needed?
    SplitApplyCombine, #TODO: is this needed?
    Plots#=}}}=#

include("QOL.jl")
include("segre manifold/Segre.jl")
include("vector-valued approximations/Approximations.jl")

include("Example1.jl")

#################### Setup ####################

using Random
Random.seed!(666)

# f : [-1, 1]^m -> M^n is the function we wish to approximate



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

################ Define approximation scheme ###############

function approximate(#={{{=#
    M::AbstractManifold,
    f::Function, # :: [-1, 1]^m -> M^n
    m::Int64;
    base_approximate=approximate1::Function # :: ([-1, 1]^m -> R^n) -> ([-1, 1]^m -> R^n)
    # TODO: chart
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
    ghat = base_approximate(g, m)
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
function main()#={{{=#
    fhat = approximate(M, f, m)

    nbr_samples = 100
    ds = zeros(nbr_samples)::Vector{Float64}
    for i in 1:nbr_samples
        x = ones(m) - 2.0 * rand(m)
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
end#=}}}=#
