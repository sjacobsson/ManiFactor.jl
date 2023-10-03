using Plots; pyplot(); import Plots: (plot!)
using Manifolds

function plot!(#={{{=#
    p::Plots.Plot,
    M::Sphere{2, ℝ};
    kwargs...
    )

	n = 50
	u = range(0, stop=2 * pi, length=n)
	v = range(0, stop=pi, length=n)
	X = cos.(u) * sin.(v)'
	Y = sin.(u) * sin.(v)'
	Z = ones(n) * cos.(v)'

	plot!(
        p,
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

function plot!(#={{{=#
    p::Plots.Plot,
    M::Hyperbolic{2};
    kwargs...
    )

    n = 30
	a = range(0, stop=1.0, length=n)
	b = range(0, stop=2 * pi, length=n)
	X = sinh.(a) * cos.(b)'
	Y = sinh.(a) * sin.(b)'
	Z = cosh.(a) * ones(n)'

	plot!(
        p,
		X,
		Y,
		Z,
		seriestype=:surface,
		colorbar=:none,
		alpha=0.4,
		c=cgrad(:grays),
		# camera=(150, 30),
		order=-1,
        ;kwargs...
	)
end#=}}}=#

# Plot image of grid in [-1, 1]^2 under f: [-1, 1]^2 -> M
function plot_grid!(#={{{=#
    p::Plots.Plot,
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
    plot!(p, M, [f([xs[1], ys[1]]), f([xs[1], ys[2]])];
        color=color, label=label, linestyle=linestyle, plot_kwargs...)
    for i in range(1, length(xs) - 1)
        for j in range(1, length(ys) - 1)
            x = xs[i]
            x_ = xs[i + 1]
            y = ys[j]
            y_ = ys[j + 1]
            plot!(p, M, [f([x, y]), f([x, y_])];
                color=color, label=false, linestyle=linestyle, plot_kwargs...)
            plot!(p, M, [f([x, y]), f([x_, y])];
                color=color, label=false, linestyle=linestyle, plot_kwargs...)
        end
    end
    xaxis!([-1., 1.]) # BODGE
    yaxis!([-1., 1.]) # BODGE
end#=}}}=#
