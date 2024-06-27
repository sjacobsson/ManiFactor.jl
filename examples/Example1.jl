# Approximate a function f: [-1, 1]^2 -> S^2
using ManiFactor
using Manifolds: Sphere, get_point, StereographicAtlas
using ApproximatingMapsBetweenLinearSpaces: chebyshev
using Plots; pyplot(); import Plots: (plot!)

function plot_sphere!(#={{{=#
    p::Plots.Plot,
    M,
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

function plot_grid!(#={{{=#
    p::Plots.Plot,
    M,
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

n = 2
M = Sphere(n)

m = 2
f(x) = get_point(M, StereographicAtlas(), :south, [x[1]^2 - x[2]^2, 2 * x[1] * x[2]])

fhat = approximate(m, M, f, univariate_scheme=chebyshev(3))
p1 = plot(; title="9 samples")
plot_sphere!(p1, M)
plot_grid!(p1, M, f; label="f", color="cyan")
plot_grid!(p1, M, fhat; label="fhat", color="magenta", linestyle=:dot)

fhat = approximate(m, M, f, univariate_scheme=chebyshev(4))
p2 = plot(; title="16 samples")
plot_sphere!(p2, M)
plot_grid!(p2, M, f; label="f", color="cyan")
plot_grid!(p2, M, fhat; label="fhat", color="magenta", linestyle=:dot)

fhat = approximate(m, M, f, univariate_scheme=chebyshev(5))
p3 = plot(; title="25 samples")
plot_sphere!(p3, M)
plot_grid!(p3, M, f; label="f", color="cyan")
plot_grid!(p3, M, fhat; label="fhat", color="magenta", linestyle=:dot)

plot(p1, p2, p3, layout=(1, 3))
