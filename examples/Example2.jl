# Approximate a function f: [-1, 1]^2 -> H^2
using Manifolds: Hyperbolic
using ManiFactor
using ApproximatingMapsBetweenLinearSpaces: chebyshev
using LinearAlgebra
using Plots; pyplot(); import Plots: (plot!)

function plot!(#={{{=#
    p::Plots.Plot,
    ::Hyperbolic{2};
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

function plot_grid!(#={{{=#
    p::Plots.Plot,
    M::Hyperbolic{2},
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

function stereographic_projection(#={{{=#
    xs::Vector{Float64};
    pole::Int64=1
    )::Vector{Float64}

    n = length(xs)
    @assert(pole <= n + 1)
    ys = zeros(n + 1) # Initialize

    for i in 1:(n + 1)
        if i < pole
            ys[i] = (2 * xs[i]) / (1 - norm(xs)^2)
        elseif i == pole
            ys[i] = (1 + norm(xs)^2) / (1 - norm(xs)^2)
        elseif i > pole
            ys[i] = (2 * xs[i - 1]) / (1 - norm(xs)^2)
        end
    end
    
    return ys
end#=}}}=#

M = Hyperbolic(2)
f(x) = stereographic_projection(0.25 * [x[1]^2 - x[2]^2, 2 * x[1] * x[2]]; pole=3)

fhat = approximate(2, M, f, univariate_scheme=chebyshev(3))
p1 = plot(; title="9 samples")
plot!(p1, M)
plot_grid!(p1, M, f; label="f", color="cyan")
plot_grid!(p1, M, fhat; label="fhat", color="magenta", linestyle=:dot)

fhat = approximate(2, M, f, univariate_scheme=chebyshev(4))
p2 = plot(; title="16 samples")
plot!(p2, M)
plot_grid!(p2, M, f; label="f", color="cyan")
plot_grid!(p2, M, fhat; label="fhat", color="magenta", linestyle=:dot)

fhat = approximate(2, M, f, univariate_scheme=chebyshev(5))
p3 = plot(; title="25 samples")
plot!(p3, M)
plot_grid!(p3, M, f; label="f", color="cyan")
plot_grid!(p3, M, fhat; label="fhat", color="magenta", linestyle=:dot)

plot(p1, p2, p3, layout=(1, 3))
