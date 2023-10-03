# Approximate a function f: [-1, 1]^2 -> S^2
include("../Approximations.jl")
include("Auxiliary.jl")

function plot!(#={{{=#
    p::Plots.Plot,
    _::Sphere{2, ℝ},
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

M = Sphere(2)
f(x) = stereographic_projection([x[1]^2 - x[2]^2, 2 * x[1] * x[2]])

fhat = approximate(2, M, f, univariate_scheme=chebfun(3))
p1 = plot(; title="9 sample points")
plot!(p1, M)
plot_grid!(p1, M, f; label="f", color="cyan")
plot_grid!(p1, M, fhat; label="fhat", color="magenta", linestyle=:dot)

fhat = approximate(2, M, f, univariate_scheme=chebfun(4))
p2 = plot(; title="16 sample points")
plot!(p2, M)
plot_grid!(p2, M, f; label="f", color="cyan")
plot_grid!(p2, M, fhat; label="fhat", color="magenta", linestyle=:dot)

fhat = approximate(2, M, f, univariate_scheme=chebfun(5))
p3 = plot(; title="25 sample points")
plot!(p3, M)
plot_grid!(p3, M, f; label="f", color="cyan")
plot_grid!(p3, M, fhat; label="fhat", color="magenta", linestyle=:dot)

plot(p1, p2, p3, layout=(1, 3))
