using Manifolds, ApproxFun, Plots


#################### Some quality of life functions ####################

function pa(f,a...) # Partial application{{{
  (b...) -> f(a...,b...)
end#=}}}=#

function sproj( # Stereographic projection from the south pole{{{
    x::Float64,
    y::Float64
    )
    
    return -[
        2 * x / (1 + x^2 + y^2),
        2 * y / (1 + x^2 + y^2),
        (-1 + x^2 + y^2) / (1 + x^2 + y^2)
        ]
end#=}}}=#

function proju( # Project upwards from xy-plane to hyperbole {{{
    x::Float64,
    y::Float64
    )
    
    return [
        x,
        y,
        sqrt(1 + x^2 + y^2)
        ]
end#=}}}=#

function normalize(x)#={{{=#
    return x / sqrt(x' * x)
end#=}}}=#

function relu(x)#={{{=#
    return min(0., x)
end#=}}}=#


#################### Setup ####################

# f : [-1, 1]x[-1, 1] -> M is the function we wish to approximate

M = Sphere(2)
f(v) = sproj((v[1]^2 - v[2]^2) / 4, v[1] * v[2] / 2) # Stereographic projection of 1/4 z^2 
# f(v) = sproj((v[1]^3 - v[2]^4) / 4, v[1] * v[2] / 2)
# f(v) = normalize([0.9580 -0.2439; 0.3942 0.8628; 0.3899 0.0580] * v + [0.2725, -0.7744, 0.8169]) # Projection onto S^2 of some random affine map : R^2 -> R^3

# M = Hyperbolic(2)
# f(v) = proju(v[1]^2 - v[2]^2, 2 * v[1] * v[2])
# f(v) = proju(exp(v[1]) * cos(pi * v[2]), exp(v[1]) * sin(pi * v[2]))
# f(v) = proju(v[1], v[2] + relu(v[1])^2)


#################### Approximate f ####################

function approximate_2d(#={{{=#
    f::Function # f: R^2 -> M
    )::Function
    # Evaluate f on a point cloud in R^2
    grid_length = 5
    domain = TensorSpace(Chebyshev(), Chebyshev())
    grid = points(domain, grid_length^2)
    fs = f.(grid)
    # TODO: Is there a better way to choose the domain, e.g. a disk or other shape?

    # Linearize M from p (for this we don't need to evaluate f specifically on grid, but why not)
    p = mean(M, fs)
    gs = pa(log, M, p).(fs) # gs subset of T_p M

    # Compute c_ij in f(x, y) = c_ij T_i(x) T_j(y)
    c1s = transform(domain, [g[1] for g in gs]) # TODO: do this transpose less bodgy
    c2s = transform(domain, [g[2] for g in gs])
    c3s = transform(domain, [g[3] for g in gs])
    # TODO: How is this best rewritten to allow for low-rank approximations of f_ij and/or c_ij?
    # TODO: Look into cross adaptive approximation, and the cheb2paper etc
    # TODO: Maybe look at LowRankFun(::Function, ::TensorSpace)

    # Approximate the linearized function g = log_p . f
    ghat1 = Fun(domain, c1s)
    ghat2 = Fun(domain, c2s)
    ghat3 = Fun(domain, c3s)
    ghat(v) = project(M, p, [ghat1(v), ghat2(v), ghat3(v)]) # project to tangent space
    # TODO: Stay in the tangent space somehow?
    # TODO: since ghat = c_ij T_i T_j, can we use the identity for products of Chebyshev polys?

    # Approximate f by exp_p . ghat
    fhat = pa(exp, M, p) ∘ ghat
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

plot_(M)
plot_image_of_unit_grid(M, f;
    color="cyan", label="f")
plot_image_of_unit_grid(M, approximate_2d(f);
    color="magenta", label="f_hat", linestyle=:dot)
plot!([], label=false) # BODGE
