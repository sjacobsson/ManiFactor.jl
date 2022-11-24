ENV["MPLBACKEND"] = "TkAgg" ;# Solves "Warning: No working GUI backend found for matplotlib"
using
    ManifoldsBase,
	Manifolds,
    Polynomials,
    Cubature,
    Optim,
    Plots
pyplot()

function pa(f,a...) # Partial application{{{
  (b...) -> f(a...,b...)
end#=}}}=#

function plot_S2(#={{{=#
    _::Manifolds.Sphere{2, ℝ},
    kwargs...
    )

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

function plotGrid(#={{{=#
    M::Manifolds.Sphere{2, ℝ},
    f::Function;
    labl=" "::String,
    colr="blue"::String,
    )
    plot_kwargs = Dict(
        :wireframe=>false,
        :geodesic_interpolation=>10,
        :linewidth=>2,
        :viewangle=>(110, 20)
        )

    xs = [x for x in 0.1:0.1:1.0]
    ys = [y for y in 0.1:0.1:1.0]
    plot!( M, [f(xs[1], ys[1]), f(xs[1], ys[2])]; color=colr, label=labl, plot_kwargs...)
    for i in range(1, length(xs) - 1)
        for j in range(1, length(ys) - 1)
            x = xs[i]
            x_ = xs[i + 1]
            y = ys[j]
            y_ = ys[j + 1]
            plot!( M, [f(x, y), f(x, y_)]; color=colr, label=false, plot_kwargs...)
            plot!( M, [f(x, y), f(x_, y)]; color=colr, label=false, plot_kwargs...)
        end
    end
end#=}}}=#

function add_S2(#={{{=#
    M::Manifolds.Sphere{2, ℝ},
    O::Vector{Float64},
    p::Vector{Float64},
    q::Vector{Float64}
    )
    check_point(M, O)
    check_point(M, p)
    check_point(M, q)

    vq = log(M, O, q); check_vector(M, O, vq)
    vq_ = parallel_transport_to(M, O, vq, p)

    return exp(M, p, vq_)
end#=}}}=#

function mult_scal_S2(#={{{=#
    M::Manifolds.Sphere{2, ℝ},
    O::Vector{Float64},
    c::Float64,
    p::Vector{Float64}
    )
    check_point(M, O)
    check_point(M, p)

    vp = log(M, O, p); check_vector(M, O, vp)

    return exp(M, O, c * vp)
end#=}}}=#

function distance( # Integrates the d(f1(x), f2(x)) on the unit square{{{
    M::Manifolds.Sphere{2, ℝ},
    f1::Function,
    f2::Function
    )
    # TODO: adjust reltol dynamically
    
    (I, ~) = hcubature(
        x -> ManifoldsBase.distance(M, f1(x[1], x[2]), f2(x[1], x[2])),
        [0.0, 0.0],
        [1.0, 1.0];
        reltol=1e-2,
        )

    return I
end#=}}}=#

# Fitting a grid
# let#={{{=#
#     M = Manifolds.Sphere(2)

#     plot_S2(M)
#     N = [0., 0, 1]::Vector{Float64}; check_point(M, N)
#     u = [0.1, 1, 0]::Vector{Float64}; check_vector(M, N, u)
#     v = [1., 0, 0]::Vector{Float64}; check_vector(M, N, v)

#     # S2 is given a linear-ish structure by add() and mult_scal():
#     add = pa(add_S2, M, N)
#     mult_scal = pa(mult_scal_S2, M, N)

#     function a_kl(#={{{=#
#         k::Int64,
#         l::Int64,
#         x::Float64,
#         y::Float64,
#         )

#         # TODO: normalize u and v?
#         return add(
#             mult_scal(sin(k * x), u),
#             mult_scal(sin(l * y), v)
#             )
#     end#=}}}=#

#     function b_kl(#={{{=#
#         k::Int64,
#         l::Int64,
#         x::Float64,
#         y::Float64,
#         )

#         # TODO: normalize u and v?
#         return add(
#             mult_scal(x^k, u),
#             mult_scal(y^l, v)
#             )
#     end#=}}}=#

#     function c_kl(#={{{=#
#         k::Int64,
#         l::Int64,
#         x::Float64,
#         y::Float64,
#         )

#         # TODO: normalize u and v?
#         return add(
#             mult_scal(ChebyshevT([i == k for i in range(1, k)])(x), u),
#             mult_scal(ChebyshevT([i == l for i in range(1, l)])(y), v)
#             )
#     end#=}}}=#

#     function f(#={{{=#
#         c::Array{Float64, 2},
#         e_kl::Function,
#         x::Float64,
#         y::Float64
#         )
#         # f is a "weighted sum" of basis functions e_kl

#         (d1, d2) = size(c)
#         s = N # Initiate s at the origin

#         for i = 1:d1
#             for j = 1:d2
#                 s = add(
#                     s,
#                     mult_scal(c[i, j], e_kl(i, j, x, y))
#                     )
#             end
#         end

#         return s
#     end#=}}}=#

#     function a(#={{{=#
#         x::Float64,
#         y::Float64
#         )

#         return add(
#             mult_scal(y, v),
#             mult_scal(x, u)
#             )
#         # return get_point(M, Manifolds.StereographicAtlas(), :south, [x, y])
#     end#=}}}=#

#     function b(#={{{=#
#         x::Float64,
#         y::Float64,
#         )

#         p1 = exp(M, N, x * v)
#         p2 = exp(M, p1, y * u)

#         return p2
#     end#=}}}=#

#     function obj_fun(c::Array{Float64, 2})::Float64
#         return distance(M, pa(f, c, a_kl), a)
#     end


#     c0 = fill(0.5, (2, 3))
#     result = optimize(obj_fun, c0, BFGS())
#     c_min = Optim.minimizer(result)

#     plotGrid(M, pa(f, c_min, a_kl), labl="f_min", colr="magenta")
#     plotGrid(M, a, labl="a", colr="cyan")
#     plot!(M, [N, exp(M, N, u)]; label="u", wireframe=false, geodesic_interpolation=10, linewidth=2)
#     plot!(M, [N, exp(M, N, v)]; label="v", wireframe=false, geodesic_interpolation=10, linewidth=2)
# end#=}}}=#

# Fitting a curve
let#={{{=#
    M = Manifolds.Sphere(2)

    plot_S2(M)
    N = [0., 0, 1]::Vector{Float64}; check_point(M, N)
    u = [0.1, 1, 0]::Vector{Float64}; check_vector(M, N, u)
    v = [1., 0, 0]::Vector{Float64}; check_vector(M, N, v)

    # S2 is given a linear-ish structure by add() and mult_scal():
    add = pa(add_S2, M, N)
    mult_scal = pa(mult_scal_S2, M, N)

    function a_kl(#={{{=#
        k::Int64,
        l::Int64,
        x::Float64,
        y::Float64,
        )

        # TODO: normalize u and v?
        return add(
            mult_scal(sin(k * x), u),
            mult_scal(sin(l * y), v)
            )
    end#=}}}=#

    function b_kl(#={{{=#
        k::Int64,
        l::Int64,
        x::Float64,
        y::Float64,
        )

        # TODO: normalize u and v?
        return add(
            mult_scal(x^k, u),
            mult_scal(y^l, v)
            )
    end#=}}}=#

    function c_kl(#={{{=#
        k::Int64,
        l::Int64,
        x::Float64,
        y::Float64,
        )

        # TODO: normalize u and v?
        return add(
            mult_scal(ChebyshevT([i == k for i in range(1, k)])(x), u),
            mult_scal(ChebyshevT([i == l for i in range(1, l)])(y), v)
            )
    end#=}}}=#

    function f(#={{{=#
        c::Array{Float64, 2},
        e_kl::Function,
        x::Float64,
        y::Float64
        )
        # f is a "weighted sum" of basis functions e_kl

        (d1, d2) = size(c)
        s = N # Initiate s at the origin

        for i = 1:d1
            for j = 1:d2
                s = add(
                    s,
                    mult_scal(c[i, j], e_kl(i, j, x, y))
                    )
            end
        end

        return s
    end#=}}}=#

    function a(#={{{=#
        x::Float64,
        y::Float64
        )

        return add(
            mult_scal(y, v),
            mult_scal(x, u)
            )
        # return get_point(M, Manifolds.StereographicAtlas(), :south, [x, y])
    end#=}}}=#

    function b(#={{{=#
        x::Float64,
        y::Float64,
        )

        p1 = exp(M, N, x * v)
        p2 = exp(M, p1, y * u)

        return p2
    end#=}}}=#

    function obj_fun(c::Array{Float64, 2})::Float64
        return distance(M, pa(f, c, a_kl), a)
    end


    c0 = fill(0.5, (2, 3))
    result = optimize(obj_fun, c0, BFGS())
    c_min = Optim.minimizer(result)

    plotGrid(M, pa(f, c_min, a_kl), labl="f_min", colr="magenta")
    plotGrid(M, a, labl="a", colr="cyan")
    plot!(M, [N, exp(M, N, u)]; label="u", wireframe=false, geodesic_interpolation=10, linewidth=2)
    plot!(M, [N, exp(M, N, v)]; label="v", wireframe=false, geodesic_interpolation=10, linewidth=2)
end#=}}}=#
