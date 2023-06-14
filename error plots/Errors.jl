# Compute errors from approximating different functions and compare to the error
# bound predicted by thm TODO.
# TODO: Move some of these methods into Tests.jl?
include("../Approximations.jl")
using Plots
using LinearAlgebra
using Random
using Printf

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

Lambda(N) = (2 / pi) * log(N + 1) + 1

# TODO: Use LowRankApprox...
function closest_rank_one_matrix(;#={{{=#
    m = 4,
    n1 = 40,
    n2 = 60,
    Ns=2:2:13,
    verbose=false,
    savefigure=false,
    figkwargs=(windowsize=(240, 160), guidefontsize=5, xtickfontsize=5, ytickfontsize=5, legendfontsize=5),
    kwargs...
    )

    n = 1 + (n1 - 1) + (n2 - 1)
    M = Segre((n1, n2))
    
    Random.seed!(420)
    a = LinearAlgebra.normalize(rand(n1))
    b = LinearAlgebra.normalize(rand(n2))
    As = [LinearAlgebra.normalize(rand(n1, n2)) for _ in 1:m]
    function f(x) # : [-1, 1]^m -> Segre((m1, m2))
        U, S, Vt = svd(
            sum([xi * A for (xi, A) in zip(x, As)]) + 
            2 * m * a * b'
            )
        return [[S[1]], U[:, 1], Vt[:, 1]]
    end

    # nth derivative of the svd along axis l #={{{=#
    # https://www.jstor.org/stable/2695472
    f1(l) = big(norm(M, f(zeros(m)), finite_difference(t -> f([i == l ? t : 0 for i in 1:m]), 0.0, 1e-5, order=1)))
    g1(l) = 1 / f1(l)
    f2(l) = big(norm(M, f(zeros(m)), finite_difference(t -> f([i == l ? t : 0 for i in 1:m]), 0.0, 1e-4, order=2)))
    g2(l) = -f2(l) / f1(l)^3
    f3(l) = big(norm(M, f(zeros(m)), finite_difference(t -> f([i == l ? t : 0 for i in 1:m]), 0.0, 1e-3, order=3)))
    g3(l) = 1 / f1(l)^5 * ( 3* f2(l)^2 - f1(l) * f3(l) )

    V1(l) = 2 * abs(f1(l))
    V2(l) = 2 * abs(f2(l))
    V3(l) = 2 * abs(f3(l))
    V4(l) = 2 * abs(1 / g1(l)^7 * (
        -15 * g2(l)^3 +
        10 * g1(l) * g2(l) * g3(l)
        ))
    V5(l) = 2 * abs(1 / g1(l)^9 * (
        105 * g2(l)^4 -
        105 * g1(l) * g2(l)^2 * g3(l) +
        10 * g1(l)^2 * g3(l)^2
        ))
    V6(l) = 2 * abs(1 / g1(l)^11 * (
        -945 * g2(l)^5 +
        1260 * g1(l) * g2(l)^3 * g3(l) -
        280 * g1(l)^2 * g2(l) * g3(l)^2
        ))
    V7(l) = 2 * abs(1 / g1(l)^13 * (
        10395 * g2(l)^6 -
        17325 * g1(l) * g2(l)^4 * g3(l) +
        6300 * g1(l)^2 * g2(l)^2 * g3(l)^2 -
        280 * g1(l)^3 * g3(l)^3
        ))
    V8(l) = 2 * abs(1 / g1(l)^15 * (
        -135135 * g2(l)^7 +
        270270 * g1(l) * g2(l)^5 * g3(l) -
        138600 * g1(l)^2 * g2(l)^3 * g3(l)^2 +
        15400 * g1(l)^3 * g2(l) * g3(l)^3
        ))
    V9(l) = 2 * abs( 1 / g1(l)^17 * (
        2027025 * g2(l)^8 -
        4729725 * g1(l) * g2(l)^6 * g3(l) +
        3153150 * g1(l)^2 * g2(l)^4 * g3(l)^2 -
        600600 * g1(l)^3 * g2(l)^2 * g3(l)^3 + 
        15400 * g1(l)^4 * g3(l)^4
        ))
    V10(l) = 2 * abs(1 / g1(l)^19 * (
        -34459425 * g2(l)^9 +
        91891800 * g1(l) * g2(l)^7 * g3(l) -
        75675600 * g1(l)^2 * g2(l)^5 * g3(l)^2 +
        21021000 * g1(l)^3 * g2(l)^3 * g3(l)^3 -
        1401400 * g1(l)^4 * g2(l) * g3(l)^4
        ))#=}}}=#

    V(nu) = maximum([[V1, V2, V3, V3, V4, V5, V6, V7, V8, V9, V10, repeat([t -> NaN], 100)...][nu](l) for l in 1:m])

    global fhat
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        # Compute bound on manifold error
        sigma = maximum([
            distance(M, f(zeros(m)), f(x))
            for x in [2 * rand(m) .- 1.0 for _ in 1:1000]])
        H = -1.0 / m
        epsilon = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
        bs[i] = epsilon + 2 / sqrt(abs(H)) * asinh(epsilon * sinh(sqrt(abs(H)) * sigma) / (2 * sigma))

        fhat = approximate(M, m, n, f; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            distance(M, f(x), fhat(x))
            for x in [2 * rand(m) .- 1.0 for _ in 1:1000]])

        if verbose
            println("error ", es[i])
            x = ones(m) - 2.0 * rand(m)
            print("evaluating f     ")
            @time(f(x))
            print("evaluating fhat  ")
            @time(fhat(x))
            println()
        end
    end

    # plot(Ns, bs; yaxis=:log, label="error bound", xlabel="N")
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        figkwargs...
        )
    scatter!(Ns, es;
        label="measured error",
        color=2,
        markersize=3,
        )
    if savefigure; savefig("closest_rank_one_matrix.pdf"); end
    display(p)
end#=}}}=#

function nullspace(;#={{{=#
    m = 4,
    n1 = 40,
    n2 = 60,
    Ns=2:2:14,
    verbose=false,
    savefigure=false,
    kwargs...
    )

    n = 1 + (n1 - 1) + (n2 - 1)
    M = Segre((n1, n2))
    
    Random.seed!(420)
    a = LinearAlgebra.normalize(rand(n1))
    b = LinearAlgebra.normalize(rand(n2))
    As = [LinearAlgebra.normalize(rand(n1, n2)) for _ in 1:m]
    function f(x) # : [-1, 1]^m -> Segre((m1, m2))
        U, S, Vt = svd(
            sum([xi * A for (xi, A) in zip(x, As)]) + 
            2 * m * a * b'
            )
        # return TODO: return element on the Grassmannian
    end

    # V(nu) = TODO

    global fhat
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        # Compute bound on manifold error
        sigma = maximum([
            distance(M, f(zeros(m)), f(x))
            for x in [2 * rand(m) .- 1.0 for _ in 1:100]])
        H = 0.0
        epsilon = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
        bs[i] = epsilon + 2 / sqrt(abs(H)) * asinh(epsilon * sinh(sqrt(abs(H)) * sigma) / (2 * sigma))

        fhat = approximate(M, m, n, f; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            distance(M, f(x), fhat(x))
            for x in [2 * rand(m) .- 1.0 for _ in 1:100]])

        if verbose
            println("error ", es[i])
            x = ones(m) - 2.0 * rand(m)
            print("evaluating f     ")
            @time(f(x))
            print("evaluating fhat  ")
            @time(fhat(x))
            println()
        end

    end

    # plot(Ns, bs; yaxis=:log, label="error bound", xlabel="N")
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        windowsize=(240, 160),
        guidefontsize=5, # TODO: How does this relate to the fontize in latex?
        xtickfontsize=5,
        ytickfontsize=5,
        legendfontsize=5,
        )
    scatter!(Ns, es;
        label="measured error",
        color=2,
        markersize=3,
        )
    if savefigure; savefig("closest_rank_one_matrix.pdf"); end
    display(p)
end#=}}}=#
