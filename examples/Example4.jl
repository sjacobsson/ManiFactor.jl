# TODO: Probably I will have to implement exponential and logarithmic maps etc for the Grassmannian myself
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
        Lambda(N) = (2 / pi) * log(N + 1) + 1
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

