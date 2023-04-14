include("Approximations.jl")

using Combinatorics, Transducers
using LinearAlgebra

### Example functions [-1, 1]^d -> R from Chertkov, Ryzhakov, and Oseledets' 2022 paper https://arxiv.org/pdf/2208.03380.pdf

function fackley(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 32.768 # rescale so that x \in [-1, 1]^d

    A = 20.0
    B = 0.2
    C = 2 * pi

    return (
        -A * exp(-B * sqrt((1.0 / d) * sum([x^2 for x in xs])))
        - exp((1.0 / d) * sum([cos(C *x) for x in xs]))
        + A + exp(1)
        )
end#=}}}=#

function falpine(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 10.0 # rescale so that x \in [-1, 1]^d

    return sum([abs(x * sin(x) + 0.1 * x) for x in xs])
end#=}}}=#

function fdixon(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 10.0 # rescale so that x \in [-1, 1]^d

    return (x[1] - 1)^2 + sum([i * (2 * x[i]^2 - x[i - 1])^2 for i in 2:d])
end#=}}}=#

function fexponential(xs::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs)

    return -exp(-(1 / 2.0) * sum([x^2 for x in xs]))
end#=}}}=#

function fgrienwank(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 600.0

    return (
        sum([x^2 / 4000.0 for x in xs])
        - prod([cos(x[i] / sqrt(i)) for i in 1:d])
        + 1
        )
end#=}}}=#

function fmichalewicz(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = (xs_ .+ 1) * pi / 2

    m = 10

    return -sum([sin(xs[i]) * sin(i * xs[i]^2 / pi)^(2 * m) for i in 1:d])
end#=}}}=#

function fpiston(xs::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs)
    assert(d == 7)
    M   = (60.0 - 30.0) * xs[1] / 2     + (60.0 + 30.0) / 2
    S   = (0.020 - 0.005) * xs[2] / 2   + (0.020 + 0.005) / 2
    V0  = (0.010 - 0.005) * xs[3] / 2   + (0.010 + 0.005) / 2
    k   = (5e3 - 1e3) * xs[4] / 2       + (5e3 + 1e3) / 2
    P0  = (11e4 - 9e4) * xs[5] / 2      + (11e4 + 9e4) / 2
    Ta  = (296.0 - 290.0) * xs[6] / 2   + (296.0 + 290.0) / 2
    T0  = (360.0 - 340.0) * xs[7] / 2   + (360.0 + 340.0) / 2

    m = 10

    # return TODO
end#=}}}=#





# TODO: Use above defined functions
# TODO: Compute required rank and number of interpolation points

""" Basis for space of degree d polynomials of m variables.  """
function polynomial_basis(m, d; with_bias=true)#={{{=#
    _m = with_bias ? m+1 : m
    exponents = multiexponents(_m, d)
    return function (x)
        _x = with_bias ? vcat(x, one(eltype(x))) : x
        exponents |> Map(exponent -> prod(_x.^exponent)) |> collect
    end
end#=}}}=#

""" Testing that approximate_scalar1 is a good fit """
function test_approximate_scalar1(#={{{=#
    ;verbose=false,
    kwargs...
    )

    d = 7
    nbr_loops = 3
    for m in 1:4
        if verbose
            println()
            println("m = ", m)
        end
        for _ in 1:nbr_loops
            cs = rand(binomial(m + d, d))

            # Let g be a random d:th degree polynomial
            function g(x::Vector{Float64})::Float64
                return dot(cs, polynomial_basis(m, d)(x))
            end
            ghat = approximate_scalar1(g, m; kwargs...)
    
            x = rand(m)
            g_ = g(x)
            ghat_ = ghat(x)
            error = abs(g_ - ghat_)

            if (error / abs(g_) > 1e-10);
                throw("approximate_scalar1 not accurate enough");
            end
            if verbose
                println("g(x)  = ", round(g_; sigdigits=2))
                println("e_rel = ", round(error / abs(g_); sigdigits=2))
                println()
            end
        end
    end
end#=}}}=#

""" Testing that approximate1 is a good fit """
function test_approximate1(#={{{=#
    ;verbose=false,
    kwargs...
    )

    d = 7
    nbr_loops = 1
    for m in 1:4
        if verbose
            println()
            println("m = ", m)
        end
        for n in 1:5
            for _ in 1:nbr_loops
                cs = [rand(binomial(m + d, d)) for _ in 1:n]
    
                # Let g be a random d:th degree polynomial
                function g(x::Vector{Float64})::Vector{Float64}
                    return [dot(cs[i], polynomial_basis(m, d)(x)) for i in 1:n]
                end
                ghat = approximate1(g, m, n; kwargs...)
        
                x = rand(m)
                g_ = g(x)
                ghat_ = ghat(x)
                error = norm(g_ - ghat_)
    
                if (error / norm(g_) > 1e-10);
                    throw("approximate1 not accurate enough");
                end
                if verbose
                    println("g(x)  = ", [round(gi; sigdigits=2) for gi in g_])
                    println("e_rel = ", round(error / norm(g_), sigdigits=2))
                    println()
                end
            end
        end
    end
end#=}}}=#

""" Testing that approximate_scalar3 is a good fit """
function test_approximate_scalar1(#={{{=#
    ;verbose=false,
    kwargs...
    )

    d = 7
    nbr_loops = 3
    for m in 1:4
        if verbose
            println()
            println("m = ", m)
        end
        for _ in 1:nbr_loops
            cs = rand(binomial(m + d, d))

            # Let g be a random d:th degree polynomial
            function g(x::Vector{Float64})::Float64
                return dot(cs, polynomial_basis(m, d)(x))
            end
            ghat = approximate_scalar1(g, m; kwargs...)
    
            x = rand(m)
            g_ = g(x)
            ghat_ = ghat(x)
            error = abs(g_ - ghat_)

            if (error / abs(g_) > 1e-10);
                throw("approximate_scalar1 not accurate enough");
            end
            if verbose
                println("g(x)  = ", round(g_; sigdigits=2))
                println("e_rel = ", round(error / abs(g_); sigdigits=2))
                println()
            end
        end
    end
end#=}}}=#

# TODO: approximate_scalar2 and approximate2
