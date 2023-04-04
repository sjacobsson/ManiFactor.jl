include("Approximations.jl")

using Combinatorics, Transducers
using LinearAlgebra

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
    ;verbose=false
    )

    d = 3
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
            ghat = approximate_scalar1(g, m)
    
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
    ;verbose=false
    )

    d = 3
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
                ghat = approximate1(g, m, n)
        
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

# TODO: approximate_scalar2 and approximate2
