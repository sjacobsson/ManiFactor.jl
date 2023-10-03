include("Segre.jl")
using StatsBase: sample

# Verbose isapprox
import Base.isapprox
function isapprox(a, b, verbose; kwargs...)#={{{=#
    if verbose; println(a, " ?≈ ", b); end

    return isapprox(a, b; kwargs...)
end#=}}}=#

""" Testing that exp maps into the manifold. """
function test_exp(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    if verbose; println("M = ", M); end

    p = rand(M)
    if verbose; println("p = ", p); end

    v = rand(M; vector_at=p)
    if verbose; println("v = ", v); println(); end
    
    e = check_point(M, p)
    if !isnothing(e); throw(e); end
    e = check_vector(M, p, v)
    if !isnothing(e); throw(e); end
    e = check_point(M, exp(M, p, v))
    if !isnothing(e); throw(e); end
end#=}}}=#

""" Testing that geodesics are unit speed. """
function test_geodesic_speed(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end
        
    v = normalize(M, p, rand(M, vector_at=p))
    if verbose; println("v = ", v); end

    geodesic_speed = norm(
        finite_difference(
            t -> embed(M, exp(M, p, t * v)),
            rand(),
            1e-6
            )
        )
    @assert(isapprox(geodesic_speed, 1.0, verbose))
    println(); 
end#=}}}=#

""" Testing that geodesics only have normal curvature. """
function test_geodesic_curvature(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end
        
    v = normalize(M, p, rand(M, vector_at=p))
    if verbose; println("v = ", v); end
            
    gamma(t) = embed(M, exp(M, p, t * v))
    n = finite_difference(gamma, 0.0, 1e-3; order=2) # Acceleration vector at p
    v_ = embed_vector(M, p, rand(M, vector_at=p)) # Random Tangent vector at p

    @assert(isapprox(dot(n, v_), 0.0, verbose; atol=1e-6))
    println();
end#=}}}=#

""" Test that log is left and right inverse of exp. """
function test_log(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    if verbose; println("M = ", M); end
    
    p = rand(M)
    q = rand(M)
    if verbose; println("p = ", p); end
        
    v = normalize(M, p, rand(M, vector_at=p))
    if verbose; println("v = ", v); end
            
    @assert(isapprox(
        embed(M, q),
        embed(M, exp(M, p, log(M, p, q))),
        verbose
        ))
    @assert(isapprox(
        embed_vector(M, p, v),
        embed_vector(M, p, log(M, p, exp(M, p, v))),
        verbose
        ))
    println();
end#=}}}=#

""" Test that get_coordinates is left and right inverse of get_vector. """
function test_get_coordinates(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    if verbose; println("M = ", M); end
    
    p = rand(M)
    if verbose; println("p = ", p); end

    v = rand(M, vector_at=p)
    if verbose; println("v = ", v); end

    X = rand(manifold_dimension(M))
    if verbose; println("X = ", X); end

    B = DefaultOrthonormalBasis()
    @assert(isapprox(v, get_vector(M, p, get_coordinates(M, p, v, B), B), verbose))
    @assert(isapprox(X, get_coordinates(M, p, get_vector(M, p, X, B), B), verbose))
    println();
end#=}}}=#

""" Test sectional curvature. """
function test_curvature(#={{{=#
    M::AbstractSegre;
    verbose=false
    )
    if verbose; println("M = ", M); end

    p = rand(M)
    if verbose; println("p = ", p); end

    u = normalize(M, p, rand(M, vector_at=p))
    v = rand(M, vector_at=p)
    v = v - inner(M, p, u, v) * u
    v = normalize(M, p, v)

    if verbose; println("u = ", u); end
    if verbose; println("v = ", v); end

    r = 1e-3
    ps = [exp(M, p, r * (cos(theta) * u + sin(theta) * v)) for theta in 0.0:1e-5:(2 * pi)]
    # ds = [distance(M, p1, p2) for (p1, p2) in zip(ps, [ps[2:end]..., ps[1]])] # TODO: wth is wrong with distance??
    ds = [norm(embed(M, p1) - embed(M, p2)) for (p1, p2) in zip(ps, [ps[2:end]..., ps[1]])]
    C = sum(ds)
    K = 3 * (2 * pi * r - C) / (pi * r^3) # https://en.wikipedia.org/wiki/Bertrand%E2%80%93Diguet%E2%80%93Puiseux_theorem

    if verbose; println(K, " ?= ", sectional_curvature(M, p, u, v)); end
    @assert(isapprox(K, sectional_curvature(M, p, u, v), rtol=2e-3, atol=1e-4))
    println()
    # TODO: This would be soo much quicker if I wouldn't have to embed anything
end#=}}}=#

function main(;#={{{=#
    max_order=4,
    nbr_tests=10,
    dimension_range=range(2, 7), # check_point not implemented for 0-spheres, which seems sane
    kwargs...
    )

    println("Testing that exp maps to the manifold.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            V = Tuple([rand(dimension_range) for _ in 1:order])
            test_exp(Segre(V); kwargs...)
        end
    end
    
    println("Testing that geodesics are unit speed.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            V = Tuple([rand(dimension_range) for _ in 1:order])
            test_geodesic_speed(Segre(V); kwargs...)
        end
    end
    
    println("Testing that geodesics only have normal curvature.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            V = Tuple([rand(dimension_range) for _ in 1:order])
            test_geodesic_curvature(Segre(V); kwargs...)
        end
    end
    
    println("Testing that log is inverse of exp.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            V = Tuple([rand(dimension_range) for _ in 1:order])
            test_log(Segre(V); kwargs...)
        end
    end

    println("Testing that get_coordinates is inverse of get_vector.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            V = Tuple([rand(dimension_range) for _ in 1:order])
            test_get_coordinates(Segre(V); kwargs...)
        end
    end

    println("Testing that sectional curvature is correct.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            V = Tuple([rand(dimension_range) for _ in 1:order])
            test_curvature(Segre(V); kwargs...)
        end
    end
end#=}}}=#
