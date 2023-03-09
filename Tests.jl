# Testing that exp maps into the manifold.
function test_exp(#={{{=#
    M::AbstractManifold;
    verbose=false
    )

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

# Testing that geodesics are unit speed
function test_geodesic_speed(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    
    p = rand(M)
    if verbose; println("p = ", p); end
        
    v = normalize(M, p, rand(M, vector_at=p))
    if verbose; println("v = ", v); println(); end

    geodesic_speed = norm(
        finite_difference(
            t -> embed(M, exp(M, p, t * v)),
            rand(),
            1e-6
            )
        )
    @assert(isapprox(geodesic_speed, 1.0))
end#=}}}=#

# Testing that geodesics only have normal curvature
function test_geodesic_curvature(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    
    p = rand(M)
    if verbose; println("p = ", p); end
        
    v = normalize(M, p, rand(M, vector_at=p))
    if verbose; println("v = ", v); println(); end
            
    gamma(t) = embed(M, exp(M, p, t * v))
    n = finite_difference(gamma, 0.0, 1e-6; order=2) # Normal curvature vector at p
    v_ = embed_vector(M, p, rand(M, vector_at=p)) # Random Tangent vector at p
    @assert(isapprox(dot(n, v_), 0.0, atol=1e-6))
end#=}}}=#

# Test that log is left and right inverse of exp
function test_log(#={{{=#
    M::AbstractManifold;
    verbose=false
    )
    
    p = rand(M)
    q = rand(M)
    if verbose; println("p = ", p); end
        
    v = normalize(M, p, rand(M, vector_at=p))
    if verbose; println("v = ", v); println(); end
            
    @assert(isapprox(v, log(M, p, exp(M, p, v))))
    @assert(isapprox(q, exp(M, p, log(M, p, q))))
end#=}}}=#

function main()#={{{=#
    max_order = 4
    nbr_tests = 20
    dimension_range = range(2, 7) # check_point not implemented for 0-spheres, which seems sane

    println("Testing that exp maps to the manifold.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            v = Tuple([rand(dimension_range) for _ in 1:order])
            test_exp(Segre(v))
        end
    end
    
    println("Testing that geodesics are unit speed.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            v = Tuple([rand(dimension_range) for _ in 1:order])
            test_geodesic_speed(Segre(v))
        end
    end
    
    println("Testing that geodesics only have normal curvature.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            v = Tuple([rand(dimension_range) for _ in 1:order])
            test_geodesic_curvature(Segre(v))
        end
    end
    
    println("Testing that log is inverse of exp.")
    for order in 1:max_order
        for _ in 1:nbr_tests
            v = Tuple([rand(dimension_range) for _ in 1:order])
            test_log(Segre(v))
        end
    end
end#=}}}=#
