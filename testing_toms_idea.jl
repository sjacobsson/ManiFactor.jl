include("QOL.jl")
include("Segre.jl")
using QuadGK

# Random.seed!(6)
M = Segre((6, 6, 3, 5))
nbr_tests = 15

g(t) = 1.0
d_g(t) = 0.0


function test_tom()#={{{=#
    p = rand(M)
    q = rand(M)
    
    # R^+ x S^{n1 - 1} x ... x S^{nd - 1} -> R^n1 x ... x R^nd
    d = length(valence(M))
    p_ = p[1][1]^(1.0 / d) * p[2:end]
    q_ = q[1][1]^(1.0 / d) * q[2:end]
    
    # gamma = g(t) [(1 - t) a + t c] ⊗  [(1 - t) b + t d] interpolates between a⊗ b and c⊗ d
    function gamma(t)#={{{=#
        return g(t) * kronecker([(1 - t) * x + t * y for (x, y) in zip(p_, q_)]...)[:, 1]
    end#=}}}=#
    
    function d_gamma(t)#={{{=#
    
        # Product rule
        return (
            d_g(t) * kronecker([(1 - t) * x + t * y for (x, y) in zip(p_, q_)]...)[:, 1] +
            g(t) * sum([
            kronecker([
                i == j ?
                -x + y :
                (1 - t) * x + t * y
                for (j, (x, y)) in enumerate(zip(p_, q_))
                ]...)[:, 1]
            for (i, _) in enumerate(p_)
            ])
            )
    end#=}}}=#

     # Check that gamma goes from p to q
    for _ in 1:nbr_tests#={{{=#
        @assert(isapprox( gamma(0.0), embed(M, p)))
        @assert(isapprox( gamma(1.0), embed(M, q)))
    end#=}}}=#
     
     # Check that gamma stays on Seg
    for _ in 1:nbr_tests#={{{=#
        _, ss, _ = svd(gamma(rand()))
        for s in ss[2:end]
            @assert(isapprox(s, 0.0; atol=1e-6))
        end
    end#=}}}=#
     
     # Check that d_gamma is the derivative of gamma
    for _ in 1:nbr_tests#={{{=#
        t = rand()
        @assert(isapprox(
            d_gamma(t),
            finite_difference(gamma, t, 1e-6)
            ))
    end#=}}}=#
     
    # Compare the length of gamma with the length of a geodesic
    println("∫|γ'(t)|dt = ", quadgk(norm ∘ d_gamma, 0.0, 1.0)[1])
     
    gamma_(t) = embed(M, exp(M, p, t * log(M, p, q)))
    d_gamma_(t) = finite_difference(gamma_, t, 1e-6)
    # println("∫|d_gamma_(t)|dt   = ", quadgk(norm ∘ d_gamma_, 0.0, 1.0)[1]) # TODO: Why doesn't this work??

    println("d(p, q)    = ", distance(M, p, q))
end#=}}}=#
