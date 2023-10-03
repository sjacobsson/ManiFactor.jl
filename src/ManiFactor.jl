module ManiFactor

using ManifoldsBase
using Manifolds
using ApproximatingMapsBetweenLinearSpaces
using Kronecker
using SplitApplyCombine #TODO: is this needed?
include("segre/Segre.jl")
# include("../approximating vector-valued maps/Approximations.jl") # TODO: Make Approximations into a package

export
    approximate

"""
    function approximate(
        m::Int,
        M::AbstractManifold,
        f::Function; # :: [-1, 1]^m -> M^n
        base_approximate=approximate_vector::Function, # :: (m::Int) -> (n::Int) -> ([-1, 1]^m -> R^n) -> ([-1, 1]^m -> R^n)
        kwargs...
        )::Function # :: [-1, 1]^m -> M^n

Approximate a manifold-valued function using Riemann normal coordinate chart.
"""
function approximate(#={{{=#
    m::Int,
    M::AbstractManifold,
    f::Function; # :: [-1, 1]^m -> M^n
    base_approximate=approximate_vector::Function, # :: (m::Int) -> (n::Int) -> ([-1, 1]^m -> R^n) -> ([-1, 1]^m -> R^n)
    # TODO: provice chart as an option?
    kwargs...
    )::Function # :: [-1, 1]^m -> M^n

    # Evaluate f on a point cloud in [-1, 1]^m
    xs = [2.0 * rand(m) .- 1.0 for _ in 1:100]
    fs = f.(xs)
    
    # Choose a point on M and linearize from there
    p = mean(M, fs)
    B = DefaultOrthonormalBasis()
    chart = (X -> get_coordinates(M, p, X, B)) ∘ (q -> log(M, p, q)) # log :: M -> T_p M, get_coordinates : T_p M -> R^n
    chart_inv = (X -> exp(M, p, X)) ∘ (X -> get_vector(M, p, X, B)) # get_vector :: R^n -> T_p M, exp : T_p M -> M^n

    g = chart ∘ f
    ghat = base_approximate(m, manifold_dimension(M), g; kwargs...)
    fhat = chart_inv ∘ ghat
    return fhat
end#=}}}=#

end