# Adaptive Cross Approximation of scalar valued functions using products of Chebyshev polynomials as a basis
import Optim: optimize, minimizer, Fminbox, Options
using ApproxFun

m = 3

function eval_(#={{{=#
    a::Approximation,
    x::Vector{Float64};
    )::Float64
    # TODO
end#=}}}=#

function cross3d_scalar(#={{{=#
    f::Function;# R^m -> R
    nbr_terms::Int64=100,
    nbr_interpolation_points::Int64=10
    )::Function# R^m -> R

    # TODO
    
end#=}}}=#

function cross3d(#={{{=#
    f::Function;# R^m -> R^n
    tol = 1e-2::Float64
    )::Function# R^m -> R^n
    
    # TODO
end#=}}}=#
