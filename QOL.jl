import LinearAlgebra:
    normalize

# Just some quality of life functions
#
# Partial application
function pa(f,a...)#={{{=#
  (b...) -> f(a...,b...)
end#=}}}=#

 # Normalize a vector in the tangent space
function normalize(#={{{=#
    M::AbstractManifold,
    p,
    v
    ) where {valence, F}

    return v / norm(M, p, v)
end#=}}}=#

# Approximate derivative of f at x
function finite_difference(#={{{=#
    f::Function, # :: ℝ -> some vector space
    x::Float64,
    h::Float64;
    order=1::Int64
    )

    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    if order == 1
        return (
            (1 / 12) *  f(x - 2 * h) +
            (-2 / 3) *  f(x - 1 * h) +
            (2 / 3) *   f(x + 1 * h) +
            (-1 / 12) * f(x + 2 * h)
            ) / h
    elseif order == 2
    return (
        (-1 / 12) *  f(x - 2 * h) +
        (4 / 3) *  f(x - 1 * h) +
        (-5 / 2) *   f(x) +
        (4 / 3) *  f(x + 1 * h) +
        (-1 / 12) * f(x + 2 * h)
        ) / h
    end
end#=}}}=#
