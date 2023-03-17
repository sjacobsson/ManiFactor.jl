# Adaptive Cross Approximation of scalar valued functions using products of Chebyshev polynomials as a basis
import Optim: optimize, minimizer, Fminbox, Options
using ApproxFun
# All of these are pretty slow currently

Aca=Array{Float64, 3}

nbr_terms = 200
nbr_interpolation_points = 100
nbr_loops = 5


# Evaluate an acapproximation
function eval_aca(#={{{=#
    a::Aca,
    x::Vector{Float64};
    )::Float64

    # TODO: asserts?
    
    t = Array{Float64, 2}(undef, (m, nbr_interpolation_points))
    for j in 1:m
        # t[j, :] = [cos((k - 1) * acos(x[j])) for k in 1:nbr_interpolation_points]
        t[j, 1] = 1
        t[j, 2] = x[j]
        for k in 3:nbr_interpolation_points
            t[j, k] = 2 * x[j] * t[j, k - 1] - t[j, k - 2]
        end
    end

    b = Array{Float64, 2}(undef, (nbr_terms, m))
    for j in 1:m
        for i in 1:nbr_terms
            b[i, j] = t[j, :]' * a[i, j, :]
        end
    end
    # TODO: compute Chebyshev polynomials more efficiently?

    return sum(mapslices(prod, b, dims=[2]))
end#=}}}=#


# A normal ACA implementation
function aca_scalar(#={{{=#
    f::Function;# R^m -> R
    nbr_terms::Int64=nbr_terms,
    nbr_interpolation_points::Int64=nbr_interpolation_points
    )::Function# R^m -> R
    
    # TODO: [T_i(x_j)] is constant, so it would suffice to compute it once?

    # Initialize
    fhat::Aca = zeros(nbr_terms, m, nbr_interpolation_points)
    error::Float64 = Inf

    # Define a domain
    lower::Float64 = -1.0
    upper::Float64 = 1.0

    # while abs(error) > tol
    for i = 1:nbr_terms

        # Estimate argmax of abs ∘ f
        xi::Vector{Float64} = rand(m)
        for j in 1:nbr_loops*m
            j_ = (j - 1)%m + 1
            xi[j_] = minimizer(optimize(
                t -> -(f - pa(eval_aca, fhat))([xi[1:j_-1]..., t, xi[j_+1:end]...])^2,
                lower,
                upper
                ))
        end

        error = (f - pa(eval_aca, fhat))(xi)
        println(abs(error))

        for j in 1:m
            if j == 1
                g = t -> (f - pa(eval_aca, fhat))([xi[1:j-1]..., t, xi[j+1:end]...])
            else
                g = t -> (f - pa(eval_aca, fhat))([xi[1:j-1]..., t, xi[j+1:end]...]) / error
            end
            fhat[i, j, :] = transform(
                Chebyshev(),
                g.(points(Chebyshev(), nbr_interpolation_points))
                )
        end
    end

    return pa(eval_aca, fhat)
end#=}}}=#

# An ACA implementation that optimizes over the whole of R^m at once rather than one dimension at the time
function aca_scalar_(#={{{=#
    f::Function;# R^m -> R
    nbr_terms::Int64=nbr_terms,
    nbr_interpolation_points::Int64=nbr_interpolation_points
    )::Function# R^m -> R
    
    # TODO: [T_i(x_j)] is constant, so it would suffice to compute it once?

    # Initialize
    fhat::Aca = zeros(nbr_terms, m, nbr_interpolation_points)
    error::Float64 = Inf

    # Define a domain
    lower::Vector{Float64} = -1.0 * ones(m)
    upper::Vector{Float64} = ones(m)

    # while abs(error) > tol
    for i = 1:nbr_terms

        # Estimate argmax of abs ∘ f
        xi::Vector{Float64} = minimizer(optimize(
            x -> -((f - pa(eval_aca, fhat))(x))^2,
            lower,
            upper,
            2.0 * rand(m) .- 1.0,
            Fminbox(),
            Options(iterations=10)
            ))

        error = (f - pa(eval_aca, fhat))(xi)
        println(abs(error))

        for j in 1:m
            if j == 1
                g = t -> (f - pa(eval_aca, fhat))([xi[1:j-1]..., t, xi[j+1:end]...])
            else
                g = t -> (f - pa(eval_aca, fhat))([xi[1:j-1]..., t, xi[j+1:end]...]) / error
            end
            fhat[i, j, :] = transform(
                Chebyshev(),
                g.(points(Chebyshev(), nbr_interpolation_points))
                )
        end
    end

    return pa(eval_aca, fhat)
end#=}}}=#

# An ACA implementation that uses a variable number of interpolation points
function aca_scalar__(#={{{=#
    f::Function;# R^m -> R
    nbr_terms::Int64=nbr_terms
    )::Function# R^m -> R
    
    # Initialize
    fhat::Function = x -> 0.0
    error::Float64 = Inf

    # Define a domain
    lower::Float64 = -1.0
    upper::Float64 = 1.0

    # while abs(error) > tol
    for i = 1:nbr_terms

        # Estimate argmax of abs ∘ f
        xi::Vector{Float64} = rand(m)
        ei::Vector{Function} = Vector{Function}(undef, m)
        for j in 1:nbr_loops*m
            j_ = (j - 1)%m + 1

            # Approximate a fiber of f - fhat with a chebfun
            ei[j_] = Fun( t -> (f - fhat)([xi[1:j_-1]..., t, xi[j_+1:end]...]) )
            xi[j_] = argmax(ei[j_]^2) # argmax is efficient for chebfuns
        end

        error = (f - fhat)(xi)
        println(abs(error))

        # fhat = fhat + ⊗_j eij
        fhat_(x) = deepcopy(fhat(x) + prod([eij(t) for (eij, t) in zip(ei, x)]) * error^(1 - m))
        fhat = deepcopy(fhat_)
    end

    return fhat
end#=}}}=#

# A recursive ACA implementation
function aca_scalar___(#={{{=#
    f::Function;# R^m -> R
    nbr_terms::Int64=nbr_terms,
    nbr_interpolation_points::Int64=nbr_interpolation_points;
    order=m
    )::Function# R^m -> R
    
    if m = 2
        return aca_scalar(f, nbr_terms, nbr_interpolation_points)
    else if m > 2
        # TODO
    end

end#=}}}=#

function aca(#={{{=#
    f::Function;# R^m -> R^n
    tol = 1e-2::Float64
    )::Function# R^m -> R^n
    
    # TODO
end#=}}}=#
