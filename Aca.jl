# Adaptive Cross Approximation of scalar valued functions using products of Chebyshev polynomials as a basis
import Optim: optimize, minimizer, Fminbox, Options
using ApproxFun

function aca_1d(#={{{=#
    f;# R^m -> R
    tol = 1e-2::Float64
    )# R^m -> R
    # The implementation is similar to Townsend's and Trefethen's cheb2paper fig. 2.1
    
    # TODO: Use @time to optimize

    # Initialize
    fhat = x -> 0.0
    error = Inf

    # Define a domain
    lower = -1.0 * ones(m)
    upper = ones(m)

    k = 1
    while abs(error) > tol

        # Estimate argmax of abs ∘ f
        x_k = minimizer(optimize(
            x -> -abs(f(x) - fhat(x)),
            lower,
            upper,
            2.0 * rand(m) .- 1.0,
            Fminbox(),
            Options(iterations=100)
            ))

        error = f(x_k) - fhat(x_k)
        println(abs(error))

        factors = [Fun(t -> f([x_k[1:j-1]..., t, x_k[j+1:end]...]) - fhat([x_k[1:j-1]..., t, x_k[j+1:end]...])) for j in 1:m]

        fhat_(x) = deepcopy(fhat(x) + prod([a(t) for (a, t) in zip(factors, x)]) * error^(1 - m))
        fhat = deepcopy(fhat_)
        k = k + 1
    end

    return fhat
end#=}}}=#

# TODO: optimize aca_1d before starting with this one!
function aca_1d_(#={{{=#
    f;# R^m -> R
    tol = 1e-2::Float64
    )# R^m -> R
    # The implementation is similar to what Raf told me about doing one variable at the time
    
    # Initialize
    fhat = x -> 0.0
    error = Inf

    # Define a domain
    lower = -1.0 * ones(m)
    upper = ones(m)

    k = 1
    while abs(error) > tol

        # Estimate argmax of abs ∘ f
        x_k = minimizer(optimize(
            x -> -abs(f(x) - fhat(x)),
            lower,
            upper,
            2.0 * rand(m) .- 1.0,
            Fminbox(),
            Optim.Options(iterations=100)
            ))

        # error = 0.0
        # x_max = rank(m)
        # for _ in 1:100
        #     x = rand(m)
        #     if abs(e(x)) > abs(error)
        #         x_max = x
        #         error = e(x_max)
        #     end
        # end

        # error = 0.0
        # x_max = rank(m)
        # for _ in 1:10
        #     x = rand(m)
            
        #     for _ in 1:10
        #     for k in 1:m
        #         x_k = minimizer(optimize(
        #             t -> -abs(e([x[1:k-1]..., t, x[k+1:end]...])),
        #             -1.0,
        #             1.0
        #             ))
    
        #         x[k] = x_k
        #     end
        #     end

        #     x_max = x
        #     error = e(x_max)
        # end

        error = f(x_k) - fhat(x_k)
        println(abs(error))

        factors = [Fun(t -> f([x_k[1:j-1]..., t, x_k[j+1:end]...]) - fhat([x_k[1:j-1]..., t, x_k[j+1:end]...])) for j in 1:m]

        fhat_(x) = deepcopy(fhat(x) + prod([a(t) for (a, t) in zip(factors, x)]) * error^(1 - m))
        fhat = deepcopy(fhat_)
        k = k + 1
    end

    return fhat
end#=}}}=#

function aca(#={{{=#
    f;# R^m -> R^n
    tol = 1e-2::Float64
    )# R^m -> R^n
    
    # TODO
end#=}}}=#
