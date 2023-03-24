function stereographic_projection(#={{{=#
    xs::Vector{Float64};
    pole::Int64=1
    )::Vector{Float64}

    n = length(xs)
    @assert(pole <= n + 1)
    ys = zeros(n + 1) # Initialize

    for i in 1:(n + 1)
        if i < pole
            ys[i] = (2 * xs[i]) / (1 + norm(xs)^2)
        elseif i == pole
            ys[i] = (-1 + norm(xs)^2) / (1 + norm(xs)^2)
        elseif i > pole
            ys[i] = (2 * xs[i - 1]) / (1 + norm(xs)^2)
        end
    end
    
    return ys
end#=}}}=#

function inverse_stereographic_projection(#={{{=#
    ys::Vector{Float64};
    pole::Int64=1
    )::Vector{Float64}

    n = length(ys) - 1
    @assert(pole <= n + 1)
    xs = zeros(n) # Initialize

    for i in 1:n
        xs[i] = ys[i] / ys[n + 1]
    end
    
    return xs
end#=}}}=#

n = 2
M = Sphere(n)
m = 3
A = rand(n, m)
f(x) = stereographic_projection(A * x)


