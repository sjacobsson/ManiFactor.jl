# Approximate functions between linear spaces

using TensorToolbox
using ApproxFun

Chebfun = Fun{Chebyshev{ChebyshevInterval{Float64}, Float64}, Float64, Vector{Float64}}

function columns(#={{{=#
    A::Matrix
    )::Vector{Vector}

    return [A[:, i] for i in 1:size(A, 2)]
end#=}}}=#

# Approximate a scalar-valued function using HOSVD and Chebyshev interpolating each term in each index
function approximate_scalar(#={{{=#
    g::Function, # :: [-1, 1]^m -> R
    m::Int64; # dimension of input space
    res::Int64=20 # nbr of interpolation points in each direction
    )::Function

    # Evaluate g on Chebyshev grid
    chebpts = chebyshevpoints(res)
    G = Array{Float64, m}(undef, (repeat([res], m)...))
    for (is, _) in pairs(G)
        G[is] = g([chebpts[i] for i in Tuple(is)])
    end

    # G_ijk = U1_ip U2_jq U3_kr C^pqr
    G_svd = hosvd(G)
    C = G_svd.cten # Core

    # TODO: do Tucker decomposition or some other nice decomposition where you don't need to compute G on the whole grid
    
    # G(x, y, z) = u1_p(x) u2_q(y) u3_r(z) C^pqr
    uss::Vector{Vector{Chebfun}} = [
        [Fun(Chebyshev(), transform(Chebyshev(), col)) for col in columns(factor_matrix)]
        for factor_matrix in G_svd.fmat]

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
        
        # Evaluate each u
        Us::Vector{Matrix{Float64}} = [
            reshape([u(t) for u in us], (1, rank))
            for (us, t, rank) in zip(uss, x, coresize(G_svd))]

        # Contract with the core tensor
        return full(ttensor(C, Us))[1]
    end

    return g_approx
end#=}}}=#

# TODO: Come up with a good name
# TODO: Explore different ways to approximate vector-valued functions
function approximate1(#={{{=#
    g::Function, # :: [-1, 1]^m -> R^n
    m::Int64; # dimension of input space
    res::Int64=20 # nbr of interpolation points in each direction
    )::Function

    n = length(g(zeros(m)))

    g_approx_::Vector{Function} = [approximate_scalar(x -> g(x)[i], m) for i in 1:n]
    g_approx::Function = x -> [h(x) for h in g_approx_]

    return g_approx
end#=}}}=#


# TODO: This method can prolly be removed
function cheb2(#={{{=#
    g::Function,# R^2 -> R^n
    )::Function# R^2 -> R^n

    grid_length = 10
    fdomain = TensorSpace(repeat([Chebyshev()], k)...)
    grid = Vector{Vector{Float64}}(points(fdomain, grid_length^k))

    gs = g.(grid)

    # Compute c_ij in f(x, y) = c_ij T_i(x) T_j(y)
    cs = [transform(fdomain, [y[i] for y in gs]) for i in 1:n]

    return v -> [Fun(fdomain, c)(v) for c in cs]
end#=}}}=#
