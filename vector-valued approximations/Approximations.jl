# Approximate functions between linear spaces

include("../QOL.jl")
using ApproxFun
using TensorToolbox
using IterTools

Chebfun = Fun{Chebyshev{ChebyshevInterval{Float64}, Float64}, Float64, Vector{Float64}}
# TODO: Avoid using the full tensors in approximate_scalar1 and approximate_scalar2

# Approximate a scalar-valued function using HOSVD and Chebyshev interpolation
function approximate_scalar1(#={{{=#
    g::Function, # :: [-1, 1]^m -> R
    m::Int64;
    res::Int64=20 # nbr of interpolation points in each direction
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(x_i, y_j, z_k)
    G = Array{Float64, m}(undef, (repeat([res], m)...))
    chebpts = chebyshevpoints(res)
    Is = IterTools.product(repeat([1:res], m)...)
    for I in Is
        G[I...] = g(chebpts[collect(I)])
    end

    # G_ijk = U1_ip U2_jq U3_kr C^pqr
    G_decomposed::ttensor = hosvd(G)
    C::Array{Float64, m} = G_decomposed.cten # Core
    Us::Vector{Array{Float64, 2}} = G_decomposed.fmat # Factor matrices

    # g(x, y, z) = u1_p(x) u2_q(y) u3_r(z) C^pqr
    us::Vector{Array{Chebfun, 2}} = Vector{Array{Chebfun, 2}}(undef, m)
    for i in 1:m
        us[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
            Us[i];
            dims=1
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
        
        # Evaluate chebfuns and contract
        return only(full(ttensor(
            C, 
            [map(f -> f(t), u) for (u, t) in zip(us, x)]
            )))
    end

    return g_approx
end#=}}}=#

# Approximate a scalar-valued function using TT decomposition and Chebyshev interpolation
function approximate_scalar2(#={{{=#
    g::Function, # :: [-1, 1]^m -> R
    m::Int64;
    res::Int64=20 # nbr of interpolation points in each direction
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(x_i, y_j, z_k)
    G = Array{Float64, m}(undef, (repeat([res], m)...))
    chebpts = chebyshevpoints(res)
    Is = IterTools.product(repeat([1:res], m)...)
    for I in Is
        G[I...] = g(chebpts[collect(I)])
    end

    # G_ijk = C1^a_ib C2^b_jc C3^c_ka
    G_decomposed::TTtensor = TTsvd(G)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # g(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
    cs::Vector{Array{Chebfun, 3}} = Vector{Array{Chebfun, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
            Cs[i];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
        
        # Evaluate chebfuns and contract
        return only(full(TTtensor(
            [map(f -> f(t), c) for (c, t) in zip(cs, x)]
            )))
    end

    return g_approx
end#=}}}=#

# Approximate vector-valued function similarly to approximate_scalar1
function approximate1(#={{{=#
    g::Function, # :: [-1, 1]^m -> R^n
    m::Int64,
    n::Int64;
    res::Int64=20 # nbr of interpolation points in each direction
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk^l = g^l(x_i, y_j, z_k)
    G = Array{Float64, m + 1}(undef, (repeat([res], m)..., n))
    chebpts = chebyshevpoints(res)
    Is = IterTools.product(repeat([1:res], m)...)
    for I in Is
        G[I..., :] = g(chebpts[collect(I)])
    end

    # G_ijk^l = U1_ip U2_jq U3_kr U4^l_s C^pqrs
    G_decomposed::ttensor = hosvd(G)
    C::Array{Float64, m + 1} = G_decomposed.cten # Core
    Us::Vector{Array{Float64, 2}} = G_decomposed.fmat # Factor matrices

    # g^l(x, y, z) = u1_p(x) u2_q(y) u3_r(z) U4^l_s C^pqrs
    us::Vector{Array{Chebfun, 2}} = Vector{Array{Chebfun, 2}}(undef, m)
    for i in 1:m
        us[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
            Us[i];
            dims=1
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
        
        # Evaluate chebfuns and contract
        return vec(full(ttensor(
            C, 
            push!([map(f -> f(t), u) for (u, t) in zip(us, x)], Us[m + 1])
            )))
    end

    return g_approx
end#=}}}=#

# Approximate vector-valued function similarly to approximate_scalar2
function approximate2(#={{{=#
    g::Function, # :: [-1, 1]^m -> R^n
    m::Int64,
    n::Int64;
    res::Int64=20 # nbr of interpolation points in each direction
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk^l = g^l(x_i, y_j, z_k)
    G = Array{Float64, m + 1}(undef, (repeat([res], m)..., n))
    chebpts = chebyshevpoints(res)
    Is = IterTools.product(repeat([1:res], m)...)
    for I in Is
        G[I..., :] = g(chebpts[collect(I)])
    end

    # G_ijk^l = C1^a_ib C2^b_jc C3^c_kd C4^dl_a
    G_decomposed::TTtensor = TTsvd(G)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # g^l(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_d(z) C4^kl_a
    cs::Vector{Array{Chebfun, 3}} = Vector{Array{Chebfun, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
            Cs[i];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
        
        # Evaluate chebfuns and contract
        return vec(full(TTtensor(
            push!([map(f -> f(t), c) for (c, t) in zip(cs, x)], Cs[m + 1])
            )))
    end

    return g_approx
end#=}}}=#

# Naive way to generalize approximate_scalar to vector-valued functions.
# TODO: Remove this
function approximate_(#={{{=#
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
