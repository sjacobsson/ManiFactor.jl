using
    Manifolds,
    ManifoldsBase,
    ApproxFun,
    LinearAlgebra,
    Kronecker,
    Plots

# Explicitly import those functions that I'm extending
import
    Manifolds: check_point, check_vector, norm, embed, rand


# Seg(P^n1 x ... P^nd) ~ R+ x S^(n1 - 1) x ... x S^(nd - 1)
struct
    Segre{V, 𝔽} <: AbstractManifold{𝔽}
end

function Segre(#={{{=#
    valence::NTuple{D, Int};
    field::AbstractNumbers=ℝ
    ) where {D}

    return Segre{valence, field}()
end#=}}}=#


###### FUNCTIONS ON Seg ######

valence(::Segre{V, 𝔽}) where {V, 𝔽} = V
ndims(::Segre{V, 𝔽}) where {V, 𝔽} = length(V)

function check_point(#={{{=#
    M::Segre{valence, 𝔽},
    p::Vector{Vector{Float64}};
    kwargs...
    ) where {valence, 𝔽}

    @assert(length.(p) == [1, valence...])
    for (x, n) in zip(p[2:end], valence)
        # check_point does not raise a DomainErorr, but returns it...
        e = check_point(Sphere(n - 1), x; kwargs...)
        if !isnothing(e)
            return e
        end
    end
    
    return nothing
end#=}}}=#

function check_vector(#={{{=#
    M::Segre{valence, 𝔽},
    p::Vector{Vector{Float64}},
    v::Vector{Vector{Float64}};
    kwargs...
    ) where {valence, 𝔽}

    e = check_point(M, p, kwargs...)
    if !isnothing(e)
        return e
    end

    @assert(size.(v) == size.(p))
    for (x, xdot, n) in zip(p[2:end], v[2:end], valence)
        # check_vector(::AbstractSphere, ...) uses isapprox to compare to a dot
        # product to 0, which by default sets atol=0...
        e = check_vector(Sphere(n - 1), x, xdot; atol=1e-14, kwargs...)
        if !isnothing(e)
            return e
        end
    end
    
    return nothing
end#=}}}=#

function to_tucker(M::Segre)::Tucker#={{{=#
    # TODO
end#=}}}=#

function norm(#={{{=#
    M::Segre{valence, F},
    p::Vector{Vector{Float64}},
    v::Vector{Vector{Float64}}
    ) where {valence, F}

    return sqrt(
        (v[1][1])^2 +
        sum([(p[1][1])^2 * norm(Sphere(n), x, xdot)^2
            for (x, xdot, n) in zip(p[2:end], v[2:end], valence)])
        )
end#=}}}=#

function rand(#={{{=#
    M::Segre{valence, F};
    vector_at=nothing,
    )::Vector{Vector{Float64}} where {valence, F}

    if isnothing(vector_at)
        xs = [normalize(rand(n)) for n in valence]
        return [rand(1), xs...]
    else
        xdots = [normalize(rand(n)) for n in valence]
        xdots = map(t -> t[2] - dot(t[2], t[1]) * t[1], zip(vector_at[2:end], xdots))
        return [rand(1), xdots...]
    end
end#=}}}=#

# Embed p ∈ Segre((n1, ..., nd), F) in F^{n1 x ... x nd}
function embed(#={{{=#
    M::Segre{valence, F},
    p::Vector{Vector{Float64}}
    ) where {valence, F}

    return collect(kronecker(p...))[:, 1]
end#=}}}=#

# Theorem 1.1 in Swijsen21
function exp(# {{{
    M::Segre{valence, F},
    p::Vector{Vector{Float64}},
    v::Vector{Vector{Float64}}
    )::Vector{Vector{Float64}} where {valence, F}

    m = sqrt(sum([norm(Sphere(n), p_i, v_i)^2 for (n, p_i, v_i) in zip(valence, p[2:end], v[2:end])]))
    if m == 0.0
        q = p # Initialize
        q[1] = q[1] .+ v[1]
        return q
    end

    t = norm(M, p, v)
    P = v[1][1] / (p[1][1] * m)
    f = atan(sqrt(P^2 + 1.0) * t / p[1][1] + P) - atan(P)

    q = zeros.(size.(p)) # Initialize
    q[1][1] = sqrt(
        t^2 +
        2 * p[1][1] * P * t / sqrt(P^2 + 1.0) +
        p[1][1]^2# / (P^2 + 1.0) # TODO: This is wrong in Swijsen21 on arxiv
        )

    for i in range(2, ndims(M) + 1)
        if all(v[i] .== 0.0)
            q[i] = p[i]
        else
            n = valence[i - 1]
            S = Sphere(n)
            a = norm(S, p[i], v[i]) * f / m
            q[i] = p[i] * cos(a) .+ v[i] * sin(a) / norm(S, p[i], v[i])
        end
    end

    return q
end# }}}

# Theorem 6.2.1 in thesisLarsSwijsen
function log(# {{{
    M::Segre{valence, F},
    p::Vector{Vector{Float64}},
    q::Vector{Vector{Float64}}
    )::Vector{Vector{Float64}} where {valence, F}

    # Check for compatability
    rho_squared = 0.0 # Initialize
    for (x, y, n) in zip(p[2:end], q[2:end], valence)
        rho_squared = rho_squared + distance(Sphere(n), x, y)^2
    end
    @assert(rho_squared < pi^2)

    v = zeros.(size.(p)) # Initialize
    for i in range(2, ndims(M) + 1)
        if p[i] == q[i]
            v[i] = zeros(size(p[i]))
        else
            a = dot(p[i], q[i])
            v[i] = (q[i] .- a * p[i]) * acos(a) / sqrt(1.0 - a^2)
        end
    end

    m = sqrt(sum([norm(Sphere(n), p_i, v_i)^2 for (n, p_i, v_i) in zip(valence, p[2:end], v[2:end])]))
    if m == 0.0
        v[1][1] = q[1][1] - p[1][1]
        return v
    end

    v[1][1] = m * p[1][1] * (q[1][1] * cos(m) - p[1][1]) / (q[1][1] * sin(m))

    return v
end# }}}


###### TESTS ######

function finite_difference(#={{{=#
    f::Function, # :: ℝ -> some vector space
    x::Float64,
    h::Float64
    )

    return (
        (1 / 12) *  f(x - 2 * h) +
        (-2 / 3) *  f(x - 1 * h) +
        (2 / 3) *   f(x + 1 * h) +
        (-1 / 12) * f(x + 2 * h)
    ) / h
end#=}}}=#

function run_tests(;verbose=false)#={{{=#
    nbr_tests = 5
    dimension_range = range(2, 7) # check_point not implemented for 0-spheres, which seems sane

    # println("Testing that exp maps into the manifold.")
    # for order in 1:3#={{{=#
    #     for _ in 1:nbr_tests
    #         valence = Tuple([rand(dimension_range) for _ in 1:order])
    #         M = Segre(valence)
    #         if verbose
    #             println("valence ", valence)
    #         end
        
    #         p = rand(M)
    #         if verbose
    #             println("p = ", map(y -> map((x -> round(x; digits=2)), y), p))
    #         end
        
    #         v = rand(M; vector_at=p)
    #         if verbose
    #             println("v = ", map(y -> map((x -> round(x; digits=2)), y), v))
    #             println()
    #         end
            
    #         e = check_point(M, p)
    #         if !isnothing(e); throw(e); end
    #         e = check_vector(M, p, v)
    #         if !isnothing(e); throw(e); end
    #         e = check_point(M, exp(M, p, v))
    #         if !isnothing(e); throw(e); end
    #     end
    # end#=}}}=#

    # println("Testing that geodesics are unit speed.")
    # for order in 1:3#={{{=#
    #     for _ in 1:nbr_tests
    #         valence = Tuple([rand(dimension_range) for _ in 1:order])
    #         M = Segre(valence)
    #         if verbose
    #             println("valence ", valence)
    #         end
    
    #         p = rand(M)
    #         if verbose
    #             println("p = ", map(y -> map((x -> round(x; digits=2)), y), p))
    #         end
        
    #         v = rand(M, vector_at=p)
    #         v = v / norm(M, p, v)
    #         if verbose
    #             println("v = ", map(y -> map((x -> round(x; digits=2)), y), v))
    #             println()
    #         end

    #         geodesic_speed = norm(
    #             finite_difference(
    #                 t -> embed(M, exp(M, p, t * v)),
    #                 rand(),
    #                 1e-6
    #                 )
    #             )
    #         @assert(isapprox(geodesic_speed, 1.0))
    #     end
    # end#=}}}=#


    println("Testing that log is a left inverse of exp.")
    for order in 2:2#={{{=# # TODO: fix the order
        for _ in 1:nbr_tests
            valence = Tuple([rand(dimension_range) for _ in 1:order])
            M = Segre(valence)
            if verbose
                println("valence ", valence)
            end
    
            p = rand(M)
            if verbose
                println("p = ", map(y -> map((x -> round(x; digits=2)), y), p))
            end
        
            v = rand(M, vector_at=p)
            v = v / norm(M, p, v)
            if verbose
                println("v = ", map(y -> map((x -> round(x; digits=2)), y), v))
                println()
            end

            println(
                "v - log_p (exp_p v) = ",
                map(y -> map((x -> round(x; digits=2)), y),
                    v - log(M, p, exp(M, p, v)))
                    )
        end
    end#=}}}=#
end#=}}}=#

M = Segre((3, 4))
p = [[1.3], [1.0;                   0.0;                0.0], [0.0; 1.0; 0.0; 0.0]]
v = [[0.15], [0.0; 0.2; -0.44], [0.13; 0.0; -0.1; 0.4]]
q = [[1.2], [0.7071067811865475,    0.7071067811865475, 0.0], [0.0; 1.0; 0.0; 0.0]]

# # def d_exp(p, v):
# #     return jacfwd(lambda x: exp(p, x))

# def metric(p):# {{{
#     return np.linalg.inv(d_embed_point(p)) @ np.diag(1., p[0] * np.ones(p.shape)) @ d_embed_point(p)# }}}

# # Christoffel symbol{{{
# def christoffel(p):
#     # https://en.wikipedia.org/wiki/Christoffel_symbols#General_definition
#     G = np.array([d_metric(p, e[i]) for i in range(0, 5)])
#     return (1. / 2.) * (-G + np.swapaxes(G, 0, 1) + np.swapaxes(G, 0, 2))# }}}

# def d_christoffel(p, v):# {{{
#     h = 1e-6
#     return (christoffel(p + h * v) / (2. * h) -\
#         christoffel(p + -h * v) / (2. * h))# }}}

# # Riemann curvature tensor
# def riemann(p):# {{{
#     dG = np.array([d_christoffel(p, e[i]) for i in range(0, 5)])
#     G = christoffel(p)
#     g = metric(p)

#     R = np.moveaxis(dG, [0, 1, 2, 3], [2, 0, 3, 1]) -\
#         np.moveaxis(dG, [0, 1, 2, 3], [3, 0, 2, 1]) +\
#         np.moveaxis(np.einsum('ija,ab,bkl', G, g, G), [0, 1, 2, 3], [0, 2, 3, 1]) -\
#         np.moveaxis(np.einsum('ija,ab,bkl', G, g, G), [0, 1, 2, 3], [0, 3, 2, 1])

#     return R# }}}

# def sectional_curvature(p, v, u):# {{{
#     g = metric(p)
#     # https://en.wikipedia.org/wiki/Sectional_curvature#Definition
#     K = np.einsum('ijka,i,j,k,ab,b', riemann(p0), u, v, v, g, u) / (np.einsum('a,ab,b', u, g, u) * np.einsum('a,ab,b', v, g, v) - np.einsum('a,ab,b', u, g, v)^2)
#     return K# }}}


# ### Is there a lower bound for the sectional curvature? ###
# from scipy.optimize import minimize

# def obj_fun(x):# {{{
#     K = sectional_curvature(p0, x[0:5] / norm(x[0:5]), x[5:10] / norm(x[5:10]))
#     print(K)
#     return K# }}}

# ### ###
# e0 = np.array([1., 0., 0., 0., 0.])
# e1 = np.array([0., 1., 0., 0., 0.])
# e2 = np.array([0., 0., 1., 0., 0.])
# e3 = np.array([0., 0., 0., 1., 0.])
# e4 = np.array([0., 0., 0., 0., 1.])
# es = [e0, e1, e2, e3, e4]

# p0 = np.array([
#     1.4,
#     3., 6.,
#     2., 7.
#     ])

# v0 = np.array([
#     -.3,
#     .2, .6,
#     .2, -.1
#     ])

# print(exp(p0, v0))
# # print(d_exp(p0, 1e-12 * v0))
# # print(v0.T @ d_exp(p0, 1e-6 * v0) @ e2)
# # print(sectional_curvature(p0, v0, e2))

# # plt.spy(metric(p0), precision=1e-10)
# # plt.show()

# ### TEST CORRECTNESS OF FUNCTIONS ###

# def finite_difference(f, p, v):# {{{
#     h = 1e-6
#     return (f(p + h * v) / (2. * h) - f(p - h * v) / (2. * h))# }}}

# def test_d_embed_point(p, v):# {{{
#     return np.allclose(
#         d_embed_point(p0) @ v,
#         finite_difference(embed_point, p0, v),
#         rtol=1e-3
#         )# }}}

# def test_embed_tangent(p, v):# {{{
#     n = len(p)
#     return np.allclose(
#         np.dot(embed_point(p), embed_tangent(p, v)),
#         np.zeros(n),
#         rtol=1e-3
#         ) & np.allclose(
#         norm(embed_tangent(p, v)),
#         norm(v),
#         rtol=1e-3
#         )# }}}

# def test_unembed_point(p, v):# {{{
#     return np.allclose(
#         unembed_point(embed_point(p)),
#         p,
#         rtol=1e-3
#         )# }}}

# # for i in range(0, 10):# {{{
# #     print(test_d_embed_point(
# #         np.random.rand(5),
# #         np.random.rand(5)
# #         ))
# #     print(test_embed_tangent(
# #         np.random.rand(5),
# #         np.random.rand(5)
# #         ))
# #     print(test_unembed_point(
# #         np.random.rand(5),
# #         np.random.rand(5)
# #         ))# }}}

# function splitat(
#     p::Vector{::T},
#     is::Vector{Int65}
#     )::Vector{Vector{::T}}
#     p_tmp = p

#     p_split = [p[1]]
#     for i in is
#         p_split = [p[i]]
#         p_tmp = p[
#     end
#     return p_split
# end
