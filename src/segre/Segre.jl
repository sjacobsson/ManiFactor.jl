import Manifolds:
    check_point,
    check_vector,
    manifold_dimension,
    exp,
    exp!,
    log,
    log!,
    get_coordinates,
    get_vector,
    norm,
    distance,
    embed,
    rand
using ManifoldsBase
using Manifolds
using LinearAlgebra
using Base.Iterators
using Kronecker

export AbstractSegre
export Segre

"""
    Seg(P^n1 x ... P^nd) ~ R^+ x S^(n1 - 1) x ... x S^(nd - 1)

is the space of rank-one tensors.
"""
abstract type AbstractSegre{𝔽} <: AbstractManifold{𝔽} end
struct Segre{V, 𝔽} <: AbstractSegre{𝔽} end

function Segre(#={{{=#
    valence::NTuple{D, Int};
    field::AbstractNumbers=ℝ
    ) where {D}

    return Segre{valence, field}()
end#=}}}=#

valence(::Segre{V, 𝔽}) where {V, 𝔽} = V
ndims(::Segre{V, 𝔽}) where {V, 𝔽} = length(V)

# Overwrite of check_point for the sphere that also checks that the input has
# the right length.
function check_point_(#={{{=#
    M::AbstractSphere,
    p;
    kwargs...
    )

    if length(p) != manifold_dimension(M) + 1
        return DomainError(
            length(p),
            "$(p) has wrong dimension."
            )
    end

    if !isapprox(norm(p), 1.0; kwargs...)
        return DomainError(
            norm(p),
            "The point $(p) does not lie on the $(M) since its norm is not 1.",
            )
    end

    return nothing
end#=}}}=#

"""
    check_point(M::Segre{V, F}, p; kwargs...)

Check whether `p` is a valid point on `M`, i.e. p[1] is a singleton containing a
positive number and p[i + 1] is a point on Sphere(V[i]). The tolerance can be
set using the `kwargs...`.
"""
function check_point(#={{{=#
    M::Segre{valence, 𝔽},
    p;
    kwargs...
    ) where {valence, 𝔽}

    # @assert(length.(p) == [1, valence...])
    if length.(p) != [1, valence...]
        return DomainError(
            length.(p),
            "$(p) has wrong dimensions."
            )
    end

    # @assert(p[1][1] > 0.0)
    if p[1][1] <= 0.0
        return DomainError(
            p[1][1],
            "$(p) has non-positive modulus."
            )
    end

    for (x, n) in zip(p[2:end], valence)
        # check_point does not raise a DomainError, but returns it...
        e = check_point_(Sphere(n - 1)::AbstractSphere{𝔽}, x; kwargs...)
        if !isnothing(e); return e; end
    end
    
    return nothing
end#=}}}=#

# Overwrite of check_vector for the sphere that also checks that the input has
# the right length.
function check_vector_(#={{{=#
    M::AbstractSphere,
    p,
    X;
    kwargs...
    )

    if length(X) != manifold_dimension(M) + 1
        return DomainError(
            length(X),
            "$(X) has wrong dimension.",
            )
    end

    if !isapprox(abs(real(LinearAlgebra.dot(p, X))), 0.0; kwargs...)
        return DomainError(
            abs(LinearAlgebra.dot(p, X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal in the embedding.",
            )
    end

    return nothing
end#=}}}=#

"""
    function check_vector(
        M::Segre{valence, 𝔽},
        p,
        v,
        kwargs...
        )

Check whether `v` is a tangent vector to `p` on `M`, i.e. after `check_point`(M, p)`, `v` has to be of same dimension as `p` and orthogonal to `p`. The tolerance can be set using the `kwargs...`.
"""
function check_vector(#={{{=#
    M::Segre{valence, 𝔽},
    p,
    v,
    kwargs...
    ) where {valence, 𝔽}

    e = check_point(M, p, kwargs...)
    if !isnothing(e); return e; end

    @assert(size.(v) == size.(p))
    for (x, xdot, n) in zip(p[2:end], v[2:end], valence)
        # check_vector(::AbstractSphere, ...) uses isapprox to compare the dot product to 0, which by default sets atol=0...
        e = check_vector_(Sphere(n - 1)::AbstractSphere{𝔽}, x, xdot; atol=1e-14, kwargs...)
        if !isnothing(e); return e; end
    end
    
    return nothing
end#=}}}=#

"""
    function manifold_dimension(
        M::Segre{valence, 𝔽}
        )
"""
function manifold_dimension(#={{{=#
    M::Segre{valence, 𝔽}
    ) where {valence, 𝔽}

    return 1 + sum([d - 1 for d in valence])
end#=}}}=#

"""
    function get_coordinates(
        M::Segre{valence, ℝ},
        p,
        v,
        B::DefaultOrthonormalBasis
        )
"""
function get_coordinates(#={{{=#
    M::Segre{valence, ℝ},
    p,
    v,
    B::DefaultOrthonormalBasis
    ) where {valence}

    @assert(check_point(M, p) == nothing)
    @assert(check_vector(M, p, v) == nothing)

    coords = eltype(p)[[] for _ in p] # Initialize
    coords[1] = v[1]
    for (i, n) in enumerate(valence)
        coords[i + 1] = get_coordinates(Sphere(n - 1), p[i + 1], v[i + 1], B)
    end

    return vcat(coords...)
end#=}}}=#

"""
    function get_vector(
        M::Segre{valence, ℝ},
        p,
        X,
        B::DefaultOrthonormalBasis
        )
"""
function get_vector(#={{{=#
    M::Segre{valence, ℝ},
    p,
    X,
    B::DefaultOrthonormalBasis
    ) where {valence}

    @assert(check_point(M, p) == nothing)
    X_ = deepcopy(X)

    v = eltype(p)[[] for _ in p] # Initialize
    v[1] = [X_[1]]
    X_ = drop(X_, 1)
    for (i, d) in enumerate(valence)
        v[i + 1] = get_vector(Sphere(d - 1), p[i + 1], take(X_, d - 1), B)
        X_ = drop(X_, d - 1)
    end

    @assert(length(X_) == 0)
    check_vector(M, p, v)

    return v
end#=}}}=#

"""
    function inner(
        M::Segre{valence, 𝔽},
        p,
        u,
        v,
        )

Inner product between two tangent vectors `u` and `v` at `p`.
"""
function inner(#={{{=#
    M::Segre{valence, 𝔽},
    p,
    u,
    v,
    ) where {valence, 𝔽}

    return u[1][1] * v[1][1] + p[1][1]^2 * (dot(u[2:end], v[2:end]))
end#=}}}=#

"""
    function normalize(
        M::AbstractManifold,
        p,
        v
        )

Normalize a tangent vector `v` at `p`.
"""
function normalize(#={{{=#
    M::AbstractManifold,
    p,
    v
    )

    return v / norm(M, p, v)
end#=}}}=#

"""
    function norm(
        M::Segre{valence, ℝ},
        p,
        v
        )

Norm of tangent vector `v` at `p`.
"""
function norm(#={{{=#
    M::Segre{valence, ℝ},
    p,
    v
    ) where {valence}

    return sqrt(
        (v[1][1])^2 +
        sum([(p[1][1])^2 * norm(Sphere(n - 1), x, xdot)^2
            for (n, x, xdot) in zip(valence, p[2:end], v[2:end])])
        )
end#=}}}=#

"""
    function rand(
        M::Segre{valence, ℝ};
        vector_at=nothing,
        )
"""
function rand(#={{{=#
    M::Segre{valence, ℝ};
    vector_at=nothing,
    ) where {valence}

    if isnothing(vector_at)
        xs = [normalize(2 * rand(n) .- 1) for n in valence]
        return [rand(1), xs...]
    else
        xdots = [normalize(2 * rand(n) .- 1) for n in valence]
        xdots = map(t -> t[2] - dot(t[2], t[1]) * t[1], zip(vector_at[2:end], xdots))
        return [2 * rand(1) .- 1.0, xdots...]
    end
end#=}}}=#

"""
    function embed_vector(
        M::Segre{valence, 𝔽},
        p,
        v
        )

Embed `p ∈ Segre((n1, ..., nd), F)` in `F^{n1 x ... x nd}`
"""
function embed(#={{{=#
    M::Segre{valence, 𝔽},
    p
    ) where {valence, 𝔽}

    return kronecker(p...)[:, 1]
end#=}}}=#

"""
    function embed_vector(
        M::Segre{valence, 𝔽},
        p,
        v
        )

Embed `v ∈ T_p Segre((n1, ..., nd), F)` in `F^{n1 x ... x nd}`
"""
function embed_vector(#={{{=#
    M::Segre{valence, 𝔽},
    p,
    v
    ) where {valence, 𝔽}

    # Product rule
    return sum([
        kronecker([
            i == j ?
            xdot :
            x
            for (j, (x, xdot)) in enumerate(zip(p, v))
            ]...)[:, 1]
        for (i, _) in enumerate(p)
        ])
end#=}}}=#


"""
    function exp(
        M::Segre{valence, ℝ},
        p,
        v
        )

Exponential map on Segre manifold. Theorem 1.1 in Swijsen 2021.
"""
function exp(#={{{=#
    M::Segre{valence, ℝ},
    p,
    v
    ) where {valence}

    q = zeros.(size.(p)) # Initialize
    exp!(M, q, p, v)

    return q
end#=}}}=#

"""
    function exp!(
        M::Segre{valence, 𝔽},
        q,
        p,
        v
        )

Exponential map on Segre manifold. Theorem 1.1 in Swijsen 2021.
"""
function exp!(#={{{=#
    M::Segre{valence, 𝔽},
    q,
    p,
    v
    ) where {valence, 𝔽}

    m = sqrt(sum([norm(Sphere(n - 1), p_i, v_i)^2 for (n, p_i, v_i) in zip(valence, p[2:end], v[2:end])]))
    if m == 0.0
        q .= deepcopy(p) # Initialize
        q[1] .= q[1] .+ v[1]
        return q
    end

    t = norm(M, p, v)
    P = v[1][1] / (p[1][1] * m)
    f = atan(sqrt(P^2 + 1.0) * t / p[1][1] + P) - atan(P)

    q[1][1] = sqrt(
        t^2 +
        2 * p[1][1] * P * t / sqrt(P^2 + 1.0) +
        p[1][1]^2 # This factor is wrong in Swijsen21 on arxiv
        )

    for i in range(2, ndims(M) + 1)
        if all(v[i] .== 0.0)
            q[i] .= deepcopy(p[i])
        else
            n = valence[i - 1]
            S = Sphere(n - 1)
            a = norm(S, p[i], v[i]) * f / m
            q[i] .= p[i] * cos(a) .+ v[i] * sin(a) / norm(S, p[i], v[i])
        end
    end

    return 0
end#=}}}=#

# Theorem 6.2.1 in thesisLarsSwijsen
"""
    function log(
        M::Segre{valence, ℝ},
        p,
        q
        )

Logarithmic map on Segre manifold.
"""
function log(#={{{=#
    M::Segre{valence, ℝ},
    p,
    q
    ) where {valence}

    # Check for compatability
    rho(a, b) = sqrt(sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(valence, a[2:end], b[2:end])]))
    if rho(p, q) < pi
        v = zeros.(size.(p)) # Initialize
        log!(M, v, p, q)
    else
        # Find closest representation by flipping an even number of signs.
        ds = [distance(Sphere(n - 1), x, y) for (n, x, y) in zip(valence, p[2:end], q[2:end])]
        flips = [false, (ds .> (pi / 2))...]
        nbr_flips = sum(flips)

        # This code is pretty ugly.
        if isodd(nbr_flips)
            if nbr_flips == length(valence)
                flips[argmin(ds) + 1] = false
            else
                is = sortperm(ds; rev=true)

                flips1 = deepcopy(flips)
                flips1[is[nbr_flips] + 1] = false
                q1 = deepcopy(q)
                q1[flips1] = -q1[flips1]

                flips2 = deepcopy(flips)
                flips2[is[nbr_flips + 1] + 1] = true
                q2 = deepcopy(q)
                q2[flips2] = -q2[flips2]

                rho(p, q1) < rho(p, q2) ? flips = flips1 : flips = flips2
            end
        end

        q_ = deepcopy(q)
        q_[flips] = -q[flips]
        @assert(iseven(sum(flips))) # Should not be necessary but you never know...
        @assert(rho(p, q_) < pi)

        v = zeros.(size.(p)) # Initialize
        log!(M, v, p, q_)
    end

    return v
end#=}}}=#

"""
    function log!(
        M::Segre{valence, 𝔽},
        v,
        p,
        q
        )

Logarithmic map on Segre manifold.
"""
function log!(#={{{=#
    M::Segre{valence, 𝔽},
    v,
    p,
    q
    ) where {valence, 𝔽}

    for i in range(2, ndims(M) + 1)
        a = LinearAlgebra.dot(p[i], q[i])
        if a >= 1.0 # Should not be able to be larger than 1, but sometimes is due to rounding
            v[i] .= zeros(size(p[i]))
        else
            v[i] .= (q[i] .- a * p[i]) * acos(a) / sqrt(1.0 - a^2)
        end
    end

    m = sqrt(sum([norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(valence, p[2:end], v[2:end])]))
    if m == 0.0
        v[1][1] = q[1][1] - p[1][1]
    else
        v[1][1] = m * p[1][1] * (q[1][1] * cos(m) - p[1][1]) / (q[1][1] * sin(m))

        t = (
                p[1][1] * tan(m + atan(v[1][1] / (p[1][1] * m))) - v[1][1] / m
            ) / (
                sqrt((v[1][1] / (p[1][1] * m))^2 + 1)
            )
        v .= t * normalize(M, p, v)
    end

    return 0
end#=}}}=#

"""
    function distance(
        M::Segre{valence, 𝔽},
        p,
        q
        )
Riemannian distance between two points `p` and `q` on the Segre manifold.
"""
function distance(#={{{=#
    M::Segre{valence, 𝔽},
    p,
    q
    ) where {valence, 𝔽}

    # TODO: Write down the closed-form expression for the distance
    return norm(M, p, log(M, p, q))
end#=}}}=#

# TODO: Check factor of p[1][1]
"""
    function second_fundamental_form(
        M::Segre{valence, 𝔽},
        p,
        u,
        v
        )

Second fundamental form of the Segre manifold embedded with `embed`.
"""
function second_fundamental_form(#={{{=#
    M::Segre{valence, 𝔽},
    p,
    u,
    v
    ) where {valence, 𝔽}

    h = 0 * embed(M, p) # Initialize
    for i in 1:length(valence)
        for j in 1:length(valence)
            if i != j
                p_ = 1 * p
                p_[i + 1] = u[i + 1]
                p_[j + 1] = v[j + 1]
                h = h + kron(p_...)[:, 1]
            end
        end
    end

    return h
end#=}}}=#

# TODO: do this without embedding
"""
    function sectional_curvature(
        M::Segre{valence, 𝔽},
        p,
        u,
        v
        )

Sectional curvature of the Segre manifold in the plane spanned by tangent vectors `u` and `v` at `p`.
"""
function sectional_curvature(#={{{=#
    M::Segre{valence, 𝔽},
    p,
    u,
    v
    ) where {valence, 𝔽}

    return (
        dot(second_fundamental_form(M, p, u, u), second_fundamental_form(M, p, v, v)) -
        dot(second_fundamental_form(M, p, u, v), second_fundamental_form(M, p, u, v))
        ) / (
            norm(M, p, u)^2 * norm(M, p, v)^2 - inner(M, p, u, v)^2
        )
end#=}}}=#
