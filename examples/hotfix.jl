import ManifoldsBase: default_retraction_method, default_inverse_retraction_method, get_coordinates_orthonormal, get_vector_orthonormal
default_retraction_method(::MetricManifold{ℝ, <:Stiefel{<:Any, <:Any, ℝ}, CanonicalMetric}, ::Type) = ExponentialRetraction()
default_inverse_retraction_method(::MetricManifold{ℝ, <:Stiefel{<:Any, <:Any, ℝ}, CanonicalMetric}, ::Type) = LogarithmicInverseRetraction()
get_coordinates_orthonormal(::MetricManifold{ℝ, <:Stiefel{<:Any, <:Any, ℝ}, CanonicalMetric}, ::Matrix{Float64}, v::Matrix{Float64}, _) = v[:]
get_vector_orthonormal(::MetricManifold{ℝ, <:Stiefel{n, k, ℝ}, CanonicalMetric}, ::Matrix{Float64}, c::Vector{Float64}, _) where {n, k} = reshape(c, (n, k))
get_coordinates_orthonormal(::Grassmann{<:Any, <:Any, ℝ}, ::Matrix{Float64}, v::Matrix{Float64}, _) = v[:]
get_vector_orthonormal(::Grassmann{n, k, ℝ}, ::Matrix{Float64}, c::Vector{Float64}, _) where {n, k} = reshape(c, (n, k))
get_coordinates_orthonormal(::MetricManifold{ℝ, <:Grassmann{<:Any, <:Any, ℝ}, CanonicalMetric}, ::Matrix{Float64}, v::Matrix{Float64}, _) = v[:]
get_vector_orthonormal(::MetricManifold{ℝ, <:Grassmann{n, k, ℝ}, CanonicalMetric}, ::Matrix{Float64}, c::Vector{Float64}, _) where {n, k} = reshape(c, (n, k))
