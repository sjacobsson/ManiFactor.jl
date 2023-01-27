include("Segre.jl")

function quad_trap(f, a, b, N) 
    h = (b-a)/N
    int = h * ( f(a) + f(b) ) / 2
    for k=1:N-1
        xk = (b-a) * k/N + a
        int = int + h*f(xk)
    end
    return int
end
    
# Random.seed!(6)
M = Segre((6, 6))
p = rand(M)
q = rand(M)
a = p[1][1] * p[2]
b = p[3]
c = q[1][1] * q[2]
d = q[3]

function gamma(t)
    return ((1 - t) * a + t * c) * transpose((1 - t) * b + t * d)
end

function d_gamma(t)
    return (-a + c) * transpose((1 - t) * b + t * d) +
        ((1 - t) * a + t * c) * transpose(-b + d)
end


# Check that gamma goes from p to q
@assert(isapprox(
    flatten(transpose(gamma(0.0))),
    embed(M, p)
    ))
@assert(isapprox(
    flatten(transpose(gamma(1.0))),
    embed(M, q)
    ))

# Check that gamma stays on Seg
_, ss, _ = svd(gamma(rand()))
for s in ss[2:end]
    @assert(isapprox(s, 0.0; atol=1e-6))
end

# Check d_gamma
t = rand()
@assert(isapprox(
    d_gamma(t),
    finite_difference(gamma, t, 1e-6)
    ))

# Compare the length of gamma with the length of a geodesic
using QuadGK
# println( quad_trap(t -> sqrt(tr(transpose(d_gamma(t)) * d_gamma(t))), 0.0, 1.0, 2000))
# println(quad_trap(norm ∘ d_gamma, 0.0, 1.0, 2000))
println("∫|γ'(t)|dt = ", quadgk(norm ∘ d_gamma, 0.0, 1.0)[1])

gamma_(t) = embed_vector(M, p, exp(M, p, t * log(M, p, q)))
d_gamma_(t) = finite_difference(gamma_, t, 1e-6)
# println("∫|d_gamma_(t)|dt   = ", quadgk(norm ∘ d_gamma_, 0.0, 1.0)[1])
#
println("d(p, q)    = ", distance(M, p, q))
