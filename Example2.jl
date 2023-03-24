using LinearAlgebra: qr

k = 2
M = OrthogonalMatrices(k)
m = k^2
n = manifold_dimension(M)
function f(x::Vector{Float64})
    # Assert that the entries of x fit in a square matrix
    l = Int64(sqrt(length(x)))
    @assert(length(x) == l^2)
    Q, _ = qr(reshape(x, l, l))
    return Q
end
