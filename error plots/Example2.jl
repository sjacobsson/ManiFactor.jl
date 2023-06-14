using LinearAlgebra

m1 = 2
m2 = 3
n = 1 + (m1 - 1) + (m2 - 1)
m = m1 * m2
M = Segre((m1, m2))

Random.seed!(420)
a = LinearAlgebra.normalize(rand(m1))
b = LinearAlgebra.normalize(rand(m2))

function f(x)
    U, S, Vt = svd(0.01 * reshape(x, m1, m2) + a * b')
    return [[S[1]], U[1, :], Vt[:, 1]]
end


