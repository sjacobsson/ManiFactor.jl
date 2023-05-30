using LinearAlgebra

m1 = 200
m2 = 300
n = 1 + (m1 - 1) + (m2 - 1)
m = 3
M = Segre((m1, m2))

Random.seed!(420)
a = LinearAlgebra.normalize(rand(m1))
b = LinearAlgebra.normalize(rand(m2))

u1 = LinearAlgebra.normalize(rand(m1))
v1 = LinearAlgebra.normalize(rand(m2))
u2 = LinearAlgebra.normalize(rand(m1))
v2 = LinearAlgebra.normalize(rand(m2))
u3 = LinearAlgebra.normalize(rand(m1))
v3 = LinearAlgebra.normalize(rand(m2))
u4 = LinearAlgebra.normalize(rand(m1))
v4 = LinearAlgebra.normalize(rand(m2))

function f(x) # : [-1, 1]^m -> Segre((m1, m2))
    U, S, Vt = svd(
        0.1 * x[1] * u1 * v1' +
        0.1 * x[2] * u2 * v2' +
        0.1 * x[3] * u3 * v3' + 
        0.0001 * a * b'
        )
    return [[S[1]], U[:, 1], Vt[:, 1]]
end


