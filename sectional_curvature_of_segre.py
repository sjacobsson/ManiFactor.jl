import numpy as np
from numpy.linalg import norm
 
# Seg(P^n1 x ... P^nd) ~ R+ x S^(n1 - 1) x ... x S^(nd - 1)

d = 2
n1 = 2
n2 = 2
ns = [1, 1 + n1, 1 + n1 + n2]
dim = 1 + (n1 - 1) + (n2 - 1)

p0 = np.array([
    1.4,
    3., 6.,
    2., 7.
    ])

v0 = np.array([
    -.3,
    .2, .6,
    .2, -.1
    ])

# Theorem 1.1 in Swijsen21
def exp(p_, v_):

    M = norm(v_[1:])
    if M == 0.:
        q_ = p_ # Initialize
        q_[0] = q_[0] + v_[0]
        return q_

    # Unflatten input
    p = np.split(p_, ns)
    v = np.split(v_, ns)

    t = norm(v_)
    P = v[0] / (p[0] * M)
    f = np.arctan(np.sqrt(P**2 + 1.) / p[0] * t + P) - np.arctan(P)

    q = p # Initialize
    q[0] = np.sqrt(
        t**2 +\
        (2 * p[0] * P) / np.sqrt(P**2 + 1.) * t +\
        p[0]**2 / (P**2 + 1.)
        )
    for i in range(1, d + 1):
        q[i] = p[i] * np.cos(norm(v[i] * f / M)) + np.sign(v[i]) * np.sin(norm(v[i]) * f / M)

    # Flatten output
    q_ = np.concatenate(q)

    return q_

# Basis for tangent space
e0 = np.array([1., 0., 0., 0., 0.])
e1 = np.array([0., 1., 0., 0., 0.])
e2 = np.array([0., 0., 1., 0., 0.])
e3 = np.array([0., 0., 0., 1., 0.])
e4 = np.array([0., 0., 0., 0., 1.])
e = [e0, e1, e2, e3, e4]

def d_exp(p, v):
    h = 1e-6
    return (exp(p, h * v) / (2. * h) - exp(p, -h * v) / (2. * h))

# Riemannian metric
def metric(p):
    g_inv = np.zeros((5, 5)) # Initialize
    for i in range(0, 5):
        for j in range(0, 5):
            g_inv[i, j] = np.dot(
                d_exp(p, e[i]),
                d_exp(p, e[j])
                )

    return np.linalg.inv(g_inv)

def metric_(p):
    n = len(p)
    d = np.nan * p # Initialize
    d[0] = 1.
    d[1:] = p[0]**2 * np.ones(n - 1)

    return np.diag(d)

def d_metric(p, v):
    h = 1e-6
    return (metric(p + h * v) / (2. * h) - metric(p + -h * v) / (2. * h))

# Christoffel symbol
def Gamma(p):
    # https://en.wikipedia.org/wiki/Christoffel_symbols#General_definition
    G = np.array([d_metric(p, e[i]) for i in range(0, 5)])
    Gamma = (1. / 2.) * (-G + np.swapaxes(G, 0, 1) + np.swapaxes(G, 0, 2))
    return Gamma

def d_Gamma(p, v):
    h = 1e-6
    return (Gamma(p + h * v) / (2. * h) - Gamma(p + -h * v) / (2. * h))

# Riemann curvature tensor
def Riemann(p):
    dG = np.array([d_Gamma(p, e[i]) for i in range(0, 5)])
    G = Gamma(p)
    g = metric(p)

    R = np.moveaxis(dG, [0, 1, 2, 3], [2, 0, 3, 1]) -\
        np.moveaxis(dG, [0, 1, 2, 3], [3, 0, 2, 1]) +\
        np.moveaxis(np.einsum('ija,ab,bkl', G, g, G), [0, 1, 2, 3], [0, 2, 3, 1]) -\
        np.moveaxis(np.einsum('ija,ab,bkl', G, g, G), [0, 1, 2, 3], [0, 3, 2, 1])

    return R

def sectional_curvature(p, v, u):
    g = metric(p)
    # https://en.wikipedia.org/wiki/Sectional_curvature#Definition
    K = np.einsum('ijka,i,j,k,ab,b', Riemann(p0), u, v, v, g, u) / (np.einsum('a,ab,b', u, g, u) * np.einsum('a,ab,b', v, g, v) - np.einsum('a,ab,b', u, g, v)**2)
    return K


### Is there a lower bound for the sectional curvature? ###
from scipy.optimize import minimize

def obj_fun(x):
    K = sectional_curvature(p0, x[0:5] / norm(x[0:5]), x[5:10] / norm(x[5:10]))
    print(K)
    return K

print(p0)
print(metric(p0))
print(metric_(p0))
print(sectional_curvature(p0, e0, e3))
# res = minimize(obj_fun, np.array(np.random.rand(10)))
# print(res)

# Hopefully there is something wrong with my implementation because rn I am finding sectional curvatures around K = -185700... Which is not too good for me since my upper bound scales as exp(|K|) for negative K.
# But there are lots of wonky things to examine like how I compute the metric as the inverse of the differential of the exponential map.
