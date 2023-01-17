import numpy as np
from numpy.linalg import norm
import numdifftools as nd
import matplotlib.pyplot as plt
from jax import jacfwd

# Seg(P^n1 x ... P^nd) ~ R+ x S^(n1 - 1) x ... x S^(nd - 1)

d = 2
n1 = 2
n2 = 2
ns = [1, 1 + n1, 1 + n1 + n2]
dim = 1 + (n1 - 1) + (n2 - 1)

###### FUNCTIONS ON S^n ######

# Stereographic projection : R^n -> S^n < R^{n + 1}
def embed_point(p):# {{{
    return np.append(
        (-1. + norm(p)**2),
        2. * p
        ) / (1. + norm(p)**2)# }}}

# Differential of stereographic projection
def d_embed_point(p):# {{{
    n = len(p)
    return np.vstack((
        4. * p.T,
        2. * (1. + norm(p)**2) * np.eye(n) - 4. * np.outer(p, p)
        )) / (1. + norm(p)**2)**2# }}}

# Tangent space to Riemann sphere
def embed_tangent(p, v):# {{{

    # embed v in R^{n + 1}
    n = len(p)
    v_ = np.append(0., v)
    p_ = embed_point(p)

    # rotate v to p
    zhat = np.append(1., np.zeros(n))
    m = p_ - zhat

    return v_ - 2 * np.dot(p_, v_) * m / norm(m)**2# }}}

def unembed_point(u):# {{{
    np.testing.assert_approx_equal(norm(u), 1.)
    return u[1:] / (1. - u[0])# }}}


###### FUNCTIONS ON Seg ######

# Theorem 1.1 in Swijsen21
def exp(p_, v_):# {{{
    if norm(v_) == 0.:
        return p_

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
        p[0]**2
        )
    for i in range(1, d + 1):
        u = embed_point(p[i])
        udot = embed_tangent(p[i], v[i])
        u =\
            u * np.cos(norm(udot) * f / M) +\
            (udot / norm(udot)) * np.sin(norm(udot) * f / M)

        q[i] = unembed_point(u)

    # Flatten output
    q_ = np.concatenate(q)

    return q_# }}}

# def d_exp(p, v):
#     return jacfwd(lambda x: exp(p, x))

def metric(p):# {{{
    # TODO: embed point is for S^n, not for Seg...
    return np.linalg.inv(d_embed_point(p)) @ np.diag(1., p[0] * np.ones(p.shape)) @ d_embed_point(p)# }}}

# Christoffel symbol{{{
def christoffel(p):
    # https://en.wikipedia.org/wiki/Christoffel_symbols#General_definition
    G = np.array([d_metric(p, e[i]) for i in range(0, 5)])
    return (1. / 2.) * (-G + np.swapaxes(G, 0, 1) + np.swapaxes(G, 0, 2))# }}}

def d_christoffel(p, v):# {{{
    h = 1e-6
    return (christoffel(p + h * v) / (2. * h) -\
        christoffel(p + -h * v) / (2. * h))# }}}

# Riemann curvature tensor
def riemann(p):# {{{
    dG = np.array([d_christoffel(p, e[i]) for i in range(0, 5)])
    G = christoffel(p)
    g = metric(p)

    R = np.moveaxis(dG, [0, 1, 2, 3], [2, 0, 3, 1]) -\
        np.moveaxis(dG, [0, 1, 2, 3], [3, 0, 2, 1]) +\
        np.moveaxis(np.einsum('ija,ab,bkl', G, g, G), [0, 1, 2, 3], [0, 2, 3, 1]) -\
        np.moveaxis(np.einsum('ija,ab,bkl', G, g, G), [0, 1, 2, 3], [0, 3, 2, 1])

    return R# }}}

def sectional_curvature(p, v, u):# {{{
    g = metric(p)
    # https://en.wikipedia.org/wiki/Sectional_curvature#Definition
    K = np.einsum('ijka,i,j,k,ab,b', riemann(p0), u, v, v, g, u) / (np.einsum('a,ab,b', u, g, u) * np.einsum('a,ab,b', v, g, v) - np.einsum('a,ab,b', u, g, v)**2)
    return K# }}}


### Is there a lower bound for the sectional curvature? ###
from scipy.optimize import minimize

def obj_fun(x):# {{{
    K = sectional_curvature(p0, x[0:5] / norm(x[0:5]), x[5:10] / norm(x[5:10]))
    print(K)
    return K# }}}

### ###
e0 = np.array([1., 0., 0., 0., 0.])
e1 = np.array([0., 1., 0., 0., 0.])
e2 = np.array([0., 0., 1., 0., 0.])
e3 = np.array([0., 0., 0., 1., 0.])
e4 = np.array([0., 0., 0., 0., 1.])
es = [e0, e1, e2, e3, e4]

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

print(exp(p0, v0))
# print(d_exp(p0, 1e-12 * v0))
# print(v0.T @ d_exp(p0, 1e-6 * v0) @ e2)
# print(sectional_curvature(p0, v0, e2))

# plt.spy(metric(p0), precision=1e-10)
# plt.show()

### TEST CORRECTNESS OF FUNCTIONS ###

def finite_difference(f, p, v):# {{{
    h = 1e-6
    return (f(p + h * v) / (2. * h) - f(p - h * v) / (2. * h))# }}}

def test_d_embed_point(p, v):# {{{
    return np.allclose(
        d_embed_point(p0) @ v,
        finite_difference(embed_point, p0, v),
        rtol=1e-3
        )# }}}

def test_embed_tangent(p, v):# {{{
    n = len(p)
    return np.allclose(
        np.dot(embed_point(p), embed_tangent(p, v)),
        np.zeros(n),
        rtol=1e-3
        ) & np.allclose(
        norm(embed_tangent(p, v)),
        norm(v),
        rtol=1e-3
        )# }}}

def test_unembed_point(p, v):# {{{
    return np.allclose(
        unembed_point(embed_point(p)),
        p,
        rtol=1e-3
        )# }}}

# for i in range(0, 10):# {{{
#     print(test_d_embed_point(
#         np.random.rand(5),
#         np.random.rand(5)
#         ))
#     print(test_embed_tangent(
#         np.random.rand(5),
#         np.random.rand(5)
#         ))
#     print(test_unembed_point(
#         np.random.rand(5),
#         np.random.rand(5)
#         ))# }}}
