type SegPoint = []

-- Theorem 1.1 in Swijsen21
def exp(p, v):
    t = np.norm(v)
    M = np.sqrt()
    P = v[0] / (p[0] * M)
    f = np.atan() - np.atan(P)
    # TODO: Initialize q
    q[0] = np.sqrt(t**2 + (2 * p[0] * P) / np.sqrt(P**2 + 1.0) * t + p[0]**2 / (P**2 + 1.))

# TODO: Riemanntensorn i termer av Christoffelsymboler (https://en.wikipedia.org/wiki/Riemann_curvature_tensor)

# TODO: Christoffelsymboler i termer av metriken (https://en.wikipedia.org/wiki/Christoffel_symbols#General_definition)

# TODO: Minimera K(u, v) med avseende på u, v (https://en.wikipedia.org/wiki/Sectional_curvature#Definition)
