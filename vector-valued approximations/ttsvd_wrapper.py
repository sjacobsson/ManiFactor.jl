import teneva
import numpy as np

def g_(x):
    return np.cos(np.linalg.norm(x)**3 + 0.5)

# TODO: kwargs
def ttsvd(g, m):
    N = 100
    
    # Let ts[i] be the i:th Chebyshev node
    ts = np.cos(2 * np.pi * (np.linspace(-1.0, 1.0, num=N) + 0.5) / (2 * N))
    
    I_trn, idx, idx_many = teneva.sample_tt([N] * m, r=15)
    print(I_trn.shape, idx.shape, idx_many.shape)
    G_trn = np.array([g(np.array([ts[i] for i in I])) for I in I_trn])
    print(G_trn.shape)
    
    Ghat = teneva.full(teneva.svd_incomplete(I_trn, G_trn, idx, idx_many, e=1e-12, r=15))

    return Ghat

print(ttsvd(g_, 3).shape)
