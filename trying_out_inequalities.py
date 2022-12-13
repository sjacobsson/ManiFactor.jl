import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

H = -10
def error(epsilon, sigma):
    return epsilon + (2. / np.sqrt(np.abs(H))) * np.arcsinh(
        (epsilon / 2) *\
        np.sinh(np.sqrt(np.abs(H)) * sigma) / sigma
        )

def error2(epsilon, sigma):
    return epsilon + (2. / np.sqrt(np.abs(H))) * np.log(
        (epsilon / 2) *\
        np.exp(np.sqrt(np.abs(H)) * sigma - 1) / sigma +\
        1
        )

es = np.logspace(-12, -6)
ss = np.linspace(0., 5.)
Es, Ss = np.meshgrid(es, ss)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Es, Ss, error(Es, Ss), alpha=0.5)
ax.plot_surface(Es, Ss, error2(Es, Ss), alpha=0.5)
ax.set_xlabel('epsilon')
ax.set_ylabel('sigma')
ax.set_zlabel('error bound')
plt.show()

print(error(1e-12, 10))
