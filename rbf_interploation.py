import numpy as np 
from test_functions import frankes
from distance_matrix import distance_matrix
import scipy.linalg as la
import matplotlib.pyplot as plt 
import os

def rbf(ep, r):
    return np.exp(-(ep * r) ** 2)

ep=21.1

N = 1089
gridtype = 'h'  
neval = 40
grid = np.linspace(0, 1, neval)
xe, ye = np.meshgrid(grid, grid)
epoints = np.column_stack([xe.ravel(), ye.ravel()])

dsites = np.random.rand(N, 2)  
ctrs = dsites

rhs = frankes(dsites[:, 0], dsites[:, 1])
DM_data = distance_matrix(dsites, ctrs)
IM = rbf(ep, DM_data)
DM_eval = distance_matrix(epoints, ctrs)
EM = rbf(ep, DM_eval)

Pf = EM @ la.solve(IM, rhs)

exact = frankes(epoints[:, 0], epoints[:, 1])

maxerr = np.linalg.norm(Pf - exact, np.inf)
rms_err = np.linalg.norm(Pf - exact) / neval

print(f'RMS error: {rms_err:e}')
print(f'Maximum error: {maxerr:e}')

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save results
np.save("results/Pf.npy", Pf)
np.save("results/exact.npy", exact)
np.save("results/errors.npy", np.array([rms_err, maxerr]))

def plot_surface(xe, ye, Pf, exact, maxerr):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xe, ye, Pf.reshape(xe.shape), cmap='viridis')
    ax.set_title("RBF Interpolant")
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xe, ye, exact.reshape(xe.shape), cmap='viridis')
    ax2.set_title("Exact Solution")
    
    plt.suptitle(f"Max Error: {maxerr:e}")
    plt.savefig("results/interpolation_plot.png")
    plt.show()

plot_surface(xe, ye, Pf, exact, maxerr)
