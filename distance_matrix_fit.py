import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import solve
from scipy.stats.qmc import Halton
from test_functions import test_function  
from distance_matrix import distance_matrix 

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def haltonseq(N, s):
    sampler = Halton(d=s, scramble=False)
    return sampler.random(N)

def make_sd_grid(s, neval):
    if s==3:
        xe, ye, ze = np.meshgrid(np.linspace(0, 1, neval), 
                             np.linspace(0, 1, neval), 
                             np.linspace(0, 1, neval))

        xe = xe.flatten()
        ye = ye.flatten()
        ze = ze.flatten()
        epoints=np.vstack((np.vstack((xe, ye)),ze)).T
    elif s==2:
        xe,ye=np.meshgrid(np.linspace(0,1,neval),np.linspace(0,1,neval))
        xe=xe.flatten()
        ye=ye.flatten()
        
        epoints=np.vstack((xe, ye)).T
    else:
        epoints=np.array(np.meshgrid(np.linspace(0,1,neval))).T
    return epoints

def test_function(x):
    n = x.shape[1]
    s = np.ones(x.shape[0])
    for i in range(x.shape[0]):
        product = 1
        for j in range(n):
            product *= x[i, j] * (1 - x[i, j])
        s[i] = 4**n * product
    return s

s = 1
k = 2
N = (2*k+1)**s
neval = 10
M = neval - s

dsites = haltonseq(N, s)
ctrs = dsites
epoints = make_sd_grid(s, neval)
#print(epoints.shape)
rhs = test_function(dsites)

IM = distance_matrix(dsites, ctrs)
EM = distance_matrix(epoints, ctrs)

Pf = EM @ solve(IM, rhs)
exact = test_function(epoints)

maxerr = np.linalg.norm(Pf - exact, np.inf)
rms_err = np.linalg.norm(Pf - exact) / np.sqrt(M)

print(f'RMS error: {rms_err:e}')
print(f'Maximum error: {maxerr:e}')

if s == 1:
    plt.figure()
    plt.plot(epoints, Pf, label="Interpolant")
    plt.plot(epoints, exact, '--', label="Exact")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "interpolant_1D.png"))
    plt.close()

    plt.figure()
    plt.plot(epoints, np.abs(Pf - exact))
    plt.title("Absolute Error")
    plt.savefig(os.path.join(results_dir, "error_1D.png"))
    plt.close()

elif s == 2:
    xe = epoints[:, 0].reshape(neval, neval)
    ye = epoints[:, 1].reshape(neval, neval)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xe, ye, Pf.reshape(neval, neval), cmap="viridis")
    plt.savefig(os.path.join(results_dir, "interpolant_2D.png"))
    plt.close()

elif s == 3:
    xe=epoints[:,0]
    ye=epoints[:,1]
    ze=epoints[:,2]

    Pf_flat = Pf.flatten()

    if Pf_flat.size != xe.size:
        Pf_flat = np.resize(Pf_flat, xe.size)  # Resize Pf_flat to match number of points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xe, ye, ze, c=Pf_flat, cmap="viridis")
    plt.savefig(os.path.join(results_dir, "interpolant_3D.png"))
    plt.close()

else:
    print("Cannot display plots for s > 3")
