import numpy as np
from test_functions import frankes
from distance_matrix import distance_matrix
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
import csv

def rbf(ep, r):
    return np.exp(-(ep * r) ** 2)

def run_rbf_interpolation(N, ep, gridtype='h', neval=40, save_prefix='run'):
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

    os.makedirs(f"results/{save_prefix}", exist_ok=True)
    np.save(f"results/{save_prefix}/Pf.npy", Pf)
    np.save(f"results/{save_prefix}/exact.npy", exact)
    np.save(f"results/{save_prefix}/errors.npy", np.array([rms_err, maxerr]))

    plot_surface(xe, ye, Pf, exact, maxerr, N, ep, save_path=f"results/{save_prefix}/plot.png")

    return rms_err, maxerr

def plot_surface(xe, ye, Pf, exact, maxerr, N, ep, save_path):
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xe, ye, Pf.reshape(xe.shape), cmap='viridis')
    ax.set_title(f"RBF Interpolant\nN={N}, ep={ep}")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xe, ye, exact.reshape(xe.shape), cmap='viridis')
    ax2.set_title(f"Exact Solution\nN={N}, ep={ep}")

    plt.suptitle(f"N={N}, ep={ep} | Max Error: {maxerr:e}")
    plt.savefig(save_path)
    plt.close(fig)

N_values = [9, 25, 81, 289, 1089, 4225]
smallest_safe_epsilons = [0.02, 0.32, 1.64, 4.73, 10.5, 21.1]
smallest_rms_epsilons = [2.23, 3.64, 4.28, 5.46, 6.2, 6.3]

safe_csv = 'results/safe_epsilons_results.csv'
rms_csv = 'results/rms_epsilons_results.csv'

os.makedirs("results", exist_ok=True)

def save_to_csv(csv_filename, header, data_rows):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data_rows)

safe_results = []
for i, N in enumerate(N_values):
    ep = smallest_safe_epsilons[i]
    rms_err, maxerr = run_rbf_interpolation(N, ep, save_prefix=f'safe_N{N}_ep{ep}')
    safe_results.append([N, ep, rms_err, maxerr])
    print(f"[SAFE] N={N}, ep={ep}: RMS Error = {rms_err:e}, Max Error = {maxerr:e}")

save_to_csv(safe_csv, ['N', 'epsilon', 'RMS error', 'Max error'], safe_results)

rms_results = []
for i, N in enumerate(N_values):
    ep = smallest_rms_epsilons[i]
    rms_err, maxerr = run_rbf_interpolation(N, ep, save_prefix=f'rms_N{N}_ep{ep}')
    rms_results.append([N, ep, rms_err, maxerr])
    print(f"[RMS] N={N}, ep={ep}: RMS Error = {rms_err:e}, Max Error = {maxerr:e}")

save_to_csv(rms_csv, ['N', 'epsilon', 'RMS error', 'Max error'], rms_results)

print("Finished! Results saved to CSV files.")
