import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

def run_stochastic_kuramoto(N, K, omega_max, sigma, T_max, dt):
    num_steps = int(T_max / dt)
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.uniform(0, omega_max, N)
    r_series = np.zeros(num_steps)

    for t in range(num_steps):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        coupling = np.sum(np.sin(diff), axis=1)
        dtheta = (omega + (K / N) * coupling) * dt + sigma * np.sqrt(dt) * np.random.randn(N)
        theta += dtheta
        r_series[t] = np.abs(np.sum(np.exp(1j * theta)) / N)

    r_avg = np.mean(r_series[int(0.8 * num_steps):])
    return r_avg

# Parameters
N = 20
omega_max = 2*np.pi
T_max = 10
dt = 0.01

# Higher resolution grid
n_sigma = 400
n_K = 400
sigma_vals = np.linspace(0, 5, n_sigma)   # X-axis
K_vals = np.linspace(0, 10, n_K)          # Y-axis

# Prepare argument pairs
param_list = [(N, K, omega_max, sigma, T_max, dt) 
              for K in K_vals for sigma in sigma_vals]

# Parallel execution with progress bar
results = Parallel(n_jobs=-1, verbose=0)(
    delayed(run_stochastic_kuramoto)(*params) for params in tqdm(param_list)
)

# Reshape result into 2D grid
R_map = np.array(results).reshape(len(K_vals), len(sigma_vals))

# Smooth contour plot
X, Y = np.meshgrid(sigma_vals, K_vals)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, R_map, cmap = "grey")
plt.xlabel('Noise Amplitude σ')
plt.ylabel('Coupling Constant K')
plt.title('Kuramoto Order Parameter Phase Boundary (r)')
plt.xticks(np.linspace(0, 5, 6))
plt.yticks(np.linspace(0, 10, 6))
plt.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar()
cbar.set_label("Order Parameter")
plt.show()
