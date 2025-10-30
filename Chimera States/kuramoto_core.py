
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

def solve_kuramoto(theta0, omega, K, alpha, T_SPAN, coupling_func, param):
    N = len(theta0)
    def kuramoto_rhs(t, theta):
        return np.array([
            omega[i] + (K / N) * sum(
                coupling_func(i, j, theta, param, N) * np.sin(theta[j] - theta[i] + alpha)
                for j in range(N) if i != j
            )
            for i in range(N)
        ])
    sol = solve_ivp(kuramoto_rhs, T_SPAN, theta0, method='RK45', t_eval=np.arange(T_SPAN[0], T_SPAN[1], 0.01))
    return sol
def compute_local_order_parameter(theta_array, neighbor_size=5):

    N, times = theta_array.shape
    r_local = np.zeros((N, times))

    for t in range(times):
        theta_t = theta_array[:, t]
        for i in range(N):
            indices = np.arange(i - neighbor_size, i + neighbor_size + 1) % N
            phases = theta_t[indices]
            r_local[i, t] = np.abs(np.mean(np.exp(1j * phases)))

    return r_local
def compute_order_parameter(theta_array):
    return np.abs(np.mean(np.exp(1j * theta_array), axis=0))
def animate_kuramoto(sol, order_param, local_order_param, title="Kuramoto Animation"):
    N = sol.y.shape[0]
    times = sol.t
    phases = sol.y
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.2)
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Phasor plot
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title("Oscillator Phases (Phasor)")
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    ax1.add_artist(circle)
    points, = ax1.plot([], [], 'o', color='tab:blue')
    text = ax1.text(-1.1, 1.1, '', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Phase vs Oscillator Index
    ax2.set_xlim(0, N - 1)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_title("Phase vs Oscillator Index")
    ax2.set_xlabel("Oscillator Index")
    ax2.set_ylabel("Phase (rad)")
    ax2.grid(True)
    phase_scatter = ax2.scatter([], [], color='purple')

    # Order parameter vs time
    ax3.set_xlim(times[0], times[-1])
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Global Order Parameter r(t)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("r(t)")
    r_line, = ax3.plot([], [], 'r-')
    r_vals, t_vals = [], []

    # Maximum local order parameter vs time
    ax4.set_xlim(times[0], times[-1])
    ax4.set_ylim(0, 1.05)
    ax4.set_title("Max Local Order Parameter")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Max r_local(t)")
    rlocal_line, = ax4.plot([], [], 'g-')
    rlocal_vals = []

    def update(frame):
        theta = phases[:, frame]
        x = np.cos(theta)
        y = np.sin(theta)
        points.set_data(x, y)

        # Phase vs index
        unwrapped = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        index = np.arange(N)
        phase_scatter.set_offsets(np.c_[index, unwrapped])

        # Global r(t)
        t_vals.append(times[frame])
        r_vals.append(order_param[frame])
        r_line.set_data(t_vals, r_vals)

        # Max Local r(t)
        rlocal_vals.append(np.max(local_order_param[:, frame]))
        rlocal_line.set_data(t_vals, rlocal_vals)

        text.set_text(f'Time: {times[frame]:.2f}\nr: {order_param[frame]:.2f}')
        return points, phase_scatter, r_line, rlocal_line, text

    ani = FuncAnimation(fig, update, frames=range(0, len(times), 2), interval=10, blit=True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
