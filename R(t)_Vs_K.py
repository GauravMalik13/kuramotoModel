import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

# ─── Kuramoto Model ────────────────────────────────────────────────────────────
def kuramoto(t, theta, N, K, omega):
    # Build an N×N matrix of phase differences: θ_j - θ_i
    diff     = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling = np.sum(np.sin(diff), axis=1)
    return omega + (K / N) * coupling

def solve_system(N, K, omega_max, t_span, t_eval):
    omega  = np.random.uniform(0, omega_max, N)
    theta0 = np.random.uniform(0, 2*np.pi, N)
    return solve_ivp(kuramoto, t_span, theta0,
                     args=(N, K, omega),
                     t_eval=t_eval)

def compute_order_parameter(theta, N):
    """Compute the Kuramoto order parameter: r = |(1/N) * sum(exp(iθ))|."""
    return np.abs(np.sum(np.exp(1j * theta)) / N)

# ─── Simulation Parameters ────────────────────────────────────────────────────
T_max          = 10
t_span         = (0, T_max)
t_eval         = np.linspace(0, T_max, 500)
N_init         = 10
N_min          = 2
N_max          = 30
K_init         = 2.0
K_min          = 0.0
K_max          = 20
omega_max_init = 2*np.pi
omega_min      = 0.5*np.pi
omega_max      = 5*np.pi

# ─── Figure & Axes Setup ───────────────────────────────────────────────────────
# Create three subplots:
#  - Top: the oscillators on the unit circle.
#  - Middle: the time series r(t) for one simulation.
#  - Bottom: the average r vs K bifurcation diagram.
fig, (ax_circle, ax_rt, ax_rK) = plt.subplots(3, 1, figsize=(10, 10),
                                              gridspec_kw={'height_ratios': [2, 1, 1]})
plt.subplots_adjust(bottom=0.3, hspace=0.6)  # leave room for sliders and buttons

# Setup the circle plot (top)
ax_circle.set_title('Kuramoto Oscillators on the Unit Circle')
ax_circle.set_xlim(-1.5, 1.5)
ax_circle.set_ylim(-1.5, 1.5)
ax_circle.set_aspect('equal')
ax_circle.axis('off')
# Text annotation for instantaneous order parameter (upper left)
r_text = ax_circle.text(-3.2, 1, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Setup the r(t) time series plot (middle)
ax_rt.set_title('Order Parameter r(t) Over Time')
ax_rt.set_xlabel('Time')
ax_rt.set_ylabel('r(t)')
ax_rt.set_xlim(0, T_max)
ax_rt.set_ylim(0, 1.1)
ax_rt.grid(True)

# Setup the r vs K diagram (bottom)
ax_rK.set_title('Average Order Parameter vs Coupling Constant K')
ax_rK.set_xlabel('K')
ax_rK.set_ylabel('Average r')
ax_rK.set_xlim(K_min, K_max)
ax_rK.set_ylim(0, 1.1)
ax_rK.grid(True)

# ─── State Variables & Initial Simulation ─────────────────────────────────────
current_time_index = 0
is_playing         = False
sol = solve_system(N_init, K_init, omega_max_init, t_span, t_eval)
# Compute r(t) for each time point for the initial simulation
r_series = np.abs(np.sum(np.exp(1j * sol.y), axis=0) / N_init)

def init_lines(N):
    """Create N point artists on the unit circle."""
    return [ax_circle.plot([], [], 'o', color='C0', ms=8)[0] for _ in range(N)]

lines = init_lines(N_init)

# Create a line for the r(t) time series (middle) and a marker for the current r.
r_line, = ax_rt.plot(sol.t, r_series, 'b-', lw=2, label="r(t)")
r_marker, = ax_rt.plot([sol.t[current_time_index]], [r_series[current_time_index]], 'ro', ms=8, label="Current r")

# Create a line for the r vs K diagram (bottom); its data will be computed later.
rK_line, = ax_rK.plot([], [], 'm-o', lw=2, label="Average r")

# ─── Function to Compute Average r vs K ───────────────────────────────────────
def compute_r_vs_K(N, omega_max, T_max, t_eval, K_values):
    r_avg_list = []
    for K_val in K_values:
        sol_tmp = solve_system(N, K_val, omega_max, (0, T_max), t_eval)
        r_tmp = np.abs(np.sum(np.exp(1j * sol_tmp.y), axis=0) / N)
        # Average the order parameter over the final 20% of the simulation
        r_avg = np.mean(r_tmp[int(0.8*len(r_tmp)):])
        r_avg_list.append(r_avg)
    return np.array(r_avg_list)

# ─── Update Functions ─────────────────────────────────────────────────────────
def update_positions():
    global current_time_index, sol, r_series
    # Update oscillator positions on the circle.
    θs = sol.y[:, current_time_index]
    N  = len(θs)
    for i, ln in enumerate(lines):
        x, y = np.cos(θs[i]), np.sin(θs[i])
        ln.set_data([x], [y])
    # Compute instantaneous order parameter and update annotation.
    inst_r = compute_order_parameter(θs, N)
    r_text.set_text(f"r = {inst_r:.3f}")
    # Update the marker on the r(t) plot (wrap scalars in lists)
    r_marker.set_data([sol.t[current_time_index]], [r_series[current_time_index]])
    fig.canvas.draw_idle()

def update_plot():
    """Reset simulation when a slider is moved or Reset is clicked."""
    global sol, lines, current_time_index, r_series, rK_line
    current_time_index = 0

    N         = int(slider_N.val)
    K         = slider_K.val  # K is used for the "live" simulation.
    omega_max = slider_omega.val

    # Run the simulation for the live r(t) and circle plots.
    sol = solve_system(N, K, omega_max, t_span, t_eval)
    r_series = np.abs(np.sum(np.exp(1j * sol.y), axis=0) / N)

    # Remove old oscillator points and create new ones.
    for ln in lines:
        ln.remove()
    lines[:] = init_lines(N)

    # Update the r(t) plot.
    r_line.set_data(sol.t, r_series)
    ax_rt.set_xlim(sol.t[0], sol.t[-1])
    r_marker.set_data([sol.t[0]], [r_series[0]])

    update_positions()

    # Also update the r vs K plot.
    K_vals = np.linspace(K_min, K_max, 20)
    r_avg_vals = compute_r_vs_K(N, omega_max, T_max, t_eval, K_vals)
    rK_line.set_data(K_vals, r_avg_vals)
    ax_rK.set_xlim(K_min, K_max)
    ax_rK.set_ylim(0, 1.1)

# ─── Sliders ──────────────────────────────────────────────────────────────────
ax_k     = plt.axes([0.1, 0.20, 0.65, 0.02], facecolor='lightgoldenrodyellow')
ax_n     = plt.axes([0.1, 0.15, 0.65, 0.02], facecolor='lightgoldenrodyellow')
ax_omega = plt.axes([0.1, 0.10, 0.65, 0.02], facecolor='lightgoldenrodyellow')

slider_K     = Slider(ax_k,     'K',    K_min, K_max,      valinit=K_init,        valstep=0.01)
slider_N     = Slider(ax_n,     'N',    N_min,   N_max,       valinit=N_init,        valstep=1)
slider_omega = Slider(ax_omega, 'ωₘₐₓ', omega_min, omega_max,  valinit=omega_max_init, valstep=0.1)

for s in (slider_K, slider_N, slider_omega):
    s.on_changed(lambda val: update_plot())

# ─── Play/Pause via Timer ─────────────────────────────────────────────────────
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(lambda: (
    globals().__setitem__('current_time_index',
        (current_time_index + 1) % len(sol.t)
    ),
    update_positions()
))

# ─── Styled Buttons (with Icon-Only Look) ─────────────────────────────────────
ax_play  = plt.axes([0.82, 0.12, 0.06, 0.06], facecolor='none')
ax_reset = plt.axes([0.90, 0.12, 0.06, 0.06], facecolor='none')

btn_play  = Button(ax_play,  '▶', color='#4CAF50', hovercolor='#45a049')
btn_reset = Button(ax_reset, '⟲', color='#f44336', hovercolor='#da190b')

for btn in (btn_play, btn_reset):
    btn.label.set_fontsize(18)
    btn.label.set_color('white')
    # Remove ticks & spines for a clean, icon‐only look
    btn.ax.set_xticks([])
    btn.ax.set_yticks([])
    for spine in btn.ax.spines.values():
        spine.set_visible(False)

def on_play(event):
    global is_playing
    is_playing = not is_playing
    btn_play.label.set_text('❚❚' if is_playing else '▶')
    if is_playing:
        timer.start()
    else:
        timer.stop()

def on_reset(event):
    slider_K.reset()
    slider_N.reset()
    slider_omega.reset()
    update_plot()

btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)

# ─── Start the Simulation ─────────────────────────────────────────────────────
update_plot()
plt.show()
