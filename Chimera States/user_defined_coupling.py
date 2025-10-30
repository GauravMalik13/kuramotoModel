import numpy as np
import time
from kuramoto_core import compute_order_parameter, animate_kuramoto, compute_local_order_parameter, solve_kuramoto
import couplings


N = 100
K = 3
alpha = 0
T_SPAN = (0, 100)

omega = np.ones(N)

x = np.linspace(-np.pi, np.pi, N, endpoint=False)
r = np.random.uniform(-1,1, N)
theta0 = 6 * r *np.exp(-1.2 * x**2)
# np.random.seed(0)
# theta0 = np.random.normal(0,2*np.pi,N) 


coupling_func = couplings.exponential_coupling
param =5

start_time = time.time()
print("solving...")

sol = solve_kuramoto(theta0, omega, K, alpha, T_SPAN,coupling_func,param)
print("The system has been sucessfully solved\n","computing order parameter",)
r_t = compute_order_parameter(sol.y)
r_local = compute_local_order_parameter(sol.y,10)

end_time = time.time()

print(f"Time taken for computation: {end_time - start_time:.4f} seconds")


animate_kuramoto(sol, r_t, r_local, title="Kuramoto with Twisted states")

