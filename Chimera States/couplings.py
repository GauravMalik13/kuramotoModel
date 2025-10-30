import numpy as np

def cosine_coupling(i, j, theta, A, N):
    distance = (2 * np.pi / N) * max(abs(i - j), N - abs(i - j))  # Periodic distance
    return (1 +A*np.cos(distance)) / (2 * np.pi)  # Normalized kerne

def cosine_coupling_twice_frequency(i, j, theta, A, N):
    distance = (2 * np.pi / N) * max(abs(i - j), N - abs(i - j))  # Periodic distance
    return (1 +A*np.cos(2*distance)) / (2 * np.pi)  # Normalized kerne

def exponential_coupling(i, j, theta, kappa, N):
    distance = 2*(np.pi/N)*np.abs(i - j)
    return np.exp(-kappa * distance)

def quadratic_coupling(i, j, theta, B, N):
    distance = np.abs(i - j)
    return 1/ (B * (distance**2))

def constant(i, j, theta, kappa, N):
    return kappa

def alternate(i, j, theta, B, N):
    if (i-j)%2==0:
        return 1
    else:
        return 0

def periodic(i, j, theta, B, N):
    if (np.abs(i-j)<=B*np.pi) & (j==50) :
        return 2*B*np.pi
    else:
        return 0

def nonlocal_exponential(i, j, theta, kappa, N, R=25):

    distance = min(abs(i - j), N - abs(i - j))
    if distance > R:
        return 0
    return np.exp(-kappa * distance)

def radius_coupling(i, j, theta, radius, N):
    
    distance = min(abs(i - j), N - abs(i - j)) 
    return 1.0 if distance <= radius else 0.0
    
def exponential_finite_range(i, j, theta, param, N,R=10):
    kappa= param
    distance = min(abs(i - j), N - abs(i - j))  # ring topology
    if distance <= R:
        return np.exp(-kappa * distance)
    else:
        return 0.0


