import numpy as np
import matplotlib.pyplot as plt

# ----- problem setup -----
a = 1.0
L = 1.0
T = 10.0
CFL = 0.9

def u0(x):
    return np.sin(2*np.pi*x)

def exact(x, t):
    return np.sin(2*np.pi*((x - a*t) % L))

# ----- schemes -----
def upwind(u, lam):
    um = np.roll(u, 1)
    return u - lam * (u - um)

def lax_wendroff(u, lam):
    up = np.roll(u, -1)
    um = np.roll(u, 1)
    return (u
            - 0.5*lam*(up - um)
            + 0.5*lam**2*(up - 2*u + um))

# ----- experiment -----
def run(N, scheme):
    dx = L / N
    dt = CFL * dx / a
    x = np.linspace(0, L, N, endpoint=False)

    u = u0(x).copy()
    nsteps = int(np.round(T/dt))
    dt = T / nsteps
    lam = a*dt/dx

    for _ in range(nsteps):
        u = scheme(u, lam)

    return x, u, exact(x, T)

# ----- plotting -----
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Upwind
for col, N in enumerate([50, 200]):
    x, u_num, u_ex = run(N, upwind)
    ax = axs[0, col]
    ax.plot(x, u_num, 'o-', label="Upwind")
    ax.plot(x, u_ex, '-', label="Exact")
    ax.set_title(f"Upwind, N={N}")
    ax.set_xlim(0,1)
    ax.legend()

# Lax–Wendroff
for col, N in enumerate([50, 200]):
    x, u_num, u_ex = run(N, lax_wendroff)
    ax = axs[1, col]
    ax.plot(x, u_num, 'o-', label="Lax-Wendroff")
    ax.plot(x, u_ex, '-', label="Exact")
    ax.set_title(f"Lax-Wendroff, N={N}")
    ax.set_xlim(0,1)
    ax.legend()

plt.tight_layout()
plt.show()