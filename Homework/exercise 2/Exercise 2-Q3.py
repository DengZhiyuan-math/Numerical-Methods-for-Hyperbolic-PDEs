import numpy as np
import matplotlib.pyplot as plt

# domain
xmin, xmax = -10.0, 10.0
N = 100
dx = (xmax - xmin) / N
x = np.linspace(xmin, xmax, N + 1)  # grid points

# advection speed
a = 2.0

# CFL choice
lam = 0.8
dt = lam * dx / a

# final time
T = 2.0
nsteps = int(round(T / dt))
dt = T / nsteps
lam = a * dt / dx

# initial data
u0 = np.where(x < 0.0, -1.0, 1.0)

def apply_bc(u):
    # ghost-cell style copying, translated to endpoint handling
    u[0] = u[1]
    u[-1] = u[-2]
    return u

def upwind(u):
    unew = u.copy()
    unew[1:-1] = u[1:-1] - lam * (u[1:-1] - u[0:-2])
    return apply_bc(unew)

def lax_wendroff(u):
    unew = u.copy()
    unew[1:-1] = (
        u[1:-1]
        - 0.5 * lam * (u[2:] - u[:-2])
        + 0.5 * lam**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    )
    return apply_bc(unew)

# exact solution at time T
uex = np.where(x < a * T, -1.0, 1.0)

# evolve
u_up = apply_bc(u0.copy())
u_lw = apply_bc(u0.copy())

for _ in range(nsteps):
    u_up = upwind(u_up)
    u_lw = lax_wendroff(u_lw)

# plot
plt.figure(figsize=(8, 5))
plt.plot(x, uex, 'k-', lw=2, label='Exact')
plt.plot(x, u_up, 'b--', lw=2, label='Upwind')
plt.plot(x, u_lw, 'r-.', lw=2, label='Lax-Wendroff')
plt.xlabel('x')
plt.ylabel('u(x,2)')
plt.title('Problem 3: Upwind vs Lax-Wendroff at t=2')
plt.legend()
plt.grid(True)
plt.show()