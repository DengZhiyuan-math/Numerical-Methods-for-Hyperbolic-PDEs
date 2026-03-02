import numpy as np
import matplotlib.pyplot as plt

# --- Problem setup (matches the book example) ---
a = 1.0
L = 1.0
T = 1.0
N = 100
dx = L / N
x = np.linspace(0, L, N, endpoint=False)

def u0(x):
    return np.sin(2*np.pi*x)

def exact_u(x, t):
    # periodic exact solution for u_t + a u_x = 0 on [0,1)
    return np.sin(2*np.pi*((x - a*t) % L))

def upwind(a, x, dx, dt, T):
    """Forward Euler in time + upwind in space for a>0, periodic BC."""
    u = u0(x).copy()
    nsteps = int(np.round(T / dt))
    dt = T / nsteps  # adjust so we land exactly at T
    lam = a * dt / dx

    for _ in range(nsteps):
        um = np.roll(u, 1)  # u_{j-1}
        u = u - lam * (u - um)
    return u, lam, dt, nsteps

# Two timesteps: dt = 1.3 dx and dt = 0.9 dx
dts = [1.3*dx, 0.9*dx]
solutions = []
for dt in dts:
    u_num, lam, dt_adj, nsteps = upwind(a, x, dx, dt, T)
    solutions.append((dt, u_num, lam, dt_adj, nsteps))

u_ex = exact_u(x, T)

# --- Plot: two separate figures (less clutter) ---
paths = []
for k, (dt_req, u_num, lam, dt_adj, nsteps) in enumerate(solutions, start=1):
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(x, u_num, 'o-', markersize=4, linewidth=1.2, label='Upwind scheme')
    plt.plot(x, u_ex, '-', linewidth=2.0, label='Exact solution')
    plt.xlim(0, 1)
    plt.xlabel('x')
    plt.ylabel('u(x,T)')
    plt.title(f'Upwind, N={N}, a={a}, T={T}\nrequested dt={dt_req/dx:.1f} dx | used dt={dt_adj/dx:.6f} dx | CFL λ={lam:.6f}')
    plt.legend()
    plt.tight_layout()
    out = f"upwind_dt_{dt_req / dx:.1f}dx.png"
    plt.savefig(out, dpi=200)
    paths.append(out)
    plt.show()

paths
