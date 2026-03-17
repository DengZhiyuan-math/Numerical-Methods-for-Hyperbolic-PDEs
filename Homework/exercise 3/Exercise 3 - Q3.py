#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Problem setup
# ============================================================

CASE = "a"          # choose "a" or "b"
CFL = 0.4
XL = -1.0
XR = 1.0
N = 200             # number of cells

# Use Dirichlet BC to match the exercise statement
BC_TYPE = "dirichlet"   # or "neumann"


# ============================================================
# Basic functions
# ============================================================

def flux(u):
    return 0.5 * u**2


def godunov_flux(u, N):
    """
    Compute Godunov fluxes at the N+1 interfaces.
    u has size N+2 because of ghost cells.
    """
    fh = np.zeros(N + 1)
    for j in range(N + 1):
        fh[j] = max(flux(max(u[j], 0.0)), flux(min(u[j + 1], 0.0)))
    return fh


def forward_euler(u, F, dt, dx, N):
    """
    One Forward Euler update for the finite volume scheme.
    """
    u_next = np.zeros(N + 2)
    for j in range(1, N + 1):
        u_next[j] = u[j] - (dt / dx) * (F[j] - F[j - 1])
    return u_next


# ============================================================
# Initial data and exact solutions
# ============================================================

def set_initial_condition(u, xc, case_name, N):
    """
    Fill the cell values u[1], ..., u[N] from the chosen case.
    """
    for i in range(1, N + 1):
        if case_name == "a":
            # u(x,0) = -1 for x<0, 1 for x>0
            if xc[i] < 0.0:
                u[i] = -1.0
            elif xc[i] > 0.0:
                u[i] = 1.0
            else:
                u[i] = 0.0

        elif case_name == "b":
            # u(x,0) = 1 for x<0, 0 for x>0
            if xc[i] < 0.0:
                u[i] = 1.0
            elif xc[i] > 0.0:
                u[i] = 0.0
            else:
                u[i] = 0.5

        else:
            raise ValueError("CASE must be 'a' or 'b'")


def exact_solution(xx, t, case_name):
    """
    Exact solution evaluated at points xx and time t.
    """
    u_exact = np.zeros_like(xx)

    if case_name == "a":
        # rarefaction
        for j in range(len(xx)):
            if xx[j] < -t:
                u_exact[j] = -1.0
            elif xx[j] > t:
                u_exact[j] = 1.0
            else:
                u_exact[j] = xx[j] / t

    elif case_name == "b":
        # shock moving with speed 1/2
        for j in range(len(xx)):
            if xx[j] < 0.5 * t:
                u_exact[j] = 1.0
            else:
                u_exact[j] = 0.0

    else:
        raise ValueError("CASE must be 'a' or 'b'")

    return u_exact


def final_time(case_name):
    if case_name == "a":
        return 0.5
    elif case_name == "b":
        return 1.0
    else:
        raise ValueError("CASE must be 'a' or 'b'")


# ============================================================
# Boundary conditions
# ============================================================

def apply_boundary_conditions(u, case_name, bc_type, N):
    """
    Set ghost cells u[0] and u[N+1].
    """
    if bc_type == "neumann":
        u[0] = u[1]
        u[N + 1] = u[N]

    elif bc_type == "dirichlet":
        if case_name == "a":
            u[0] = -1.0
            u[N + 1] = 1.0
        elif case_name == "b":
            u[0] = 1.0
            u[N + 1] = 0.0
        else:
            raise ValueError("CASE must be 'a' or 'b'")

    else:
        raise ValueError("BC_TYPE must be 'dirichlet' or 'neumann'")


# ============================================================
# Grid construction
# ============================================================

dx = (XR - XL) / N

# cell interfaces: x[0], ..., x[N]
x = np.zeros(N + 1)
for j in range(N + 1):
    x[j] = XL + j * dx

# cell centers: xc[1], ..., xc[N]
# xc[0] and xc[N+1] are ghost locations and not used for plotting
xc = np.zeros(N + 2)
for i in range(1, N + 1):
    xc[i] = 0.5 * (x[i] + x[i - 1])


# ============================================================
# Initialize
# ============================================================

u = np.zeros(N + 2)
set_initial_condition(u, xc, CASE, N)
apply_boundary_conditions(u, CASE, BC_TYPE, N)

t = 0.0
t_max = final_time(CASE)


# ============================================================
# Time loop
# ============================================================

while t < t_max:
    max_speed = np.max(np.abs(u[1:N + 1]))
    if max_speed < 1e-14:
        dt = t_max - t
    else:
        dt = min(t_max - t, CFL * dx / max_speed)

    F = godunov_flux(u, N)
    u_next = forward_euler(u, F, dt, dx, N)

    u = u_next
    apply_boundary_conditions(u, CASE, BC_TYPE, N)

    t += dt


# ============================================================
# Remove ghost cells for plotting
# ============================================================

xx = np.zeros(N)
uc = np.zeros(N)

for j in range(N):
    xx[j] = xc[j + 1]
    uc[j] = u[j + 1]

u_exact = exact_solution(xx, t_max, CASE)

order = np.argsort(xx)
xx_ordered = xx[order]
uc_ordered = uc[order]
u_exact_ordered = u_exact[order]


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(xx_ordered, uc_ordered, "--r", linewidth=2, label="Godunov")
plt.plot(xx_ordered, u_exact_ordered, "-k", linewidth=2, label="Exact")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Case ({CASE}) at t = {t_max}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()