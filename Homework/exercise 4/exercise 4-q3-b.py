import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Problem setup
# ============================================================
A = -1.0
B = 1.0
N = 50
DX = (B - A) / N
X = A + (np.arange(1, N + 1) - 0.5) * DX   # cell centers
CFL = 0.9
T_FINAL = 0.5


# ============================================================
# Numerical fluxes for Burgers' equation: f(u) = u^2 / 2
# ============================================================
def flux_godunov(a, b):
    return np.maximum(0.5 * np.maximum(a, 0.0)**2,
                      0.5 * np.minimum(b, 0.0)**2)


def flux_roe(a, b):
    return np.where(a + b >= 0.0, 0.5 * a**2, 0.5 * b**2)


def flux_rusanov(a, b):
    return 0.25 * (a**2 + b**2) - 0.5 * np.maximum(np.abs(a), np.abs(b)) * (b - a)


# ============================================================
# Initial condition for part (b)
#   u(x,0) = sin(4*pi*x)
# Use exact cell averages
# ============================================================
def initial_data():
    x_left = A + np.arange(N) * DX
    x_right = x_left + DX
    return (np.cos(4.0 * np.pi * x_left) - np.cos(4.0 * np.pi * x_right)) / (4.0 * np.pi * DX)


# ============================================================
# Boundary conditions: U_0 = U_1, U_{N+1} = U_N
# ============================================================
def apply_bc(U):
    U[0] = U[1]
    U[-1] = U[-2]


# ============================================================
# Finite-volume solver
# ============================================================
def solve(flux):
    U = np.zeros(N + 2)          # includes ghost cells
    U[1:-1] = initial_data()
    t = 0.0

    while t < T_FINAL - 1e-14:
        apply_bc(U)

        max_speed = np.max(np.abs(U[1:-1]))
        if max_speed < 1e-14:
            dt = T_FINAL - t
        else:
            dt = CFL * DX / max_speed
            if t + dt > T_FINAL:
                dt = T_FINAL - t

        F = flux(U[:-1], U[1:])   # interface fluxes
        U[1:-1] = U[1:-1] - (dt / DX) * (F[1:] - F[:-1])

        t += dt

    return U[1:-1]


# ============================================================
# Run all three schemes
# ============================================================
U_god = solve(flux_godunov)
U_roe = solve(flux_roe)
U_rus = solve(flux_rusanov)


# ============================================================
# Print representative values
# ============================================================
sample_points = np.array([-0.82, -0.42, -0.02, 0.38, 0.78])
indices = [np.argmin(np.abs(X - s)) for s in sample_points]

print("Representative values for Exercise 3(b), t = 0.5")
print("x =", X[indices])
print("Godunov =", U_god[indices])
print("Roe     =", U_roe[indices])
print("Rusanov =", U_rus[indices])


# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(X, U_god, 'o-', markersize=3, label='Godunov')
plt.plot(X, U_roe, 's-', markersize=3, label='Roe')
plt.plot(X, U_rus, '^-', markersize=3, label='Rusanov')
plt.xlabel("x")
plt.ylabel("u(x,0.5)")
plt.title("Exercise 3(b): numerical solutions at t = 0.5")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()