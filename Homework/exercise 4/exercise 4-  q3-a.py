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
T_FINAL = 1.0


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
# Initial condition for part (a)
#   u(x,0) = -1 for x<0, 1 for x>0
# ============================================================
def initial_data():
    return np.where(X < 0.0, -1.0, 1.0)


# ============================================================
# Exact solution for part (a)
# ============================================================
def exact_solution(x, t):
    u = np.empty_like(x)
    u[x < -t] = -1.0
    u[x > t] = 1.0
    mask = (x >= -t) & (x <= t)
    u[mask] = x[mask] / t
    return u


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

        F = flux(U[:-1], U[1:])   # interface fluxes, length N+1
        U[1:-1] = U[1:-1] - (dt / DX) * (F[1:] - F[:-1])

        t += dt

    return U[1:-1]


# ============================================================
# Run all three schemes
# ============================================================
U_god = solve(flux_godunov)
U_roe = solve(flux_roe)
U_rus = solve(flux_rusanov)
U_exact = exact_solution(X, T_FINAL)


# ============================================================
# Print representative values
# ============================================================
sample_points = np.array([-0.82, -0.42, -0.02, 0.38, 0.78])
indices = [np.argmin(np.abs(X - s)) for s in sample_points]

print("Representative values for Exercise 3(a), t = 1")
print("x =", X[indices])
print("Godunov =", U_god[indices])
print("Roe     =", U_roe[indices])
print("Rusanov =", U_rus[indices])

print("\nMean absolute errors against exact solution:")
print("Godunov =", np.mean(np.abs(U_god - U_exact)))
print("Roe     =", np.mean(np.abs(U_roe - U_exact)))
print("Rusanov =", np.mean(np.abs(U_rus - U_exact)))


# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(X, U_exact, 'k-', linewidth=2, label='Exact')
plt.plot(X, U_god, 'o-', markersize=3, label='Godunov')
plt.plot(X, U_roe, 's-', markersize=3, label='Roe')
plt.plot(X, U_rus, '^-', markersize=3, label='Rusanov')
plt.xlabel("x")
plt.ylabel("u(x,1)")
plt.title("Exercise 3(a): numerical solutions at t = 1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()