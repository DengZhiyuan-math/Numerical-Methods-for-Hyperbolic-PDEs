"""
Second-order finite volume scheme for the linear advection equation

    u_t + u_x = 0,  x in [0, 1].

Spatial discretization:
    Piecewise linear reconstruction with minmod, Superbee, or MC limiter.

Time discretization:
    Second-order SSP Runge--Kutta method.

Boundary conditions:
    (a) free / extrapolation boundary condition,
    (b) periodic boundary condition.

The code compares numerical cell averages with exact cell averages.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Basic limiter utilities
# -----------------------------------------------------------------------------

def minmod_many(*values: float) -> float:
    """Scalar minmod of several numbers."""
    signs = [np.sign(v) for v in values]
    if all(s > 0 for s in signs):
        return min(values)
    if all(s < 0 for s in signs):
        return max(values)
    return 0.0


def maxmod(a: float, b: float) -> float:
    """Scalar maxmod of two numbers."""
    if a > 0.0 and b > 0.0:
        return max(a, b)
    if a < 0.0 and b < 0.0:
        return min(a, b)
    return 0.0


def limited_slope_increment(du_left: float, du_right: float, limiter: str) -> float:
    """
    Return the limited increment sigma_j * dx.

    Here
        du_left  = U_j - U_{j-1},
        du_right = U_{j+1} - U_j.

    The reconstructed edge values are
        U_j^- = U_j - 0.5 * increment,
        U_j^+ = U_j + 0.5 * increment.
    """
    name = limiter.lower()

    if name == "minmod":
        return minmod_many(du_left, du_right)

    if name == "superbee":
        return maxmod(
            minmod_many(2.0 * du_left, du_right),
            minmod_many(du_left, 2.0 * du_right),
        )

    if name == "mc":
        return minmod_many(
            2.0 * du_left,
            0.5 * (du_left + du_right),
            2.0 * du_right,
        )

    raise ValueError(f"Unknown limiter {limiter!r}. Use 'minmod', 'superbee', or 'mc'.")


# -----------------------------------------------------------------------------
# Boundary conditions and reconstruction
# -----------------------------------------------------------------------------

def apply_boundary_conditions(u: np.ndarray, bc: str) -> np.ndarray:
    """
    Add two ghost cells on each side.

    The physical cells are ext[2], ..., ext[N+1].
    Two ghost cells are used so that the boundary ghost cell also has a slope.
    """
    n = u.size
    ext = np.empty(n + 4, dtype=float)
    ext[2 : n + 2] = u

    if bc == "periodic":
        ext[0] = u[-2]
        ext[1] = u[-1]
        ext[n + 2] = u[0]
        ext[n + 3] = u[1]
    elif bc in {"free", "outflow", "extrapolation"}:
        ext[0] = u[0]
        ext[1] = u[0]
        ext[n + 2] = u[-1]
        ext[n + 3] = u[-1]
    else:
        raise ValueError(f"Unknown boundary condition {bc!r}.")

    return ext


def reconstruct_edges(u_ext: np.ndarray, limiter: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise linear reconstruction on the extended array.

    Returns arrays u_minus and u_plus with the same size as u_ext:
        u_minus[i] = left edge value of cell i,
        u_plus[i]  = right edge value of cell i.
    """
    u_minus = u_ext.copy()
    u_plus = u_ext.copy()

    for i in range(1, u_ext.size - 1):
        du_left = u_ext[i] - u_ext[i - 1]
        du_right = u_ext[i + 1] - u_ext[i]
        inc = limited_slope_increment(du_left, du_right, limiter)
        u_minus[i] = u_ext[i] - 0.5 * inc
        u_plus[i] = u_ext[i] + 0.5 * inc

    return u_minus, u_plus


# -----------------------------------------------------------------------------
# Finite volume operator and SSP-RK2 time stepping
# -----------------------------------------------------------------------------

def rhs_advection(u: np.ndarray, dx: float, limiter: str, bc: str, speed: float = 1.0) -> np.ndarray:
    """
    Semi-discrete finite volume operator L(U):

        dU_j/dt = L(U)_j = -(F_{j+1/2} - F_{j-1/2}) / dx.

    For linear advection f(u) = a u with a = speed, the Godunov/upwind flux is
        F(u_L, u_R) = a u_L,  if a >= 0,
                    = a u_R,  if a < 0.
    """
    if speed == 0.0:
        return np.zeros_like(u)

    n = u.size
    u_ext = apply_boundary_conditions(u, bc)
    u_minus, u_plus = reconstruct_edges(u_ext, limiter)

    # Physical interfaces are between extended indices
    # (1,2), (2,3), ..., (N+1,N+2), hence there are N+1 fluxes.
    flux = np.empty(n + 1, dtype=float)
    for j in range(n + 1):
        i_left = j + 1
        i_right = j + 2
        if speed > 0.0:
            flux[j] = speed * u_plus[i_left]
        else:
            flux[j] = speed * u_minus[i_right]

    return -(flux[1:] - flux[:-1]) / dx


def ssp_rk2_step(u: np.ndarray, dt: float, dx: float, limiter: str, bc: str, speed: float = 1.0) -> np.ndarray:
    """Second-order SSP Runge--Kutta step."""
    u_star = u + dt * rhs_advection(u, dx, limiter, bc, speed)
    u_star_star = u_star + dt * rhs_advection(u_star, dx, limiter, bc, speed)
    return 0.5 * (u + u_star_star)


def solve_advection(
    u0: np.ndarray,
    dx: float,
    t_final: float,
    limiter: str,
    bc: str,
    cfl: float = 0.5,
    speed: float = 1.0,
) -> np.ndarray:
    """Integrate the semi-discrete scheme up to exactly t_final."""
    u = u0.copy()
    t = 0.0
    max_speed = abs(speed)

    if max_speed == 0.0:
        return u

    while t < t_final - 1.0e-14:
        dt = cfl * dx / max_speed
        dt = min(dt, t_final - t)
        u = ssp_rk2_step(u, dt, dx, limiter, bc, speed)
        t += dt

    return u


# -----------------------------------------------------------------------------
# Exact cell averages
# -----------------------------------------------------------------------------

def cell_grid(x_left: float, x_right: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return cell centers, left faces, right faces, and dx."""
    dx = (x_right - x_left) / n
    faces_left = x_left + dx * np.arange(n)
    faces_right = faces_left + dx
    centers = 0.5 * (faces_left + faces_right)
    return centers, faces_left, faces_right, dx


def step_cell_average(faces_left: np.ndarray, faces_right: np.ndarray, threshold: float) -> np.ndarray:
    """
    Cell averages of the step function
        u(x) = 2 for x < threshold,
             = 1 for x >= threshold.
    """
    dx = faces_right - faces_left
    avg = np.empty_like(faces_left)

    fully_left = faces_right <= threshold
    fully_right = faces_left >= threshold
    crossing = ~(fully_left | fully_right)

    avg[fully_left] = 2.0
    avg[fully_right] = 1.0
    avg[crossing] = (
        2.0 * (threshold - faces_left[crossing])
        + 1.0 * (faces_right[crossing] - threshold)
    ) / dx[crossing]

    return avg


def sine_cell_average(faces_left: np.ndarray, faces_right: np.ndarray, t: float = 0.0) -> np.ndarray:
    """
    Exact cell averages of sin(4*pi*(x - t)).

    Periodicity is automatic because sine is periodic.
    """
    dx = faces_right - faces_left
    return (
        np.cos(4.0 * np.pi * (faces_left - t))
        - np.cos(4.0 * np.pi * (faces_right - t))
    ) / (4.0 * np.pi * dx)


# -----------------------------------------------------------------------------
# Error and plotting utilities
# -----------------------------------------------------------------------------

def l1_error(u: np.ndarray, u_exact: np.ndarray, dx: float) -> float:
    """Discrete L1 error of cell averages."""
    return dx * np.sum(np.abs(u - u_exact))


def total_variation(u: np.ndarray) -> float:
    """Discrete total variation on the physical cells."""
    return np.sum(np.abs(np.diff(u)))


def plot_comparison(
    x: np.ndarray,
    exact: np.ndarray,
    numerical: dict[str, np.ndarray],
    title: str,
    filename: str,
) -> None:
    """Plot exact and numerical cell averages."""
    plt.figure(figsize=(8, 5))
    plt.plot(x, exact, "k-", linewidth=2.5, label="Exact")

    for limiter, u in numerical.items():
        plt.plot(x, u, marker="o", markersize=3, linewidth=1.5, label=limiter)

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def main() -> None:
    n = 50
    cfl = 0.5
    speed = 1.0
    limiters = ["minmod", "superbee", "mc"]

    # ------------------------------------------------------------------
    # (a) Free / extrapolation boundary condition, discontinuous data
    # ------------------------------------------------------------------
    x, xl, xr, dx = cell_grid(0.0, 1.0, n)
    t_free = 0.5

    u0_free = step_cell_average(xl, xr, threshold=0.1)
    # Exact solution: u(x,t)=u0(x-t), hence the jump is at 0.1 + t.
    exact_free = step_cell_average(xl, xr, threshold=0.1 + t_free)

    numerical_free = {}
    print("Case (a): free / extrapolation boundary condition")
    for limiter in limiters:
        u_num = solve_advection(u0_free, dx, t_free, limiter, bc="free", cfl=cfl, speed=speed)
        numerical_free[limiter] = u_num
        print(
            f"  {limiter:8s}  L1 error = {l1_error(u_num, exact_free, dx):.6e}, "
            f"TV = {total_variation(u_num):.6e}"
        )

    plot_comparison(
        x,
        exact_free,
        numerical_free,
        title="Linear advection, free boundary, discontinuous data, N=50, t=0.5",
        filename="advection_free_limiters.png",
    )

    # ------------------------------------------------------------------
    # (b) Periodic boundary condition, smooth data
    # ------------------------------------------------------------------
    # The exercise statement does not specify a final time for (b).  We use
    # t=0.5 to match part (a).  Change this number if another final time is
    # required by the instructor.
    t_periodic = 0.5

    u0_periodic = sine_cell_average(xl, xr, t=0.0)
    exact_periodic = sine_cell_average(xl, xr, t=t_periodic)

    numerical_periodic = {}
    print("\nCase (b): periodic boundary condition")
    for limiter in limiters:
        u_num = solve_advection(u0_periodic, dx, t_periodic, limiter, bc="periodic", cfl=cfl, speed=speed)
        numerical_periodic[limiter] = u_num
        print(
            f"  {limiter:8s}  L1 error = {l1_error(u_num, exact_periodic, dx):.6e}, "
            f"TV = {total_variation(u_num):.6e}"
        )

    plot_comparison(
        x,
        exact_periodic,
        numerical_periodic,
        title="Linear advection, periodic boundary, sine data, N=50, t=0.5",
        filename="advection_periodic_limiters.png",
    )

    print("\nSaved figures:")
    print("  advection_free_limiters.png")
    print("  advection_periodic_limiters.png")


if __name__ == "__main__":
    main()
