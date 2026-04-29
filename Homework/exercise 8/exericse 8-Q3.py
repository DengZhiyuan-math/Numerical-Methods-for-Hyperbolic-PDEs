"""
Second-order finite volume scheme for Burgers' equation
    u_t + (u^2/2)_x = 0
using MC reconstruction, Rusanov flux, and SSP-RK2 time stepping.

Cases:
(a) Riemann shock, Neumann-type non-reflecting boundary condition
(b) Riemann rarefaction, Neumann-type non-reflecting boundary condition
(c) smooth periodic data, periodic boundary condition
"""

import numpy as np
import matplotlib.pyplot as plt


def flux(u):
    """Burgers flux f(u)=u^2/2."""
    return 0.5 * u**2


def make_grid(x_min, x_max, N):
    """Uniform finite-volume grid with N cells."""
    dx = (x_max - x_min) / N
    edges = x_min + dx * np.arange(N + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, dx


def initial_cell_average(case, x_left, x_right):
    """Exact initial cell averages for the three test cases."""
    dx = x_right - x_left

    if case == "a":
        # u0=1 for x<0, u0=0 for x>0
        return np.where(
            x_right <= 0.0,
            1.0,
            np.where(x_left >= 0.0, 0.0, (0.0 - x_left) / dx),
        )

    if case == "b":
        # u0=-1 for x<0, u0=1 for x>0
        return np.where(
            x_right <= 0.0,
            -1.0,
            np.where(
                x_left >= 0.0,
                1.0,
                ((0.0 - x_left) * (-1.0) + (x_right - 0.0) * 1.0) / dx,
            ),
        )

    if case == "c":
        # u0=(1+sin(4*pi*x))/2.  We use the exact cell average.
        return 0.5 + (np.cos(4.0 * np.pi * x_left) - np.cos(4.0 * np.pi * x_right)) / (
            8.0 * np.pi * dx
        )

    raise ValueError("case must be 'a', 'b', or 'c'")


def fill_ghost(U, bc, ng=2):
    """Fill ghost cells. We use two ghost cells because MC reconstruction uses neighbors."""
    N = len(U)
    V = np.empty(N + 2 * ng)
    V[ng : ng + N] = U

    if bc == "neumann":
        # Non-reflecting Neumann-type condition: U_0=U_1, U_{N+1}=U_N.
        V[:ng] = U[0]
        V[ng + N :] = U[-1]
    elif bc == "periodic":
        V[:ng] = U[-ng:]
        V[ng + N :] = U[:ng]
    else:
        raise ValueError("bc must be 'neumann' or 'periodic'")

    return V


def minmod3(a, b, c):
    """Three-argument minmod function, vectorized."""
    same_pos = (a > 0.0) & (b > 0.0) & (c > 0.0)
    same_neg = (a < 0.0) & (b < 0.0) & (c < 0.0)

    out = np.zeros_like(a)
    out[same_pos] = np.minimum(np.minimum(a[same_pos], b[same_pos]), c[same_pos])
    out[same_neg] = np.maximum(np.maximum(a[same_neg], b[same_neg]), c[same_neg])
    return out


def mc_slope(V):
    """
    MC limiter in cell-average units:
        sigma_j = minmod(2(U_j-U_{j-1}), (U_{j+1}-U_{j-1})/2, 2(U_{j+1}-U_j)).

    Interface reconstructed values are:
        U_j^- = U_j - sigma_j/2,
        U_j^+ = U_j + sigma_j/2.
    """
    backward = V[1:-1] - V[:-2]
    forward = V[2:] - V[1:-1]
    centered = 0.5 * (V[2:] - V[:-2])

    sigma_inner = minmod3(2.0 * backward, centered, 2.0 * forward)

    sigma = np.zeros_like(V)
    sigma[1:-1] = sigma_inner
    return sigma


def rusanov_flux(uL, uR):
    """
    Rusanov / local Lax-Friedrichs flux:
        F(uL,uR)=1/2(f(uL)+f(uR))-1/2*a*(uR-uL),
    where a=max(|f'(uL)|,|f'(uR)|)=max(|uL|,|uR|) for Burgers.
    """
    a = np.maximum(np.abs(uL), np.abs(uR))
    return 0.5 * (flux(uL) + flux(uR)) - 0.5 * a * (uR - uL)


def rhs(U, dx, bc, ng=2):
    """Semi-discrete finite-volume operator L(U)."""
    V = fill_ghost(U, bc, ng=ng)
    sigma = mc_slope(V)

    # Physical cells are V[ng],...,V[ng+N-1].
    # Required interfaces are between V[ng-1+i] and V[ng+i], i=0,...,N.
    left_cells = np.arange(ng - 1, ng + len(U))
    right_cells = left_cells + 1

    uL = V[left_cells] + 0.5 * sigma[left_cells]
    uR = V[right_cells] - 0.5 * sigma[right_cells]

    F = rusanov_flux(uL, uR)
    return -(F[1:] - F[:-1]) / dx


def ssp_rk2_step(U, dt, dx, bc):
    """Second-order SSP Runge-Kutta step."""
    U1 = U + dt * rhs(U, dx, bc)
    U2 = U1 + dt * rhs(U1, dx, bc)
    return 0.5 * (U + U2)


def solve(case, N=50, t_final=0.1, cfl=0.45):
    """Solve one test case up to t_final."""
    if case in ("a", "b"):
        x_min, x_max, bc = -1.0, 1.0, "neumann"
    elif case == "c":
        x_min, x_max, bc = 0.0, 1.0, "periodic"
    else:
        raise ValueError("case must be 'a', 'b', or 'c'")

    edges, x, dx = make_grid(x_min, x_max, N)
    U = initial_cell_average(case, edges[:-1], edges[1:])

    t = 0.0
    while t < t_final - 1.0e-14:
        max_speed = max(np.max(np.abs(U)), 1.0e-14)  # Burgers: f'(u)=u
        dt = cfl * dx / max_speed
        if t + dt > t_final:
            dt = t_final - t

        U = ssp_rk2_step(U, dt, dx, bc)
        t += dt

    return x, U, dx


def exact_case_a(x, t):
    """Shock solution: speed s=(f(0)-f(1))/(0-1)=1/2."""
    return np.where(x < 0.5 * t, 1.0, 0.0)


def exact_case_b(x, t):
    """Rarefaction solution for left state -1 and right state 1."""
    return np.where(x < -t, -1.0, np.where(x > t, 1.0, x / t))


def phi0_case_c(y):
    """Potential Phi_0 with Phi_0'(y)=(1+sin(4*pi*y))/2."""
    return 0.5 * y - np.cos(4.0 * np.pi * y) / (8.0 * np.pi)


def exact_case_c_hopf_lax(x, t, y_min=-2.0, y_max=3.0, Ny=40000):
    """
    Numerical evaluation of the exact Hopf-Lax variational formula.

    For convex Burgers flux, if Phi_x=u, then
        Phi(x,t)=min_y { Phi_0(y)+(x-y)^2/(2t) },
        u(x,t)=(x-y_*)/t
    at points where the minimizer is unique.

    At shock points the entropy solution has two traces; this sampling picks
    one minimizer, so do not interpret pointwise values exactly at jumps too literally.
    """
    x = np.asarray(x)
    y = np.linspace(y_min, y_max, Ny)
    phi_y = phi0_case_c(y)
    out = np.empty_like(x, dtype=float)

    for i, xi in enumerate(x):
        objective = phi_y + (xi - y) ** 2 / (2.0 * t)
        k = int(np.argmin(objective))
        y_star = y[k]
        out[i] = (xi - y_star) / t

    return out


def exact_solution(case, x, t):
    if case == "a":
        return exact_case_a(x, t)
    if case == "b":
        return exact_case_b(x, t)
    if case == "c":
        return exact_case_c_hopf_lax(x, t)
    raise ValueError("case must be 'a', 'b', or 'c'")


def discrete_l1_error(U, U_exact, dx):
    return dx * np.sum(np.abs(U - U_exact))


def run_all(N=50, cfl=0.45, savefig=True):
    experiments = [("a", 0.1), ("b", 0.1), ("c", 0.3), ("c", 0.6)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, (case, t_final) in zip(axes, experiments):
        x, U, dx = solve(case, N=N, t_final=t_final, cfl=cfl)

        if case in ("a", "b"):
            x_plot = np.linspace(-1.0, 1.0, 1200)
        else:
            x_plot = np.linspace(0.0, 1.0, 1200)

        U_exact_plot = exact_solution(case, x_plot, t_final)
        U_exact_cells = exact_solution(case, x, t_final)
        err = discrete_l1_error(U, U_exact_cells, dx)

        ax.plot(x_plot, U_exact_plot, label="exact", linewidth=2)
        ax.plot(x, U, "o--", label=f"MC-Rusanov, N={N}", markersize=4)
        ax.set_title(f"case ({case}), t={t_final}, L1 error ~ {err:.3e}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True)
        ax.legend()

    if savefig:
        fig.savefig("burgers_mc_rusanov_results.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    run_all()
