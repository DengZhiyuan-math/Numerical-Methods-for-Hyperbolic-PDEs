import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x: np.ndarray) -> np.ndarray:
    """
    Initial data:
        u0(x) = 1 + cos(4*pi*(x - 1/2))
    """
    return 1.0 + np.cos(4.0 * np.pi * (x - 0.5))


def conservative_step(u: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """
    Conservative scheme:
        U_j^{n+1} = U_j^n - dt/(2dx) * ( (U_j^n)^2 - (U_{j-1}^n)^2 )

    Periodic boundary conditions are handled with np.roll.
    """
    u_left = np.roll(u, 1)
    return u - (dt / (2.0 * dx)) * (u**2 - u_left**2)


def nonconservative_step(u: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """
    Non-conservative scheme:
        U_j^{n+1} = U_j^n - dt/dx * U_j^n * (U_j^n - U_{j-1}^n)

    Periodic boundary conditions are handled with np.roll.
    """
    u_left = np.roll(u, 1)
    return u - (dt / dx) * u * (u - u_left)


def compute_dt(u: np.ndarray, dx: float, cfl: float) -> float:
    """
    Burgers characteristic speed is |u|.
    Use a small safeguard to avoid division by zero.
    """
    max_speed = max(np.max(np.abs(u)), 1e-14)
    return cfl * dx / max_speed


def run_scheme(
    step_function,
    N: int,
    T: float,
    cfl: float = 0.4,
    domain=(0.0, 1.0),
):
    """
    Run one scheme up to time T on a periodic grid.
    """
    a, b = domain
    dx = (b - a) / N
    x = a + (np.arange(N) + 0.5) * dx   # cell centers

    u = initial_condition(x)
    t = 0.0

    mass_history = [dx * np.sum(u)]
    time_history = [t]

    while t < T:
        dt = compute_dt(u, dx, cfl)
        if t + dt > T:
            dt = T - t

        u = step_function(u, dt, dx)
        t += dt

        mass_history.append(dx * np.sum(u))
        time_history.append(t)

    return x, u, np.array(time_history), np.array(mass_history)


def main():
    # You can change these parameters
    N = 1600
    T = 0.3
    cfl = 0.4

    x0 = (np.arange(N) + 0.5) / N
    u_init = initial_condition(x0)

    x_cons, u_cons, t_cons, m_cons = run_scheme(
        conservative_step, N=N, T=T, cfl=cfl
    )
    x_nonc, u_nonc, t_nonc, m_nonc = run_scheme(
        nonconservative_step, N=N, T=T, cfl=cfl
    )

    mass0 = np.mean([m_cons[0], m_nonc[0]])

    print(f"Initial mass                 = {mass0:.12f}")
    print(f"Final mass (conservative)    = {m_cons[-1]:.12f}")
    print(f"Final mass (non-conservative)= {m_nonc[-1]:.12f}")
    print(f"Mass error (conservative)    = {abs(m_cons[-1] - mass0):.12e}")
    print(f"Mass error (non-conservative)= {abs(m_nonc[-1] - mass0):.12e}")
    print(f"L_inf difference at T        = {np.max(np.abs(u_cons - u_nonc)):.12e}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

    # Top plot: solution profiles
    axes[0].plot(x0, u_init, label="Initial condition", linewidth=2)
    axes[0].plot(x_cons, u_cons, label="Conservative scheme", linewidth=2)
    axes[0].plot(x_nonc, u_nonc, "--", label="Non-conservative scheme", linewidth=2)
    axes[0].set_title(f"Burgers equation at T = {T}, N = {N}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].grid(True)
    axes[0].legend()

    # Bottom plot: mass history
    axes[1].plot(t_cons, m_cons, label="Conservative mass", linewidth=2)
    axes[1].plot(t_nonc, m_nonc, "--", label="Non-conservative mass", linewidth=2)
    axes[1].axhline(mass0, color="k", linestyle=":", linewidth=1.5, label="Initial mass")
    axes[1].set_title("Discrete mass over time")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Δx Σ_j U_j")
    axes[1].grid(True)
    axes[1].legend()

    plt.show()


if __name__ == "__main__":
    main()