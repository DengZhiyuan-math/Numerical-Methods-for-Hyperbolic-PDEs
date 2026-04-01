import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x):
    return np.sin(4.0 * np.pi * x)


def exact_solution(x, t, a):
    return np.sin(4.0 * np.pi * (x - a * t))


def godunov_flux(uL, uR, a):
    """
    Godunov flux for linear advection f(u)=a u.
    For linear flux, this is exactly the upwind flux:
        F(uL,uR) = a^+ uL + a^- uR
    """
    a_plus = max(a, 0.0)
    a_minus = min(a, 0.0)
    return a_plus * uL + a_minus * uR


def godunov_step(u, a, dx, dt):
    """
    One time step of the Godunov finite volume scheme
    with periodic boundary conditions.
    """
    flux_right = godunov_flux(u, np.roll(u, -1), a)
    return u - (dt / dx) * (flux_right - np.roll(flux_right, 1))


def upwind_step(u, a, dx, dt):
    """
    One time step of the upwind scheme with periodic boundary conditions.
    """
    nu = a * dt / dx
    if a >= 0.0:
        return u - nu * (u - np.roll(u, 1))
    else:
        return u - nu * (np.roll(u, -1) - u)


def run_scheme(stepper, N, a=1.0, T=10.0, cfl=0.9):
    """
    Run a scheme up to time T on [0,1] with periodic boundary conditions.
    We use cell centers x_j = (j+1/2) dx.
    """
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx

    u = initial_condition(x).copy()

    # Choose dt from the CFL condition and then adjust so that nsteps*dt = T exactly
    if a == 0.0:
        dt = T
        nsteps = 1
    else:
        dt_cfl = cfl * dx / abs(a)
        nsteps = int(np.ceil(T / dt_cfl))
        dt = T / nsteps

    for _ in range(nsteps):
        u = stepper(u, a, dx, dt)

    return x, u, dt, nsteps


def main():
    a = 1.0
    T = 10.0
    Ns = [40, 80, 200]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, N in zip(axes, Ns):
        x, u_god, dt, nsteps = run_scheme(godunov_step, N, a=a, T=T, cfl=0.9)
        _, u_upw, _, _ = run_scheme(upwind_step, N, a=a, T=T, cfl=0.9)
        u_ex = exact_solution(x, T, a)

        # Check that Godunov and upwind agree numerically
        diff = np.max(np.abs(u_god - u_upw))
        print(f"N = {N:3d}, steps = {nsteps:5d}, dt = {dt:.6e}, "
              f"max |Godunov - Upwind| = {diff:.3e}")

        ax.plot(x, u_ex, label="Exact", linewidth=2)
        ax.plot(x, u_god, "--", label="Godunov / Upwind", linewidth=2)
        ax.set_title(f"N = {N}")
        ax.set_xlabel("x")
        ax.grid(True)

    axes[0].set_ylabel("u(x,10)")
    axes[0].legend()
    fig.suptitle(r"Linear advection: $u_t + a u_x = 0$,  $u_0(x)=\sin(4\pi x)$,  $t=10$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()