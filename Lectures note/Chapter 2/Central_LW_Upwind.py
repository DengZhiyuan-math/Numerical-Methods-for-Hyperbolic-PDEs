import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数
# =========================
a = 1.0
N = 500
L = 1.0
dx = L / N
CFL = 0.8
dt = CFL * dx / abs(a)
T = 1.0
x = np.linspace(0, L, N, endpoint=False)
nsteps = int(T / dt)

# =========================
# 初值
# =========================
U0 = np.sin(2*np.pi*x)

U_central = U0.copy()
U_upwind = U0.copy()
U_lw = U0.copy()

lam = a * dt / dx

# =========================
# 时间推进
# =========================
for n in range(nsteps):

    Uc_new = np.zeros_like(U_central)
    Uu_new = np.zeros_like(U_upwind)
    Ulw_new = np.zeros_like(U_lw)

    for j in range(N):
        jp = (j+1) % N
        jm = (j-1) % N

        # Central
        Uc_new[j] = (
            U_central[j]
            - lam/2 * (U_central[jp] - U_central[jm])
        )

        # Upwind
        if a > 0:
            Uu_new[j] = (
                U_upwind[j]
                - lam * (U_upwind[j] - U_upwind[jm])
            )
        else:
            Uu_new[j] = (
                U_upwind[j]
                - lam * (U_upwind[jp] - U_upwind[j])
            )

        # Lax-Wendroff
        Ulw_new[j] = (
            U_lw[j]
            - lam/2 * (U_lw[jp] - U_lw[jm])
            + lam**2/2 * (U_lw[jp] - 2*U_lw[j] + U_lw[jm])
        )

    U_central = Uc_new
    U_upwind = Uu_new
    U_lw = Ulw_new

U_exact = np.sin(2*np.pi*(x - a*T))

# =========================
# 分开画三张
# =========================
fig, axs = plt.subplots(3, 1, figsize=(8,10), sharex=True)

axs[0].plot(x, U_exact, 'k', label="Exact")
axs[0].plot(x, U_central, 'r--', label="Central")
axs[0].set_title("Central scheme")
axs[0].legend()

axs[1].plot(x, U_exact, 'k')
axs[1].plot(x, U_upwind, 'b--')
axs[1].set_title("Upwind scheme")

axs[2].plot(x, U_exact, 'k')
axs[2].plot(x, U_lw, 'g--')
axs[2].set_title("Lax-Wendroff scheme")

plt.tight_layout()
plt.show()