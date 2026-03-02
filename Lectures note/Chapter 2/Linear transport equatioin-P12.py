import numpy as np
import matplotlib.pyplot as plt

# 参数
a = 1.0
N = 500                 # 网格点数
L = 1.0
dx = L / N
CFL = 1              # 可以随便选
dt = CFL * dx / abs(a)
T = 3                 # 终止时间
# 网格
x = np.linspace(0, L, N, endpoint=False)

# 初值
U = np.sin(2*np.pi*x)

# 时间步数
nsteps = int(T/dt)

# 中心格式推进
for n in range(nsteps):
    U_new = np.zeros_like(U)

    # 周期边界
    for j in range(N):
        jp = (j+1) % N
        jm = (j-1) % N
        U_new[j] = (
            U[j]
            - a*dt/(2*dx)*(U[jp] - U[jm])
        )

    U = U_new

# 精确解
U_exact = np.sin(2*np.pi*(x - a*T))

# 作图
plt.figure(figsize=(8,4))
plt.plot(x, U_exact, label="Exact")
plt.plot(x, U, '--', label="Central scheme")
plt.legend()
plt.title("Central scheme instability")
plt.show()