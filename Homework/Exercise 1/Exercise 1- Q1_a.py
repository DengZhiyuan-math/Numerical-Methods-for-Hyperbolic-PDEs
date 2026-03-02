import numpy as np
import matplotlib.pyplot as plt

# 定义解函数
def u(x, t):
    return np.exp((x - (8/3)*t)**2)

# x 取值范围
x = np.linspace(-0.5, 0.5, 400)

# 选几个时间
times = [0, 0.5, 1.0]

plt.figure(figsize=(8,5))

for t in times:
    plt.plot(x, u(x, t), label=f"t = {t}")

plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.yscale("log")
plt.title("Solution of u_t + (8/3)u_x = 0")
plt.grid(True)
plt.show()