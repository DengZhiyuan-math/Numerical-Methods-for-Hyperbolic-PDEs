import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 演示说明
# ============================================================
# 目标：
#   演示“相邻局部 Riemann 波在一个时间步内发生相互作用”这件事。
#
# 方程：
#   Burgers 方程
#       u_t + (u^2 / 2)_x = 0
#
# 初值：
#       u(x,0) = 1,    x < 0
#               -1,   0 < x < 1
#                1,    x > 1
#
# 于是有两个相邻界面：
#
#   (1) 左界面 x = 0:
#       左右状态为 (1, -1)
#       这产生一个 shock，其速度为
#           s = [f(1)-f(-1)] / [1-(-1)] = 0
#       所以 shock 轨迹就是
#           x = 0
#
#   (2) 右界面 x = 1:
#       左右状态为 (-1, 1)
#       这产生一个 rarefaction，其左右边界速度分别为 -1 和 1
#       所以 rarefaction 扇形边界为
#           x = 1 - t
#           x = 1 + t
#
# 关键观察：
#   右界面发出的 rarefaction 左边界 x = 1 - t 向左传播。
#   当它到达 x = 0 时，说明右界面发出的波已经跑到了左界面，
#   也就是两个相邻局部 Riemann 问题已经开始相互作用。
#
# 由
#       1 - t = 0
# 得到第一次相互作用时刻
#       t_interact = 1
#
# 因此：
#   - 若 dt < 1，则下一时层前还没有发生相互作用
#   - 若 dt > 1，则在一个时间步内已经发生相互作用
#
# 这正是 CFL 几何意义的一部分：
#   dt 不能大到让局部波在一个时间步内跑到相邻界面。
# ============================================================


# ----------------------------
# 参数设置
# ----------------------------
t_max = 1.6
t_interact = 1.0

# 一个“安全”的时间层，一个“过大”的时间层
t_safe = 0.45
t_bad = 1.20

t = np.linspace(0.0, t_max, 600)

# ----------------------------
# 三条关键轨迹
# ----------------------------
# 左界面 shock
x_shock = np.zeros_like(t)

# 右界面 rarefaction 左右边界
x_rare_left = 1.0 - t
x_rare_right = 1.0 + t

# 两个时间层上对应的位置
x_safe_left = 1.0 - t_safe
x_safe_right = 1.0 + t_safe

x_bad_left = 1.0 - t_bad
x_bad_right = 1.0 + t_bad


# ============================================================
# 图 1：全局时空图
# ============================================================
plt.figure(figsize=(10, 7))

# 画三条传播轨迹
plt.plot(x_shock, t, linewidth=3, label="shock from x=0: x = 0")
plt.plot(x_rare_left, t, linewidth=3, label="left edge of rarefaction from x=1: x = 1 - t")
plt.plot(x_rare_right, t, linewidth=3, label="right edge of rarefaction from x=1: x = 1 + t")

# 画两个界面
plt.axvline(0.0, linestyle="--", linewidth=1.5, color="gray")
plt.axvline(1.0, linestyle="--", linewidth=1.5, color="gray")

# 画三个关键时间层
plt.axhline(t_safe, linestyle=":", linewidth=2.5, label=f"safe time layer: t = {t_safe}")
plt.axhline(t_interact, linestyle="--", linewidth=2.5, label="first interaction time: t = 1")
plt.axhline(t_bad, linestyle=":", linewidth=2.5, label=f"too large time layer: t = {t_bad}")

# 标出安全时间层的关键点
plt.scatter([0.0, x_safe_left, x_safe_right], [t_safe, t_safe, t_safe], s=70)
plt.text(
    0.05, t_safe + 0.035,
    f"safe layer:\nleft edge at x = {x_safe_left:.2f} > 0,\nso it has NOT reached x=0",
    fontsize=10
)

# 标出第一次相互作用点
plt.scatter([0.0], [t_interact], s=90)
plt.text(
    0.05, t_interact + 0.04,
    "first interaction:\nx = 1 - t first reaches x = 0",
    fontsize=10
)

# 标出过大时间层的关键点
plt.scatter([0.0, x_bad_left, x_bad_right], [t_bad, t_bad, t_bad], s=70)
plt.text(
    0.15, t_bad + 0.04,
    f"large layer:\nleft edge at x = {x_bad_left:.2f} < 0,\nso it has already crossed x=0",
    fontsize=10
)

plt.xlim(-1.1, 2.7)
plt.ylim(0.0, t_max)
plt.xlabel("x", fontsize=12)
plt.ylabel("t", fontsize=12)
plt.title("Space-time picture: adjacent local Riemann waves start interacting", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()


# ============================================================
# 图 2：放大关键区域
# ============================================================
plt.figure(figsize=(10, 7))

plt.plot(x_shock, t, linewidth=3, label="shock: x = 0")
plt.plot(x_rare_left, t, linewidth=3, label="rarefaction left edge: x = 1 - t")

plt.axvline(0.0, linestyle="--", linewidth=1.5, label="left interface x = 0")
plt.axhline(t_safe, linestyle=":", linewidth=2.5, label=f"safe layer t = {t_safe}")
plt.axhline(t_interact, linestyle="--", linewidth=2.5, label="interaction starts at t = 1")
plt.axhline(t_bad, linestyle=":", linewidth=2.5, label=f"large layer t = {t_bad}")

plt.scatter([x_safe_left], [t_safe], s=90)
plt.scatter([0.0], [t_interact], s=90)
plt.scatter([x_bad_left], [t_bad], s=90)

plt.annotate(
    "still on the right of x=0",
    xy=(x_safe_left, t_safe),
    xytext=(0.33, 0.28),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=11
)

plt.annotate(
    "first reaches x=0",
    xy=(0.0, t_interact),
    xytext=(0.30, 0.92),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=11
)

plt.annotate(
    "already crossed to the left side",
    xy=(x_bad_left, t_bad),
    xytext=(-0.55, 1.33),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=11
)

plt.xlim(-0.9, 1.1)
plt.ylim(0.2, 1.45)
plt.xlabel("x", fontsize=12)
plt.ylabel("t", fontsize=12)
plt.title("Zoomed view near the first interaction point", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()