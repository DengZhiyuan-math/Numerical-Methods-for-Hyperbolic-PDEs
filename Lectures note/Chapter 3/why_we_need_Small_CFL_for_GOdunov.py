import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 演示目标
# ============================================================
# 这个脚本独立演示：在 Godunov / 有限体积方法的几何解释中，
# 如果一个时间步 dt 取得过大，那么“相邻界面”发出的局部 Riemann 波
# 可能会在下一时层之前进入彼此的影响区域，从而破坏
# “每个界面在这个时间步内都可以被看成独立 Riemann 问题”的假设。
#
# 我们考虑 Burgers 方程
#     u_t + (u^2 / 2)_x = 0
# 其特征速度为
#     f'(u) = u .
#
# 下面使用一个最容易观察的三段常值初值：
#     u(x,0) = 1,   x < 0
#              -1,  0 < x < 1
#               1,   x > 1
#
# 于是有两个相邻界面：
#   1) x = 0 处，左状态(1) 右状态(-1)，产生一个 shock；
#      shock 速度
#          s = [f(1)-f(-1)] / [1-(-1)] = 0
#      所以 shock 轨迹是 x = 0 .
#
#   2) x = 1 处，左状态(-1) 右状态(1)，产生一个 rarefaction；
#      其左右边界速度分别为 -1 和 1，
#      所以 rarefaction 扇形边界为
#          x = 1 - t,   x = 1 + t .
#
# 关键观察：
#   右侧 rarefaction 的左边界 x = 1 - t 会向左移动。
#   当它移动到 x = 0 时，就说明右界面发出的波已经到达左界面位置，
#   也就是两个相邻局部 Riemann 问题已经发生相互作用。
#
#   令 1 - t = 0，可得第一次相互作用时间为
#          t_interact = 1 .
#
# 因此：
#   - 若 dt < 1，则下一时层前尚未发生相互作用；
#   - 若 dt > 1，则在一个时间步内已经发生相互作用。
#
# 这正是 CFL 条件背后的几何意义之一：
#   时间步不能大到让局部波在一个时间步内跑到相邻界面。
# ============================================================


# ----------------------------
# 基本参数
# ----------------------------
dx = 1.0                    # 相邻两个界面的距离
t_max = 1.6                 # 图像时间上限
t_interact = 1.0            # 第一次相互作用时刻：1 - t = 0

# 取两个时间层：一个安全，一个过大
t_safe = 0.45               # 尚未发生相互作用
t_bad = 1.20                # 已经发生相互作用

# 连续时间变量，用来画时空轨迹
t = np.linspace(0.0, t_max, 500)

# ----------------------------
# 三条关键轨迹
# ----------------------------
# 左界面 shock: x = 0
x_shock = np.zeros_like(t)

# 右界面 rarefaction 左、右边界
x_rare_left = 1.0 - t
x_rare_right = 1.0 + t

# 在两个时间层上的对应位置
x_safe_left = 1.0 - t_safe
x_safe_right = 1.0 + t_safe

x_bad_left = 1.0 - t_bad
x_bad_right = 1.0 + t_bad

# ============================================================
# 图 1：主图 —— 直接看“什么时候进入相邻界面”
# ============================================================
plt.figure(figsize=(10, 7))

# 轨迹
plt.plot(x_shock, t, linewidth=3, label="shock from x=0: x = 0")
plt.plot(x_rare_left, t, linewidth=3, label="left edge of rarefaction from x=1: x = 1 - t")
plt.plot(x_rare_right, t, linewidth=3, label="right edge of rarefaction from x=1: x = 1 + t")

# 两个界面位置
plt.axvline(0.0, linestyle="--", linewidth=1.5)
plt.axvline(1.0, linestyle="--", linewidth=1.5)

# 三个关键时间层
plt.axhline(t_safe, linestyle=":", linewidth=2.5, label=f"safe time layer: t = {t_safe}")
plt.axhline(t_interact, linestyle="--", linewidth=2.5, label="first interaction time: t = 1")
plt.axhline(t_bad, linestyle=":", linewidth=2.5, label=f"too large time layer: t = {t_bad}")

# 标记安全时间层上的波前位置
plt.scatter([0.0, x_safe_left, x_safe_right], [t_safe, t_safe, t_safe], s=70)
plt.text(0.05, t_safe + 0.03,
         f"safe layer: rarefaction left edge at x = {x_safe_left:.2f} > 0,\n"
         "so it has NOT reached the left interface yet",
         fontsize=10)

# 标记过大时间层上的波前位置
plt.scatter([0.0, x_bad_left, x_bad_right], [t_bad, t_bad, t_bad], s=70)
plt.text(0.08, t_bad + 0.03,
         f"large layer: rarefaction left edge at x = {x_bad_left:.2f} < 0,\n"
         "so it has already crossed the neighboring interface",
         fontsize=10)

# 交点
plt.scatter([0.0], [t_interact], s=90)
plt.text(0.05, t_interact + 0.04,
         "here x = 1 - t first reaches x = 0\n=> first interaction occurs",
         fontsize=10)

plt.xlim(-1.1, 2.7)
plt.ylim(0.0, t_max)
plt.xlabel("x", fontsize=12)
plt.ylabel("t", fontsize=12)
plt.title("Space-time picture: a too-large dt makes adjacent local Riemann waves interact", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()


# ============================================================
# 图 2：放大关键区域 —— 更容易观察 crossing
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

plt.annotate("still on the right of x=0",
             xy=(x_safe_left, t_safe),
             xytext=(0.35, 0.28),
             arrowprops=dict(arrowstyle="->", lw=1.5),
             fontsize=11)

plt.annotate("first reaches x=0",
             xy=(0.0, t_interact),
             xytext=(0.30, 0.92),
             arrowprops=dict(arrowstyle="->", lw=1.5),
             fontsize=11)

plt.annotate("already crossed to the left side",
             xy=(x_bad_left, t_bad),
             xytext=(-0.55, 1.33),
             arrowprops=dict(arrowstyle="->", lw=1.5),
             fontsize=11)

plt.xlim(-0.9, 1.1)
plt.ylim(0.2, 1.45)
plt.xlabel("x", fontsize=12)
plt.ylabel("t", fontsize=12)
plt.title("Zoomed view near the interaction point", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
