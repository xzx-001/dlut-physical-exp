#!/usr/bin/env python3
"""小倾角夹持几何示意（过孔轴的剖面，单幅）

本实验的插入角实际未测，但只有 θ<45° 的一支与“光纤不被夹断”相容：
ε_b≤0.2% ⟺ 孔/纤直径比 x≤1.28 ⟺ θ≤8.8°。本图按 x=1.15（θ≈4.9°）作图。

几何：孔径 d_h=2R_h、纤径 d_f=2r_f、板厚 2T；光纤与孔轴成 θ 角斜穿，
与上孔口棱边 A、下孔口棱边 B 两点相切，接触处有法向力 N 与沿纤轴的摩擦 f。

用法：python3 scripts/plot_clamp.py [--figures DIR]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

_ap = argparse.ArgumentParser(description="小倾角夹持几何示意图")
_ap.add_argument("--figures", default=None)
_ap.add_argument("--theta", type=float, default=4.95, help="倾角/度")
_ap.add_argument("--radius", type=float, default=1.15, help="孔径/纤径比 x=R_h/r_f")
_args, _ = _ap.parse_known_args()

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
FIG = _args.figures or os.path.join(REPO, "output", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False, "font.size": 9.0, "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "legend.frameon": False,
})

# ==================== 几何参数 ====================
R_FIB = 0.50                      # 光纤半径（示意单位）
X_RATIO = _args.radius            # R_h / r_f
R_HOLE = X_RATIO * R_FIB
THETA = np.deg2rad(_args.theta)   # 纤轴与孔轴夹角
T_HALF = 0.85                     # 板半厚
S_HALF = 1.45                     # 画幅内板料半宽
U = np.array([np.sin(THETA), -np.cos(THETA)])   # 纤轴方向 (x, z)
N = np.array([np.cos(THETA), np.sin(THETA)])    # 截面内垂直于纤轴

assert T_HALF <= (R_HOLE - R_FIB) / np.tan(THETA) + 1e-9, (
    "板太厚：θ=%.1f° 时要求半厚 ≤ %.2f，当前 %.2f"
    % (_args.theta, (R_HOLE - R_FIB) / np.tan(THETA), T_HALF))

# 接触点：上下孔口棱边处离纤轴最近的点（本截面内即 x=∓R_h, z=±T）
P_A = np.array([-R_HOLE, +T_HALF])
P_B = np.array([+R_HOLE, -T_HALF])
for P, tag in ((P_A, "A"), (P_B, "B")):
    d = abs(U[0] * P[1] - U[1] * P[0])   # 二维叉积
    assert abs(d - R_FIB) < 0.02, "%s 点距纤轴 %.3f，应等于 r_f=%.2f" % (tag, d, R_FIB)
L_C = float(np.linalg.norm(P_B - P_A))

# ==================== 作图 ====================
fig, ax = plt.subplots(figsize=(4.3, 3.6))

for x0, x1 in ((-S_HALF, -R_HOLE), (R_HOLE, S_HALF)):
    ax.add_patch(Rectangle((x0, -T_HALF), x1 - x0, 2 * T_HALF,
                           facecolor="#ece9e3", edgecolor="#4a4a4a", lw=0.9,
                           hatch="////", zorder=3))

L_ROD = 2.4
p_hi, p_lo = +L_ROD * U, -L_ROD * U
rod = [p_hi + R_FIB * N, p_lo + R_FIB * N, p_lo - R_FIB * N, p_hi - R_FIB * N]
ax.add_patch(Polygon(rod, closed=True, facecolor="#f2914a",
                     edgecolor="#a34700", lw=1.0, zorder=5))

ax.plot([0, 0], [-1.25, 1.15], ls=(0, (5, 3)), color="#1f4e79", lw=1.0, zorder=2)
ax.plot([0], [0], "o", ms=3.6, mfc="white", mec="#1f4e79", mew=1.0, zorder=7)

for P, nm, off in ((P_A, "A", (-0.17, 0.11)), (P_B, "B", (0.13, -0.14))):
    ax.plot([P[0]], [P[1]], "o", ms=6.0, mfc="#c00000", mec="white", mew=0.9, zorder=9)
    ax.annotate(nm, P, xytext=P + np.array(off), fontsize=10.5, color="#c00000",
                zorder=9, arrowprops=dict(arrowstyle="-", lw=0.8, color="#c00000"))

for P, sgn in ((P_A, +1), (P_B, -1)):
    n_in = sgn * N
    tip = 0.30
    ax.annotate("", xy=P + tip * n_in, xytext=P,
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#1f4e79", shrinkA=2), zorder=8)
    ax.text(*(P + (tip + 0.17) * n_in), "$N$", color="#1f4e79", fontsize=9.5,
            ha="center", va="center", zorder=9)

for P in (P_A, P_B):
    ax.annotate("", xy=P - 0.40 * U, xytext=P,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#2e7d32", shrinkA=2), zorder=8)
ax.text(*(P_A - 0.52 * U + np.array([0.15, 0.03])), "$f$", color="#2e7d32",
        fontsize=9.5, zorder=9)

ax.annotate("", xy=P_A, xytext=P_B,
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.4", shrinkA=10, shrinkB=10))
ax.text(*(0.5 * (P_A + P_B) + np.array([0.22, 0.0])), "$L_c$", fontsize=9.5, color="0.25")

# 倾角 θ：以孔中心为顶点（角弧放大 8 倍以便看清）
th = np.linspace(-np.pi / 2, -np.pi / 2 + 8 * THETA, 120)
ax.plot(0.62 * np.cos(th), 0.62 * np.sin(th), "-", color="0.3", lw=1.1, zorder=8)
ax.text(0.26, -0.70, r"$\theta\approx5^\circ$", fontsize=9.0, color="0.2")
ax.text(-0.10, -1.08, "孔轴", fontsize=8.0, color="#1f4e79", ha="right")

ax.annotate("", xy=(-R_HOLE, 0.99), xytext=(R_HOLE, 0.99),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.45"))
ax.text(0, 1.05, r"孔径 $d_h=2R_h$", fontsize=8.0, color="0.3", ha="center")
ax.annotate("", xy=(-1.30, -T_HALF), xytext=(-1.30, T_HALF),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.45"))
ax.text(-1.34, 0, "板厚 $2T$", fontsize=8.0, color="0.3", ha="right",
        va="center", rotation=90)

ax.text(-0.40, 0.28, "孔", fontsize=8.0, color="0.45", ha="center", va="center", zorder=4)
ax.text(R_HOLE + 0.09, -0.52, "夹持件\n（剖面）", fontsize=8.0, color="0.28",
        ha="left", va="center", zorder=4)
ax.annotate("光纤", xy=(-0.04, 0.92), xytext=(0.52, 1.02), fontsize=8.6, color="#a34700",
            ha="left", va="center", zorder=9,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#a34700"))

ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.22, 1.18)
ax.set_aspect("equal")
ax.set_axis_off()

EPS_B = (X_RATIO - 1) * np.sin(THETA) ** 2 / (2 * X_RATIO ** 2)
print("θ = %.2f°, x = R_h/r_f = %.3f (R_h = %.3f)，r_f = %.2f" %
      (_args.theta, X_RATIO, R_HOLE, R_FIB))
print("穿出判据：半厚 ≤ (R_h − r_f)/tanθ = %.2f，实际 %.2f → 通过" %
      ((R_HOLE - R_FIB) / np.tan(THETA), T_HALF))
print("接触点 A=(%.3f,%.3f) B=(%.3f,%.3f)，距纤轴 = r_f，跨距 L_c = %.2f"
      % (P_A[0], P_A[1], P_B[0], P_B[1], L_C))
print("ε_b = (x−1)sin²θ/(2x²) = %.4f%% (%.0f με)" % (100 * EPS_B, EPS_B * 1e6))

out = os.path.join(FIG, "fig9b_clamp3d.pdf")
fig.savefig(out)
fig.savefig(os.path.join(FIG, "fig9b_clamp3d_preview.png"), dpi=150)
print("写出", out)
