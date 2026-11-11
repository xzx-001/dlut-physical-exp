#!/usr/bin/env python3
"""小倾角分支的补充图：(a) 相切几何下窗口内应变变化量 Delta eps_b；(b) 为解释 293 pm 弯曲所需的 K_eps

数据来源：建在 analyze.py 给出的同一个几何函数上（此处独立重写一份，便于单文件复跑）。
用法：python3 scripts/plot_small_angle.py [--figures DIR]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})

K_RATIO = 0.85 / 0.50          # k = T_half / r_f，取示意图比例
DT_WIN = np.array([0.0, 28.7])  # 窗口：26.17 -> 54.87 °C
CURV_PM = 293.0                 # 需解释的曲率幅度（二次项 -0.355 pm/°C² × 窗口²）
KE_LIT = 1.2                    # 文献量级 pm/με


def theta_of_x(x, k=K_RATIO):
    A = np.hypot(x, k)
    return np.arccos(np.clip(1.0 / A, -1.0, 1.0)) - np.arctan(k / x)


def eps_b(x, k=K_RATIO):
    return (x - 1.0) * np.sin(theta_of_x(x, k)) ** 2 / (2.0 * x ** 2)


def deps_micro(x0, alpha):
    """窗口内 eps_b 的变化量（με）"""
    x = x0 * (1.0 + alpha * DT_WIN)
    return (eps_b(x[-1]) - eps_b(x[0])) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "figures"))
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "results"))
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)

    S = json.load(open(os.path.join(args.results, "summary.json"), encoding="utf-8"))
    alpha = S["mcmc"]["joint_small"]["alpha"]["median"]
    x_med = S["mcmc"]["joint_small"]["x0"]["median"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    TT = np.linspace(0.0, DT_WIN[1], 120)

    ax = axes[0]
    for x0, c in ((1.05, "#8e24aa"), (1.15, "#6a1b9a"), (1.30, "#1f4e79")):
        x = x0 * (1.0 + alpha * TT)
        d = (eps_b(x) - eps_b(x0)) * 1e6
        ax.plot(TT, d, "-", lw=1.3, color=c,
                label=r"$x_0=%.2f$（$\theta_0=%.1f^\circ$）" % (x0, np.degrees(theta_of_x(x0))))
    ax.set_xlabel(r"$\Delta T$ / $^\circ$C")
    ax.set_ylabel(r"$\varepsilon_b(T)-\varepsilon_b(T_0)$ / $\mu\varepsilon$")
    ax.set_title("(a) 小倾角分支：窗口内应变只变几 $\\mu\\varepsilon$", fontsize=9)
    ax.legend(fontsize=6.6, loc="upper left")
    ax.grid(alpha=0.22, lw=0.5)
    d_small = deps_micro(x_med, alpha)
    ax.axhline(d_small, ls=":", color="0.4", lw=0.9)

    ax = axes[1]
    xs = np.linspace(1.005, 8.0, 400)
    need = np.array([CURV_PM / max(deps_micro(x, alpha), 1e-9) for x in xs])
    ax.semilogy(xs, need, "-", color="#c62828", lw=1.5, label=r"需 $K_\varepsilon=293\,\mathrm{pm}/\Delta\varepsilon_b$")
    ax.axhline(KE_LIT, ls="--", color="#2e7d32", lw=1.1, label=r"文献量级 1.2 pm/$\mu\varepsilon$")
    ax.axvline(x_med, ls=":", color="0.4", lw=1.0)
    ax.annotate(r"$x_0=%.2f$ → $K_\varepsilon\approx%.0f$" % (x_med, CURV_PM / d_small),
                xy=(x_med, CURV_PM / d_small), xytext=(x_med + 0.5, 8),
                fontsize=7.2, color="#c62828",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#c62828"))
    ax.set_xlabel(r"孔/纤直径比 $R_h/r_f$（小角度在左端）")
    ax.set_ylabel(r"所需 $K_\varepsilon$ / (pm/$\mu\varepsilon$)")
    ax.set_ylim(0.5, 3e3)
    ax.set_title("(b) 要靠夹持应变解释 293 pm 所需的灵敏度", fontsize=9)
    ax.legend(fontsize=6.6, loc="upper right")
    ax.grid(alpha=0.22, lw=0.5, which="both")

    fig.tight_layout()
    out = os.path.join(args.figures, "fig_small_angle.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(args.figures, "fig_small_angle_preview.png"), dpi=150)
    print("写出", out)
    print("x_med=%.3f  theta=%.2f deg  d_eps=%.2f me  need_Ke=%.0f pm/me"
          % (x_med, np.degrees(theta_of_x(x_med)), d_small, CURV_PM / d_small))
    print("x=5.0 -> d_eps=%.2f me, need_Ke=%.0f" % (deps_micro(5.0, alpha), CURV_PM / deps_micro(5.0, alpha)))


if __name__ == "__main__":
    main()
