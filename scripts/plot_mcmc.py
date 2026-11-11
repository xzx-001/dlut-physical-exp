#!/usr/bin/env python3
"""图 8：MCMC 后验（1×3）
(a) 三条链的 K_T 边缘分布：模型 A 宽先验 / 模型 A 理论先验 / 模型 B 小倾角联合拟合
(b) 小倾角分支中应变项在窗口末端被用掉的量（后验），与需解释的弯曲幅度 293 pm 对比
(c) 小倾角几何：相切条件给出的弯曲应变 vs 孔/纤直径比，标出许用值与两个代表点

数据来源：<repo>/output/results/ 下的 summary.json 与 chains_*.npy（由 analyze.py 保存）。
用法：python3 scripts/plot_mcmc.py [--figures DIR] [--results DIR]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False, "font.size": 8.0, "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})
K_RATIO = 0.85 / 0.50


def theta_of_x(x, k=K_RATIO):
    A = np.hypot(x, k)
    return np.arccos(np.clip(1.0 / A, -1.0, 1.0)) - np.arctan(k / x)


def eps_b(x, k=K_RATIO):
    return (x - 1.0) * np.sin(theta_of_x(x, k)) ** 2 / (2.0 * x ** 2)


def draw(summary, figdir, resultsdir):
    S = summary
    ch_u = np.load(os.path.join(resultsdir, "chains_a_wide.npy"))
    ch_p = np.load(os.path.join(resultsdir, "chains_a_phys.npy"))
    ch_s = np.load(os.path.join(resultsdir, "chains_joint_small.npy"))
    curv = float(S["mcmc"]["compare"]["quad_span_pm"])
    eps_allow = float(S["clamp"]["eps_allow"])
    x_small = float(S["clamp"]["small"]["x_med"])

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))

    # (a) K_T 边缘分布
    ax = axes[0]
    for v, c, lab in ((ch_u[:, 1] * 1000, "#c00000", "A：松弛（宽先验）"),
                      (ch_p[:, 1] * 1000, "#1f4e79", "A：松弛（理论先验）"),
                      (ch_s[:, 1], "#6a1b9a", "B：联合（小倾角）")):
        ax.hist(v, bins=60, density=True, histtype="step", lw=1.2, color=c, label=lab)
    ax.axvline(10.9, ls=":", color="0.4", lw=1.0)
    ax.set_xlabel("$K_T$ / (pm/$^\\circ$C)")
    ax.set_ylabel("后验密度")
    ax.legend(fontsize=6.0, labelspacing=0.45, handletextpad=0.4, loc="upper right")
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_title("(a) $K_T$ 边缘分布", fontsize=8.2)

    # (b) 应变项实际用掉的量
    ax = axes[1]
    contrib = ch_s[:, 2] * np.array([(eps_b(float(x * (1 + a * 28.7))) - eps_b(float(x)))
                                     for x, a in zip(ch_s[:, 3], ch_s[:, 4])]) * 1e6
    ax.hist(contrib, bins=60, density=True, histtype="stepfilled", color="#6a1b9a",
            alpha=0.35, lw=0)
    ax.hist(contrib, bins=60, density=True, histtype="step", color="#6a1b9a", lw=1.1,
            label="小倾角链")
    ax.axvline(curv, ls="--", color="#c00000", lw=1.1, label="需解释的弯曲 %.0f pm" % curv)
    ax.set_xlabel("窗口末应变项贡献 $K_\\varepsilon\\Delta\\varepsilon_b$ / pm")
    ax.set_ylabel("后验密度")
    ax.legend(fontsize=6.0, labelspacing=0.45, handletextpad=0.4)
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_title("(b) 应变项实际被用掉多少", fontsize=8.2)

    # (c) 小倾角几何：ε_b vs x
    ax = axes[2]
    xs = np.linspace(1.005, 1.6, 400)
    ax.plot(xs, np.array([eps_b(x) for x in xs]) * 100, "-", color="#1f4e79", lw=1.4,
            label="$\\varepsilon_b$（相切几何）")
    ax.axhline(eps_allow * 100, ls="--", color="#c00000", lw=1.0,
               label="许用 $%.1f\\%%$" % (eps_allow * 100))
    ax.plot([x_small], [eps_b(x_small) * 100], "o", ms=4, color="#6a1b9a")
    ax.annotate("$x_0=%.2f$\n$\\theta=%.1f^\\circ$" % (x_small, np.degrees(theta_of_x(x_small))),
                xy=(x_small, eps_b(x_small) * 100), xytext=(x_small + 0.06, eps_b(x_small) * 100 + 0.06),
                fontsize=6.2, color="#6a1b9a",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#6a1b9a"))
    ax.set_xlabel("孔/纤直径比 $R_h/r_f$")
    ax.set_ylabel("$\\varepsilon_b$ / %")
    ax.legend(fontsize=6.0, labelspacing=0.45, handletextpad=0.4, loc="upper left")
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_title("(c) 小倾角分支的应变水平", fontsize=8.2)

    fig.tight_layout()
    out = os.path.join(figdir, "fig8_mcmc.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(figdir, "fig8_mcmc_preview.png"), dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--figures", default=os.path.join(base, "output", "figures"))
    ap.add_argument("--results", default=os.path.join(base, "output", "results"))
    args = ap.parse_args()
    S = json.load(open(os.path.join(args.results, "summary.json"), encoding="utf-8"))
    print("写出", draw(S, args.figures, args.results))


if __name__ == "__main__":
    main()
