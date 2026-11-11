#!/usr/bin/env python3
"""图：取峰算法在双峰重叠下的读数偏差 vs 两峰间距（用真实谱形）

数据来源：sim_real_profile.py 的同一套“真实线型 + 叠加双峰 + 噪声”模拟。
两条振幅比：1.0（实测谱里两峰 66.2/67.1 dBm，近等幅）与 2.7（弱峰明显较弱）。

用法：python3 scripts/plot_bias.py [--n 40] [--figures DIR]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_real_profile import (  # noqa: E402
    BASE, NOISE, RNG, bias_argmax, bias_bwmid, bias_centroid, digitize, real_line_shape)
from digitize_spectrum import _largest_component  # noqa: F401,E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False, "font.size": 8.5, "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "legend.frameon": False,
})

SEPS = np.linspace(0.15, 1.10, 16)
SEP_LO, SEP_HI = 0.267, 0.598      # 实测稳定段的间距范围
CURV_PM = 293.0                    # 观测弯曲幅度


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--figures", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "figures"))
    ap.add_argument("--png", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "rainbow_spectrum-温度.png"))
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)

    lam, pw = digitize(args.png)
    c_main = float(lam[int(np.argmax(pw))])
    f, info = real_line_shape(lam, pw, c_main)
    npk = lambda d: f(d) / info["peak"]
    a2 = 60000.0 - BASE
    x = np.arange(-1.3, 1.3, 0.0012)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    styles = {"argmax": ("#c62828", "极大值+抛物线"),
              "centroid": ("#2e7d32", "固定窗质心"),
              "bwmid": ("#1f4e79", "半高带宽中点")}
    for ax, ratio in zip(axes, (1.0, 2.7)):
        for k, (c, lab) in styles.items():
            mu, sd = [], []
            for sep in SEPS:
                vals = []
                for _ in range(args.n):
                    y = a2 * npk(x) + a2 / ratio * npk(x + sep) + BASE
                    y = y + RNG.normal(0, NOISE, len(x))
                    if k == "argmax":
                        v = bias_argmax(x, y, 0.0)
                    elif k == "centroid":
                        v = bias_centroid(x, y, 0.0, 0.6)
                    else:
                        v = bias_bwmid(x, y, 0.0, 0.5)
                    vals.append(v * 1000)
                mu.append(np.mean(vals))
                sd.append(np.std(vals))
            mu, sd = np.array(mu), np.array(sd)
            ax.fill_between(SEPS, mu - sd, mu + sd, color=c, alpha=0.10, lw=0)
            ax.plot(SEPS, mu, "-", color=c, lw=1.5, label=lab)
        ax.axvspan(SEP_LO, SEP_HI, color="0.75", alpha=0.30, lw=0)
        ax.axhspan(-CURV_PM / 2, CURV_PM / 2, color="#b8860b", alpha=0.10, lw=0)
        ax.axhline(0, color="0.4", lw=0.7)
        ax.set_xlabel(r"两峰间距 / nm（实测 $0.267\to0.598$，灰带）")
        ax.set_title(r"(a) 振幅比 1.0（实测谱近等幅）" if ratio == 1.0
                     else r"(b) 振幅比 2.7", fontsize=8.5)
        ax.grid(alpha=0.2, lw=0.5)
    axes[0].set_ylabel("主峰读数偏差 / pm")
    axes[0].legend(fontsize=6.8, labelspacing=0.45, handletextpad=0.4, loc="lower left")
    axes[1].text(0.60, 120, "±146 pm ≈ 观测弯曲的一半", fontsize=6.6, color="#8a6d1a")
    fig.tight_layout()
    out = os.path.join(args.figures, "fig_peak_bias.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(args.figures, "fig_peak_bias_preview.png"), dpi=150)
    print("写出", out)


if __name__ == "__main__":
    main()
