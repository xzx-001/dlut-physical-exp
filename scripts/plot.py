"""PECVD 石墨烯拉曼光谱绘图脚本。

输出（figures/，PDF + PNG 各一份）：
    fig01_spectra.pdf   三个样品的全谱与硅标准样参考峰
    fig02_baseline.pdf  基线扣除示例
    fig03_fits.pdf      G 带（含空气 O2 线）与 2D 带的洛伦兹拟合
    fig04_dband.pdf     三谱平均后的 D 带区
    fig05_summary.pdf   峰位 / 半高宽 / 强度比对比
    fig06_air.pdf       空气 O2 / N2 拉曼线的识别
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raman_io import FIG, RES, lorentz

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                        "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

COLORS = {"g1": "#1f4e79", "g2": "#c0504d", "g3": "#4f8a3d"}
LABEL = {"g1": "S1", "g2": "S2", "g3": "S3"}
TAGS = ("g1", "g2", "g3")


def load():
    d = np.load(RES / "arrays.npz")
    with open(RES / "summary.json", encoding="utf-8") as f:
        s = json.load(f)
    return d, s


def fig01(d, s):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [2.05, 1]})
    ax = axes[0]
    for k, off in zip(TAGS, (0, 1500, 3000)):
        yk = d[f"{k}_y"] + off
        ax.plot(d[f"{k}_x"], yk, lw=0.7, color=COLORS[k])
        ax.text(3012, yk[-1] + 240, f"{LABEL[k]}", color=COLORS[k],
                fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("强度（计数）")
    ax.set_xlim(1000, 3130)
    ymax = max((d[f"{k}_y"]).max() for k in TAGS) + 3000.0
    ax.set_ylim(-400, ymax * 1.48)
    marks = ((1582, "G"), (2330, r"N$_2$（空气）"), (2722, r"2D$_2$"))
    for pos, lab in marks:
        ax.axvline(pos, color="0.45", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(pos + 22, ymax * 1.16, lab, fontsize=7, color="0.2",
                ha="left", va="top", zorder=5)
    ax.set_title("(a) 剥离样品的拉曼谱（纵向错开；S1/S2/S3 为三个测量位置）", fontsize=9)

    ax = axes[1]
    ax.plot(d["si_x"], d["si_y"], lw=0.7, color="#333333")
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("强度（计数）")
    ax.set_xlim(430, 620)
    ax.set_title("(b) 硅标准样（仅作刻度核对）", fontsize=9)
    p = s["si"]["peaks"][0]
    ax.text(0.04, 0.92,
            f"{p['pos']:.1f} cm$^{{-1}}$\n半高宽 {p['fwhm']:.1f} cm$^{{-1}}$",
            transform=ax.transAxes, fontsize=7, va="top")
    fig.savefig(FIG / "fig01_spectra.pdf")
    fig.savefig(FIG / "fig01_spectra.png", dpi=200)
    plt.close(fig)


def fig02(d, s):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    x, y, b, yc = (d["g1_x"], d["g1_y"], d["g1_base"], d["g1_yc"])
    ax = axes[0]
    ax.plot(x, y, lw=0.6, color="#444444", label="原始谱")
    ax.plot(x, b, lw=1.0, color="#c0504d", label="ALS 基线")
    for lo, hi in ((1540, 1570), (2315, 2345)):
        ax.axvspan(lo, hi, color="#999999", alpha=0.3, lw=0)
    ax.set_ylabel("强度（计数）")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(a) S1 谱与拟合基线", fontsize=9)
    ax = axes[1]
    ax.plot(x, yc, lw=0.6, color="#1f4e79")
    for lo, hi in ((1540, 1570), (2315, 2345)):
        ax.axvspan(lo, hi, color="#999999", alpha=0.3, lw=0)
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("扣基线后强度（计数）")
    ax.set_title(
        "(b) 扣基线后；灰带为空气 O$_2$ 与 N$_2$ 线的位置",
        fontsize=9)
    fig.savefig(FIG / "fig02_baseline.pdf")
    fig.savefig(FIG / "fig02_baseline.png", dpi=200)
    plt.close(fig)


def _draw(ax, x, yc, band, colors):
    """数据点 + 线性基线 + 各分量（虚线）+ 总拟合（红线）。"""
    w = band["window"]
    m = (x >= w[0]) & (x <= w[1])
    ax.plot(x[m], yc[m], "o", ms=1.6, color="#444444", mfc="none")
    base = band["baseline"]
    xs = np.linspace(w[0], w[1], 700)
    bl = base[0] + base[1] * (xs - np.mean(x[m]))
    ax.plot(xs, bl, ls=":", lw=0.7, color="#888888")
    tot = bl.copy()
    for c, col in zip(band["peaks"], colors):
        comp = lorentz(xs, c["pos"], c["fwhm"], c["height"])
        tot = tot + comp
        ax.plot(xs, comp + bl, lw=0.7, ls="--", color=col)
    ax.plot(xs, tot, lw=1.0, color="#c0504d")
    ax.set_xlim(w[0], w[1])
    # 刻度密度压到 4 个，避免相邻子图的刻度标签在角落相碰
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def fig03(d, s):
    """G 带（含空气 O2 线）与 2D 带的拟合：图例明确标出每条线。"""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.945, bottom=0.255,
                        hspace=0.62, wspace=0.26)
    for j, k in enumerate(TAGS):
        x, yc = d[f"{k}_x"], d[f"{k}_yc"]
        bands = s["graphene"][k]["bands"]
        sm = s["graphene"][k]["summary"]
        ax = axes[0, j]
        _draw(ax, x, yc, bands["G"], ("#4f8a3d", "#1f4e79"))
        ax.set_title(f"{LABEL[k]}：G  {sm['G_pos']:.1f} cm$^{{-1}}$", fontsize=8)
        ax.set_ylabel("强度（计数）" if j == 0 else "")
        ax = axes[1, j]
        _draw(ax, x, yc, bands["2D"], ("#4f8a3d", "#1f4e79"))
        ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
        ax.set_ylabel("强度（计数）" if j == 0 else "")
        ax.set_title(
            f"{LABEL[k]}：2D  $I_{{2D}}/I_G$={sm['I2D_over_IG_height']:.2f}",
            fontsize=8)

    hl = [
        Line2D([], [], ls="none", marker="o", ms=3.5, mfc="none", mec="#444444",
               label="实测谱"),
        Line2D([], [], color="#c0504d", lw=1.0, label="总拟合"),
        Line2D([], [], color="#888888", ls=":", lw=0.9, label="线性基线"),
        Line2D([], [], color="#4f8a3d", ls="--", lw=0.9,
               label="绿虚线：O$_2$（空气，上排）/ 2D$_1$（下排）"),
        Line2D([], [], color="#1f4e79", ls="--", lw=0.9,
               label="蓝虚线：$G$（上排）/ 2D$_2$（下排）"),
    ]
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=3, fontsize=6.8, frameon=False,
               handlelength=1.8, columnspacing=1.6, labelspacing=0.45)
    fig.savefig(FIG / "fig03_fits.pdf")
    fig.savefig(FIG / "fig03_fits.png", dpi=200)
    plt.close(fig)


def fig04(d, s):
    """三谱平均后的 D 带区：统计窗口、包络与 D 带位置分开标注。"""
    from matplotlib.transforms import blended_transform_factory
    x = d["g1_x"]
    ys = np.vstack([d[f"{k}_yc"] for k in TAGS])
    mean = ys.mean(axis=0)
    sem = ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
    m = (x >= 1250) & (x <= 1520)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.94, bottom=0.30)
    ax.axvspan(1320, 1440, color="#9aa4b0", alpha=0.16, lw=0,
               label="统计窗口 1320--1440 cm$^{-1}$")
    ax.axhline(0, lw=0.6, color="#888888")
    ax.fill_between(x[m], (mean - sem)[m], (mean + sem)[m],
                    color="#1f4e79", alpha=0.25, lw=0,
                    label=r"平均值 $\pm$ 标准误（$n=3$）")
    ax.plot(x[m], mean[m], lw=1.0, color="#1f4e79", label="平均谱")
    ax.axvline(1350, ls=":", lw=1.0, color="#c0504d")
    lo, hi = mean[m].min() - sem[m].max(), mean[m].max() + sem[m].max()
    ax.set_ylim(lo - 0.20 * (hi - lo), hi + 0.42 * (hi - lo))
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1354, 0.985, "D 带（1350 cm$^{-1}$）", transform=tr, fontsize=7,
            color="#c0504d", va="top", ha="left")
    ax.set_xlim(1250, 1520)
    ax.set_xticks(np.arange(1300, 1501, 50))
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("扣基线后强度（计数）")
    ax.legend(fontsize=6.8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=3, handlelength=1.6,
              columnspacing=1.2)
    fig.savefig(FIG / "fig04_dband.pdf")
    fig.savefig(FIG / "fig04_dband.png", dpi=200)
    plt.close(fig)


def fig05(d, s):
    pos2d = [s["graphene"][t]["bands"]["2D"]["peaks"][1]["pos"] for t in TAGS]
    fw2d = [s["graphene"][t]["bands"]["2D"]["peaks"][1]["fwhm"] for t in TAGS]
    ratio = [s["graphene"][t]["summary"]["I2D_over_IG_height"] for t in TAGS]
    idig = [s["graphene"][t]["summary"]["ID_over_IG_height"] for t in TAGS]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    data = [(pos2d, r"2D$_2$ 峰位 (cm$^{-1}$)", (2700, 2740)),
            (fw2d, r"2D$_2$ 半高宽 (cm$^{-1}$)", (25, 50)),
            (ratio, r"$I_{2D}/I_G$（峰高比）", (0, 1.0)),
            (idig, r"$I_D/I_G$（峰高比）", (0, 0.1))]
    for ax, (vals, lab, ylim) in zip(axes, data):
        ax.bar(range(3), vals, color=[COLORS[t] for t in TAGS], width=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels([LABEL[t] for t in TAGS])
        ax.set_ylabel(lab, fontsize=8)
        ax.set_ylim(*ylim)
    fig.savefig(FIG / "fig05_summary.pdf")
    fig.savefig(FIG / "fig05_summary.png", dpi=200)
    plt.close(fig)


def fig06(d, s):
    """空气 O2（1554.6）与 N2（2330.7）拉曼线的识别：图例标出每条线。"""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.085, right=0.98, top=0.90, bottom=0.30,
                        wspace=0.22)
    x, yc = d["g1_x"], d["g1_yc"]
    b = s["graphene"]["g1"]["bands"]
    sm = s["graphene"]["g1"]["summary"]

    ax = axes[0]
    _draw(ax, x, yc, b["G"], ("#4f8a3d", "#1f4e79"))
    ax.set_xlim(1535, 1610)
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_ylabel("强度（计数）")
    ax.set_title(r"(a) O$_2$ 线与 G 带（S1）", fontsize=9)
    ax.margins(y=0.22)
    ax.text(0.03, 0.97,
            rf"O$_2$ 峰位 {sm['O2_pos']:.1f} cm$^{{-1}}$，半高宽 {sm['O2_fwhm']:.1f} cm$^{{-1}}$"
            "\n" rf"$G$ 峰位 {sm['G_pos']:.1f} cm$^{{-1}}$，半高宽 {sm['G_fwhm']:.1f} cm$^{{-1}}$",
            transform=ax.transAxes, fontsize=7, va="top", ha="left")

    ax = axes[1]
    _draw(ax, x, yc, b["N2"], ("#4f8a3d",))
    ax.set_xlabel(r"拉曼位移 (cm$^{-1}$)")
    ax.set_title(r"(b) N$_2$ 线（S1）", fontsize=9)
    ax.margins(y=0.22)
    ax.text(0.03, 0.97,
            rf"N$_2$ 峰位 {sm['N2_pos']:.1f} cm$^{{-1}}$，半高宽 {sm['N2_fwhm']:.1f} cm$^{{-1}}$"
            "\n" rf"$I_{{N_2}}/I_{{O_2}}$ = {sm['IN2_over_IO2']:.2f}",
            transform=ax.transAxes, fontsize=7, va="top", ha="left")

    hl = [
        Line2D([], [], ls="none", marker="o", ms=3.5, mfc="none", mec="#444444",
               label="实测谱"),
        Line2D([], [], color="#c0504d", lw=1.0, label="总拟合"),
        Line2D([], [], color="#888888", ls=":", lw=0.9, label="线性基线"),
        Line2D([], [], color="#4f8a3d", ls="--", lw=0.9,
               label="绿虚线：O$_2$（a）/ N$_2$（b）"),
        Line2D([], [], color="#1f4e79", ls="--", lw=0.9, label="蓝虚线：$G$（a）"),
    ]
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=6.8, frameon=False,
               handlelength=1.8, columnspacing=1.5)
    fig.savefig(FIG / "fig06_air.pdf")
    fig.savefig(FIG / "fig06_air.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    d, s = load()
    fig01(d, s)
    fig02(d, s)
    fig03(d, s)
    fig04(d, s)
    fig05(d, s)
    fig06(d, s)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
