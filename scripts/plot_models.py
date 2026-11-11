#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""温度稳定段：六种拟合模型的可视化对比（对应 tab:tempfit）
每个子图只画该模型实际使用的数据点（其余稳定段点作灰色背景参考），
拟合线与注释数值与 analyze.py / results/summary.json 一致。
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

_ap = argparse.ArgumentParser(description="温度稳定段六种拟合模型对比图")
_ap.add_argument("--figures", default=None, help="图输出目录（默认 <repo>/output/figures）")
_args, _ = _ap.parse_known_args()

BASE = os.path.dirname(os.path.abspath(__file__))        # scripts/
REPO = os.path.dirname(BASE)
SRC = os.path.join(REPO, "data")
FIG = _args.figures or os.path.join(REPO, "output", "figures")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False, "font.size": 8.5, "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "legend.frameon": False,
})

def load(fn, enc="utf-8-sig"):
    rows = list(csv.reader(open(os.path.join(SRC, fn), encoding=enc)))
    h = [x.strip() for x in rows[0]]
    return [dict(zip(h, r)) for r in rows[1:] if r]

def num(rows, key):
    return np.array([float(x[key]) if x.get(key, "") not in ("", None) else np.nan for x in rows])

def linfit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    ssr = float(res @ res); sst = float((y - y.mean()) @ (y - y.mean()))
    return dict(a=float(a), b=float(b), r2=1 - ssr / sst,
                rms=float(np.sqrt(ssr / (len(x) - 2))), n=int(len(x)))

warm = load("温度正向.csv")
Tw, W1, W2 = num(warm, "pt100_temperature_c"), num(warm, "fbg_peak1_nm"), num(warm, "fbg_peak2_nm")

seg = [i for i in range(2, 20) if i != 12 and np.isfinite(W2[i])]
T0 = float(Tw[seg[0]])
DT = Tw - T0

models = []
# 1 全段线性
f = linfit(DT[seg], W2[seg])
models.append(dict(name="（a）全段线性", idx=seg, x=DT[seg], y=W2[seg], kind="lin", fit=f,
                   txt="$K_T=%.2f$ pm/$^\\circ$C" % (f["a"] * 1000)))
# 2 二次多项式
q = np.polyfit(DT[seg], W2[seg], 2)
qr = W2[seg] - np.polyval(q, DT[seg])
r2q = 1 - (qr @ qr) / ((W2[seg] - W2[seg].mean()) @ (W2[seg] - W2[seg].mean()))
models.append(dict(name="（b）二次多项式", idx=seg, x=DT[seg], y=W2[seg], kind="quad", coef=q,
                   r2=float(r2q), rms=float(np.sqrt((qr @ qr) / (len(seg) - 3))),
                   vertex=float(-q[1] / (2 * q[0])),
                   txt="$a=%.3f$ pm/$^\\circ$C$^2$，顶点 %+.1f $^\\circ$C" % (q[0] * 1000, -q[1] / (2 * q[0]))))
# 3--5 分段
for nm, idx, lab in [("（c）分段：第 3--12 行（去第 8 行）", range(2, 12), "26--42"),
                     ("（d）分段：第 9--20 行（去第 13 行）", [i for i in range(8, 20) if i != 12], "37--55"),
                     ("（e）分段：第 15--20 行", range(14, 20), "47--55")]:
    idx = [i for i in idx if np.isfinite(W2[i])]
    f = linfit(DT[idx], W2[idx])
    models.append(dict(name=nm, idx=idx, x=DT[idx], y=W2[idx], kind="lin", fit=f,
                       txt="$K_T=%.2f$ pm/$^\\circ$C（%s $^\\circ$C）" % (f["a"] * 1000, lab)))
# 6 并入第 21--23 行（取峰值 1 列）
idx6 = list(seg) + [20, 21, 22]
x6 = np.r_[DT[seg], DT[20:23]]
y6 = np.r_[W2[seg], W1[20:23]]
f6 = linfit(x6, y6)
models.append(dict(name="（f）并入第 21--23 行（取峰值 1 列）", idx=idx6, x=x6, y=y6, kind="lin",
                   fit=f6, txt="$K_T=%.2f$ pm/$^\\circ$C" % (f6["a"] * 1000)))

fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.9))
bg_x, bg_y = DT[seg], (W2[seg] - 1544.0) * 1000

for ax, m in zip(axes.ravel(), models):
    ax.plot(bg_x, bg_y, "o", ms=2.6, mfc="0.72", mec="none", zorder=1, label="稳定段其余点")
    x = m["x"]; y = (m["y"] - 1544.0) * 1000
    ax.plot(x, y, "s", ms=4.2, mfc="#c00000", mec="#c00000", zorder=3, label="参与拟合的点")
    xg = np.linspace(x.min(), x.max(), 60)
    if m["kind"] == "lin":
        yg = (m["fit"]["a"] * xg + m["fit"]["b"] - 1544.0) * 1000
        ann = (m["txt"] + "\n$R^2=%.4f$，rms$=%.1f$ pm" % (m["fit"]["r2"], m["fit"]["rms"] * 1000))
    else:
        yg = (np.polyval(m["coef"], xg) - 1544.0) * 1000
        ann = (m["txt"] + "\n$R^2=%.4f$，rms$=%.1f$ pm" % (m["r2"], m["rms"] * 1000))
    ax.plot(xg, yg, "-", color="#1f4e79", lw=1.3, zorder=2, label="拟合")
    ax.set_title(m["name"], fontsize=8.6)
    ax.text(0.03, 0.96, ann, transform=ax.transAxes, va="top", fontsize=7.4, color="#333333")
    ax.text(0.97, 0.06, "$n=%d$" % len(x), transform=ax.transAxes, ha="right", fontsize=7.4, color="#333333")
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_xlabel("$T-T_0$ / $^\\circ$C  ($T_0=%.2f$ $^\\circ$C)" % T0)
    ax.set_ylabel("$\\lambda_2 - 1544$ / pm")

axes[0, 0].legend(loc="lower right", fontsize=7.0, handletextpad=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig9_temp_models.pdf"))
fig.savefig(os.path.join(FIG, "fig9_temp_models_preview.png"), dpi=140)
print("写出 fig9_temp_models.pdf / .png")
for m in models:
    if m["kind"] == "lin":
        print("%-34s n=%2d  K=%.2f pm/C  R2=%.4f  rms=%.1f pm" %
              (m["name"], m["fit"]["n"], m["fit"]["a"] * 1000, m["fit"]["r2"], m["fit"]["rms"] * 1000))
    else:
        print("%-34s n=%2d  a=%.3f  R2=%.4f  rms=%.1f pm" % (m["name"], len(m["x"]), m["coef"][0] * 1000, m["r2"], m["rms"] * 1000))
