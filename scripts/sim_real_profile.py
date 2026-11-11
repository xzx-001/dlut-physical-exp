#!/usr/bin/env python3
"""用“真实的谱形”（从 rainbow_spectrum*.png 反解）而不是理想高斯，重做峰检测偏差模拟

为什么必须用真实谱形：理想高斯的双峰拟合偏差 ≈ 0，但如果真实谱形不对称、有边带、
顶部更尖，质心/极大值/带宽中点这些粗糙算法就会被拉得更偏。

做法：
  1. 反解 PNG 得到 (λ, P) 曲线（复用 digitize_spectrum 的标定与提取）；
  2. 抠出主峰附近的实测线型 → 基线归零、两端渐变到零 → 得到“单位线型” f(Δλ)；
  3. 用两个 f 按间距 sep 叠加（振幅比可调）+ 实测噪声，得到任意间距下的合成谱；
  4. 三种算法取峰，量偏差；再把偏差按实测 sep(T) 映射，看能造出多大的弯曲。

用法：python3 scripts/sim_real_profile.py [--n 40] [--ratio 2.7]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digitize_spectrum import _largest_component, _blue_mask, calibrate, extract_trace  # noqa: E402

RNG = np.random.default_rng(20261025)
BASE, NOISE = 3500.0, 300.0
SEP0, SEP1 = 0.267, 0.598
T0, T1, N_T = 26.17, 54.89, 16
K_TRUE = 10.0


def digitize(path):
    a = np.array(Image.open(path).convert("RGB"))
    cal = calibrate(a)
    xc = [c[0] for c in cal["xlab"]]
    vals = 1.525 + 0.005 * np.arange(len(xc))
    kx, bx = np.polyfit(xc, vals, 1)
    yc = [c[0] for c in cal["ylab"]]
    yv = 60000 - 10000 * np.arange(len(yc))
    ky, by = np.polyfit(yc, yv, 1)
    cols, top, _med = extract_trace(cal["blue"], cal["box"])
    lam = (bx + kx * cols) * 1000.0        # nm
    pw = by + ky * top                     # dBm（谱功率）
    ok = np.isfinite(pw) & np.isfinite(lam)
    return lam[ok], pw[ok]


def real_line_shape(lam, pw, c_main, half=0.55, step=0.0012):
    """抠出主峰附近的实测线型（基线归零，两端 cosine 渐变到 0），返回可调用的 f(Δλ)"""
    m = np.abs(lam - c_main) < half
    x, y = lam[m] - c_main, pw[m]
    o = np.argsort(x)
    x, y = x[o], y[o]
    base = np.percentile(y, 3)
    y = y - base
    y = np.clip(y, 0, None)
    # 两端 15% 长度做 cosine 收尾，避免截断造成的台阶
    n = len(y)
    k = max(4, int(0.15 * n))
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    y[:k] *= w
    y[-k:] *= w[::-1]
    grid = np.arange(-half, half, step)
    prof = np.interp(grid, x, y, left=0.0, right=0.0)
    prof = np.clip(prof, 0, None)

    def f(d):
        return np.interp(d, grid, prof, left=0.0, right=0.0)
    return f, dict(fwhm_nm=fwhm_of(grid, prof), peak=float(prof.max()), half=half)


def fwhm_of(grid, prof):
    half = prof.max() / 2
    idx = np.where(prof > half)[0]
    return float(grid[idx[-1]] - grid[idx[0]]) if len(idx) > 1 else np.nan


def bias_argmax(x, y, c0):
    i = int(np.argmax(y))
    c = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
    return (-c[1] / (2 * c[0]) if c[0] < 0 else x[i]) - c0


def bias_centroid(x, y, c0, rad):
    m = np.abs(x - c0) < rad
    w = y[m] - y[m].min()
    return float((x[m] * w).sum() / w.sum()) - c0


def bias_bwmid(x, y, c0, rad):
    m = np.abs(x - c0) < rad
    xs, ys = x[m], y[m]
    half = (ys.max() + ys.min()) / 2
    idx = np.where(ys > half)[0]
    return (xs[idx[0]] + xs[idx[-1]]) / 2 - c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ratio", type=float, default=2.7, help="主峰/弱峰振幅比")
    ap.add_argument("--rad", type=float, default=0.6)
    ap.add_argument("--png", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "rainbow_spectrum-温度.png"))
    args = ap.parse_args()

    lam, pw = digitize(args.png)
    # 主峰取实测最强峰位置
    i = int(np.argmax(pw))
    c_main = float(lam[i])
    f, info = real_line_shape(lam, pw, c_main)
    print("反解谱：%d 点，强峰 λ=%.3f nm，线型 FWHM=%.3f nm（实测）"
          % (len(lam), c_main, info["fwhm_nm"]))

    # 先在实测谱上直接看“三种算法差多少”（该谱间距 0.862 nm）
    x = np.arange(-1.3, 1.3, 0.0012)
    npk = lambda d: f(d) / info["peak"]          # 归一到 1
    sep_meas = 0.862
    a2 = 60000.0 - BASE
    y = a2 * npk(x) + a2 / 1.01 * npk(x + sep_meas) + BASE
    y = y + RNG.normal(0, NOISE, len(x))
    print("\n实测谱（间距 %.3f nm）各算法读数相对真值：" % sep_meas)
    print("  argmax %+.1f pm   质心(±%.1f) %+.1f pm   带宽中点 %+.1f pm"
          % (bias_argmax(x, y, 0.0) * 1000, args.rad, bias_centroid(x, y, 0.0, args.rad) * 1000,
             bias_bwmid(x, y, 0.0, args.rad) * 1000))

    seps = np.linspace(0.15, 1.10, 20)
    print("\n间距/nm   argmax    centroid   bwmid      （pm，振幅比 %.1f）" % args.ratio)
    table = {}
    for sep in seps:
        acc = {"argmax": [], "centroid": [], "bwmid": []}
        for _ in range(args.n):
            y = a2 * npk(x) + a2 / args.ratio * npk(x + sep) + BASE
            y = y + RNG.normal(0, NOISE, len(x))
            acc["argmax"].append(bias_argmax(x, y, 0.0))
            acc["centroid"].append(bias_centroid(x, y, 0.0, args.rad))
            acc["bwmid"].append(bias_bwmid(x, y, 0.0, args.rad - 0.1))
        table[float(sep)] = {k: float(np.mean(v)) * 1000 for k, v in acc.items()}
        print("%6.3f  %+8.1f  %+9.1f  %+9.1f"
              % (sep, table[sep]["argmax"], table[sep]["centroid"], table[sep]["bwmid"]))

    print("\n按实测 间距(T) 映射（真 K_T=%.1f pm/°C）：" % K_TRUE)
    T = np.linspace(T0, T1, N_T)
    sepT = SEP0 + (SEP1 - SEP0) * (T - T0) / (T1 - T0)
    sp = np.array(sorted(table))
    for k in ("argmax", "centroid", "bwmid"):
        v = np.array([table[s][k] for s in sp])
        lamT = K_TRUE * (T - T0) + np.interp(sepT, sp, v)
        c1 = np.polyfit(T - T0, lamT, 1)
        c2 = np.polyfit(T - T0, lamT, 2)
        print("  %-9s 表观 K_T=%.2f pm/°C   弯曲 %.0f pm   线性残差 rms %.1f pm"
              % (k, c1[0], abs(c2[0]) * (T1 - T0) ** 2,
                 (lamT - np.polyval(c1, T - T0)).std(ddof=2)))
    print("\n观测：K_T=9.75 pm/°C，弯曲 293 pm，线性残差 rms 29.5 pm")


if __name__ == "__main__":
    main()
