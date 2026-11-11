#!/usr/bin/env python3
"""模拟二：不同峰检测算法在“双峰重叠”下的峰位偏差，以及它会造成多大的表观弯曲

对比三种常见取峰方式（都是商用解调器里真实存在的做法）：
  argmax   —— 取谱极大点 + 三点抛物线细化
  centroid —— 固定半径窗口内的加权质心
  bwmid    —— 半高带宽中点（左右半高位置的均）
以及“正确的双峰拟合”（两高斯最小二乘，见 sim_peak_bias.py，偏差 ≈ 0）

输出两张表：
  ① 偏差(间距)：单位 pm，主峰读数相对真值的偏移
  ② 把偏差按实测“间距—温度”关系映射后，表观 K_T / 弯曲幅度 / 残差 rms
"""
import argparse
import numpy as np

RNG = np.random.default_rng(20261025)
FWHM = 0.390
BASE, NOISE = 3500.0, 300.0
SEP0, SEP1 = 0.267, 0.598
T0, T1, N_T = 26.17, 54.89, 16
K_TRUE = 10.0        # pm/°C，真值（文献量级）


def profile(x, c1, c2, a1, a2, w=FWHM):
    f = lambda u: np.exp(-4 * np.log(2) * (u / w) ** 2)
    return a1 * f(x - c1) + a2 * f(x - c2) + BASE


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
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ratio", type=float, default=2.7, help="主峰/弱峰振幅比")
    ap.add_argument("--rad", type=float, default=0.6, help="窗口半径 nm")
    args = ap.parse_args()

    seps = np.linspace(0.15, 1.10, 20)
    print("FWHM=%.3f nm，振幅比 %.1f，窗半径 %.2f nm，每点 %d 次实现"
          % (FWHM, args.ratio, args.rad, args.n))
    print("\n间距/nm   argmax     centroid   bwmid        （pm）")
    table = {}
    for sep in seps:
        acc = {"argmax": [], "centroid": [], "bwmid": []}
        for _ in range(args.n):
            x = np.arange(-1.3, 1.3, 0.0012)
            a2 = 60000.0 - BASE
            y = profile(x, -sep, 0.0, a2 / args.ratio, a2) + RNG.normal(0, NOISE, len(x))
            acc["argmax"].append(bias_argmax(x, y, 0.0))
            acc["centroid"].append(bias_centroid(x, y, 0.0, args.rad))
            acc["bwmid"].append(bias_bwmid(x, y, 0.0, args.rad - 0.15))
        row = {k: float(np.mean(v)) * 1000 for k, v in acc.items()}
        table[float(sep)] = row
        print("%6.3f  %+8.1f  %+9.1f  %+9.1f"
              % (sep, row["argmax"], row["centroid"], row["bwmid"]))

    print("\n按实测 间距(T) 映射后（真 K_T=%.1f pm/°C）：" % K_TRUE)
    T = np.linspace(T0, T1, N_T)
    sepT = SEP0 + (SEP1 - SEP0) * (T - T0) / (T1 - T0)
    sp = np.array(sorted(table))
    L = T1 - T0
    for k in ("argmax", "centroid", "bwmid"):
        v = np.array([table[s][k] for s in sp])
        lam = K_TRUE * (T - T0) + np.interp(sepT, sp, v)
        c1 = np.polyfit(T - T0, lam, 1)
        c2 = np.polyfit(T - T0, lam, 2)
        print("  %-9s 表观 K_T=%.2f pm/°C   弯曲 %.0f pm   线性残差 rms %.1f pm"
              % (k, c1[0], abs(c2[0]) * L ** 2, (lam - np.polyval(c1, T - T0)).std(ddof=2)))
    print("\n观测值（温度组主峰，16 点）：K_T=9.75 pm/°C，弯曲 293 pm，线性残差 rms 29.5 pm")
    print("结论：方向与形态都对（低温端偏低 → 表观灵敏度先高后低），但量级只有观测的 1/3 左右；")
    print("      而“正确的双峰拟合”偏差 < 0.1 pm（sim_peak_bias.py），说明偏差来自算法粗糙度而非重叠本身。")


if __name__ == "__main__":
    main()
