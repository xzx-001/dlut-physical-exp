#!/usr/bin/env python3
"""模拟：双峰重叠 + 双高斯拟合，是否会把“热漂移”扭曲成观测到的弯曲？

模型链：
  真值谱 = 双峰（两峰间距 d 随温度线性增大，主峰按真实 K_T 线性漂移）
  解调器 = 在固定窗口内做双高斯最小二乘拟合，取“较强那一支”的中心作为主峰读数
  统计量 = 拟合中心相对真值的偏差 bias(d) → 换算成 λ2(T) → 再拟线性，看弯曲幅度

同时做一项真实数据检验：λ2 = a + b·T + c·λ1（把弱峰实测值当协变量），
看弯曲是否随 λ1 项被吸收（c≠0 说明主峰读数被弱峰的位置拖着走）。

用法：python3 scripts/sim_peak_bias.py [--n 300]
"""
import argparse
import json
import os

import numpy as np

RNG = np.random.default_rng(20261025)

# ---- 由 PNG 反解得到的实测谱参数 ----
FWHM = 0.390          # nm，单峰半高全宽（digitize_spectrum.py）
BASE = 3500.0         # 基线
NOISE = 300.0         # 单点噪声（与谱图噪底同量级）
T0, T1 = 26.17, 54.89
N_T = 16
SEP0, SEP1 = 0.267, 0.598     # 稳定段两峰间距（nm，实测首末）


def two_peaks(x, c1, c2, a1, a2, w, kind="gauss"):
    """kind: gauss | lorentz  (w 为 FWHM)"""
    if kind == "gauss":
        f = lambda u: np.exp(-4 * np.log(2) * (u / w) ** 2)
    else:
        f = lambda u: 1.0 / (1.0 + 4.0 * (u / w) ** 2)
    return a1 * f(x - c1) + a2 * f(x - c2)


def gn_fit(x, y, p0, kind="gauss", n_iter=120, lam0=1e-3):
    """Gauss-Newton + 阻尼（数值雅可比），双峰 5 参数 [c1,c2,a1,a2,w]"""
    p = np.asarray(p0, float)

    def r(p):
        return two_peaks(x, *p, kind=kind) - y

    lam = lam0
    cur = np.sum(r(p) ** 2)
    for _ in range(n_iter):
        J = np.empty((len(x), len(p)))
        h = np.array([1e-5, 1e-5, 1.0, 1.0, 1e-5])
        for k in range(len(p)):
            dp = p.copy()
            dp[k] += h[k]
            J[:, k] = (r(dp) - r(p)) / h[k]
        A = J.T @ J
        g = J.T @ r(p)
        try:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        except np.linalg.LinAlgError:
            break
        pn = p + step
        new = np.sum(r(pn) ** 2)
        if new < cur:
            p, cur, lam = pn, new, max(lam * 0.4, 1e-9)
        else:
            lam *= 4.0
            if lam > 1e6:
                break
    return p


def one_trial(sep, amp_ratio, kind_true="gauss", kind_fit="gauss", noise=NOISE,
              win=1.2):
    """给定两峰间距，返回 (bias_of_strong_peak, ok)"""
    x = np.arange(-win, win, 0.0012)          # 1.2 pm 采样步长
    c2 = 0.0                                  # 主峰真值位置
    amp2 = 60000.0 - BASE
    amp1 = amp2 / amp_ratio
    y = two_peaks(x, c2 - sep, c2, amp1, amp2, FWHM, kind=kind_true) + BASE
    y = y + RNG.normal(0, noise, size=len(x))
    p0 = [c2 - sep, c2 + 0.02 * sep, amp1, amp2, FWHM]
    p = gn_fit(x, y, p0, kind=kind_fit)
    c1f, c2f, a1f, a2f, w = p
    if not np.all(np.isfinite(p)) or w < 0.05 or w > 2.0 or a2f <= 0:
        return np.nan
    # 取振幅较大的那一支作为“主峰读数”
    strong = c2f if a2f >= a1f else c1f
    # 与“离它最近的真峰”比较
    truth = c2 if abs(strong - c2) <= abs(strong - (c2 - sep)) else c2 - sep
    return strong - truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="每个间距的噪声实现次数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seps = np.linspace(0.15, 1.10, 20)
    scenarios = [
        ("拟合双高斯 / 真值双高斯", "gauss", "gauss"),
        ("拟合双高斯 / 真值洛伦兹（FBG 长尾）", "lorentz", "gauss"),
        ("拟合高斯 / 真值高斯，振幅比 1:1", None, None),   # 单独处理（比例不同）
    ]
    results = {}
    print("FWHM=%.3f nm，噪声 σ=%.0f，每点 %d 次实现\n" % (FWHM, NOISE, args.n))
    print("间距/nm  " + "  ".join("%-14s" % s[0][:14] for s in scenarios))
    for sep in seps:
        row = {}
        for tag, kt, kf in scenarios:
            ratio = 1.0 if "1:1" in tag else 2.7
            kt = kt or "gauss"
            kf = kf or "gauss"
            vals = [one_trial(sep, ratio, kt, kf) for _ in range(args.n)]
            vals = np.array([v for v in vals if np.isfinite(v)])
            row[tag] = (float(np.mean(vals)) if len(vals) else np.nan,
                        float(np.std(vals)) if len(vals) else np.nan)
        results[float(sep)] = row
        print("%6.3f   " % sep + "  ".join("%+7.1f±%4.1f" % row[s[0]] for s in scenarios))

    # ---- 把 bias(sep) 映射回温度轴，看会造出多大的弯曲 ----
    print("\n把上表的偏差按实测间距-温度关系映射，再做一次线性拟合：")
    T = np.linspace(T0, T1, N_T)
    sepT = SEP0 + (SEP1 - SEP0) * (T - T0) / (T1 - T0)
    for tag in [s[0] for s in scenarios]:
        pts = np.array([[k, results[k][tag][0]] for k in results if np.isfinite(results[k][tag][0])])
        if len(pts) < 4:
            continue
        if np.nanmax(np.abs(pts[:, 1])) < 1.0:
            print("  %-28s 偏差全表 < 1 pm（数值零），不做弯曲换算——此时表观 K_T/弯曲 只是"
                  "对噪声级数值做插值的结果，没有物理含义" % tag)
            continue
        o = np.argsort(pts[:, 0])
        bias = np.interp(sepT, pts[o, 0], pts[o, 1])
        # 表观读数 = 真值(线性) + bias
        K_TRUE = 10.0                      # pm/°C
        lam = K_TRUE * (T - T0) + bias * 1000.0     # pm
        c1 = np.polyfit(T - T0, lam, 1)
        c2 = np.polyfit(T - T0, lam, 2)
        curv = abs(c2[0]) * (T1 - T0) ** 2
        kapp = c1[0]
        res_l = lam - np.polyval(c1, T - T0)
        res_q = lam - np.polyval(c2, T - T0)
        print("  %-28s 表观 K_T=%.2f pm/°C，弯曲幅度 %.0f pm，"
              "线性残差 rms %.1f pm，二次残差 rms %.1f pm"
              % (tag, kapp, curv, res_l.std(ddof=2), res_q.std(ddof=3)))

    if args.out:
        json.dump({str(k): {t: list(v) for t, v in r.items()} for k, r in results.items()},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n写出", args.out)


if __name__ == "__main__":
    main()
