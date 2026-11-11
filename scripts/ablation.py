"""G 带建模变体与窗口敏感度的消融实验（报告 §4.7 引用的数值由此脚本产生）。

正文 §4.7「拟合模型的取舍」「拟合窗口与初值影响半高宽」两段所引的替代模型
数值，必须能由本脚本复现，因此单列一个脚本并把结果写入
results/ablation.json，而不是手工记录。

变体（G 带窗口默认 1525--1615 cm^-1）：
    main      2 个洛伦兹（空气 O2 + G）+ 线性基线   ← 正文主模型
    no_o2     1 个洛伦兹（G）+ 线性基线             ← 不计 O2 分量
    no_base   2 个洛伦兹 + 常数偏置                 ← 不加线性基线
    no_o2_no_base 1 个洛伦兹 + 常数偏置
    wide      main 模型，窗口放宽到 1490--1625 cm^-1 ← 窗口敏感度

另外统计 2D 带的展宽区间（9 点平滑后超过 3 倍背景噪声的范围）。

用法：
    python3 scripts/ablation.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import curve_fit

from analyze import FILES
from raman_io import DATA, RES, als_baseline, load_spectrum, lorentz

G_WIN = (1525.0, 1615.0)
G_WIN_WIDE = (1490.0, 1625.0)
D2_WIN = (2560.0, 2790.0)
O2_BOUNDS = (1538.0, 1575.0)
G_BOUNDS = (1570.0, 1600.0)


def fit(x, y, win, npeak, baseline):
    """窗口内 npeak 个洛伦兹 + baseline（'linear' 或 'const'）拟合。"""
    m = (x >= win[0]) & (x <= win[1])
    xs, ys = x[m], y[m]
    xm = xs.mean()

    def model(xx, *p):
        out = p[0] + (p[1] * (xx - xm) if baseline == "linear" else 0.0)
        for k in range(npeak):
            out = out + lorentz(xx, *p[2 + 3 * k: 5 + 3 * k])
        return out

    p0, lo, hi = [float(np.median(ys)), 0.0], [-np.inf, -np.inf], [np.inf, np.inf]
    peaks = [{"p0": 1555.0, "b": O2_BOUNDS}, {"p0": 1582.0, "b": G_BOUNDS}]
    if npeak == 1:            # 只有 G 峰时用 G 的位置边界，不能用 O2 的
        peaks = [peaks[1]]
    for spec in peaks:
        p0 += [spec["p0"], 20.0, float(np.interp(spec["p0"], xs, ys) - np.median(ys))]
        lo += [spec["b"][0], 2.0, -np.inf]
        hi += [spec["b"][1], 250.0, np.inf]
    popt, _ = curve_fit(model, xs, ys, p0=p0, bounds=(lo, hi), maxfev=60000)
    resid = ys - model(xs, *popt)
    g = popt[2 + 3 * (npeak - 1): 5 + 3 * (npeak - 1)]
    return {"g_pos": float(g[0]), "g_fwhm": float(g[1]),
            "rms": float(np.sqrt(np.mean(resid ** 2)))}


def band_span(x, yc, win, noise, smooth=9, k_sigma=3.0):
    """平滑后超过 k_sigma 倍噪声的区域端点（判断 2D 带的展宽范围）。"""
    m = (x >= win[0]) & (x <= win[1])
    xs, ys = x[m], yc[m]
    kern = np.ones(smooth) / smooth
    ss = np.convolve(ys, kern, mode="same")
    idx = np.where(ss > k_sigma * noise)[0]
    if idx.size == 0:
        return None
    return {"lo": float(xs[idx[0]]), "hi": float(xs[idx[-1]])}


def main():
    variants = {
        "main": (G_WIN, 2, "linear"),
        "no_o2": (G_WIN, 1, "linear"),
        "no_base": (G_WIN, 2, "const"),
        "no_o2_no_base": (G_WIN, 1, "const"),
        "wide": (G_WIN_WIDE, 2, "linear"),
    }
    out = {"variants": {k: {} for k in variants}, "span_2d": {}}
    for tag, fname in FILES.items():
        x, y = load_spectrum(DATA / fname)
        yc = y - als_baseline(y, lam=1e6, p=0.003, niter=15)
        for name, (win, npeak, base) in variants.items():
            out["variants"][name][tag] = fit(x, yc, win, npeak, base)
        noise = float(np.std(yc[x > 2850]))
        out["span_2d"][tag] = band_span(x, yc, D2_WIN, noise)

    RES.mkdir(exist_ok=True)
    with open(RES / "ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'变体':<16}{'谱':<5}{'G位':>9}{'G宽':>8}{'rms':>8}")
    for name, d in out["variants"].items():
        for tag, v in d.items():
            print(f"{name:<16}{tag:<5}{v['g_pos']:9.2f}{v['g_fwhm']:8.2f}{v['rms']:8.1f}")
    for tag, s in out["span_2d"].items():
        print(f"2D 展宽区间 {tag}: {s['lo']:.0f}--{s['hi']:.0f} cm^-1" if s else f"{tag}: 无")


if __name__ == "__main__":
    main()
