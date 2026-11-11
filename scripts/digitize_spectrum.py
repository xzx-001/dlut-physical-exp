#!/usr/bin/env python3
"""从 rainbow_spectrum*.png 里反解光谱曲线与峰位（坐标轴标定 + 轨迹提取 + 峰拟合）

思路：
  1. 用坐标轴刻度文字（深灰 110,110,115）的包围盒中心做线性标定，不依赖网格线；
  2. 用“蓝色”掩码逐列取轨迹（蓝线抗锯齿，取列内蓝像素的中位行，陡沿取最上行做包络）；
  3. 峰：包络上找局部极大 → 抛物线细化 → 半高全宽（FWHM）与两峰间距。

用法：
  python3 scripts/digitize_spectrum.py ../rainbow_spectrum-温度.png
  python3 scripts/digitize_spectrum.py ../rainbow_spectrum.png --xlabel-unit um
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


# ---------------- 通用：文字/颜色掩码 ----------------
def _text_mask(a):
    r, g, b = (a[:, :, i].astype(int) for i in range(3))
    return (r < 175) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25)


def _blue_mask(a):
    """曲线蓝（#378ADD）及其抗锯齿边缘；用色距把图例里的浅蓝样本排除掉"""
    r, g, b = (a[:, :, i].astype(int) for i in range(3))
    loose = (b > 120) & (b - r > 40) & (b - g > 25)
    d_curve = (r - 55) ** 2 + (g - 138) ** 2 + (b - 221) ** 2      # 曲线色
    d_legend = (r - 136) ** 2 + (g - 185) ** 2 + (b - 233) ** 2    # 图例浅蓝
    return loose & (d_curve <= d_legend)


def _largest_component(mask):
    """取最大连通域（曲线是贯穿全图的折线，图例/标注是孤立小块）"""
    from collections import deque
    seen = np.zeros_like(mask, bool)
    best = None
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q, comp = deque([(y0, x0)]), []
        seen[y0, x0] = True
        while q:
            y, x = q.popleft()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] \
                            and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
        if best is None or len(comp) > len(best):
            best = comp
    out = np.zeros_like(mask, bool)
    for y, x in best:
        out[y, x] = True
    return out


def _clusters(idx, gap=3):
    """把一维像素下标聚成连通簇，返回每簇的中心与跨度"""
    idx = np.sort(idx)
    out, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(cur)
            cur = [v]
    out.append(cur)
    return [(float(np.mean(c)), int(min(c)), int(max(c))) for c in out]


def calibrate(a):
    """返回 x→波长、y→功率 的线性映射锚点（用刻度标签中心，不用网格线）

    关键点：标签行/列里往往还夹着轴标题（“波长(knm)”），所以先按行(列)分 band，
    再在每个 band 内部按列(行)分组，取“分组数最多”的那个 band 作为刻度标签。
    """
    tm = _text_mask(a)
    blue = _largest_component(_blue_mask(a))
    ys, xs = np.nonzero(blue)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

    # ---- x 轴标签（绘图区下方）----
    below = tm.copy()
    below[: y1 + 6, :] = False
    rows = np.where(below.sum(axis=1) > 2)[0]
    bands = _clusters(rows, gap=4) if len(rows) else []
    best, best_n = None, 0
    for _, r0, r1 in bands:
        cols = np.where(below[r0 : r1 + 1, :].sum(axis=0) > 0)[0]
        g = [c for c in _clusters(cols, gap=10) if c[2] - c[1] >= 6]
        if len(g) > best_n:
            best, best_n = g, len(g)

    # ---- y 轴标签（绘图区左侧）----
    left = tm.copy()
    left[:, x0 - 4 :] = left[:, x0 - 4 :] * False  # 只留左侧，右侧全清
    left[:, x0 - 3 :] = False
    cols_l = np.where(left.sum(axis=0) > 2)[0]
    cbands = _clusters(cols_l, gap=4) if len(cols_l) else []
    ybest, ybest_n = None, 0
    for _, c0, c1 in cbands:
        rr = np.where(left[:, c0 : c1 + 1].sum(axis=1) > 0)[0]
        g = [v for v in _clusters(rr, gap=10) if v[2] - v[1] >= 4]
        if len(g) > ybest_n:
            ybest, ybest_n = g, len(g)
    return dict(xlab=best, ylab=ybest, blue=blue, box=(x0, x1, y0, y1))


def fit_linear_axis(centers, values):
    c = np.asarray(centers, float)
    v = np.asarray(values, float)
    k, b = np.polyfit(c, v, 1)
    res = v - (k * c + b)
    return k, b, float(np.abs(res).max())


def extract_trace(blue, box):
    x0, x1, y0, y1 = box
    cols = np.arange(x0, x1 + 1)
    top = np.full(len(cols), np.nan)
    med = np.full(len(cols), np.nan)
    for i, c in enumerate(cols):
        r = np.where(blue[:, c])[0]
        if len(r):
            top[i] = r.min()
            med[i] = np.median(r)
    return cols, top, med


def smooth(y, win=5):
    """中值滑动平滑，压掉噪声毛刺（NaN 不参与）"""
    out = y.copy()
    h = win // 2
    for i in range(len(y)):
        seg = y[max(0, i - h) : min(len(y), i + h + 1)]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def find_peaks(y_top, min_prom=8.0, min_sep=6):
    """在包络（像素行号，越小越高）上找局部极大"""
    y = -y_top  # 转成“越高越大”，NaN 保持
    idx = []
    n = len(y)
    for i in range(1, n - 1):
        if not np.isfinite(y[i - 1 : i + 2]).all():
            continue
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            idx.append(i)
    # 按凸起高度过滤
    keep = []
    for i in idx:
        lo, hi = max(0, i - 12), min(n, i + 13)
        seg = y[lo:hi]
        if not np.isfinite(seg).any():
            continue
        if y[i] - np.nanmin(seg) >= min_prom:
            keep.append(i)
    # 合并过近的峰（保留更高者）
    keep.sort(key=lambda i: -y[i])
    out = []
    for i in keep:
        if all(abs(i - j) >= min_sep for j in out):
            out.append(i)
    return sorted(out)


def refine_peak(y_top, i, half=4):
    """抛物线拟合细化峰位（像素亚精度）"""
    lo, hi = max(0, i - half), min(len(y_top), i + half + 1)
    y = -y_top[lo:hi]
    x = np.arange(lo, hi, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float(i)
    c = np.polyfit(x[ok], y[ok], 2)
    if c[0] >= 0:
        return float(i)
    return float(-c[1] / (2 * c[0]))


def fwhm_px(y_top, i, level_frac=0.5, span=90):
    """半高全宽（像素）。基线取两侧 span 内的最低值；走到窗口边缘仍未降到半高 → 判为“不可分”。"""
    n = len(y_top)
    y = -y_top
    lo, hi = max(0, i - span), min(n, i + span + 1)
    base = np.nanmin(y[lo:hi])
    peak = y[i]
    half = base + level_frac * (peak - base)
    res = {}
    for tag, step in (("right", +1), ("left", -1)):
        j = i
        while 0 <= j + step < n and abs(j - i) < span and \
                np.isfinite(y[j + step]) and y[j + step] > half:
            j += step
        res[tag] = (j, abs(j - i) < span and np.isfinite(y[j + step]))
    width = res["right"][0] - res["left"][0]
    resolved = res["right"][1] and res["left"][1]
    return width, base, peak, half, resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--xlabel-unit", default="um", choices=["um", "nm"],
                    help="x 轴标签的物理单位：um 表示 1.525=1525 nm")
    ap.add_argument("--json", default=None, help="把结果写到该 json")
    args = ap.parse_args()

    a = np.array(Image.open(args.png).convert("RGB"))
    cal = calibrate(a)
    print("x 轴标签簇数 %d，y 轴标签簇数 %d"
          % (len(cal["xlab"] or []), len(cal["ylab"] or [])))

    # x 标定：标签值按顺序排列（图像左→右递增）
    xc = [c[0] for c in (cal["xlab"] or [])]
    if len(xc) < 3:
        raise SystemExit("未能识别 x 轴刻度，请检查图片或改用其他标定方式")
    # 由标签文本宽度估算值：这里用等间隔假设，取首末已知值
    # 刻度值需人工给出：从图上看是 1.525 起、步长 0.005
    n = len(xc)
    if abs(n - 9) <= 1:
        vals = 1.525 + 0.005 * np.arange(n)
    else:
        vals = np.linspace(1.525, 1.565, n)
    kx, bx, ex = fit_linear_axis(xc, vals)
    print("x 标定：每像素 %.3e（残差 %.4f）→ λ = %.4f + %.5f·col"
          % (kx, ex, bx, kx))

    # y 标定：标签 0..60000 共 7 个
    yc = [c[0] for c in (cal["ylab"] or [])]
    yv = 60000 - 10000 * np.arange(len(yc))  # 自上而下递减
    ky, by, ey = fit_linear_axis(yc, yv)
    print("y 标定：共 %d 个标签，残差 %.1f dBm" % (len(yc), ey))

    cols, top, med = extract_trace(cal["blue"], cal["box"])
    lam = bx + kx * cols
    if args.xlabel_unit == "um":
        lam_nm = lam * 1000.0
    else:
        lam_nm = lam

    peaks = find_peaks(smooth(top, 5))
    ys_s = -smooth(top, 5)
    yseg = ys_s
    def _prom_px(i):
        seg = ys_s[max(0, i - 60) : i + 61]
        seg = seg[np.isfinite(seg)]
        return ys_s[i] - seg.min() if len(seg) else 0.0
    peaks = [i for i in peaks if _prom_px(i) * abs(ky) >= 2500]
    print("\n识别到 %d 个峰（按波长排序）：" % len(peaks))
    res = []
    for i in peaks:
        ip = refine_peak(top, i)
        lam_p = bx + kx * cols[int(round(ip))]
        lam_p_nm = lam_p * 1000 if args.xlabel_unit == "um" else lam_p
        w, base, pk, half, resolved = fwhm_px(top, i)
        _seg = -smooth(top, 5)[max(0, i - 60) : i + 61]
        _seg = _seg[np.isfinite(_seg)]
        _prom_px = (-smooth(top, 5)[i] - _seg.min()) if len(_seg) else 0.0
        fwhm_nm = abs(bx + kx * (cols[i] + w / 2)) - abs(bx + kx * (cols[i] - w / 2))
        fwhm_nm = fwhm_nm * 1000 if args.xlabel_unit == "um" else fwhm_nm
        pw = by + ky * top[i]
        prom = abs(ky) * _prom_px
        res.append(dict(lam_nm=float(lam_p_nm), power_dbm=float(pw), prom_dbm=float(prom),
                        fwhm_nm=float(fwhm_nm) if resolved else None,
                        px_col=float(cols[int(round(ip))])))
        print("  λ = %.3f nm  峰功率 %.0f dBm  凸起 %.0f dBm  %s"
              % (lam_p_nm, pw, prom,
                 ("FWHM ≈ %.3f nm" % fwhm_nm) if resolved else "FWHM 不可分（与邻峰粘连）"))

    if len(res) >= 2:
        order = np.argsort([r["lam_nm"] for r in res])
        r = [res[i] for i in order]
        print("\n相邻峰间距：", ", ".join("%.3f nm" % (r[i + 1]["lam_nm"] - r[i]["lam_nm"])
                                     for i in range(len(r) - 1)))
        print("最强峰（最高 dBm）：λ = %.3f nm，功率 %.0f dBm"
              % (max(res, key=lambda x: x["power_dbm"])["lam_nm"],
                 max(x["power_dbm"] for x in res)))
        print("基线 ≈ %.0f dBm，主峰最高 ≈ %.0f dBm"
              % (by + ky * np.nanpercentile(top, 95), by + ky * np.nanmin(top)))

    if args.json:
        json.dump(dict(peaks=res, x_cal=dict(k=kx, b=bx, resid=ex),
                       y_cal=dict(k=ky, b=by, resid=ey)),
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("写出", args.json)


if __name__ == "__main__":
    main()
