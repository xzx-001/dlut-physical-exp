"""条纹相位随行号的线性/二次成分分析 —— 判别"倾角" vs "光盘轨道曲率"。

用法: python3 curve_check.py
输出: 每个样品的主周期、相位漂移(周期数)、线/二次分解、隐含曲率半径、分带局部倾角
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csmio, afmproc

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '20261026')

out = {}
for tag in ('cd1', 'cd2', 'cd3'):
    f = f'{DATA}/{tag}-形貌000.csm'
    img = csmio.CSMImage(f)
    z, _unit = img.physical()
    z = z.astype(np.float64)
    dx = img.px_size_nm
    a = afmproc.flatten_rows(z, 1)

    # 主周期
    pers = []
    x = np.arange(z.shape[1])
    for i in range(0, z.shape[0], 4):
        p = a[i] - a[i].mean()
        p -= np.polyval(np.polyfit(x, p, 1), x)
        per, _, _ = afmproc.fft_period(p, dx)
        if np.isfinite(per):
            pers.append(per)
    P = float(np.median(pers))
    f0 = dx / P

    # 逐行相位（不抽样，全部 1024 行）
    idx, ph = [], []
    for i in range(z.shape[0]):
        p = a[i] - a[i].mean()
        p -= np.polyval(np.polyfit(x, p, 1), x)
        idx.append(i)
        ph.append(np.angle(np.sum(p * np.exp(-2j * np.pi * f0 * x))))
    ph = np.unwrap(np.array(ph))
    idx = np.array(idx, dtype=float)

    # 线性拟合
    c1 = np.polyfit(idx, ph, 1)
    r1 = ph - np.polyval(c1, idx)
    # 二次拟合
    c2 = np.polyfit(idx, ph, 2)
    r2 = ph - np.polyval(c2, idx)

    ny = float(z.shape[0] - 1)
    drift_lin = c1[0] * ny / (2 * np.pi)          # 周期数
    quad_at_half = c2[0] * (ny / 2) ** 2 / (2 * np.pi)  # 中点到边缘的二次贡献（周期数）
    quad_total = c2[0] * ny ** 2 / (2 * np.pi)

    # 二次项等效的曲率半径：δx(y) = y²/(2R)
    shift_nm_half = -quad_at_half * P            # 半幅处的横移 nm
    y_half = ny * dx / 2
    R_implied = y_half ** 2 / (2 * shift_nm_half) if shift_nm_half != 0 else np.inf

    # 分带局部倾角（8 带 × 128 行）
    bands = []
    nband = 8
    L = z.shape[0] // nband
    for b in range(nband):
        sl = slice(b * L, (b + 1) * L)
        bl = ph[sl]
        ii = idx[sl]
        s = np.polyfit(ii, bl, 1)[0]
        shift_px = -s / (2 * np.pi * f0)
        bands.append(dict(band=b, y_nm=float(ii.mean() * dx),
                          tilt_deg=float(np.degrees(np.arctan(shift_px)))))

    bt = np.array([d['tilt_deg'] for d in bands])
    by = np.array([d['y_nm'] for d in bands])
    bfit = np.polyfit(by, bt, 1)

    out[tag] = dict(
        period_nm=P, dx_nm=dx,
        drift_linear_periods=float(drift_lin),
        quad_total_periods=float(quad_total),
        quad_halfamp_periods=float(quad_at_half),
        rms_lin_periods=float(np.std(r1) / (2 * np.pi)),
        rms_quad_periods=float(np.std(r2) / (2 * np.pi)),
        implied_R_from_quad_mm=float(R_implied / 1e6),
        tilt_global_deg=float(np.degrees(np.arctan(-c1[0] / (2 * np.pi * f0)))),
        tilt_band_min=float(bt.min()), tilt_band_max=float(bt.max()),
        tilt_band_spread=float(bt.max() - bt.min()),
        tilt_vs_y_slope_deg_per_um=float(bfit[0] * 1e3),
        bands=bands,
    )

print(json.dumps(out, indent=2, ensure_ascii=False))
