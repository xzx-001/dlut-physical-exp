"""PECVD 石墨烯拉曼光谱数据处理主流程。

处理链：
    读入两列光谱 → Eilers 非对称最小二乘去除荧光/瑞利尾 → 分窗口多峰
    洛伦兹拟合（含空气 O2 / N2 拉曼线）→ 峰位 / 半高宽 / 峰高 / 峰面积
    → 比值 I_D/I_G、I_2D/I_G、I_N2/I_O2 → 结果落盘 results/

空气拉曼线说明：
    光路中的空气会给出 O2（1554.6 cm^-1）与 N2（2330.7 cm^-1）两条
    气体拉曼线。本数据的 G 带窗口内 O2 线与 G 峰相距约 27 cm^-1，其
    宽度（FWHM 6.7-7.2）不足 G 峰（17.0-17.1）的一半，必须作为独立
    分量一并拟合：不建模时该窗口的残差均方根由约 16-21 计数升到
    30-33 计数，而 G 峰半高宽只变化 0.1 cm^-1 量级（复算见
    scripts/ablation.py）；2330 cm^-1 处的宽峰（FWHM 6.3 cm^-1）即
    N2 线，不是探测器坏列（坏列不会呈对称的洛伦兹型）。两条空气线
    同时给出强度比的独立校验：I_N2/I_O2 ≈ 4，与空气组成 78:21 相符。

用法：
    python3 scripts/analyze.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import curve_fit

from raman_io import (DATA, RES, als_baseline, fwhm_area, load_spectrum,
                      lorentz)

FILES = {
    "g1": "graphene_100mW_50X-.txt",
    "g2": "graphene_100mW_50X-1.txt",
    "g3": "graphene_100mW_50X-2.txt",
}
SI_FILE = "Si.txt"

# 文献公认的拉曼位移（用于刻度核对，见报告 §3.3）
CALIB = {
    "Si": 520.5,    # Si 一阶光学声子，300 K
    "O2": 1554.6,   # 空气 O2 v=1 伸缩
    "N2": 2330.7,   # 空气 N2 v=1 伸缩
}

# 拟合窗口 (cm^-1) 与每个窗口内的峰初值 / 位置边界
BANDS = {
    "G": {"win": (1525.0, 1615.0),
          "peaks": [{"p0": 1555.0, "bounds": (1538.0, 1575.0)},   # O2(空气)
                    {"p0": 1582.0, "bounds": (1570.0, 1600.0)}]},  # G
    "Dp": {"win": (1600.0, 1690.0), "peaks": [{"p0": 1620.0}]},
    "2D": {"win": (2560.0, 2790.0), "peaks": [{"p0": 2680.0},
                                              {"p0": 2720.0}]},
    "N2": {"win": (2300.0, 2360.0),
           "peaks": [{"p0": 2330.0, "bounds": (2315.0, 2345.0)}]},
}


def fit_window(x, y, win, peaks):
    """窗口内拟合若干洛伦兹峰 + 线性基线。"""
    m = (x >= win[0]) & (x <= win[1])
    xs, ys = x[m], y[m]
    npeaks = len(peaks)
    span = (win[1] - win[0]) / 2.0

    def model(xx, *p):
        out = p[0] + p[1] * (xx - xs.mean())
        for k in range(npeaks):
            x0, fw, amp = p[2 + 3 * k: 5 + 3 * k]
            out = out + lorentz(xx, x0, fw, amp)
        return out

    p0 = [float(np.median(ys)), 0.0]
    lo = [-np.inf, -np.inf]
    hi = [np.inf, np.inf]
    for spec in peaks:
        pos = spec["p0"]
        b = spec.get("bounds", (pos - span / 2, pos + span / 2))
        p0 += [pos, 20.0, float(np.interp(pos, xs, ys) - np.median(ys))]
        lo += [b[0], 2.0, -np.inf]
        hi += [b[1], 250.0, np.inf]

    popt, pcov = curve_fit(model, xs, ys, p0=p0, bounds=(lo, hi), maxfev=60000)
    perr = np.sqrt(np.abs(np.diag(pcov)))
    resid = ys - model(xs, *popt)
    rss = float(np.sum(resid ** 2))
    aic = len(xs) * np.log(rss / len(xs)) + 2 * len(popt)
    out_peaks = []
    for i in range(npeaks):
        x0, fw, amp = popt[2 + 3 * i: 5 + 3 * i]
        e = perr[2 + 3 * i: 5 + 3 * i]
        out_peaks.append({"pos": float(x0), "pos_err": float(e[0]),
                          "fwhm": float(fw), "fwhm_err": float(e[1]),
                          "height": float(amp), "height_err": float(e[2]),
                          "area": float(fwhm_area(fw, amp)),
                          "shape": "lorentz"})
    return {"peaks": out_peaks, "aic": float(aic),
            "rms": float(np.sqrt(rss / len(xs))), "window": list(win),
            "baseline": [float(popt[0]), float(popt[1])]}


def band_stat(x, yc, win, shoulder=(1300.0, 1460.0)):
    """窗口内以两端连线的线性基线统计宽弱带的峰高与积分面积。

    D 带区只有宽弱鼓包时，多点洛伦兹拟合会退化到边界，改用不依赖
    峰形的统计量：峰高 = 平滑后最大值相对基线，面积 = 基线以上积分。
    """
    m = (x >= win[0]) & (x <= win[1])
    xs, ys = x[m], yc[m]
    left, right = shoulder
    y0 = np.interp(left, xs, ys)
    y1 = np.interp(right, xs, ys)
    lin = y0 + (y1 - y0) * (xs - left) / (right - left)
    d = ys - lin
    k = np.ones(7) / 7.0
    ds = np.convolve(d, k, mode="same")
    i = int(np.argmax(ds))
    return {"pos": float(xs[i]), "height": float(ds[i]),
            "area": float(np.trapezoid(np.clip(d, 0, None), xs)),
            "window": [float(win[0]), float(win[1])],
            "noise": float(np.std(d)), "noise_smooth": float(np.std(ds))}


def process_graphene(tag, fname):
    x, y = load_spectrum(DATA / fname)
    base = als_baseline(y, lam=1e6, p=0.003, niter=15)
    yc = y - base
    out = {"file": fname, "n_points": int(len(x)),
           "x_min": float(x.min()), "x_max": float(x.max()),
           "x_step": float(np.median(np.diff(x))),
           "raw_max": float(y.max()), "raw_min": float(y.min()),
           "noise_bg": float(np.std(yc[x > 2850])),
           "noise_low": float(np.std(yc[(x < 1150)]))}
    fits = {}
    for band, cfg in BANDS.items():
        fits[band] = fit_window(x, yc, cfg["win"], cfg["peaks"])
    out["bands"] = fits
    D = band_stat(x, yc, (1320.0, 1440.0))

    GO2 = fits["G"]["peaks"]
    O2, G = GO2[0], GO2[1]
    p2 = fits["2D"]["peaks"]
    N2 = fits["N2"]["peaks"][0]
    A2D = sum(p["area"] for p in p2)
    H2D = max(p["height"] for p in p2)
    out["summary"] = {
        "D_pos": D["pos"], "D_height": D["height"], "D_area": D["area"],
        "D_noise": D["noise"], "D_noise_smooth": D["noise_smooth"],
        "G_pos": G["pos"], "G_fwhm": G["fwhm"], "G_height": G["height"],
        "O2_pos": O2["pos"], "O2_fwhm": O2["fwhm"], "O2_height": O2["height"],
        "N2_pos": N2["pos"], "N2_fwhm": N2["fwhm"], "N2_height": N2["height"],
        "2D_pos_max": p2[int(np.argmax([p["height"] for p in p2]))]["pos"],
        "2D_components": [{"pos": p["pos"], "fwhm": p["fwhm"],
                           "height": p["height"]} for p in p2],
        "I2D_over_IG_height": H2D / G["height"],
        "I2D_over_IG_area": A2D / G["area"],
        "ID_over_IG_height": D["height"] / G["height"],
        "ID_over_IG_area": D["area"] / G["area"],
        "IO2_over_IG": O2["height"] / G["height"],
        "IN2_over_IG": N2["height"] / G["height"],
        "IN2_over_IO2": N2["height"] / O2["height"],
        "D_is_resolved": False,
    }
    return out, dict(x=x, y=y, base=base, yc=yc)


def process_si():
    x, y = load_spectrum(DATA / SI_FILE)
    base = als_baseline(y, lam=1e6, p=0.003, niter=15)
    yc = y - base
    fit = fit_window(x, yc, (495.0, 552.0), [{"p0": 520.0}])
    return ({"file": SI_FILE, "n_points": int(len(x)),
             "x_min": float(x.min()), "x_max": float(x.max()),
             "x_step": float(np.median(np.diff(x))), **fit},
            dict(x=x, y=y, base=base, yc=yc))


def main():
    RES.mkdir(exist_ok=True)
    result = {"graphene": {}, "si": None, "calib_ref": CALIB}
    cache = {}
    for tag, fname in FILES.items():
        result["graphene"][tag], cache[tag] = process_graphene(tag, fname)
    result["si"], cache["si"] = process_si()

    # 刻度核对：实测 vs 公认值
    si = result["si"]["peaks"][0]
    dev = {"Si": si["pos"] - CALIB["Si"]}
    for line, tag in (("O2", "O2_pos"), ("N2", "N2_pos")):
        vals = [result["graphene"][t]["summary"][tag] for t in FILES]
        dev[line] = float(np.mean(vals)) - CALIB[line]
    result["calib_dev"] = dev

    with open(RES / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    hdr = (f"{'样品':<4}{'G位':>9}{'G宽':>7}{'G高':>7}{'O2位':>8}{'N2位':>8}"
           f"{'N2宽':>7}{'2D₁':>8}{'2D₂':>8}{'I2D/IG':>8}{'ID/IG':>8}"
           f"{'A2D/AG':>8}{'N2/O2':>7}")
    print(hdr)
    for tag, out in result["graphene"].items():
        s, b = out["summary"], out["bands"]
        print(f"{tag:<4}{s['G_pos']:9.2f}{s['G_fwhm']:7.2f}{s['G_height']:7.0f}"
              f"{s['O2_pos']:8.2f}{s['N2_pos']:8.2f}{s['N2_fwhm']:7.2f}"
              f"{b['2D']['peaks'][0]['pos']:8.1f}{b['2D']['peaks'][1]['pos']:8.1f}"
              f"{s['I2D_over_IG_height']:8.2f}{s['ID_over_IG_height']:8.3f}"
              f"{s['I2D_over_IG_area']:8.2f}{s['IN2_over_IO2']:7.2f}")
    print(f"\nSi 参考峰：{si['pos']:.2f}±{si['pos_err']:.2f} cm^-1，"
          f"FWHM {si['fwhm']:.2f} cm^-1，步长 {result['si']['x_step']:.3f} cm^-1")
    print("刻度偏差 (cm^-1)：" +
          "  ".join(f"{k}{v:+.2f}" for k, v in result["calib_dev"].items()))
    np.savez(RES / "arrays.npz",
             **{f"{k}_{n}": v[n] for k, v in cache.items()
                for n in ("x", "y", "base", "yc")})


if __name__ == "__main__":
    main()
