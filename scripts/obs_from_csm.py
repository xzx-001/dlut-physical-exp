"""从原始 .csm 重算观测向量（不读 results/*.json）。

产出 observations.json，供 mcmc_fit.py 使用。
可独立重跑：python3 obs_from_csm.py
"""
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
from csmio import CSMImage          # noqa: E402
from afmproc import flatten_rows, fft_period, phase_tilt  # noqa: E402

DATA = os.path.join(PROJ, 'data', '20261026')
DISCS = ['cd1', 'cd2', 'cd3']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'observations.json')


def row_periods(z, dx, block=None):
    """逐行 FFT 主峰周期（nm）。block=(i0,i1) 时只取该区间行。"""
    ny, nx = z.shape
    x = np.arange(nx, dtype=float)
    pers = []
    rows = range(ny) if block is None else range(block[0], block[1])
    for i in rows:
        p = z[i] - z[i].mean()
        c = np.polyfit(x, p, 1)
        p = p - np.polyval(c, x)
        per, _, _ = fft_period(p, dx)
        if np.isfinite(per):
            pers.append(per)
    return np.array(pers)


def smooth_depth(z, dx, smooth_nm=150.0, lo=1, hi=99):
    """逐行 150 nm 平滑剖面 P99-P1（nm），返回中位数与逐行数组。"""
    ny, nx = z.shape
    n_s = max(3, int(round(smooth_nm / dx)))
    kern = np.ones(n_s) / n_s
    out = []
    for i in range(ny):
        ps = np.convolve(z[i] - z[i].mean(), kern, mode='same')
        m = ps[n_s:-n_s] if n_s * 2 < nx else ps
        out.append(np.percentile(m, hi) - np.percentile(m, lo))
    return np.array(out)


def main():
    res = {'discs': {}, 'meta': {}}
    for d in DISCS:
        f = os.path.join(DATA, f'{d}-形貌000.csm')
        im = CSMImage(f)
        z, unit = im.physical()
        assert unit == 'nm', unit
        dx = im.px_size_nm
        zp = flatten_rows(z, 1)                      # 去逐行线性倾斜（不改变周期）

        P = row_periods(zp, dx)
        depth = smooth_depth(zp, dx)
        P_med, tilt = phase_tilt(zp, dx)[0], phase_tilt(zp, dx)[1]

        # 位置相关：把 1024 行分 8 块，各块周期中位数 → 检验标定是否为纯乘性常数
        blocks = []
        for b in range(8):
            i0, i1 = b * 128, (b + 1) * 128
            blocks.append(float(np.median(row_periods(zp, dx, (i0, i1)))))

        # 密致抽样：整幅重采样到 800x800 后的周期（检验边缘 vs 中心）
        res['discs'][d] = {
            'file': os.path.basename(f),
            'scan_size_nm': im.scan_size_nm,
            'px_nm': dx,
            'row_period_median': float(np.median(P)),
            'row_period_mean': float(np.mean(P)),
            'row_period_std': float(np.std(P)),
            'row_period_p16': float(np.percentile(P, 16)),
            'row_period_p25': float(np.percentile(P, 25)),
            'row_period_p75': float(np.percentile(P, 75)),
            'row_period_p84': float(np.percentile(P, 84)),
            'row_period_n': int(len(P)),
            'row_periods_sample': [float(v) for v in P[::16]],   # 每 16 行取 1，共 64 点
            'block_periods_8': blocks,
            'block_rel_spread': float((max(blocks) - min(blocks)) / np.median(blocks)),
            'depth_pp_median': float(np.median(depth)),
            'depth_pp_std': float(np.std(depth)),
            'tilt_px_per_row': tilt,
            'tilt_deg': float(np.degrees(np.arctan(tilt))),
            'meta': {k: im.meta.get(k) for k in
                     ('ScanSpeed', 'ZLimit', 'Sensitivity', 'HeightScale', 'Scan Angle',
                      'X Fast', 'Y Fast', 'X Slow', 'Y Slow', 'Max Scope', 'Scanner Name',
                      'AX', 'BX', 'CX', 'DX', 'Series', 'Date')},
        }
        print(f'{d}: P={np.median(P):.2f} nm  IQR=({np.percentile(P,25):.1f},'
              f'{np.percentile(P,75):.1f})  depth={np.median(depth):.1f} nm  '
              f'tilt={np.degrees(np.arctan(tilt)):+.2f} deg  '
              f'block spread={100*(max(blocks)-min(blocks))/np.median(blocks):.2f}%')

    res['meta']['scanner_coeffs'] = {'X_fast': 251.087, 'Y_fast': 251.943,
                                     'X_slow': 318.454, 'Y_slow': 317.546}
    res['meta']['standards'] = {
        'CD': {'pitch_nm': 1600.0, 'tol_nm': 100.0, 'src': 'ECMA-130 11.3'},
        'DVD': {'pitch_nm': 740.0, 'tol_nm': 30.0, 'src': 'ECMA-267 10.6.2'},
    }
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
