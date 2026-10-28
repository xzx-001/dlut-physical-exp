"""AFM CD-R 实验数据处理主流程 —— 对应 Imager 4.7 后处理的操作：
    消除扫描线 → 2 阶面整平 → 粗糙度 → 剖面线 → FFT/PSD → 三维图

用法：
    python3 analyze.py <数据目录> <输出目录>
"""
import json
import os
import sys
import glob

import numpy as np

import afmproc as ap
from csmio import CSMImage

SAMPLES = ['cd1', 'cd2', 'cd3']
CHANNELS = ['形貌', '探针起伏', '横向力']


def process(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = {'data_dir': data_dir, 'samples': {}}
    arrays = {}
    extra = {}

    for s in SAMPLES:
        summary['samples'][s] = {}
        for c in CHANNELS:
            path = os.path.join(data_dir, f'{s}-{c}000.csm')
            im = CSMImage(path)
            z, unit = im.physical()
            dx = im.px_size_nm

            # ---- 处理链：去扫描线(逐行1阶) → 面整平(2阶) ----
            z_lines = ap.flatten_rows(z, 1)
            z_flat, coef = ap.plane_fit(z_lines, 2)

            rec = {
                'file': os.path.basename(path),
                'unit': unit,
                'size_px': [im.width, im.height],
                'scan_size_nm': im.scan_size_nm,
                'px_size_nm': dx,
                'z_fullscale': im.z_scale_text,
                'header': {k: im.meta.get(k) for k in
                           ('Date', 'Source', 'ScanSpeed', 'Ref', 'Signal Gain',
                            'Sensitivity', 'P.G.', 'I.G.', 'ZLimit', 'Filter',
                            'Surface Fit', 'ScanSize', 'Machine', 'Scanner Name',
                            'X Fast', 'Y Fast', 'X Slow', 'Y Slow', 'Max Scope')},
                'raw': {'min': float(z.min()), 'max': float(z.max()),
                        'mean': float(z.mean()), 'std': float(z.std())},
                'after_lines': {'std': float(z_lines.std()),
                                'row_mean_std': float(z_lines.mean(axis=1).std())},
                'after_flatten': {'min': float(z_flat.min()), 'max': float(z_flat.max()),
                                  'mean': float(z_flat.mean()), 'std': float(z_flat.std())},
            }

            if c == '形貌':
                rec['roughness_raw'] = ap.roughness(z, dx)
                rec['roughness_flat'] = ap.roughness(z_flat, dx)
                g = ap.groove_stats(z_flat, dx)
                skip = ('profile', 'prof_raw', 'psd_freq', 'psd_pow',
                        'row_periods', 'row_depths', 'period_iqr')
                rec['groove'] = {k: float(v) for k, v in g.items()
                                 if k not in skip and np.isscalar(v)}
                rec['groove']['period_iqr'] = [float(x) for x in g['period_iqr']]
                rec['groove']['row_period_std'] = float(np.std(g['row_periods']))
                rec['groove']['row_depth_median'] = float(np.median(g['row_depths']))
                # 附加统计（供报告 §3.2/§3.3/§3.7 引用，可复现）
                rm = z.mean(axis=1)
                extra[s] = dict(
                    row_mean_std=float(rm.std()),
                    col_mean_std=float(z.mean(axis=0).std()),
                    row_jump=float(np.abs(np.diff(rm)).mean()),
                    row_jump_max=float(np.abs(np.diff(rm)).max()),
                    raw_std=float(z.std()),
                    raw_col_mean=float(z.mean()),
                    ridge_width=float(g['duty'] * g['period_fft']),
                    trench_width=float((1 - g['duty']) * g['period_fft']),
                    profile_ptp=float(g['depth_pp_avg']),
                )
                # 未做行程对齐时的平均剖面峰谷（用于说明倾角的影响量级）
                p_un = z_flat.mean(axis=0)
                n_s = max(3, int(round(150.0 / dx)))
                ps_un = np.convolve(p_un - p_un.mean(), np.ones(n_s) / n_s, 'same')
                extra[s]['profile_p99_p1_unaligned'] = float(
                    np.percentile(ps_un[n_s:-n_s], 99) - np.percentile(ps_un[n_s:-n_s], 1))
                extra[s]['tilt_attenuation'] = float(
                    g['depth_pp_avg'] / extra[s]['profile_p99_p1_unaligned'])
                # 基频峰的归一化半高宽与二次谐波幅度比（报告 §3.4，可复现）
                f_psd = np.asarray(g['psd_freq'], float)
                P_psd = np.asarray(g['psd_pow'], float)
                f0_psd = 1.0 / g['period_fft']
                band = (f_psd > 0.6 * f0_psd) & (f_psd < 1.6 * f0_psd)
                if band.any():
                    k_pk = np.argmax(np.where(band, P_psd, -np.inf))
                    half = P_psd[k_pk] / 2.0
                    i = k_pk
                    while i > 0 and P_psd[i] > half:
                        i -= 1
                    j = k_pk
                    while j < len(P_psd) - 1 and P_psd[j] > half:
                        j += 1
                    extra[s]['fund_fwhm_norm'] = float(
                        (f_psd[j] - f_psd[i]) / f_psd[k_pk])
                    k2 = np.argmin(np.abs(f_psd - 2 * f_psd[k_pk]))
                    extra[s]['harmonic2_amp_ratio'] = float(
                        np.sqrt(P_psd[k2] / P_psd[k_pk]))
                arrays[f'{s}_{c}_raw'] = z
                arrays[f'{s}_{c}_flat'] = z_flat
                arrays[f'{s}_{c}_profile'] = g['profile']
                arrays[f'{s}_{c}_profraw'] = g['prof_raw']
            else:
                # 力/误差信号通道：不做面整平（信号本就围绕零点），只去行直流
                zc = ap.flatten_rows(z, 0)
                rec['signal'] = {'std': float(zc.std()), 'rms': float(np.sqrt(np.mean(zc ** 2))),
                                 'min': float(zc.min()), 'max': float(zc.max()),
                                 'ptp': float(zc.ptp() if hasattr(zc, 'ptp') else np.ptp(zc))}
                arrays[f'{s}_{c}_raw'] = z
                arrays[f'{s}_{c}_flat'] = zc

            summary['samples'][s][c] = rec

    # ---- 跨样品横向对比（形貌通道） ----
    cmp = {}
    for s in SAMPLES:
        r = summary['samples'][s]['形貌']
        cmp[s] = {
            'Sa_flat': r['roughness_flat']['Sa'],
            'Sq_flat': r['roughness_flat']['Sq'],
            'Sy_flat': r['roughness_flat']['Sy'],
            'period_fft_nm': r['groove']['period_fft'],
            'period_peaks_nm': r['groove']['period_peaks'],
            'tilt_deg': r['groove']['tilt_deg'],
            'depth_pp_nm': r['groove']['depth_pp'],
            'depth_pp_avg_nm': r['groove']['depth_pp_avg'],
            'duty': r['groove']['duty'],
            'z_fullscale': r['z_fullscale'],
            'sensitivity': r['header']['Sensitivity'],
            'signal_gain': r['header']['Signal Gain'],
        }
    summary['comparison_topo'] = cmp

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 只存处理后的 float32 数组（raw 可从 .csm 复算，不必重复占空间）
    small = {k: v.astype(np.float32) for k, v in arrays.items()
             if not k.endswith('_raw')}
    np.savez_compressed(os.path.join(out_dir, 'arrays.npz'), **small)
    with open(os.path.join(out_dir, 'extra_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/20261026'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '../results'
    sm = process(data_dir, out_dir)
    for s, v in sm['comparison_topo'].items():
        print(f"{s}: Sa={v['Sa_flat']:.2f}nm Sq={v['Sq_flat']:.2f}nm Sy={v['Sy_flat']:.2f}nm "
              f"周期={v['period_fft_nm']:.0f}nm 深度={v['depth_pp_nm']:.1f}nm "
              f"倾斜={v['tilt_deg']:+.1f}° duty={v['duty']:.2f}")
