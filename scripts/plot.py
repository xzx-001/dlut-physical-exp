"""AFM CD-R 数据处理 —— 出图（初步处理阶段）"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import afmproc as ap
from csmio import CSMImage

plt.rcParams.update({
    'font.family': 'Noto Sans CJK SC',
    'font.sans-serif': ['Noto Sans CJK SC', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

SAMPLES = ['cd1', 'cd2', 'cd3']
CHANNELS = ['形貌', '探针起伏', '横向力']
DATA = os.environ.get('AFM_DATA', '../data/20261026')      # .csm 数据目录
OUT = os.environ.get('AFM_FIG', '../figures')              # 出图目录
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    """同时输出 PNG（快速查看）与 PDF（LaTeX 矢量插图）"""
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight')


def load(s, c):
    im = CSMImage(os.path.join(DATA, f'{s}-{c}000.csm'))
    z, unit = im.physical()
    return im, z, unit


def fig_overview():
    fig, axes = plt.subplots(3, 3, figsize=(12, 12.5))
    for i, s in enumerate(SAMPLES):
        for j, c in enumerate(CHANNELS):
            im, z, unit = load(s, c)
            ax = axes[i, j]
            lo, hi = np.percentile(z, [0.5, 99.5])
            cmap = 'afmhot' if c == '形貌' else ('RdYlBu_r' if c == '横向力' else 'gray')
            m = ax.imshow(z, cmap=cmap, vmin=lo, vmax=hi, extent=[0, im.scan_size_nm / 1000,
                                                                 0, im.scan_size_nm / 1000])
            ax.set_title(f'{s} · {c}   [{lo:.1f}~{hi:.1f} {unit}]', fontsize=10)
            ax.set_xlabel('x / µm'); ax.set_ylabel('y / µm')
            cb = plt.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(f'{unit}', fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle('光盘表面 AFM 原始图像（接触模式，10 µm×10 µm，1024×1024）', fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, 'fig01_overview_raw')
    plt.close(fig)


def fig_processing():
    s = 'cd1'
    im, z, unit = load(s, '形貌')
    zl = ap.flatten_rows(z, 1)
    zf, coef = ap.plane_fit(zl, 2)
    stages = [('(a) 原始图像', z), ('(b) 逐行 1 阶去倾斜', zl), ('(c) 2 阶多项式面整平', zf)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (t, d) in zip(axes, stages):
        lo, hi = np.percentile(d, [1, 99])
        m = ax.imshow(d, cmap='afmhot', vmin=lo, vmax=hi,
                      extent=[0, im.scan_size_nm / 1000, 0, im.scan_size_nm / 1000])
        ax.set_title(f'{t}\nSq = {d.std():.2f} nm', fontsize=11)
        ax.set_xlabel('x / µm'); ax.set_ylabel('y / µm')
        plt.colorbar(m, ax=ax, fraction=0.046, pad=0.02).set_label('nm', fontsize=8)
    fig.suptitle(f'{s} 形貌通道处理链（对应 Imager「消除扫描线 + 面整平」）', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'fig03_processing_chain')
    plt.close(fig)


def fig_profiles(summary):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5))
    for ax, s in zip(axes, SAMPLES):
        im, z, unit = load(s, '形貌')
        zf, _ = ap.plane_fit(ap.flatten_rows(z, 1), 2)
        g = ap.groove_stats(zf, im.px_size_nm)
        x = np.arange(len(g['profile'])) * im.px_size_nm / 1000
        ok = np.isfinite(g['prof_raw'])
        ax.plot(x, g['prof_raw'], lw=0.3, color='#9ecae1', label='倾角对齐后逐列')
        ax.plot(x, g['profile'], lw=1.6, color='#08519c', label='150 nm 平滑')
        ax.set_xlim(0, x[-1])
        ax.set_xlabel('x / µm'); ax.set_ylabel('高度 / nm')
        ax.grid(alpha=.3, lw=.5)
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.set_title(f'{s} 平均剖面（倾角对齐） '
                     f'(周期 {g["period_fft"]:.0f} nm, 峰谷 {g["depth_pp"]:.0f} nm, '
                     f'条纹倾斜 {g["tilt_deg"]:+.1f}°)', fontsize=10)
        # 标注周期
        if np.isfinite(g['period_fft']):
            P = g['period_fft'] / 1000
            y0 = np.nanmax(g['profile'])
            ax.annotate('', xy=(1.0, y0), xytext=(1.0 + P, y0),
                        arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.4))
            ax.text(1.0 + P / 2, y0, f'{g["period_fft"]:.0f} nm', color='crimson',
                    fontsize=9, ha='center', va='bottom')
    fig.suptitle('形貌图平均剖面线（去除扫描线+面整平+倾角对齐后沿 y 平均）', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, 'fig04_profiles')
    plt.close(fig)


def fig_spectra():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, s in enumerate(SAMPLES):
        im, z, unit = load(s, '形貌')
        zf, _ = ap.plane_fit(ap.flatten_rows(z, 1), 2)
        dx = im.px_size_nm
        # 1D PSD
        g = ap.groove_stats(zf, dx)
        f, P = g['psd_freq'], g['psd_pow']
        m = (f > 0) & (f < 0.01)
        ax = axes[0, j]
        ax.loglog(1 / f[m] if np.isscalar(f) is False else f[m], P[m] * 1000, lw=1, color='#08519c')
        ax.set_xlabel('波长 / nm'); ax.set_ylabel('PSD / (nm³)')
        ax.set_title(f'{s} 一维功率谱（平均剖面）', fontsize=10)
        ax.axvline(g['period_fft'], color='crimson', ls='--', lw=1)
        ax.text(g['period_fft'] * 1.15, np.max(P[m] * 1000) * 0.5,
                f'{g["period_fft"]:.0f} nm', color='crimson', fontsize=8)
        ax.grid(alpha=.3, which='both', lw=.4)
        # 2D FFT
        A = zf - zf.mean()
        F = np.abs(np.fft.fftshift(np.fft.fft2(A)))
        F = np.log1p(F)
        ny, nx = F.shape
        ext = [-(nx / 2) / (nx * dx) * 1000, (nx / 2) / (nx * dx) * 1000,
               -(ny / 2) / (ny * dx) * 1000, (ny / 2) / (ny * dx) * 1000]
        ax2 = axes[1, j]
        m2 = ax2.imshow(F, origin='lower', extent=ext, cmap='inferno',
                        vmin=np.percentile(F, 70), vmax=np.percentile(F, 99.9))
        ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2)
        ax2.set_xlabel('k$_x$ / µm$^{-1}$'); ax2.set_ylabel('k$_y$ / µm$^{-1}$')
        ax2.set_title(f'{s} 二维 FFT（对数强度）', fontsize=10)
        plt.colorbar(m2, ax=ax2, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle('周期性结构频谱分析', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, 'fig05_spectra')
    plt.close(fig)


def fig_3d():
    fig = plt.figure(figsize=(15, 5))
    for j, s in enumerate(SAMPLES):
        im, z, unit = load(s, '形貌')
        zf, _ = ap.plane_fit(ap.flatten_rows(z, 1), 2)
        ds = 2
        zz = zf[::ds, ::ds]
        ny, nx = zz.shape
        X, Y = np.meshgrid(np.arange(nx) * im.px_size_nm * ds / 1000,
                           np.arange(ny) * im.px_size_nm * ds / 1000)
        ax = fig.add_subplot(1, 3, j + 1, projection='3d')
        ax.plot_surface(X, Y, zz, cmap='afmhot', rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=True)
        ax.view_init(elev=42, azim=-58)
        ax.set_xlabel('x / µm', fontsize=8); ax.set_ylabel('y / µm', fontsize=8)
        ax.set_zlabel('z / nm', fontsize=8)
        ax.set_title(f'{s} 三维形貌  (Sq={zf.std():.1f} nm)', fontsize=10)
        ax.tick_params(labelsize=6)
    fig.suptitle('光盘表面三维形貌（2 阶整平后，2×降采样显示）', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'fig02_3d_surface')
    plt.close(fig)


def fig_channels():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    s = 'cd1'
    for j, c in enumerate(CHANNELS):
        im, z, unit = load(s, c)
        ax = axes[0, j]
        lo, hi = np.percentile(z, [1, 99])
        cmap = 'afmhot' if c == '形貌' else ('RdYlBu_r' if c == '横向力' else 'gray')
        m = ax.imshow(z, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(f'{s} · {c}（原始，{lo:.3g}~{hi:.3g} {unit}）', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(m, ax=ax, fraction=0.046, pad=0.02).set_label(unit, fontsize=8)
        # 第一行同一行剖面
        zc = ap.flatten_rows(z, 0) if c != '形貌' else ap.flatten_rows(z, 1)
        ax2 = axes[1, j]
        x = np.arange(z.shape[1]) * im.px_size_nm / 1000
        for k in (300, 500, 700):
            ax2.plot(x, zc[k], lw=0.8, alpha=.8, label=f'第 {k} 行')
        ax2.set_xlabel('x / µm'); ax2.set_ylabel(f'信号 / {unit}')
        ax2.grid(alpha=.3, lw=.5); ax2.legend(fontsize=7)
        ax2.set_title(f'{s} · {c} 单行剖面', fontsize=10)
    fig.suptitle('三通道对照：形貌（反馈高度） / 探针起伏（误差信号） / 横向力（摩擦）', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, 'fig06_channels')
    plt.close(fig)


def fig_roughness(summary):
    keys = ['Sa', 'Sq', 'Sy', 'Sz']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.26
    xs = np.arange(len(keys))
    for i, s in enumerate(SAMPLES):
        r = summary['samples'][s]['形貌']['roughness_flat']
        axes[0].bar(xs + (i - 1) * width, [r[k] for k in keys], width, label=s)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(keys)
    axes[0].set_ylabel('nm'); axes[0].legend(); axes[0].grid(axis='y', alpha=.3)
    axes[0].set_title('面粗糙度参数（2 阶整平后）', fontsize=10)

    # 沟槽参数
    ls = ['周期 周期/nm', '周期/nm', '深度/nm']
    per = [summary['samples'][s]['形貌']['groove']['period_fft'] for s in SAMPLES]
    dep = [summary['samples'][s]['形貌']['groove']['depth_pp'] for s in SAMPLES]
    ax = axes[1]
    ax.bar(np.arange(3) - 0.2, per, 0.4, label='沟槽周期 / nm', color='#3182bd')
    ax2 = ax.twinx()
    ax2.bar(np.arange(3) + 0.2, dep, 0.4, label='峰谷深度 / nm', color='#e6550d')
    ax.set_xticks(range(3)); ax.set_xticklabels(SAMPLES)
    ax.set_ylabel('周期 / nm', color='#3182bd'); ax2.set_ylabel('深度 / nm', color='#e6550d')
    ax.set_title('沟槽周期与深度', fontsize=10)
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    save(fig, 'fig07_roughness_groove')
    plt.close(fig)


if __name__ == '__main__':
    summary = json.load(open(os.path.join(os.environ.get('AFM_RES', '../results'), 'summary.json'), encoding='utf-8'))
    fig_overview(); print('fig01 ok')
    fig_processing(); print('fig02 ok')
    fig_profiles(summary); print('fig03 ok')
    fig_spectra(); print('fig04 ok')
    fig_3d(); print('fig05 ok')
    fig_channels(); print('fig06 ok')
    fig_roughness(summary); print('fig07 ok')
