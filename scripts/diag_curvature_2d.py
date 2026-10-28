"""曲率项的独立核验：二维带通解调（无滑窗/无 Hann 窗偏置）。

做法
----
1. 先用全局单斜率把条纹摆正（行位移插值，边缘镜像填充避免 NaN）；
2. 二维 FFT，只在 (f0, 0) 附近（含共轭一侧）取高斯带通，逆变换得到复场，其幅角即条纹相位；
3. 相位场沿 x 解缠绕后扣除均值，对 [1, v, uv, v^2, u, u^2] 回归：
   v^2 系数即曲率项，u 系数即残余倾角，u*v 即剪切。
4. 与 scripts/diag_phase_factors.py 的滑窗结果对照（两者方法完全独立）。

半径换算：ν^2 项（ν=(x-nx/2)/(nx-1)）对应横向位移 Δx = b3·ν^2/f0；
圆轨道 Δx = x^2/(2R) ⇒ R(px) = f0·(nx-1)^2/(2·b3)。

运行：python3 diag_curvature_2d.py
输出：../results/diag_curvature_2d.json
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import csmio
import afmproc

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', '20261026')
OUT = os.path.join(ROOT, 'results', 'diag_curvature_2d.json')

DS = 4          # 拟合时降采样步长（相位场本身是平滑的）


def deskew_mirror(a, d_px):
    """按逐行位移 d(x 方向) 重采样，边缘镜像填充（不产生 NaN）。"""
    ny, nx = a.shape
    xs = np.arange(nx)
    d = np.broadcast_to(np.asarray(d_px, float), (ny,))
    out = np.empty_like(a)
    for i in range(ny):
        out[i] = np.interp(xs - d[i], xs, a[i], left=np.nan, right=np.nan)
        bad = ~np.isfinite(out[i])
        if bad.any():
            out[i][bad] = out[i][~bad][0] if bad[0] else out[i][~bad][-1]
    return out


def phase_2d(z, f0, sig_kx=1.2e-3, sig_ky=8e-3):
    """二维带通解调，返回相位场（周期）。"""
    ny, nx = z.shape
    P = z - z.mean()
    F = np.fft.fft2(P)
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky)
    G = np.exp(-((KX - f0) ** 2) / (2 * sig_kx ** 2) - (KY ** 2) / (2 * sig_ky ** 2))
    c = np.fft.ifft2(F * G)
    ph = np.unwrap(np.angle(c), axis=1) / (2 * np.pi)
    ph = ph - ph.mean(axis=1, keepdims=True)     # 去掉沿 x 的整体偏置（不影响 v^2 之外的项）
    return ph


def analyse(tag):
    img = csmio.CSMImage(os.path.join(DATA, f'{tag}-形貌000.csm'))
    z, _ = img.physical()
    dx = img.px_size_nm
    a = afmproc.flatten_rows(z.astype(float), 1)
    st = afmproc.groove_stats(a, dx)
    P, f0 = st['period_fft'], dx / st['period_fft']
    ny, nx = a.shape

    ad = deskew_mirror(a, st['tilt_px_per_row'] * np.arange(ny))
    ph = phase_2d(ad, f0)

    u = (np.arange(ny) - (ny - 1) / 2.0) / (ny - 1)
    x = (np.arange(nx) - nx / 2.0) / (nx - 1)
    U, X = np.meshgrid(u, x, indexing='ij')
    sel = (slice(None, None, DS), slice(None, None, DS))
    cols = [np.ones_like(U)[sel], U[sel], U[sel] ** 2, U[sel] * X[sel], X[sel], X[sel] ** 2]
    v = ph[sel]
    M = np.stack([c.ravel() for c in cols], axis=1)
    scale = np.linalg.norm(M, axis=0)
    coef_n, *_ = np.linalg.lstsq(M / scale, v.ravel(), rcond=None)
    coef = coef_n / scale
    resid = v.ravel() - (M / scale) @ coef_n
    dof = max(1, v.size - M.shape[1])
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv((M / scale).T @ (M / scale))
    sig = np.sqrt(np.clip(np.diag(cov), 0, None)) / scale
    c0, a1, a2, b2, b1, b3 = coef

    R_px = f0 * (nx - 1) ** 2 / (2 * b3) if b3 != 0 else np.inf
    sag_2sig_px = 0.25 * 2 * sig[5] / f0
    R_limit_px = (nx / 2) ** 2 / (2 * sag_2sig_px) if sag_2sig_px > 0 else np.inf

    # 对照：未 deskew 时的原图也跑一遍，确认 deskew 不引入曲率
    ph_raw = phase_2d(a, f0, sig_ky=12e-3)
    cols2 = [np.ones_like(U)[sel], U[sel], U[sel] ** 2, U[sel] * X[sel], X[sel], X[sel] ** 2]
    M2 = np.stack([c.ravel() for c in cols2], axis=1)
    sc2 = np.linalg.norm(M2, axis=0)
    cf2, *_ = np.linalg.lstsq(M2 / sc2, ph_raw[sel].ravel(), rcond=None)
    b3_raw = (cf2 / sc2)[5]

    return dict(
        sample=tag, period_nm=float(P), dx_nm=float(dx),
        b3_cycles_per_v2=float(b3), sb3=float(sig[5]),
        R_from_x2_mm=float(R_px * dx / 1e6),
        R_limit_mm=float(R_limit_px * dx / 1e6),
        sag_2sig_nm=float(sag_2sig_px * dx),
        b2_xy=float(b2), sb2=float(sig[3]),
        residual_tilt_deg=float(np.degrees(np.arctan(a1 / (ny - 1) / f0))),
        b3_raw_undeskewed=float(b3_raw),
        rms_resid_periods=float(np.sqrt(np.mean(resid ** 2))),
        n_points=int(v.size),
    )


if __name__ == '__main__':
    res = [analyse(t) for t in ('cd1', 'cd2', 'cd3')]
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    for r in res:
        print(f"{r['sample']}: b3={r['b3_cycles_per_v2']:+.3e}±{r['sb3']:.1e} 周期/v² | "
              f"R={r['R_from_x2_mm']:.4g} mm | 2σ 检出限 R={r['R_limit_mm']:.3g} mm（矢高 {r['sag_2sig_nm']:.2f} nm）| "
              f"未deskew 时 b3={r['b3_raw_undeskewed']:+.3e} | 残差 {r['rms_resid_periods']:.4f} 周期 | 残余倾角 {r['residual_tilt_deg']:+.3f}°")
    print(f"\n写出 {OUT}", file=sys.stderr)
