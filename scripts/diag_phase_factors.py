"""条纹相位漂移的多因素分解、曲率项灵敏度标定与三种对齐方案比较。

模型
----
相位场按物理上可区分的基函数分离：

    phi(x,y)/2pi = c0 + a1*u + a2*u^2 + a3*u^3        （与 x 无关，逐行全行相位给出）
                 + b1*v + b2*u*v + b3*v^2             （与 x 有关，滑窗相位给出）

    u = (y-(ny-1)/2)/(ny-1)      v = (x-nx/2)/(nx-1)

    a1  固定倾角（扫描轴与沟槽走向的夹角）
    a2,a3  非线性 y 漂移（压电蠕变 / 慢扫轴非线性）；a2/4 记二次垂度、a3/8 记三次端项
    b2  条纹方向随 x 的剪切（扫描头弓形轨迹）
    b3  沿 x 的曲率项：半径 R 的圆弧给出横向位移 dx = x^2/(2R)，故 R = f0*(nx-1)^2/(2*b3)

⚠ 关键实现细节（2026-10-27 首版曾在此失明，独立审计抓出）
    滑窗投影必须用**全局坐标的载波** exp(-2πi f0 (x0+j))，否则各窗口的相位相差一个
    任意常数 2π f0 x0。首版用「每个窗口沿 y 去均值」消这个常数，代价是把
    「随 x 二次变化、沿 y 恒定」的分量（正是轨道曲率的形状）一并扣掉，b3 恒为
    机器精度零。现在改为按载波归算，窗间常数由 b1（线性于 v）吸收。

灵敏度标定
----------
曲率项的检出限不能由拟合残差直接给（那样等于假设模型正确），故做**注入-恢复**：
在形貌场上叠加已知的二次横向位移场 dx(x)=s*(x/x_max)^2，重跑整条流程，检查
b3 是否按注入量线性恢复；由此给出该尺度的响应斜率与 1σ/2σ 检出限。

三种对齐方案：全局单斜率（与 afmproc.row_shift 同口径，作为基线）、逐行相位、
全二维相位场；深度统一用 150 nm 平滑剖面的 P99-P1，边缘按 row_shift 的 NaN 处理。

运行：python3 diag_phase_factors.py
输出：../results/diag_phase_factors.json
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
OUT = os.path.join(ROOT, 'results', 'diag_phase_factors.json')

W, HOP = 512, 128
INJECT_SAG_NM = (0.5, 2.0, 8.0)


def lsq(cols, v):
    """列归一化最小二乘，返回 (系数, 残差, sigma_b) —— sigma_b 为各系数 1σ 误差。"""
    v = np.asarray(v, dtype=float).ravel()
    M = np.stack([np.asarray(c, dtype=float).ravel() for c in cols], axis=1)
    scale = np.linalg.norm(M, axis=0)
    scale[scale == 0] = 1.0
    Mn = M / scale
    coef_n, *_ = np.linalg.lstsq(Mn, v, rcond=None)
    resid = v - Mn @ coef_n
    dof = max(1, v.size - Mn.shape[1])
    s2 = float(resid @ resid) / dof
    cov_n = s2 * np.linalg.pinv(Mn.T @ Mn)
    sig_n = np.sqrt(np.clip(np.diag(cov_n), 0, None))
    return coef_n / scale, resid, sig_n / scale


def row_phase(z, f0):
    """逐行（全行）解调相位，单位=周期，沿 y 解缠绕。"""
    ny, nx = z.shape
    xs = np.arange(nx)
    kern = np.exp(-2j * np.pi * f0 * xs)
    ph = np.empty(ny)
    for i in range(ny):
        p = z[i] - z[i].mean()
        p = p - np.polyval(np.polyfit(xs, p, 1), xs)
        ph[i] = np.angle(np.sum(p * kern))
    return np.unwrap(ph) / (2 * np.pi)


def strip_phase(z, f0, W=W, HOP=HOP):
    """滑窗解调（载波用全局坐标）：返回 (窗口中心 x[ns], 相位[ns, ny]，单位=周期)。"""
    ny, nx = z.shape
    win = np.hanning(W)
    j = np.arange(W)
    xc, out = [], []
    for x0 in range(0, nx - W + 1, HOP):
        seg = z[:, x0:x0 + W]
        seg = seg - seg.mean(axis=1, keepdims=True)
        kern = np.exp(-2j * np.pi * f0 * (x0 + j))
        c = np.sum((seg * win) * kern, axis=1)
        out.append(np.unwrap(np.angle(c)) / (2 * np.pi))
        xc.append(x0 + W / 2.0)
    return np.array(xc, float), np.array(out)


def fit_x_terms(a, f0, c0, a1, a2, a3):
    """扣除与 x 无关的拟合面后，对 v / uv / v^2 回归，返回系数与 1σ。"""
    ny, nx = a.shape
    xc, PHI = strip_phase(a, f0)
    u = (np.arange(ny) - (ny - 1) / 2.0) / (ny - 1)
    v = (xc - nx / 2.0) / (nx - 1)
    U = np.repeat(u[:, None], len(xc), axis=1)
    V = np.repeat(v[None, :], ny, axis=0)
    base = c0 + a1 * U + a2 * U ** 2 + a3 * U ** 3
    coef, resid, sig = lsq([np.ones_like(U), V, U * V, V ** 2], PHI.T - base)
    return dict(xc=xc, v=v, PHI=PHI, U=U, V=V,
                b1=coef[1], b2=coef[2], b3=coef[3],
                sb1=sig[1], sb2=sig[2], sb3=sig[3],
                rms=float(np.sqrt(np.mean(resid ** 2))))


def depth_pp(prof, n_s):
    k = np.ones(n_s) / n_s
    ps = np.convolve(prof, k, 'same')
    return float(np.percentile(ps, 99) - np.percentile(ps, 1))


def align_nan(a, d_px):
    """按逐行位移 d（px）重采样，边缘置 NaN（与 afmproc.row_shift 同口径），返回平均剖面。"""
    ny, nx = a.shape
    xs = np.arange(nx)
    d = np.broadcast_to(np.asarray(d_px, float), (ny,))
    out = np.empty_like(a)
    for i in range(ny):
        out[i] = np.interp(xs + d[i], xs, a[i], left=np.nan, right=np.nan)
    return np.nanmean(out, axis=0)


def analyse(tag):
    img = csmio.CSMImage(os.path.join(DATA, f'{tag}-形貌000.csm'))
    z, _ = img.physical()
    dx = img.px_size_nm
    a = afmproc.flatten_rows(z.astype(float), 1)
    st = afmproc.groove_stats(a, dx)
    P, f0 = st['period_fft'], dx / st['period_fft']
    ny, nx = a.shape
    u = (np.arange(ny) - (ny - 1) / 2.0) / (ny - 1)

    # ---- (A) 与 x 无关的成分 ----
    phi = row_phase(a, f0)
    phi = phi - phi.mean()
    (c0, a1, a2, a3), res_poly, _ = lsq([np.ones(ny), u, u ** 2, u ** 3], phi)
    (g0, g1), res_lin, _ = lsq([np.ones(ny), u], phi)
    tilt_deg = float(np.degrees(np.arctan(-(g1 / (ny - 1)) / f0)))
    rms_lin = float(np.sqrt(np.mean(res_lin ** 2)))

    # ---- (B) 与 x 有关的成分 ----
    fx = fit_x_terms(a, f0, c0, a1, a2, a3)
    b3, sb3 = fx['b3'], fx['sb3']
    R_from_b3 = f0 * (nx - 1) ** 2 / (2 * b3) if b3 != 0 else np.inf
    # 检出限：2σ 的 b3 → 边缘矢高（v=0.5 处 Δx = 0.25*b3/f0）→ 等效半径
    sag_2sig_px = 0.25 * (2 * sb3) / f0
    R_limit = (nx / 2) ** 2 / (2 * sag_2sig_px) if sag_2sig_px > 0 else np.inf
    shear_deg = float(np.degrees(np.arctan(abs(fx['b2']) / (ny - 1) / f0)))
    shear_2sig_deg = float(np.degrees(np.arctan(abs(2 * fx['sb2']) / (ny - 1) / f0)))

    # ---- (C) 注入-恢复：曲率项灵敏度 ----
    inj = []
    xs = np.arange(nx)
    for sag_nm in INJECT_SAG_NM:
        # 叠加横向位移场 dx(x) = sag*(x/xmax)^2（单位 nm → px），用线性插值重采样
        dx_px = (sag_nm / dx) * ((xs - nx / 2) / (nx / 2)) ** 2
        zi = np.empty_like(a)
        for i in range(ny):
            zi[i] = np.interp(xs - dx_px, xs, a[i], left=a[i, 0], right=a[i, -1])
        ai = afmproc.flatten_rows(zi, 1)
        phi_i = row_phase(ai, f0)
        phi_i = phi_i - phi_i.mean()
        (ci0, ci1, ci2, ci3), _, _ = lsq([np.ones(ny), u, u ** 2, u ** 3], phi_i)
        fxi = fit_x_terms(ai, f0, ci0, ci1, ci2, ci3)
        inj.append(dict(sag_injected_nm=sag_nm, b3=float(fxi['b3']),
                        d_b3=float(fxi['b3'] - b3)))
    b3_per_nm = np.polyfit([d['sag_injected_nm'] for d in inj],
                           [d['b3'] for d in inj], 1)[0] if len(inj) > 1 else np.nan

    # ---- (D) 三种对齐方案（统一 NaN 边缘口径） ----
    n_s = max(3, int(round(150.0 / dx)))
    d_base = depth_pp(align_nan(a, st['tilt_px_per_row'] * np.arange(ny)), n_s)
    x0 = -phi / f0
    x0 = x0 - x0.mean()
    d_perrow = depth_pp(align_nan(a, x0), n_s)
    # 全二维：与 x 无关部分（逐行相位）+ 与 x 有关部分（b1/b2/b3 拟合面）
    model_x = -(fx['b1'] * fx['V'] + fx['b2'] * fx['U'] * fx['V']
                + fx['b3'] * fx['V'] ** 2) / f0
    za = np.empty_like(a)
    for i in range(ny):
        d_i = -phi[i] / f0 + np.interp(xs, fx['xc'], model_x[i])
        za[i] = np.interp(xs + d_i, xs, a[i], left=np.nan, right=np.nan)
    d_full = depth_pp(np.nanmean(za, axis=0), n_s)

    return dict(
        sample=tag, period_nm=float(P), dx_nm=float(dx),
        tilt_deg=tilt_deg,
        drift_total_periods=float(a1 + 0.25 * a3),
        drift_linear_periods=float(a1),
        quad_sag_periods=float(a2 / 4.0),
        cubic_end_periods=float(a3 / 8.0),
        quad_share=float(abs(a2 / 4.0) / abs(a1 + 0.25 * a3)),
        cubic_share=float(abs(a3 / 8.0) / abs(a1 + 0.25 * a3)),
        rms_lin_periods=rms_lin,
        rms_poly_periods=float(np.sqrt(np.mean(res_poly ** 2))),
        x_terms=dict(b1=fx['b1'], b2=fx['b2'], b3=fx['b3'],
                     sb1=fx['sb1'], sb2=fx['sb2'], sb3=fx['sb3'],
                     rms=fx['rms']),
        R_from_x2_mm=float(R_from_b3 * dx / 1e6),
        R_limit_mm=float(R_limit * dx / 1e6),
        sag_2sig_nm=float(sag_2sig_px * dx),
        shear_deg=shear_deg,
        shear_2sig_deg=shear_2sig_deg,
        injection=inj,
        b3_per_nm_sag=float(b3_per_nm),
        depth_baseline_nm=float(d_base),
        depth_perrow_nm=float(d_perrow),
        depth_fullfield_nm=float(d_full),
        gain_perrow_pct=float((d_perrow / d_base - 1) * 100),
        gain_fullfield_pct=float((d_full / d_base - 1) * 100),
    )


if __name__ == '__main__':
    res = [analyse(t) for t in ('cd1', 'cd2', 'cd3')]
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    for r in res:
        print(f"{r['sample']}: 倾角 {r['tilt_deg']:+.2f}° | 漂移 {r['drift_total_periods']:+.2f} 周期 "
              f"(线性 {r['drift_linear_periods']:+.2f} | 二次垂度 {r['quad_sag_periods']:+.3f}={r['quad_share']*100:.1f}% "
              f"| 三次端项 {r['cubic_end_periods']:+.3f}={r['cubic_share']*100:.1f}% | 一次残差 {r['rms_lin_periods']:.3f})")
        print(f"   x 向: b3={r['x_terms']['b3']:+.3e}±{r['x_terms']['sb3']:.1e} 周期/v² → R={r['R_from_x2_mm']:.3g} mm"
              f" | 2σ 检出限 R={r['R_limit_mm']:.3g} mm（矢高 {r['sag_2sig_nm']:.2f} nm）"
              f" | 剪切 {r['shear_deg']:.2f}°(2σ {r['shear_2sig_deg']:.2f}°)")
        print(f"   注入-恢复: " + ', '.join(f"{d['sag_injected_nm']}nm→Δb3={d['d_b3']:+.2e}" for d in r['injection'])
              + f" | 斜率 {r['b3_per_nm_sag']:+.2e} /nm")
        print(f"   深度: 基线 {r['depth_baseline_nm']:.2f} | 逐行 {r['depth_perrow_nm']:.2f}({r['gain_perrow_pct']:+.2f}%)"
              f" | 全二维 {r['depth_fullfield_nm']:.2f}({r['gain_fullfield_pct']:+.2f}%) nm")
    print(f"\n写出 {OUT}", file=sys.stderr)
