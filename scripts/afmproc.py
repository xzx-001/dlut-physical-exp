"""AFM / CSPM5500 数据处理核心库 —— CD-R 光盘形貌实验

数据格式见 csmio.py。本模块提供与 Imager 4.7 后处理软件对应的处理算子：
    flatten_rows  逐行多项式减除（去扫描线倾斜 / 行漂移）
    plane_fit     全局二维多项式面整平（slope correction）
    roughness     二维粗糙度参数（ISO 25178，Imager §47.2 定义）
    profile_stats 沿 x 平均剖面 → 沟槽周期 / 峰谷深度
    psd_1d        一维功率谱密度（剖面）
所有长度单位 nm，扫描面由 header 的 ScanSize 给出。
"""
import numpy as np


# ---------------- 基本处理 ----------------
def flatten_rows(z, order=1):
    """逐行拟合多项式并减除：order=1 去每行倾斜+直流，order=0 只去行均值。

    对应 Imager 的 "消除扫描线 / Flatten"。
    """
    ny, nx = z.shape
    x = np.linspace(-1.0, 1.0, nx)
    out = np.empty_like(z, dtype=np.float64)
    for i in range(ny):
        coef = np.polyfit(x, z[i], order)
        out[i] = z[i] - np.polyval(coef, x)
    return out


def plane_fit(z, order=2):
    """全局二维多项式面整平（slope correction），返回残差。

    拟合 z ≈ Σ_{i+j<=order} c_ij x^i y^j，x,y 归一化到 [-1,1]。
    """
    ny, nx = z.shape
    xv = np.linspace(-1, 1, nx)
    yv = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(xv, yv)
    terms = [X ** i * Y ** j for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack([t.ravel() for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    return z - (A @ coef).reshape(z.shape), coef


def plane_correct_both(z, order_rows=1, order_plane=2):
    """标准流程：先去扫描线（逐行 1 阶）→ 再 2 阶面整平。"""
    a = flatten_rows(z, order_rows)
    return plane_fit(a, order_plane)


# ---------------- 粗糙度 ----------------
def _local_extrema(z, kind='max'):
    """8 邻域严格极值（Imager §47.2 定义：8 个邻居全部更低=局部峰）。"""
    ny, nx = z.shape
    c = z[1:-1, 1:-1]
    nb = [z[0:-2, 0:-2], z[0:-2, 1:-1], z[0:-2, 2:],
          z[1:-1, 0:-2],                 z[1:-1, 2:],
          z[2:,   0:-2], z[2:,   1:-1], z[2:,   2:]]
    if kind == 'max':
        m = np.ones_like(c, bool)
        for n in nb:
            m &= c > n
    else:
        m = np.ones_like(c, bool)
        for n in nb:
            m &= c < n
    yy, xx = np.nonzero(m)
    return yy + 1, xx + 1


def roughness(z, dx_nm):
    """二维（面）粗糙度参数，按 Imager §47.2 / ISO 25178 定义。"""
    mu = z.mean()
    d = z - mu
    sq = np.sqrt(np.mean(d ** 2))
    sa = np.mean(np.abs(d))
    sy = z.max() - z.min()
    # Sz 十点高度: 5 个最高局部峰均值 - 5 个最低局部谷均值
    py, px = _local_extrema(z, 'max')
    vy, vx = _local_extrema(z, 'min')
    if len(py) >= 5 and len(vy) >= 5:
        sz = np.sort(z[py, px])[-5:].mean() - np.sort(z[vy, vx])[0:5].mean()
    else:
        sz = np.nan
    ssk = np.mean(d ** 3) / sq ** 3 if sq > 0 else np.nan
    sku = np.mean(d ** 4) / sq ** 4 if sq > 0 else np.nan
    # 梯度类
    gy, gx = np.gradient(z, dx_nm, dx_nm)
    s = np.hypot(gx, gy)
    sdq = np.sqrt(np.mean(s ** 2))
    sdr = np.mean(s ** 2)          # 表面积比增量（一阶近似）
    res = dict(Sa=sa, Sq=sq, Sy=sy, Sz=sz, Ssk=ssk, Sku=sku,
               Sdq=sdq, Sdr=sdr, Sds=len(py) / ((z.shape[1] * dx_nm / 1000) ** 2),
               n_peaks=len(py), n_valleys=len(vy), mean=mu)
    # 一维（剖面）粗糙度：沿 x 每行 Ra 的平均 —— 便于与 Imager 剖面线结果对照
    ra_rows = np.mean([np.mean(np.abs(r - r.mean())) for r in z])
    rq_rows = np.mean([np.sqrt(np.mean((r - r.mean()) ** 2)) for r in z])
    res['Ra_row'] = ra_rows
    res['Rq_row'] = rq_rows
    return res


# ---------------- 剖面 / 周期 ----------------
def mean_profile(z, axis=1):
    """沿某轴平均得到一维剖面（默认沿 y 平均 → x 方向剖面），已去均值。"""
    p = z.mean(axis=0) if axis == 1 else z.mean(axis=1)
    return p - p.mean()


def find_extrema_1d(p, min_sep, kind='max'):
    """一维极值检测（最小间距 min_sep，按幅值排序取优）。"""
    idx = []
    for i in range(1, len(p) - 1):
        if kind == 'max' and p[i] > p[i - 1] and p[i] >= p[i + 1]:
            idx.append(i)
        elif kind == 'min' and p[i] < p[i - 1] and p[i] <= p[i + 1]:
            idx.append(i)
    idx.sort(key=lambda i: -abs(p[i] - np.median(p)))
    keep = []
    for i in idx:
        if all(abs(i - k) >= min_sep for k in keep):
            keep.append(i)
    return np.array(sorted(keep))


def autocorr_period(p, dx, max_lag_frac=0.5):
    """一维自相关首峰 → 周期（nm），用多段加窗平均抑制噪声。"""
    N = len(p)
    acs = []
    seg = N // 4
    for s in range(0, N - seg + 1, seg):
        a = p[s:s + seg] * np.hanning(seg)
        ac = np.correlate(a, a, 'full')[seg - 1:]
        if ac[0] > 0:
            acs.append(ac / ac[0])
    ac = np.mean(acs, axis=0)
    lo = max(3, int(0.02 * seg))
    hi = int(max_lag_frac * seg)
    peaks = [i for i in range(lo, hi) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.1]
    if not peaks:
        return np.nan, np.nan, ac
    lag = peaks[0]
    # 抛物线插值亚像素
    if 0 < lag < len(ac) - 1:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        den = (y0 - 2 * y1 + y2)
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        off = 0.0
    return (lag + off) * dx, ac[int(lag)], ac


def fft_period(p, dx, pad=16):
    """零填充 FFT 主峰 → 周期（nm），带抛物线亚 bin 插值。"""
    N = len(p)
    w = p * np.hanning(N)
    P = np.abs(np.fft.rfft(w, n=N * pad))
    f = np.fft.rfftfreq(N * pad, d=dx)
    P[:3] = 0
    k = int(np.argmax(P))
    if 0 < k < len(P) - 1:
        y0, y1, y2 = P[k - 1], P[k], P[k + 1]
        den = (y0 - 2 * y1 + y2)
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        off = 0.0
    kk = k + off
    if kk <= 0:
        return np.nan, f, P
    return 1.0 / (f[1] * kk), f, P


def row_shift(z, slope_px_per_row):
    """按每行水平位移 slope（像素/行）做整数+线性插值对齐，使倾斜条纹竖直。

    等价于 Imager 里把图像旋转到条纹与 y 轴平行；用于跨行平均求高信噪比剖面。
    """
    ny, nx = z.shape
    out = np.empty_like(z)
    x = np.arange(nx)
    for i in range(ny):
        out[i] = np.interp(x, x - slope_px_per_row * i, z[i], left=np.nan, right=np.nan)
    return out


def phase_tilt(z, dx, stride=4):
    """逐行 DFT 相位斜率法估计条纹相对 y 轴的倾斜（px/行，正值=条纹随 y 向 +x 漂移）。

    原理：对每行做去趋势后，在主场频 f0 处计算复投影 Σz·exp(-2πi f0 x)，其相位
    随行号线性变化，斜率 = -2π f0 · Δx（Δx 为每行水平位移）。比二维 FFT 峰位
    （受 1/ScanSize 分辨率限制）精确得多，也比逐行互相关稳健。
    """
    ny, nx = z.shape
    x = np.arange(nx)
    pers = []
    for i in range(0, ny, stride):
        p = z[i] - z[i].mean()
        c = np.polyfit(x, p, 1)
        p = p - np.polyval(c, x)
        per, _, _ = fft_period(p, dx)
        if np.isfinite(per):
            pers.append(per)
    P = float(np.median(pers))
    f0 = dx / P                                    # cycles / px
    idx, ph = [], []
    for i in range(0, ny, stride):
        p = z[i] - z[i].mean()
        c = np.polyfit(x, p, 1)
        p = p - np.polyval(c, x)
        idx.append(i)
        ph.append(np.angle(np.sum(p * np.exp(-2j * np.pi * f0 * x))))
    ph = np.unwrap(np.array(ph))
    slope = np.polyfit(idx, ph, 1)[0]
    shift_px = -slope / (2 * np.pi * f0)
    return P, float(shift_px), float(np.degrees(np.arctan(shift_px))), ph


def estimate_tilt(z, dx, smooth_nm=150.0):
    """条纹倾斜（px/行）—— 相位斜率法。"""
    return phase_tilt(z, dx)[1]


def groove_stats(z, dx, smooth_nm=150.0, lo=1, hi=99):
    """沟槽几何（稳健版）。

    周期  : 逐行 1D FFT 主峰中位数（跨行相位漂移不会抹平单行信息）
    倾斜  : 逐行 DFT 相位斜率 → px/行、度
    深度  : 逐行平滑剖面的 P(hi)-P(lo) 中位数（峰谷差，抗单点噪声）
    剖面  : deskew 对齐后沿 y 平均（信噪比提高 1~2 个量级），供作图/占宽比
    """
    ny, nx = z.shape
    n_s = max(3, int(round(smooth_nm / dx)))
    kern = np.ones(n_s) / n_s
    x = np.arange(nx)

    periods, dephs = [], []
    for i in range(ny):
        p = z[i] - z[i].mean()
        c = np.polyfit(x, p, 1)
        p = p - np.polyval(c, x)
        per, _, _ = fft_period(p, dx)
        if np.isfinite(per):
            periods.append(per)
        ps = np.convolve(p, kern, mode='same')
        m = ps[n_s:-n_s] if n_s * 2 < nx else ps
        dephs.append(np.percentile(m, hi) - np.percentile(m, lo))

    periods = np.array(periods); dephs = np.array(dephs)
    period_fft = float(np.median(periods)) if len(periods) else np.nan
    period_iqr = ((float(np.percentile(periods, 25)), float(np.percentile(periods, 75)))
                  if len(periods) else (np.nan, np.nan))
    depth_pp = float(np.median(dephs)) if len(dephs) else np.nan

    # ---- deskew + 跨行平均剖面 ----
    P_ph, tilt, tilt_deg, phase = phase_tilt(z, dx)
    zs = row_shift(z, tilt)
    prof = np.nanmean(zs, axis=0)
    prof = prof - np.nanmean(prof)
    ok = np.isfinite(prof)
    pf = np.where(ok, prof, 0.0)
    ps = np.convolve(pf, kern, mode='same')

    sep = (period_fft / dx) * 0.5 if np.isfinite(period_fft) else 20
    mx = find_extrema_1d(ps[ok], max(4, sep), 'max')
    mn = find_extrema_1d(ps[ok], max(4, sep), 'min')
    depth_pp_avg = float(np.percentile(ps[ok], hi) - np.percentile(ps[ok], lo))
    period_peaks = float(np.median(np.diff(mx)) * dx) if len(mx) >= 2 else np.nan
    thr = (np.percentile(ps[ok], hi) + np.percentile(ps[ok], lo)) / 2
    duty = float(np.mean(ps[ok] > thr))

    f, Pw = psd_1d(prof[ok], dx)
    return dict(period_fft=period_fft, period_iqr=period_iqr,
                period_peaks=period_peaks, depth_pp=depth_pp,
                depth_pp_avg=depth_pp_avg, tilt_px_per_row=tilt,
                tilt_deg=tilt_deg, phase_drift_rad=float(phase[-1] - phase[0]),
                n_peaks=int(len(mx)), n_valleys=int(len(mn)), duty=duty,
                profile=ps, prof_raw=prof, psd_freq=f, psd_pow=Pw,
                row_periods=periods, row_depths=dephs)


def psd_1d(p, dx):
    """一维功率谱密度（nm²·nm），Hanning 窗，归一化到 Parseval。"""
    N = len(p)
    w = np.hanning(N)
    F = np.fft.rfft((p - p.mean()) * w)
    f = np.fft.rfftfreq(N, d=dx)
    psd = (np.abs(F) ** 2) / (N * np.sum(w ** 2)) * 2 * dx
    psd[0] /= 2
    return f, psd


def radial_psd_2d(z, dx):
    """二维径向平均功率谱（用于面粗糙度频谱分析）。"""
    ny, nx = z.shape
    A = z - z.mean()
    F = np.fft.fftshift(np.fft.fft2(A))
    P = (np.abs(F) ** 2) / (nx * ny)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    nb = 60
    edges = np.linspace(0, K.max(), nb + 1)
    idx = np.digitize(K.ravel(), edges) - 1
    kb, pb = [], []
    for b in range(nb):
        m = idx == b
        if m.sum() > 0:
            kb.append(K.ravel()[m].mean())
            pb.append(P.ravel()[m].mean())
    return np.array(kb), np.array(pb)


def structure_orientation(z):
    """结构张量法估条纹方向（角度相对 x 轴，度）与各向异性。"""
    gy, gx = np.gradient(z)
    Jxx = np.mean(gx * gx); Jyy = np.mean(gy * gy); Jxy = np.mean(gx * gy)
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)          # 主梯度方向
    ev = np.linalg.eigvalsh(np.array([[Jxx, Jxy], [Jxy, Jyy]]))
    aniso = (ev[1] - ev[0]) / (ev[1] + ev[0]) if ev[1] + ev[0] > 0 else np.nan
    return np.degrees(theta), aniso
