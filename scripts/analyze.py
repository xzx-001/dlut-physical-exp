#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""光纤传感实验（位移/应变 + 温度）数据分析 —— 修订版
- 只用 PE 与 FBG 两条链路（不含 CMCU）
- 主测峰 = 峰值 2（长波峰），峰值 1 作为扩展讨论
- 温度部分加入 MCMC（Metropolis–Hastings）拟合：温度项 + 夹持应变松弛项
输出：figures/*.pdf，results/summary.json，控制台数值表
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

_ap = argparse.ArgumentParser(description="光纤传感实验数据分析（位移/应变 + 温度）")
_ap.add_argument("--figures", default=None, help="图输出目录（默认 <repo>/output/figures）")
_ap.add_argument("--results", default=None, help="数值输出目录（默认 <repo>/output/results）")
_args, _unknown = _ap.parse_known_args()

BASE = os.path.dirname(os.path.abspath(__file__))        # scripts/
REPO = os.path.dirname(BASE)                            # 仓库根目录
SRC = os.path.join(REPO, "data")                        # 原始记录 CSV
FIG = _args.figures or os.path.join(REPO, "output", "figures")
RES = _args.results or os.path.join(REPO, "output", "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

def load(fn, enc="utf-8-sig"):
    rows = list(csv.reader(open(os.path.join(SRC, fn), encoding=enc)))
    h = [x.strip() for x in rows[0]]
    return h, [dict(zip(h, r)) for r in rows[1:] if r]

def num(rows, key):
    return np.array([float(x[key]) if x.get(key, "") not in ("", None) else np.nan for x in rows])

def linfit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    ss_res = np.sum(res ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(a=float(a), b=float(b), r2=float(1 - ss_res / ss_tot),
                rms=float(np.sqrt(ss_res / (len(x) - 2))), maxres=float(np.max(np.abs(res))),
                n=int(len(x)), res=res.tolist())

S = {}

# ======================= 位移/应变 =======================
_, fwd = load("应变正向60-5.csv")
_, rev = load("应变反向.csv")
_, ver = load("应变验证值.csv", "gb18030")

x_f = np.arange(10, 61, 5.0)
x_r = np.arange(60, 9, -5.0)
PE_f, PE_r = num(fwd, "pe_raw_value"), num(rev, "pe_raw_value")
L1f, L1r = num(fwd, "fbg_peak1_nm"), num(rev, "fbg_peak1_nm")
L2f, L2r = num(fwd, "fbg_peak2_nm"), num(rev, "fbg_peak2_nm")
x_v = np.array([float(list(r.values())[-1]) for r in ver])
PE_v, L1v, L2v = num(ver, "pe_raw_value"), num(ver, "fbg_peak1_nm"), num(ver, "fbg_peak2_nm")

for tag, x, y in [("PE_f", x_f, PE_f), ("PE_r", x_r, PE_r),
                  ("L1_f", x_f, L1f), ("L1_r", x_r, L1r),
                  ("L2_f", x_f, L2f), ("L2_r", x_r, L2r)]:
    S[tag] = linfit(x, y)
x_all = np.r_[x_f, x_r]
S["PE_all"] = linfit(x_all, np.r_[PE_f, PE_r])
S["L1_all"] = linfit(x_all, np.r_[L1f, L1r])
S["L2_all"] = linfit(x_all, np.r_[L2f, L2r])
S["n_fwd"], S["n_rev"], S["n_ver"] = len(fwd), len(rev), len(ver)
S["PE_span"] = float(PE_f[-1] - PE_f[0])
S["L2_span_pm"] = float((L2f[-1] - L2f[0]) * 1000)
S["L1_span_pm"] = float((L1f[-1] - L1f[0]) * 1000)
sp = np.r_[(L2f - L1f), (L2r - L1r)] * 1000
S["spacing"] = dict(mean=float(sp.mean()), std=float(sp.std(ddof=1)),
                    min=float(sp.min()), max=float(sp.max()))
S["skew_max"] = float(max(float(x["max_time_skew_ms"]) for x in fwd + rev + ver))

hyst = dict(x=x_f.tolist(),
            PE=(PE_f - PE_r[::-1]).tolist(),
            L1=((L1f - L1r[::-1]) * 1000).tolist(),
            L2=((L2f - L2r[::-1]) * 1000).tolist())
S["hyst"] = hyst

pred = {}
for tag, yv, k in [("PE", PE_v, "PE_all"), ("L1", L1v, "L1_all"), ("L2", L2v, "L2_all")]:
    a, b = S[k]["a"], S[k]["b"]
    xp = (yv - b) / a
    pred[tag] = dict(x_true=x_v.tolist(), x_pred=xp.tolist(), err=(xp - x_v).tolist())
S["verify"] = pred

# ======================= 温度 =======================
_, warm = load("温度正向.csv")
_, cool = load("温度降温.csv")
Tw, W1, W2 = num(warm, "pt100_temperature_c"), num(warm, "fbg_peak1_nm"), num(warm, "fbg_peak2_nm")
Tc, C1, C2 = num(cool, "pt100_temperature_c"), num(cool, "fbg_peak1_nm"), num(cool, "fbg_peak2_nm")
tw = np.array([np.datetime64(x["cycle_started_at"][:19]) for x in warm])

seg = [i for i in range(2, 20) if i != 12 and np.isfinite(W2[i])]
fit_w2 = linfit(Tw[seg] - Tw[seg[0]], W2[seg])
fit_w1 = linfit(Tw[seg] - Tw[seg[0]], W1[seg])
fit_diff = linfit(Tw[seg] - Tw[seg[0]], (W2 - W1)[seg])

def jumps(y, thr=500.0):
    d = np.diff(y) * 1000
    return [(int(i + 2), float(d[i])) for i in range(len(d)) if abs(d[i]) > thr]

S["temp"] = dict(
    n_rows=len(warm), n_partial=int(sum(1 for r in warm if r["cycle_status"] != "complete")),
    T_min=float(np.nanmin(Tw)), T_max=float(np.nanmax(Tw)),
    T_a=float(Tw[seg[0]]), T_b=float(Tw[seg[-1]]), n_seg=len(seg),
    L2a=float(W2[seg[0]]), L2b=float(W2[seg[-1]]),
    endpoint_sens=float((W2[seg[-1]] - W2[seg[0]]) * 1000 / (Tw[seg[-1]] - Tw[seg[0]])),
    fit2=fit_w2, fit1=fit_w1, fitdiff=fit_diff,
    jumps_w1=jumps(W1), jumps_w2=jumps(W2),
    jumps_c1=jumps(C1), jumps_c2=jumps(C2),
    cooling_n=len(cool), cooling_T=(float(Tc[0]), float(Tc[-1])),
    cooling_dl1=float((C1[4] - C1[0]) * 1000), cooling_dl2=float((C2[4] - C2[0]) * 1000),
    cooling_dT=float(Tc[4] - Tc[0]),
    warm_dur_min=float((tw[-1] - tw[0]) / np.timedelta64(1, "m")),
)

# ---------------- 温度模型对比与剖面（补充：分段/二次/列迁移救回） ----------------
def _fit(idx):
    ii = np.array([i for i in idx if np.isfinite(W2[i])], dtype=int)
    f = linfit(Tw[ii] - 26.17, W2[ii])
    return dict(n=int(len(ii)), K=float(f["a"] * 1000), r2=float(f["r2"]),
                rms_pm=float(f["rms"] * 1000))

# 二次拟合（16 点）
q = np.polyfit(Tw[seg] - 26.17, W2[seg], 2)
qr = W2[seg] - np.polyval(q, Tw[seg] - 26.17)
S["temp"]["quad"] = dict(a=float(q[0] * 1000), vertex_C=float(26.17 - q[1] / (2 * q[0])),
                         r2=float(1 - (qr @ qr) / ((W2[seg] - W2[seg].mean()) @ (W2[seg] - W2[seg].mean()))),
                         rms_pm=float(np.sqrt((qr @ qr) / (len(qr) - 3)) * 1000))
# 分段
S["temp"]["seg_models"] = dict(
    seg_26_42=_fit(range(2, 12)),
    seg_37_55=_fit([i for i in range(8, 20) if i != 12]),
    seg_47_55=_fit(range(14, 20)),
)
# 第 21--23 行（index 20..22）传感峰落在峰值 1 列：按 W1 救回
res_idx = list(seg) + [20, 21, 22]
yr = np.r_[W2[seg], W1[20:23]]
xr = np.r_[Tw[seg], Tw[20:23]] - 26.17
fr = linfit(xr, yr)
S["temp"]["rescue_21_23"] = dict(n=int(len(xr)), K=float(fr["a"] * 1000),
                                 r2=float(fr["r2"]), rms_pm=float(fr["rms"] * 1000),
                                 rows=[21, 22, 23])
# 指数松弛模型的最小二乘剖面（固定 tau 扫描 KT，对 lambda0/A 解析消去）
_TTp = ((tw[seg] - tw[seg][0]) / np.timedelta64(1, "m")).astype(float)
_DTp = (Tw[seg] - Tw[seg][0]).astype(float)
_YYp = (W2[seg] - 1544.0).astype(float) * 1000.0
_best = None
for _kt in np.arange(0.0, 20.001, 0.1):
    for _tau in np.arange(0.3, 10.001, 0.1):
        _X = np.column_stack([np.ones_like(_TTp), _kt * _DTp + (1 - np.exp(-_TTp / _tau))])
        _c, *_r = np.linalg.lstsq(_X, _YYp, rcond=None)
        _e = _YYp - _X @ _c
        _sse = float(_e @ _e)
        if _best is None or _sse < _best[0]:
            _best = (_sse, float(_kt), float(_tau), float(_c[0]), float(_c[1]))
S["temp"]["expo_profile"] = dict(K_T_pm_per_C=_best[1], tau_min=_best[2],
                                 l0_pm=_best[3], A_pm=_best[4],
                                 rms_pm=float(np.sqrt(_best[0] / (len(_TTp) - 2))))

# ---------------- MCMC ----------------
T0 = float(Tw[seg][0])
TT = ((tw[seg] - tw[seg][0]) / np.timedelta64(1, "m")).astype(float)
DT = (Tw[seg] - T0).astype(float)
YY = (W2[seg] - 1544.0).astype(float)

def logpost(p, prior_a=None):
    l0, a, A, tau, sig = p
    if not (0.02 < l0 < 0.35 and 0.0 <= a < 0.030 and -0.5 < A < 0.5
            and 0.3 < tau < 10.0 and 0.004 < sig < 0.08):
        return -np.inf
    lp = 0.0 if prior_a is None else -0.5 * ((a - prior_a[0]) / prior_a[1]) ** 2
    model = l0 + a * DT + A * (1.0 - np.exp(-TT / tau))
    r = YY - model
    return lp - 0.5 * np.sum(r ** 2) / sig ** 2 - len(YY) * np.log(sig)

def mcmc(prior_a=None, nsteps=60000, burn=15000, thin=10, seeds=(1, 2, 3)):
    """随机游走 Metropolis–Hastings，预热段按目标接受率 0.3 自适应调步长。"""
    step0 = np.array([0.004, 0.0015, 0.03, 0.35, 0.0015])
    p0 = np.array([0.155, 0.005, 0.30, 3.5, 0.012])
    out, acc_tot, n_tot = [], 0, 0
    for sd in seeds:
        rng = np.random.default_rng(sd)
        p = p0.copy(); lp = logpost(p, prior_a)
        step = step0.copy(); acc_w = 0; n_w = 0
        for i in range(nsteps):
            prop = p + rng.normal(0, step)
            lq = logpost(prop, prior_a)
            ok = np.log(rng.random()) < lq - lp
            if ok:
                p, lp = prop, lq
            if i < burn:
                acc_w += ok; n_w += 1
                if i % 1000 == 999:
                    rate = acc_w / n_w
                    step *= np.exp(np.clip((rate - 0.30) * 0.8, -0.4, 0.4))
                    acc_w = 0; n_w = 0
            else:
                acc_tot += ok; n_tot += 1
                if i % thin == 0:
                    out.append(p.copy())
    return np.array(out), acc_tot / n_tot

ch_u, acc_u = mcmc(None)
ch_p, acc_p = mcmc((0.0109, 0.0015))

def summ(ch):
    d = {}
    for j, nm, sc in zip(range(5), ("l0", "K_T", "A", "tau", "sigma"),
                         (1e3, 1e3, 1e3, 1.0, 1e3)):
        v = ch[:, j] * sc
        d[nm] = dict(median=float(np.median(v)), lo=float(np.percentile(v, 2.5)),
                     hi=float(np.percentile(v, 97.5)))
    d["corr_a_A"] = float(np.corrcoef(ch[:, 1], ch[:, 2])[0, 1])
    return d

S["mcmc"] = dict(uniform=summ(ch_u), prior=summ(ch_p),
                 acc_u=float(acc_u), acc_p=float(acc_p), N=int(len(ch_u)))

# ---------------- 联合拟合：温度项 + 几何夹持应变项（模型 B）----------------
# 几何前提：光纤与上、下孔口棱边同时相切（图 1）；孔半径随温度膨胀 R_h(T)=x0·r_f·(1+α_mΔT)
#   记 x=R_h/r_f、k=T_HALF/r_f：
#     A=sqrt(x²+k²)，θ=arccos(r_f/A)−arctan(k/x)            （相切条件）
#     L_c=2R_h/sinθ，δ=R_h−r_f
#     ε_b=(d_f/L_c²)δ=r_f(R_h−r_f)sin²θ/(2R_h²)=(x−1)sin²θ/(2x²)
# 联合模型（温度与应变同时拟合，K_T 与 K_ε 都在链里）：
#     λ(T)=λ0+K_T(T−T0)+K_ε[ε_b(T)−ε_b(T0)]
K_RATIO = 0.85 / 0.50      # k=T_HALF/r_f，取图 1 的示意比例
EPS_ALLOW = 0.002          # 石英光纤工程许用弯曲应变（保守取 0.2%）


def theta_of_x(x, k=K_RATIO):
    A = np.hypot(x, k)
    return np.degrees(np.arccos(np.clip(1.0 / A, -1.0, 1.0)) - np.arctan(k / x))


def eps_b_of(DT_, x0, alpha, k=K_RATIO):
    """相切几何给出的弯曲应变（无量纲）"""
    x = x0 * (1.0 + alpha * np.asarray(DT_, dtype=float))
    A = np.hypot(x, k)
    th = np.arccos(np.clip(1.0 / A, -1.0, 1.0)) - np.arctan(k / x)
    return (x - 1.0) * np.sin(th) ** 2 / (2.0 * x ** 2)


def d_eps_micro(x0, alpha):
    """ε_b(T)−ε_b(T0)，单位 με"""
    return (eps_b_of(DT, x0, alpha) - eps_b_of(0.0, x0, alpha)) / 1e-6


_dT_win = float(DT[-1] - DT[0])
_g = {}
for _tag, _ang in (("theta45", 45.0), ("theta65", 65.0)):
    _lo, _hi = 1.0 + 1e-9, 60.0
    for _ in range(300):
        _mid = 0.5 * (_lo + _hi)
        if theta_of_x(_mid) > _ang:
            _hi = _mid
        else:
            _lo = _mid
    _x = 0.5 * (_lo + _hi)
    _eb = float(eps_b_of(0.0, _x, 0.0))
    _g[_tag] = dict(theta_deg=_ang, x_ratio=float(_x), eps_b=_eb,
                    eps_micro=float(_eb * 1e6), dlam_pm=float(_eb * 1e6 * 1.2))
# 许用应变对应的最大直径比 x 与倾角
_lo, _hi = 1.0 + 1e-9, 3.0
for _ in range(200):
    _mid = 0.5 * (_lo + _hi)
    if float(eps_b_of(0.0, _mid, 0.0)) > EPS_ALLOW:
        _hi = _mid
    else:
        _lo = _mid
_x_allow = 0.5 * (_lo + _hi)
S["clamp"] = dict(k_ratio=K_RATIO, dT_window=_dT_win, eps_allow=EPS_ALLOW,
                  x_at_allow=float(_x_allow),
                  theta_at_allow_deg=float(theta_of_x(_x_allow)), **_g)

# ---- 联合 MCMC ----
YY_pm = YY * 1000.0


def logpost_joint(p, ke_prior=None, x_bounds=(1.0005, 8.0)):
    """p = (l0, K_T, K_eps, x0, alpha, sigma)"""
    l0, kt, ke, x0, al, sig = p
    if not (0.0 < l0 < 400.0 and 0.0 <= kt < 30.0 and 0.0 <= ke <= 10.0
            and x_bounds[0] <= x0 <= x_bounds[1] and 5e-6 <= al <= 3e-5 and 1.0 < sig < 60.0):
        return -np.inf
    lp = 0.0
    if ke_prior is not None:
        lp = -0.5 * ((ke - ke_prior[0]) / ke_prior[1]) ** 2
    model = l0 + kt * DT + ke * d_eps_micro(x0, al)
    r = YY_pm - model
    return lp - 0.5 * np.sum(r ** 2) / sig ** 2 - len(YY_pm) * np.log(sig)


def mcmc_joint(ke_prior=None, seeds=(1, 2, 3), nsteps=60000, burn=15000, thin=10,
               x_bounds=(1.0005, 8.0), x_start=2.0):
    step0 = np.array([5.0, 0.3, 0.3, 0.05, 2e-6, 1.0])
    p0 = np.array([170.0, 9.0, 2.0, x_start, 2e-5, 25.0])
    out, acc_tot, n_tot = [], 0, 0
    for sd in seeds:
        rng = np.random.default_rng(sd)
        p, lp = p0.copy(), logpost_joint(p0, ke_prior, x_bounds)
        step, acc_w, n_w = step0.copy(), 0, 0
        for i in range(nsteps):
            prop = p + rng.normal(0, step)
            lq = logpost_joint(prop, ke_prior, x_bounds)
            ok = np.log(rng.random()) < lq - lp
            if ok:
                p, lp = prop, lq
            if i < burn:
                acc_w += ok
                n_w += 1
                if i % 1000 == 999:
                    step *= np.exp(np.clip((acc_w / n_w - 0.30) * 0.8, -0.4, 0.4))
                    acc_w = n_w = 0
            else:
                acc_tot += ok
                n_tot += 1
                if i % thin == 0:
                    out.append(p.copy())
    return np.array(out), acc_tot / n_tot


ch_jw, acc_jw = mcmc_joint(None)              # K_ε 宽均匀先验 [0,10]
ch_jp, acc_jp = mcmc_joint((1.2, 0.3))        # K_ε ~ N(1.2, 0.3)（文献量级）
# 小倾角分支：相切几何下 θ 小 ⟺ 孔径只比纤径略大，x0 ≤ 1.30（θ ≲ 9°）
X_SMALL = 1.30
ch_js, acc_js = mcmc_joint(None, x_bounds=(1.0005, X_SMALL), x_start=1.1)


def summ_joint(ch, ke_prior=None, x_bounds=(1.0005, 8.0)):
    d = {}
    for j, nm in ((1, "K_T"), (2, "K_eps"), (3, "x0"), (4, "alpha")):
        v = ch[:, j]
        d[nm] = dict(median=float(np.median(v)), lo=float(np.percentile(v, 2.5)),
                     hi=float(np.percentile(v, 97.5)))
    d["sigma"] = dict(median=float(np.median(ch[:, 4 + 1])), lo=float(np.percentile(ch[:, 5], 2.5)),
                      hi=float(np.percentile(ch[:, 5], 97.5)))
    d["corr_kt_eps"] = float(np.corrcoef(ch[:, 1], ch[:, 2])[0, 1])
    d["corr_kt_x0"] = float(np.corrcoef(ch[:, 1], ch[:, 3])[0, 1])
    # 应变项在窗口末端的实际贡献（pm）
    contrib = np.array([ch[i, 2] * d_eps_micro(ch[i, 3], ch[i, 4])[-1] for i in range(len(ch))])
    d["contrib_pm"] = dict(median=float(np.median(contrib)),
                           lo=float(np.percentile(contrib, 2.5)),
                           hi=float(np.percentile(contrib, 97.5)))
    # 最优点（后验密度最大的样本）
    lps = np.array([logpost_joint(p, ke_prior, x_bounds) for p in ch])
    b = ch[int(np.argmax(lps))]
    d["map"] = dict(K_T=float(b[1]), K_eps=float(b[2]), x0=float(b[3]),
                    alpha=float(b[4]), sigma=float(b[5]))
    # 最优点处的残差与 AIC/BIC（σ̂ = sqrt(SSE/n)）
    _m = b[0] + b[1] * DT + b[2] * d_eps_micro(b[3], b[4])
    _sse = float(np.sum((YY_pm - _m) ** 2))
    _n = len(YY_pm)
    d["map"]["sse"] = _sse
    d["map"]["rms_pm"] = float(np.sqrt(_sse / _n))
    d["map"]["aic"] = float(_n * np.log(_sse / _n) + 2 * 6)
    d["map"]["bic"] = float(_n * np.log(_sse / _n) + 6 * np.log(_n))
    return d


S["mcmc"]["joint_wide"] = summ_joint(ch_jw, None)
S["mcmc"]["joint_phys"] = summ_joint(ch_jp, (1.2, 0.3))
np.save(os.path.join(RES, "chains_a_wide.npy"), ch_u)
np.save(os.path.join(RES, "chains_a_phys.npy"), ch_p)
np.save(os.path.join(RES, "chains_joint_small.npy"), ch_js)
S["mcmc"]["joint_small"] = summ_joint(ch_js, None, (1.0005, X_SMALL))
S["mcmc"]["acc_js"] = float(acc_js)
_dT_win_s = float(DT[-1] - DT[0])
_eps_s = S["mcmc"]["joint_small"]
# 小倾角分支的几何量（取后验中位 x0、α_m）
_xm = _eps_s["x0"]["median"]
_amu = _eps_s["alpha"]["median"]
S["clamp"]["small"] = dict(x_med=float(_xm), alpha_med=float(_amu),
                            theta_deg=float(theta_of_x(_xm)),
                            eps_b=float(eps_b_of(0.0, _xm, 0.0)),
                            d_eps_micro=float(d_eps_micro(_xm, _amu)[-1]),
                            dlam_at_1p2=float(d_eps_micro(_xm, _amu)[-1] * 1.2))
S["mcmc"]["acc_jw"] = float(acc_jw)
S["mcmc"]["acc_jp"] = float(acc_jp)

# ---- 与线性/二次模型同口径比较（AIC/BIC，n=16）----
_n = len(DT)
_lin = np.polyfit(DT, YY, 1)
_lin_m = np.polyval(_lin, DT)
_sse_lin = float(np.sum((YY * 1000.0 - _lin_m * 1000.0) ** 2))
_quad_m = np.polyval(np.polyfit(DT, YY, 2), DT)
_sse_quad = float(np.sum((YY * 1000.0 - _quad_m * 1000.0) ** 2))
S["mcmc"]["compare"] = dict(
    lin_rms_pm=float(np.sqrt(np.sum((YY * 1000.0 - _lin_m * 1000.0) ** 2) / (_n - 2))),
    quad_rms_pm=float(S["temp"]["quad"]["rms_pm"]),
    quad_span_pm=float(abs(S["temp"]["quad"]["a"]) * _dT_win ** 2),
    lin_KT_pm=float(_lin[0] * 1000.0),
    lin_aic=float(_n * np.log(_sse_lin / _n) + 2 * 3),
    lin_bic=float(_n * np.log(_sse_lin / _n) + 3 * np.log(_n)),
    lin_sigma_ml=float(np.sqrt(_sse_lin / _n)),
    quad_sigma_ml=float(np.sqrt(_sse_quad / _n)),
    quad_aic=float(_n * np.log(_sse_quad / _n) + 2 * 4),
    quad_bic=float(_n * np.log(_sse_quad / _n) + 4 * np.log(_n)),
)

# ======================= 图 1：PE =======================
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.4, 4.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.3, 1]})
a, b = S["PE_all"]["a"], S["PE_all"]["b"]
xx = np.linspace(8, 62, 120)
ax.plot(xx, a * xx + b, "-", color="0.55", lw=1.2,
        label="$S=%.2f\\,x%+.0f$，$R^2=%.5f$" % (a, b, S["PE_all"]["r2"]))
ax.plot(x_f, PE_f, "o", ms=5, mfc="white", mec="#1f4e79", mew=1.2, label="正程 10→60 mm（11 点）")
ax.plot(x_r, PE_r, "s", ms=4.2, mfc="#c00000", mec="#c00000", label="反程 60→10 mm（11 点）")
ax.set_ylabel("PE 原始输出 $S_{PE}$ / count")
ax.legend(loc="upper left", fontsize=7.8)
ax.grid(alpha=0.22, lw=0.5)
ax.text(0.40, 0.07, "50 mm 内变化 %.1f kcount\n灵敏度 %.2f count/mm\n1 count ≈ 0.99 µm" %
        (S["PE_span"] / 1000, a), transform=ax.transAxes, fontsize=7.8, color="#333333")
ax2.axhline(0, color="k", lw=0.6)
res_all = np.r_[S["PE_f"]["res"], S["PE_r"]["res"]]
ax2.plot(x_all, res_all, "o", ms=4, mfc="white", mec="#1f4e79")
ax2.axhspan(-S["PE_all"]["rms"], S["PE_all"]["rms"], color="#1f4e79", alpha=0.12,
            label="$\\pm$rms $=%.1f$ count" % S["PE_all"]["rms"])
ax2.set_xlabel("参考位置 $x$ / mm"); ax2.set_ylabel("残差 / count")
ax2.legend(fontsize=7.5, loc="lower right"); ax2.grid(alpha=0.22, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_pe_position.pdf")); plt.close(fig)

# ======================= 图 2：FBG 主峰 =======================
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.6, 4.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.3, 1]})
k = S["L2_all"]; kx, kb = k["a"], k["b"]
ax.plot(xx, kx * xx + kb, "-", color="0.55", lw=1.2,
        label="$\\lambda_2=%.4f+%.4f\\,x$（nm）" % (kb, kx))
ax.plot(x_f, L2f, "o", ms=5, mfc="white", mec="#1f4e79", mew=1.2, label="正程")
ax.plot(x_r, L2r, "s", ms=4.2, mfc="#c00000", mec="#c00000", label="反程")
ax.set_ylabel("主测峰 $\\lambda_2$ / nm")
ax.legend(loc="upper left", fontsize=7.8)
ax.grid(alpha=0.22, lw=0.5)
ax.text(0.50, 0.07, "$K_x=%.3f$ pm/mm，$R^2=%.5f$\n残差 rms $=%.1f$ pm\n50 mm 漂移 %.0f pm" %
        (kx * 1000, k["r2"], k["rms"] * 1000, S["L2_span_pm"]),
        transform=ax.transAxes, fontsize=7.8, color="#333333")
ax2.axhline(0, color="k", lw=0.6)
ax2.plot(x_all, np.r_[S["L2_f"]["res"], S["L2_r"]["res"]] * 1000, "o", ms=4,
         mfc="white", mec="#1f4e79")
ax2.axhspan(-k["rms"] * 1000, k["rms"] * 1000, color="#1f4e79", alpha=0.12,
            label="$\\pm$rms $=%.1f$ pm" % (k["rms"] * 1000))
ax2.set_xlabel("参考位置 $x$ / mm"); ax2.set_ylabel("残差 / pm")
ax2.legend(fontsize=7.5, loc="lower right"); ax2.grid(alpha=0.22, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_fbg_position.pdf")); plt.close(fig)

# ======================= 图 3：回程差 =======================
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
ax = axes[0]
ax.bar(x_f, np.array(hyst["PE"]) / 1000, width=3.4, color="#1f4e79", alpha=0.85)
ax.axhline(0, color="k", lw=0.6)
mx = max(abs(np.array(hyst["PE"])))
ax.text(0.05, 0.85, "最大 %.0f count = %.2f mm" % (mx, mx / S["PE_all"]["a"]),
        transform=ax.transAxes, fontsize=8)
ax.set_ylabel("$\\Delta S_{PE}$ / $10^3$ count"); ax.set_xlabel("参考位置 $x$ / mm")
ax.set_title("PE：正程$-$反程", fontsize=8.5)
ax = axes[1]
ax.bar(x_f - 0.95, hyst["L1"], width=1.8, color="#1f4e79", alpha=0.85, label="次峰（峰值 1）")
ax.bar(x_f + 0.95, hyst["L2"], width=1.8, color="#c00000", alpha=0.85, label="主峰（峰值 2）")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("$\\Delta\\lambda$ / pm"); ax.set_xlabel("参考位置 $x$ / mm")
ax.set_title("FBG：正程$-$反程", fontsize=8.5)
ax.legend(fontsize=7.2, ncol=2, loc="upper center")
ax.set_ylim(-20, 26)
for ax in axes: ax.grid(alpha=0.2, lw=0.5, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_hysteresis.pdf")); plt.close(fig)

# ======================= 图 4：验证 =======================
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
lim = [18, 60]
for ax, tag, ylab, ttl in [(axes[0], "PE", "PE 反算位置 / mm", "PE 链路"),
                           (axes[1], "L2", "主峰反算位置 / mm", "FBG 主峰")]:
    xp = np.array(pred[tag]["x_pred"])
    ax.fill_between(lim, np.array(lim) - 1, np.array(lim) + 1, color="0.88",
                    alpha=0.7, lw=0, label="$\\pm1$ mm（参考尺分辨力）")
    ax.plot(lim, lim, "--", color="0.45", lw=1.0, label="$y=x$")
    ax.plot(x_v, xp, "o", ms=5.5, mfc="#c00000", mec="#c00000", label="反算值")
    for xi, yi in zip(x_v, xp):
        ax.annotate("%+.2f" % (yi - xi), (xi, yi), textcoords="offset points",
                    xytext=(5, -11), fontsize=7.5, color="#333333")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("%s：最大偏差 %.2f mm" % (ttl, np.abs(xp - x_v).max()), fontsize=8.5)
    ax.set_xlabel("刻度尺参考位置 / mm"); ax.set_ylabel(ylab)
    ax.grid(alpha=0.22, lw=0.5)
axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_verification.pdf")); plt.close(fig)

# ======================= 图 5：温度原始 =======================
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
ax = axes[0]
ax.axvspan(S["temp"]["T_a"], S["temp"]["T_b"], color="#e8f0f8", zorder=0,
           label="稳定跟踪段（16 点）")
ax.plot(Tw, W2, "o-", ms=3.4, lw=0.9, mfc="#c00000", mec="#c00000",
        color="#c00000", label="主峰（峰值 2）")
ax.plot(Tw, W1, "s--", ms=3.0, lw=0.8, mfc="white", mec="#1f4e79",
        color="#1f4e79", label="次峰（峰值 1）")
for r, dl, off in [(S["temp"]["jumps_w2"][0][0], S["temp"]["jumps_w2"][0][1], (26, 10)),
                   (S["temp"]["jumps_w2"][-1][0], S["temp"]["jumps_w2"][-1][1], (-16, 10))]:
    ax.annotate("%+.1f nm" % (dl / 1000), xy=(Tw[r - 1], W2[r - 1]),
                xytext=off, textcoords="offset points", fontsize=6.8,
                color="#c00000", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#c00000"))
ax.text(0.03, 0.95, "其余跳变：$-0.5$ / $+0.5$ / $+0.7$ nm\n（第 13--14、21 行）",
        transform=ax.transAxes, fontsize=6.6, color="#c00000", va="top")
_lo = float(np.nanmin([np.nanmin(W1), np.nanmin(W2)]))
_hi = float(np.nanmax([np.nanmax(W1), np.nanmax(W2)]))
ax.set_ylim(_lo - 0.30, _hi + 0.80)
ax.set_xlabel("PT100 温度 $T$ / $^\\circ$C"); ax.set_ylabel("$\\lambda$ / nm")
ax.set_title("升温（30 s 采样，26 组）", fontsize=8.5)
ax.legend(fontsize=6.6, loc="lower right"); ax.grid(alpha=0.22, lw=0.5)
ax = axes[1]
ax.plot(Tc, C2, "s-", ms=3.4, lw=0.9, mfc="#c00000", mec="#c00000",
        color="#c00000", label="主峰")
ax.plot(Tc, C1, "o--", ms=3.0, lw=0.8, mfc="white", mec="#1f4e79",
        color="#1f4e79", label="次峰")
ax.annotate("失锁\n$-14.2$ / $-6.1$ nm", xy=(Tc[5], C2[5]),
            xytext=(59.2, 1541.5), fontsize=6.8, color="#333333",
            arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
ax.set_xlabel("PT100 温度 $T$ / $^\\circ$C"); ax.set_ylabel("$\\lambda$ / nm")
ax.set_title("降温（60 s 采样，8 组，全部失锁）", fontsize=8.5)
ax.invert_xaxis(); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.22, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_temp_raw.pdf")); plt.close(fig)

# ======================= 图 6：主峰温度拟合 =======================
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.9),
                              gridspec_kw={"width_ratios": [1.15, 1]})
ax.plot(Tw[seg], W2[seg], "s", ms=4.5, mfc="#c00000", mec="#c00000", label="主峰数据（16 点）")
xg = np.linspace(Tw[seg][0] - 0.5, Tw[seg][-1] + 0.5, 50)
py = fit_w2["a"] * (xg - Tw[seg][0]) + fit_w2["b"]
ax.plot(xg, py, "-", color="0.45", lw=1.2, label="线性拟合")
ax.fill_between(xg, py - fit_w2["rms"], py + fit_w2["rms"], color="0.45",
                alpha=0.15, lw=0, label="$\\pm$rms 带")
ax.set_xlabel("$T$ / $^\\circ$C"); ax.set_ylabel("$\\lambda_2$ / nm")
ax.ticklabel_format(axis="y", useOffset=False)
ax.set_title("$K_T=%.2f$ pm/$^\\circ$C，$R^2=%.4f$" % (fit_w2["a"] * 1000, fit_w2["r2"]), fontsize=8.5)
ax.text(0.03, 0.06, "端点法 %.2f pm/$^\\circ$C\n残差 rms %.1f pm" %
        (S["temp"]["endpoint_sens"], fit_w2["rms"] * 1000), transform=ax.transAxes,
        fontsize=7.5, color="#333333")
ax.legend(fontsize=7.2, loc="upper left"); ax.grid(alpha=0.22, lw=0.5)
ax2.axhline(0, color="k", lw=0.6)
ax2.plot(Tw[seg], np.array(fit_w2["res"]) * 1000, "o", ms=4.2, mfc="white", mec="#1f4e79")
ax2.set_xlabel("$T$ / $^\\circ$C"); ax2.set_ylabel("残差 / pm")
ax2.set_title("线性拟合残差（非随机起伏）", fontsize=8.5)
ax2.grid(alpha=0.22, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_temp_fit.pdf")); plt.close(fig)

# ======================= 图 7：双峰对比 =======================
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
ax = axes[0]
ax.plot(Tw[seg], W2[seg] - 1543.0, "s-", ms=3.4, lw=0.8, mfc="#c00000", mec="#c00000",
        color="#c00000", label="主峰 $\\lambda_2$（%.2f pm/$^\\circ$C）" % (fit_w2["a"] * 1000))
ax.plot(Tw[seg], W1[seg] - 1543.0, "o-", ms=3.4, lw=0.8, mfc="white", mec="#1f4e79",
        color="#1f4e79", label="次峰 $\\lambda_1$（%.2f pm/$^\\circ$C）" % (fit_w1["a"] * 1000))
ax.set_xlabel("$T$ / $^\\circ$C"); ax.set_ylabel("$\\lambda-1543$ / nm")
ax.set_title("两峰各自的温度漂移", fontsize=8.5)
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.22, lw=0.5)
ax = axes[1]
ax.plot(Tw[seg], (W2 - W1)[seg] * 1000, "o", ms=4, mfc="white", mec="#7f6000")
xg3 = np.linspace(Tw[seg][0], Tw[seg][-1], 10)
ax.plot(xg3, fit_diff["a"] * (xg3 - Tw[seg][0]) + fit_diff["b"] * 1000, "-",
        color="0.45", lw=1.1, label="差分拟合 %.2f pm/$^\\circ$C" % (fit_diff["a"] * 1000))
ax.set_xlabel("$T$ / $^\\circ$C"); ax.set_ylabel("$\\lambda_2-\\lambda_1$ / pm")
ax.set_title("差分波长（扣除公共漂移）", fontsize=8.5)
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.22, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig7_peak_compare.pdf")); plt.close(fig)

# ======================= 图 8：MCMC（画图逻辑见 scripts/plot_mcmc.py） =======================
import plot_mcmc  # noqa: E402
plot_mcmc.draw(S, FIG, RES)

# ======================= 输出 =======================
json.dump(S, open(os.path.join(RES, "summary.json"), "w"), ensure_ascii=False, indent=1)

print("=== 位移 ===")
for kk in ["PE_f", "PE_r", "PE_all", "L1_all", "L2_all"]:
    s = S[kk]
    print("%-6s a=%.6g b=%.6f R2=%.6f rms=%.4g max=%.4g n=%d" %
          (kk, s["a"], s["b"], s["r2"], s["rms"], s["maxres"], s["n"]))
print("PE span %.0f  L2 span %.1f pm  L1 span %.1f pm" % (S["PE_span"], S["L2_span_pm"], S["L1_span_pm"]))
print("spacing %.4f +- %.4f nm (%.3f..%.3f)" % (S["spacing"]["mean"], S["spacing"]["std"],
                                                S["spacing"]["min"], S["spacing"]["max"]))
print("skew max %.1f ms" % S["skew_max"])
print("hyst PE max %.0f (=%.3f mm)  L2 %.1f pm  L1 %.1f pm" %
      (max(abs(np.array(hyst["PE"]))), max(abs(np.array(hyst["PE"]))) / S["PE_all"]["a"],
       max(abs(np.array(hyst["L2"]))), max(abs(np.array(hyst["L1"])))))
for tag in ["PE", "L1", "L2"]:
    print("verify", tag, np.round(pred[tag]["x_pred"], 2), "err", np.round(pred[tag]["err"], 2),
          "rms %.3f max %.3f" % (np.sqrt(np.mean(np.square(pred[tag]["err"]))),
                                 np.abs(pred[tag]["err"]).max()))
t = S["temp"]
print("=== 温度 ===")
print("rows %d partial %d T %.2f..%.2f dur %.2f min" % (t["n_rows"], t["n_partial"], t["T_min"], t["T_max"], t["warm_dur_min"]))
print("seg %.2f..%.2f n=%d endpoint %.2f" % (t["T_a"], t["T_b"], t["n_seg"], t["endpoint_sens"]))
for nm in ["fit2", "fit1", "fitdiff"]:
    s = t[nm]
    print("%-8s a=%.3f pm/C b=%.6f R2=%.4f rms=%.2f max=%.2f n=%d" %
          (nm, s["a"] * 1000, s["b"], s["r2"], s["rms"] * 1000, s["maxres"] * 1000, s["n"]))
print("jumps_w2", t["jumps_w2"]); print("jumps_w1", t["jumps_w1"])
print("cooling dT %.2f dL1 %.0f pm dL2 %.0f pm" % (t["cooling_dT"], t["cooling_dl1"], t["cooling_dl2"]))
print("=== MCMC ===")
for key in ["uniform", "prior"]:
    m = S["mcmc"][key]
    print(key, "K_T %.2f [%.2f, %.2f]  A %.1f [%.1f, %.1f] pm  tau %.2f [%.2f, %.2f]  sigma %.1f [%.1f, %.1f] pm  corr %.2f" %
          (m["K_T"]["median"], m["K_T"]["lo"], m["K_T"]["hi"],
           m["A"]["median"], m["A"]["lo"], m["A"]["hi"],
           m["tau"]["median"], m["tau"]["lo"], m["tau"]["hi"],
           m["sigma"]["median"], m["sigma"]["lo"], m["sigma"]["hi"], m["corr_a_A"]))
print("acc_u %.3f acc_p %.3f N %d" % (S["mcmc"]["acc_u"], S["mcmc"]["acc_p"], S["mcmc"]["N"]))
for tag, key in (("joint-wide", "joint_wide"), ("joint-phys", "joint_phys"),
                 ("joint-small", "joint_small")):
    mj = S["mcmc"][key]
    print("%s K_T %.2f [%.2f, %.2f]  K_eps %.2f [%.2f, %.2f]  x0 %.2f [%.2f, %.2f]  "
          "sigma %.1f  应变项 %.0f pm [%.0f, %.0f]  corr(K_T,K_eps) %.2f  corr(K_T,x0) %.2f"
          % (tag, mj["K_T"]["median"], mj["K_T"]["lo"], mj["K_T"]["hi"],
             mj["K_eps"]["median"], mj["K_eps"]["lo"], mj["K_eps"]["hi"],
             mj["x0"]["median"], mj["x0"]["lo"], mj["x0"]["hi"], mj["sigma"]["median"],
             mj["contrib_pm"]["median"], mj["contrib_pm"]["lo"], mj["contrib_pm"]["hi"],
             mj["corr_kt_eps"], mj["corr_kt_x0"]))
    print("   最优点 K_T %.2f  K_eps %.2f  x0 %.2f  rms %.1f pm  AIC %.1f  BIC %.1f"
          % (mj["map"]["K_T"], mj["map"]["K_eps"], mj["map"]["x0"],
             mj["map"]["rms_pm"], mj["map"]["aic"], mj["map"]["bic"]))
cp = S["mcmc"]["compare"]
print("compare 线性 rms %.1f AIC %.1f | 二次 rms %.1f AIC %.1f | 需解释曲率 %.0f pm"
      % (cp["lin_rms_pm"], cp["lin_aic"], cp["quad_rms_pm"], cp["quad_aic"], cp["quad_span_pm"]))
c = S["clamp"]
print("几何自洽性：θ>45° ⟹ x=%.2f, ε_b=%.2f%% (%.0f με)；θ=65° ⟹ x=%.2f；ε_b≤%.1f%% ⟹ x≤%.3f, θ≤%.1f°"
      % (c["theta45"]["x_ratio"], c["theta45"]["eps_b"] * 100, c["theta45"]["eps_micro"],
         c["theta65"]["x_ratio"], c["eps_allow"] * 100, c["x_at_allow"], c["theta_at_allow_deg"]))
