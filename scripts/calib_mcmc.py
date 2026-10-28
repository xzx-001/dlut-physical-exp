"""贝叶斯联合拟合：以 CD/DVD 标称道距为约束估计横向标定因子 k（报告 §3.5）。

数据流：obs_from_csm.py → observations.json → 本脚本 → ../results/calib_mcmc.json

已知限制（2026-10-27 独立审计发现，未修改采样器本身）：
  stretch move 的接受率缺 z^(d-1) 雅可比因子（第 75 行附近），导致接受率偏高
  （0.70--0.72，理想 0.2--0.5）、后验区间偏窄约 2--3 倍；另 lnL_max 在
  "未接受的试探点 + 忽略先验" 上取极大，故 mcmc_summary.json 中的 lnL 绝对值不可用。
  报告正文采用的 k 中心值（稳健）与 1σ（拉普拉斯近似 ±0.018）已按审计结论修正，
  不要用本脚本输出的 ± 区间直接替换正文数字。

"""
"""贝叶斯联合拟合：以 CD/DVD 标准约束三张光盘的表观横向周期。

无 scipy / 无 emcee → 自写仿射不变系综采样器（Goodman & Weare 2010 stretch move）。

核心关系
    P_obs,i = k · p_i / cos α_i
    k = P_obs / p_true = S_nominal / S_true = C_nominal / C_true
      （表观过冲因子；等于"真实横向伸缩系数相对铭牌值的相对缺失"的倒数）
    p_i = 该盘真实道距（由假设固定 or 自由）
    α_i = 条纹相对慢扫轴（y）的倾角 → 沿快扫轴截取的周期被 1/cosα 拉长

关键不可辨识性
    k 与全部 p_i 只以乘积出现。若 p_i 自由且 k 亦自由，后验是严格的一条脊（完全退化），
    数据本身无法把"仪器标定偏大"与"盘不是标准盘"分开。因此：
      · H1/H2/H3 固定 p_i → 用"三盘共享同一 k"的自洽性做模型比较；
      · H4 固定 k=1 → 问"若仪器是对的，三张盘各是什么道距"，再与已知格式对照。

跑法：python3 mcmc_fit.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(HERE, 'diag.json')
OUTJ = os.path.join(HERE, 'mcmc_summary.json')
CHAINDIR = os.path.join(HERE, 'chains')

DISCS = ['cd1', 'cd2', 'cd3']
STD_CD, TOL_CD = 1600.0, 100.0      # ECMA-130 §11.3  (1.6 µm)
STD_DVD, TOL_DVD = 740.0, 30.0      # ECMA-267 §10.6.2 (0.74 ± 0.03 µm)
SIG_K_PRIOR = 0.35                  # lognormal(0, 0.35)：k 先验，约 0.5–2.0
SIG_ALPHA_PRIOR = 0.3               # 度


def load():
    d = json.load(open(DIAG))
    D = {'names': DISCS, 'P': [], 'sP': [], 'alpha': [], 'depth': [],
         'block_spread_pct': [], 'fwhm_ratio': []}
    for nm in DISCS:
        P = d[nm]['A_row_fft_median']
        D['P'].append(P)
        D['sP'].append(0.010 * P)                  # 1% 方法/系统误差
        D['alpha'].append(d[nm]['tilt_deg'])
        D['depth'].append(d[nm]['G_depth_topo'])
        D['block_spread_pct'].append(d[nm]['E_block_rel_spread_pct'])
        D['fwhm_ratio'].append(d[nm]['D_fwhm_ratio'])
    for k in ('P', 'sP', 'alpha', 'depth', 'block_spread_pct', 'fwhm_ratio'):
        D[k] = np.array(D[k], float)
    return D


# ---------------------------------------------------------------- 采样器
def stretch_sample(logp, logL, x0, nsteps, rng, a=2.0):
    nw, nd = x0.shape
    chain = np.empty((nsteps, nw, nd))
    lnp_hist = np.empty((nsteps, nw))
    lnp = np.array([logp(x) for x in x0], float)
    lnL = np.array([logL(x) for x in x0], float)
    fin = lnL[np.isfinite(lnL)]
    lnL_max = float(fin.max()) if fin.size else -np.inf
    Z = x0.copy()
    nacc = 0
    for t in range(nsteps):
        for kk in range(nw):
            j = rng.integers(0, nw - 1)
            j = j + 1 if j >= kk else j
            z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a
            prop = Z[j] + z * (Z[kk] - Z[j])
            lp = logp(prop)
            lL = logL(prop)
            if np.isfinite(lL) and lL > lnL_max:
                lnL_max = lL
            if np.log(rng.random()) < lp - lnp[kk]:
                Z[kk] = prop
                lnp[kk] = lp
                lnL[kk] = lL
                nacc += 1
        chain[t] = Z
        lnp_hist[t] = lnp
    return chain, lnp_hist, lnL_max, nacc / (nsteps * nw)


def autocorr_time(x):
    n = len(x)
    x = x - x.mean()
    v = np.dot(x, x) / n
    if v <= 0:
        return np.nan
    tau = 0.5
    for w in range(1, n // 2):
        c = np.dot(x[:-w], x[w:]) / (n - w) / v
        tau += c
        if w >= 5.0 * tau:
            break
    return tau


def ess_and_tau(chain):
    ns, nw, nd = chain.shape
    taus, esss = [], []
    for d in range(nd):
        ts = [autocorr_time(chain[:, w, d]) for w in range(nw)]
        tau = float(np.nanmedian(ts))
        taus.append(tau)
        esss.append(float(ns * nw / tau))
    return taus, esss


def q3(a):
    return [float(np.percentile(a, 16)), float(np.median(a)), float(np.percentile(a, 84))]


# ---------------------------------------------------------------- 模型
class Model:
    def __init__(self, D, name, p_fix=None, k_free=True, p_free=False,
                 sigma_free=False, sig_k=SIG_K_PRIOR):
        self.D = D
        self.name = name
        self.p_fix = p_fix
        self.k_free = k_free
        self.p_free = p_free
        self.sigma_free = sigma_free
        self.sig_k = sig_k

    def dim(self):
        return ((1 if self.k_free else 0) + 3
                + (1 if self.sigma_free else 0)
                + (3 if self.p_free else 0))

    def i0p(self):
        return (1 if self.k_free else 0) + 3 + (1 if self.sigma_free else 0)

    def ial(self, j):
        return (1 if self.k_free else 0) + j

    def unpack(self, th):
        i = 0
        k = np.exp(th[i]) if self.k_free else 1.0
        i += 1 if self.k_free else 0
        al = th[i:i + 3]
        i += 3
        sig = np.exp(th[i]) if self.sigma_free else 0.0
        i += 1 if self.sigma_free else 0
        pp = (np.exp(th[i:i + 3]) if self.p_free
              else (self.p_fix if self.p_fix is not None else np.ones(3)))
        return k, al, sig, pp

    def logprior(self, th):
        D = self.D
        k, al, sig, pp = self.unpack(th)
        lp = 0.0
        if self.k_free:
            lp += -0.5 * (np.log(k) / self.sig_k) ** 2 - np.log(k)
        for j in range(3):
            lp += -0.5 * ((al[j] - D['alpha'][j]) / SIG_ALPHA_PRIOR) ** 2
        if self.sigma_free:
            Pbar = D['P'].mean()
            if not (1.0 <= sig <= 0.2 * Pbar):
                return -np.inf
        if self.p_free:
            for j in range(3):
                lp += -0.5 * ((np.log(pp[j]) - np.log(1000.0)) / 0.5) ** 2 - np.log(pp[j])
        return float(lp)

    def logL(self, th):
        D = self.D
        k, al, sig, pp = self.unpack(th)
        if self.p_free and np.any(pp <= 0):
            return -np.inf
        if self.k_free and k <= 0:
            return -np.inf
        mod = k * pp / np.cos(np.radians(al))
        s2 = D['sP'] ** 2 + sig ** 2
        return float(np.sum(-0.5 * (D['P'] - mod) ** 2 / s2
                            - 0.5 * np.log(2 * np.pi * s2)))

    def logp(self, th):
        lp = self.logprior(th)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.logL(th)

    def init(self, rng, nw):
        D = self.D
        X = np.zeros((nw, self.dim()))
        i = 0
        if self.k_free:
            X[:, 0] = np.log(1.3) + 0.15 * rng.standard_normal(nw)
            i = 1
        for j in range(3):
            X[:, i + j] = D['alpha'][j] + 0.1 * rng.standard_normal(nw)
        i += 3
        if self.sigma_free:
            X[:, i] = np.log(0.01 * D['P'].mean()) + 0.3 * rng.standard_normal(nw)
            i += 1
        if self.p_free:
            for j in range(3):
                X[:, i + j] = np.log(D['P'][j] / 1.34) + 0.1 * rng.standard_normal(nw)
        return X


class ModelKi(Model):
    """每盘独立 k_i（检验"三盘共享 k"是否成立）。"""
    def dim(self):
        return 3 + 3 + 1

    def unpack(self, th):
        return np.exp(th[0:3]), th[3:6], np.exp(th[6]), self.p_fix

    def logprior(self, th):
        D = self.D
        kk, al, sig, pp = self.unpack(th)
        lp = 0.0
        for j in range(3):
            lp += -0.5 * (np.log(kk[j]) / self.sig_k) ** 2 - np.log(kk[j])
            lp += -0.5 * ((al[j] - D['alpha'][j]) / SIG_ALPHA_PRIOR) ** 2
        Pbar = D['P'].mean()
        if not (1.0 <= sig <= 0.2 * Pbar):
            return -np.inf
        return float(lp)

    def logL(self, th):
        D = self.D
        kk, al, sig, pp = self.unpack(th)
        mod = kk * pp / np.cos(np.radians(al))
        s2 = D['sP'] ** 2 + sig ** 2
        return float(np.sum(-0.5 * (D['P'] - mod) ** 2 / s2
                            - 0.5 * np.log(2 * np.pi * s2)))

    def init(self, rng, nw):
        X = np.zeros((nw, 7))
        X[:, 0:3] = np.log(1.3) + 0.15 * rng.standard_normal((nw, 3))
        for j in range(3):
            X[:, 3 + j] = self.D['alpha'][j] + 0.1 * rng.standard_normal(nw)
        X[:, 6] = np.log(0.01 * self.D['P'].mean()) + 0.3 * rng.standard_normal(nw)
        return X


def run_model(m, nsteps, burn, seed=20261027):
    rng = np.random.default_rng(seed)
    nw = max(2 * m.dim() + 2, 20)
    x0 = m.init(rng, nw)
    ch, lph, lnLmax, acc = stretch_sample(m.logp, m.logL, x0, nsteps, rng)
    post = ch[burn:]
    flat = post.reshape(-1, m.dim())
    taus, esss = ess_and_tau(post)
    k, al, sig, pp = m.unpack(np.median(flat, axis=0))
    mod = k * pp / np.cos(np.radians(al))
    s = {'tag': m.name.split()[0], 'model': m.name, 'ndim': m.dim(),
         'nwalkers': nw, 'nsteps': nsteps, 'burn': burn, 'accept': float(acc),
         'lnL_max': float(lnLmax),
         'tau': [float(t) for t in taus], 'ess': [float(e) for e in esss],
         'pred_period': [float(v) for v in mod],
         'resid_rel_pct': [float(100 * (m.D['P'][j] - mod[j]) / m.D['P'][j]) for j in range(3)]}
    if isinstance(m, ModelKi):
        for j in range(3):
            s[f'k{j+1}'] = q3(np.exp(flat[:, j]))
    elif m.k_free:
        s['k'] = q3(np.exp(flat[:, 0]))
    if m.p_free:
        i0 = m.i0p()
        for j in range(3):
            s[f'p{j+1}'] = q3(np.exp(flat[:, i0 + j]))
        if m.k_free:
            s['corr_logk_logp'] = [float(np.corrcoef(flat[:, 0], flat[:, i0 + j])[0, 1])
                                   for j in range(3)]
    if m.sigma_free:
        iS = (1 if m.k_free else 0) + 3
        s['sigma_extra_nm'] = q3(np.exp(flat[:, iS]))
    s['perdisc'] = []
    for j in range(3):
        if isinstance(m, ModelKi):
            kk = np.exp(flat[:, j])
        elif m.k_free:
            kk = np.exp(flat[:, 0])
        else:
            kk = 1.0
        pj = (np.exp(flat[:, m.i0p() + j]) if m.p_free
              else (m.p_fix[j] if m.p_fix is not None else 1.0))
        aj = flat[:, (3 if isinstance(m, ModelKi) else (1 if m.k_free else 0)) + j]
        s['perdisc'].append(q3(kk * pj / np.cos(np.radians(aj))))
    s['bic'] = -2 * s['lnL_max'] + m.dim() * np.log(3)
    s['aic'] = -2 * s['lnL_max'] + 2 * m.dim()
    s['chain_file'] = os.path.join(CHAINDIR, s['tag'] + '.npy')
    np.save(s['chain_file'], post)
    return s


def main():
    os.makedirs(CHAINDIR, exist_ok=True)
    D = load()
    print('观测向量（由原始 .csm 复算）:')
    for j, nm in enumerate(DISCS):
        print(f'  {nm}: P={D["P"][j]:.2f} ± {D["sP"][j]:.2f} nm  α={D["alpha"][j]:+.2f}°  '
              f'depth={D["depth"][j]:.1f} nm  blk_spread={D["block_spread_pct"][j]:.2f}%  '
              f'FWHM/ref={D["fwhm_ratio"][j]:.3f}')

    P_CD = np.array([STD_CD] * 3)
    P_DVD = np.array([STD_DVD] * 3)
    P_H1 = np.array([STD_CD, STD_CD, STD_DVD])

    specs = [
        ('H1', Model(D, 'H1 CD,CD,DVD sigfixed', p_fix=P_H1), 12000, 3000),
        ('H2', Model(D, 'H2 allCD sigfixed', p_fix=P_CD), 12000, 3000),
        ('H3', Model(D, 'H3 allDVD sigfixed', p_fix=P_DVD), 12000, 3000),
        ('H4', Model(D, 'H4 pfree k=1 sigfixed', p_fix=None, k_free=False, p_free=True),
         14000, 3500),
        ('B1', Model(D, 'B1 H1 sigmafree', p_fix=P_H1, sigma_free=True), 12000, 3000),
        ('S1', ModelKi(D, 'S1 H1 perdisck', p_fix=P_H1), 12000, 3000),
        ('S2', Model(D, 'S2 H1 widek', p_fix=P_H1, sig_k=0.7), 10000, 3000),
        ('S3', Model(D, 'S3 H1 pfree+kfree', p_fix=None, k_free=True, p_free=True),
         10000, 2500),
    ]
    res = []
    for tag, m, ns, nb in specs:
        s = run_model(m, ns, nb)
        s['tag'] = tag
        res.append(s)

    with open(OUTJ, 'w') as f:
        json.dump({'data': {k: D[k].tolist() for k in
                            ('P', 'sP', 'alpha', 'depth', 'block_spread_pct', 'fwhm_ratio')},
                   'results': res}, f, indent=1, ensure_ascii=False)

    print('\n=== 后验（中位数 [16%, 84%]）+ 收敛诊断 ===')
    for s in res:
        print(f"\n[{s['tag']}] {s['model']}  dim={s['ndim']} acc={s['accept']:.2f} "
              f"tau_max={max(s['tau']):.1f} ESS_min={min(s['ess']):.0f}")
        for key in ('k', 'k1', 'k2', 'k3', 'p1', 'p2', 'p3', 'sigma_extra_nm'):
            if key in s:
                v = s[key]
                print(f"   {key:14s} = {v[1]:10.4f}   [{v[0]:.4f}, {v[2]:.4f}]")
        print(f"   每盘模型周期 = {[round(v[1], 1) for v in s['perdisc']]}")
        print(f"   相对残差 %   = {[round(v, 2) for v in s['resid_rel_pct']]}")
        print(f"   lnL_max={s['lnL_max']:.3f} BIC={s['bic']:.2f} AIC={s['aic']:.2f}"
              + (f" corr(logk,logp)={[round(v, 3) for v in s['corr_logk_logp']]}"
                 if 'corr_logk_logp' in s else ''))

    print('\n=== 模型比较（n=3；ΔBIC 相对最优）===')
    cand = [s for s in res if s['tag'] in ('H1', 'H2', 'H3', 'H4')]
    bmin = min(s['bic'] for s in cand)
    for s in cand:
        print(f"  {s['model']:<22s} lnL_max={s['lnL_max']:8.3f} BIC={s['bic']:7.2f} "
              f"dBIC={s['bic']-bmin:7.2f}")


if __name__ == '__main__':
    main()
