#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""光纤传感实验（位移/应变 + 温度）数据分析
输入：../*.csv   输出：figures/*.pdf + results/summary.json + 控制台表格
所有写入报告的数字均来自本脚本输出。
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(BASE)
FIG = os.path.join(BASE, "figures")
RES = os.path.join(BASE, "results")
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
    out = []
    for x in rows:
        v = x.get(key, "")
        out.append(float(v) if v not in ("", None) else np.nan)
    return np.array(out)

# ---------------- 位移/应变数据 ----------------
_, fwd = load("应变正向60-5.csv")
_, rev = load("应变反向.csv")
hv, ver = load("应变验证值.csv", "gb18030")

x_f = np.arange(10, 61, 5.0)          # 正程 10→60
x_r = np.arange(60, 9, -5.0)          # 反程 60→10
PE_f, PE_r = num(fwd, "pe_raw_value"), num(rev, "pe_raw_value")
L1f, L1r = num(fwd, "fbg_peak1_nm"), num(rev, "fbg_peak1_nm")
L2f, L2r = num(fwd, "fbg_peak2_nm"), num(rev, "fbg_peak2_nm")
ADf, ADr = num(fwd, "cmcu_ad_value"), num(rev, "cmcu_ad_value")

x_v = np.array([float(list(r.values())[-1]) for r in ver])   # 位置(mm) 列
PE_v, L1v, L2v = num(ver, "pe_raw_value"), num(ver, "fbg_peak1_nm"), num(ver, "fbg_peak2_nm")

def linfit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    a, b = np.polyfit(x, y, 1)
    yh = a * x + b
    res = y - yh
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(a=float(a), b=float(b), r2=float(1 - ss_res / ss_tot),
                rms=float(np.sqrt(ss_res / (len(x) - 2))), maxres=float(np.max(np.abs(res))),
                n=int(len(x)), res=res.tolist())

S = {}
S["n_fwd"], S["n_rev"], S["n_ver"] = len(fwd), len(rev), len(ver)
S["status_all_complete"] = all(r["cycle_status"] == "complete" for r in fwd + rev + ver)

for tag, x, y in [("PE_f", x_f, PE_f), ("PE_r", x_r, PE_r),
                  ("L1_f", x_f, L1f), ("L1_r", x_r, L1r),
                  ("L2_f", x_f, L2f), ("L2_r", x_r, L2r),
                  ("AD_f", x_f, ADf), ("AD_r", x_r, ADr)]:
    S[tag] = linfit(x, y)

x_all = np.r_[x_f, x_r]
S["PE_all"] = linfit(x_all, np.r_[PE_f, PE_r])
S["L1_all"] = linfit(x_all, np.r_[L1f, L1r])
S["L2_all"] = linfit(x_all, np.r_[L2f, L2r])
S["AD_all"] = linfit(x_all, np.r_[ADf, ADr])
S["AD_std"] = float(np.std(np.r_[ADf, ADr]))
S["AD_mean"] = float(np.mean(np.r_[ADf, ADr]))

# 回程差：同一参考位置 正程-反程
hyst = dict(x=x_f.tolist(),
            PE=(PE_f - PE_r[::-1]).tolist(),
            L1=((L1f - L1r[::-1]) * 1000).tolist(),
            L2=((L2f - L2r[::-1]) * 1000).tolist(),
            AD=(ADf - ADr[::-1]).tolist())
S["hyst"] = hyst
S["PE_span"] = float(PE_f[-1] - PE_f[0])
S["L2_span_pm"] = float((L2f[-1] - L2f[0]) * 1000)
S["L1_span_pm"] = float((L1f[-1] - L1f[0]) * 1000)
S["L2_spacing_mean"] = float(np.mean((L2f - L1f) * 1000))

# 验证点反算
pred = {}
for tag, yv, fitk in [("PE", PE_v, "PE_all"), ("L1", L1v, "L1_all"), ("L2", L2v, "L2_all")]:
    a, b = S[fitk]["a"], S[fitk]["b"]
    xp = (yv - b) / a
    pred[tag] = dict(x_true=x_v.tolist(), x_pred=xp.tolist(),
                     err=(xp - x_v).tolist())
S["verify"] = pred

# ---------------- 温度数据 ----------------
_, warm = load("温度正向.csv")
_, cool = load("温度降温.csv")
Tw, W1, W2 = num(warm, "pt100_temperature_c"), num(warm, "fbg_peak1_nm"), num(warm, "fbg_peak2_nm")
Tc, C1, C2 = num(cool, "pt100_temperature_c"), num(cool, "fbg_peak1_nm"), num(cool, "fbg_peak2_nm")

# 稳定跟踪段：第 3~20 行，剔除第 13 行（峰识别跳变）
seg = [i for i in range(2, 20) if i != 12 and np.isfinite(W2[i])]
fit_w2 = linfit(Tw[seg], W2[seg])
fit_w1 = linfit(Tw[seg], W1[seg])
fit_diff = linfit(Tw[seg], W2[seg] - W1[seg])
# 端点法平均灵敏度
T_a, T_b = Tw[seg[0]], Tw[seg[-1]]
S["temp"] = dict(seg_rows=[3, 20], seg_excluded=[13], n=len(seg),
                 T_a=float(T_a), T_b=float(T_b),
                 L2a=float(W2[seg[0]]), L2b=float(W2[seg[-1]]),
                 endpoint_sens=float((W2[seg[-1]] - W2[seg[0]]) * 1000 / (T_b - T_a)),
                 fit2=fit_w2, fit1=fit_w1, fitdiff=fit_diff,
                 T_min=float(np.nanmin(Tw)), T_max=float(np.nanmax(Tw)),
                 n_rows=len(warm), n_partial=int(sum(1 for r in warm if r["cycle_status"] != "complete")),
                 cooling_T=(float(Tc[0]), float(Tc[-1])), cooling_n=len(cool))

# 温度原始曲线的峰识别跳变
def jumps(y):
    d = np.diff(y) * 1000
    return [(int(i + 1), float(d[i])) for i in range(len(d)) if abs(d[i]) > 500]
S["temp"]["jumps_p1"] = jumps(W1)
S["temp"]["jumps_p2"] = jumps(W2)
S["temp"]["jumps_cool_p1"] = jumps(C1)
S["temp"]["jumps_cool_p2"] = jumps(C2)

# ---------------- 图 1：PE 标定 ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
a, b = S["PE_all"]["a"], S["PE_all"]["b"]
xx = np.linspace(8, 62, 100)
ax.plot(xx, a * xx + b, "-", color="0.55", lw=1.2,
        label="整体线性拟合 $y=%.1fx%+.0f$" % (a, b))
ax.plot(x_f, PE_f, "o", ms=4.5, mfc="white", mec="#1f4e79", label="正程 10→60 mm")
ax.plot(x_r, PE_r, "s", ms=4.0, mfc="#c00000", mec="#c00000", label="反程 60→10 mm")
ax.set_xlabel("参考位置 $x$ / mm")
ax.set_ylabel("PE 原始输出 / count")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.25, lw=0.5)
fig.savefig(os.path.join(FIG, "fig1_pe_position.pdf"))
plt.close(fig)

# ---------------- 图 2：FBG 波长-位置 ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
for ax, L1, L2, nm in [(axes[0], L1f, L1r, "峰值 1"), (axes[1], L2f, L2r, "峰值 2")]:
    ax.plot(x_f, L1, "o", ms=4, mfc="white", mec="#1f4e79", label="正程")
    ax.plot(x_r, L2, "s", ms=3.6, mfc="#c00000", mec="#c00000", label="反程")
    k = S["L1_all"] if nm == "峰值 1" else S["L2_all"]
    ax.plot(xx, k["a"] * xx + k["b"], "-", color="0.55", lw=1.0)
    ax.set_xlabel("参考位置 $x$ / mm")
    ax.set_ylabel("$\\lambda$ / nm")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_title("%s：$%.2f$ pm/mm，$R^2=%.4f$" % (nm, k["a"] * 1000, k["r2"]), fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_fbg_position.pdf"))
plt.close(fig)

# ---------------- 图 3：回程差 ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
axes[0].bar(x_f, np.array(hyst["PE"]) / 1000, width=3.2, color="#1f4e79", alpha=0.85)
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("$\\Delta S_{PE}$ / $10^3$ count")
axes[0].set_xlabel("参考位置 $x$ / mm")
axes[1].bar(x_f - 0.9, hyst["L1"], width=1.7, color="#1f4e79", alpha=0.85, label="峰值 1")
axes[1].bar(x_f + 0.9, hyst["L2"], width=1.7, color="#c00000", alpha=0.85, label="峰值 2")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_ylabel("$\\Delta\\lambda$ / pm")
axes[1].set_xlabel("参考位置 $x$ / mm")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.22, lw=0.5, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_hysteresis.pdf"))
plt.close(fig)

# ---------------- 图 4：验证点 ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
lim = [18, 60]
for ax, tag, ylab in [(axes[0], "PE", "PE 反算位置 / mm"), (axes[1], "L2", "FBG 反算位置 / mm")]:
    xp = np.array(pred[tag]["x_pred"])
    ax.plot(lim, lim, "--", color="0.6", lw=1.0, label="$y=x$")
    ax.plot(x_v, xp, "o", ms=5, mfc="#c00000", mec="#c00000", label="验证点反算值")
    for xi, yi in zip(x_v, xp):
        ax.annotate("%+.2f" % (yi - xi), (xi, yi), textcoords="offset points",
                    xytext=(4, -10), fontsize=7.5, color="#333333")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("刻度尺参考位置 / mm")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_verification.pdf"))
plt.close(fig)

# ---------------- 图 5：CMCU 通道 ----------------
fig, ax = plt.subplots(figsize=(5.2, 2.9))
ax.plot(x_f, ADf / 1e6, "o", ms=4.5, mfc="white", mec="#1f4e79", label="正程")
ax.plot(x_r, ADr / 1e6, "s", ms=4.0, mfc="#c00000", mec="#c00000", label="反程")
ax.axhline(S["AD_mean"] / 1e6, color="0.55", ls="--", lw=1.0, label="均值")
ax.set_xlabel("参考位置 $x$ / mm")
ax.set_ylabel("CMCU 原始 AD 值 / $10^6$ count")
ax.legend(fontsize=8, loc="upper right", ncol=3)
ax.grid(alpha=0.25, lw=0.5)
fig.savefig(os.path.join(FIG, "fig5_cmcu_ad.pdf"))
plt.close(fig)

# ---------------- 图 6：温度原始曲线 ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
ax = axes[0]
ax.axvspan(T_a, T_b, color="#e8f0f8", zorder=0)
ax.plot(Tw, W1, "o-", ms=3, lw=0.8, mfc="white", mec="#1f4e79", color="#1f4e79", label="峰值 1")
ax.plot(Tw, W2, "s-", ms=3, lw=0.8, mfc="#c00000", mec="#c00000", color="#c00000", label="峰值 2")
ax.set_xlabel("PT100 温度 $T$ / $^\\circ$C")
ax.set_ylabel("$\\lambda$ / nm")
ax.set_title("升温过程（阴影为稳定跟踪段）", fontsize=8.5)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, lw=0.5)
ax = axes[1]
ax.plot(Tc, C1, "o-", ms=3.5, lw=0.9, mfc="white", mec="#1f4e79", color="#1f4e79", label="峰值 1")
ax.plot(Tc, C2, "s-", ms=3.2, lw=0.9, mfc="#c00000", mec="#c00000", color="#c00000", label="峰值 2")
ax.set_xlabel("PT100 温度 $T$ / $^\\circ$C")
ax.set_ylabel("$\\lambda$ / nm")
ax.set_title("降温过程（峰识别失锁）", fontsize=8.5)
ax.invert_xaxis()
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_temp_raw.pdf"))
plt.close(fig)

# ---------------- 图 7：温度拟合与残差 ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
ax.plot(Tw[seg], W2[seg], "s", ms=4, mfc="#c00000", mec="#c00000", label="峰值 2（参与拟合）")
ax.plot(np.linspace(T_a - 1, T_b + 1, 50),
        fit_w2["a"] * np.linspace(T_a - 1, T_b + 1, 50) + fit_w2["b"],
        "-", color="0.5", lw=1.2, label="线性拟合")
ax.set_xlabel("$T$ / $^\\circ$C"); ax.set_ylabel("$\\lambda_2$ / nm")
ax.ticklabel_format(axis="y", useOffset=False)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25, lw=0.5)
ax.set_title("$K_T=%.2f$ pm/$^\\circ$C，$R^2=%.4f$" % (fit_w2["a"] * 1000, fit_w2["r2"]), fontsize=8.5)
ax = axes[1]
ax.axhline(0, color="k", lw=0.6)
ax.plot(Tw[seg], np.array(fit_w2["res"]) * 1000, "o", ms=4, mfc="white", mec="#1f4e79")
ax.set_xlabel("$T$ / $^\\circ$C"); ax.set_ylabel("残差 / pm")
ax.grid(alpha=0.25, lw=0.5); ax.set_title("拟合残差", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig7_temp_fit.pdf"))
plt.close(fig)

# ---------------- 输出 ----------------
def fmt(v, n=4):
    return round(float(v), n)

print("=== 位移/应变 ===")
for k in ["PE_f", "PE_r", "PE_all", "L1_f", "L1_r", "L1_all", "L2_f", "L2_r", "L2_all", "AD_f", "AD_r", "AD_all"]:
    s = S[k]
    print("%-6s a=%.6g b=%.6g R2=%.6f rms=%.4g max=%.4g" % (k, s["a"], s["b"], s["r2"], s["rms"], s["maxres"]))
print("PE span", S["PE_span"], "count over 50mm")
print("L2 span %.1f pm" % S["L2_span_pm"], "L1 span %.1f pm" % S["L1_span_pm"],
      "spacing mean %.1f pm" % S["L2_spacing_mean"])
print("AD mean %.0f std %.0f (%.4f%%)" % (S["AD_mean"], S["AD_std"], S["AD_std"] / S["AD_mean"] * 100))
print("hyst PE max %.0f count (=%.3f mm), L2 max %.1f pm, L1 max %.1f pm" %
      (max(abs(np.array(hyst["PE"]))), max(abs(np.array(hyst["PE"]))) / S["PE_all"]["a"],
       max(abs(np.array(hyst["L2"]))), max(abs(np.array(hyst["L1"])))))
for tag in ["PE", "L1", "L2"]:
    print("verify", tag, np.round(pred[tag]["x_pred"], 2), "err", np.round(pred[tag]["err"], 2))
print("=== 温度 ===")
t = S["temp"]
print("N=%d rows, partial=%d, T %.2f..%.2f" % (t["n_rows"], t["n_partial"], t["T_min"], t["T_max"]))
print("seg %.2f..%.2f n=%d endpoint K=%.2f pm/C" % (t["T_a"], t["T_b"], t["n"], t["endpoint_sens"]))
for nm in ["fit2", "fit1", "fitdiff"]:
    s = t[nm]
    print("%-8s a=%.3f pm/C b=%.6f R2=%.4f rms=%.2f pm max=%.2f pm" %
          (nm, s["a"] * 1000, s["b"], s["r2"], s["rms"] * 1000, s["maxres"] * 1000))
print("jumps p1", t["jumps_p1"]); print("jumps p2", t["jumps_p2"])
print("cooling jumps p1", t["jumps_cool_p1"], "p2", t["jumps_cool_p2"])

json.dump(S, open(os.path.join(RES, "summary.json"), "w"), ensure_ascii=False, indent=1)
print("\nsaved figures ->", FIG)
