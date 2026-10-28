# 基于原子力显微镜的光盘表面沟槽形貌测量

大连理工大学 物理创新实验报告（2026 秋）· 本仓库是**实验分支**，索引见 [`main`](https://github.com/xzx-001/dlut-physical-exp/tree/main) 分支。

用接触模式原子力显微镜测量三张光盘数据面的预刻沟槽，建立从仪器原始 `.csm` 到沟槽周期、峰谷深度与面粗糙度的完整离线处理链，并讨论横向标定与针尖卷积两类系统误差。

## 实验条件

| 项 | 值 |
|:--|:--|
| 仪器 | 本原 CSPM5500 扫描探针显微镜，Imager 4.7 后处理 |
| 模式 | 接触模式（恒力，Ref = 0.30） |
| 视场 / 采样 | 10 μm × 10 μm，1024 × 1024 点 |
| 扫描频率 | 1.0 Hz（快扫 x，慢扫 y） |
| 样品 | 光盘 3 张（cd1 / cd2 / cd3） |
| 通道 | 形貌、误差信号（探针起伏）、横向力，各 1 幅/张 |

## 主要结果

| 样品 | 沟槽周期 (nm) | 峰谷深度 (nm) | 面均方根高度 $S_q$ (nm) | 条纹倾角 (°) |
|:--|--:|--:|--:|--:|
| cd1 | 2170 ± 4 | 55.2 | 20.26 | +8.1 |
| cd2 | 2143 ± 7 | 60.4 | 19.94 | ≈ 0 |
| cd3 | 1090 ± 2 | 93.4 | 31.02 | −13.2 |

- 周期误差为 1024 行的行间四分位半宽（稳健量，对应的行间标准差分别为 5.5 / 17.1 / 2.0 nm）；深度与粗糙度均在二阶面整平后统计。
- 以 CD 标称道距 1600 nm 为参照反推的快扫轴横向标定因子 $k = 1.34$，贝叶斯联合拟合给出后验 1.34--1.43；据此 cd3 修正后周期 792 nm，与 DVD 道距 740 nm 相差 7.0%。
- 标定因子与道距在数据中不可分离，$k$ 的成立依赖「盘片道距等于标称值」这一前提，详见报告 §3.5。
- 条纹相对慢扫轴的倾角需先修正再沿 $y$ 平均，否则条纹在平均中相互抵消（cd3 平均剖面峰谷由 88.6 nm 降至 8.1 nm）。

## 目录结构

```
report/     XeLaTeX 源（afm-report.tex，720 行）与编译产物 afm-report.pdf（24 页）
  src/      早期分块草稿，已与正式稿不同步，仅供追溯
  figures/  正文插图（jpg / pdf）
figures/    同一批插图的高分辨率 png + pdf 版本
data/
  20261026/ 原始 .csm 扫描（9 个文件：3 张盘 × 3 通道）
scripts/    .csm 解析、处理链、诊断与拟合脚本
results/    复算输出（summary.json、各诊断 json、arrays.npz 中间量）
```

`.csm` 文件名规则：`<样品>-<通道>000.csm`，通道为 `形貌` / `探针起伏`（误差信号）/ `横向力`。

## 复现

依赖 Python 3 + `numpy` + `matplotlib`（无 scipy）；报告编译需 TeX Live（XeLaTeX + ctex + 中文字体）。

```bash
# 1. 主处理链：去扫描线 → 二阶面整平 → 粗糙度 / 剖面 / FFT → 出图
cd scripts
python3 analyze.py ../data/20261026 ../results

# 2. 横向标定的贝叶斯联合拟合
python3 obs_from_csm.py          # 从原始 .csm 重算观测向量
python3 calib_mcmc.py            # 输出 ../results/calib_mcmc.json

# 3. 诊断脚本（相位漂移分解、曲率核验、结构张量、倾角因子）
python3 diag_phase_drift.py
python3 diag_phase_factors.py
python3 diag_curvature_2d.py
python3 diag_orientation_tensor.py

# 4. 编译报告
cd ../report
latexmk -xelatex afm-report.tex
```

脚本用相对路径定位 `../data` 与 `../results`，克隆后无需改动即可运行（`plot.py` 可用环境变量 `AFM_DATA` / `AFM_FIG` 覆盖路径）。

## 说明

- 报告正文与数据均可复现；`results/arrays.npz` 是处理链的中间数组缓存（34 MB），删掉后重跑 `analyze.py` 会重新生成。
- 已做去标识化处理：报告封面不含姓名与学号。
- 原始 `.csm` 由仪器导出，未做任何改动；`.csm` 的逆向解析细节见 `scripts/csmio.py` 与报告附录 A。
