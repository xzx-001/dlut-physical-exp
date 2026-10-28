# 大连理工大学 · 物理创新实验报告（DLUT Physical Experiment Reports）

本仓库按**实验**组织：`main` 分支只作为索引，**每个实验的内容放在各自的分支里**。点击下表的分支名即可跳到该实验的目录。

## 实验列表

| # | 实验名称 | 分支（点击进入） | 内容 |
|:--|:--|:--|:--|
| 1 | 基于原子力显微镜的光盘表面沟槽形貌测量 | [`afm-cd-groove-morphology`](https://github.com/xzx-001/dlut-physical-exp/tree/afm-cd-groove-morphology) | `report/`（XeLaTeX 源 + 24 页 PDF）· `data/`（三张光盘 .csm 原始扫描）· `scripts/`（解析与处理链）· `figures/` · `results/` |

## 使用方式

```bash
# 克隆某个实验（--branch 指定分支）
git clone -b afm-cd-groove-morphology https://github.com/xzx-001/dlut-physical-exp.git

# 或先克隆索引，再切换到实验分支
git clone https://github.com/xzx-001/dlut-physical-exp.git
cd dlut-physical-exp
git branch -r          # 查看全部分支
git switch afm-cd-groove-morphology
```

## 分支命名约定

分支名取实验的英文短名，便于在 URL 与命令行中引用：

| 分支 | 实验 |
|:--|:--|
| `afm-cd-groove-morphology` | AFM 光盘表面沟槽形貌测量 |

## 说明

- 各实验分支彼此独立、历史互不干扰；`main` 不含实验数据，只维护本索引。
- 报告已做去标识化处理（不含姓名、学号等个人信息）；`.csm` 数据为仪器原始导出，未做改动。
- 每个实验的数据都可由该分支 `scripts/` 下的脚本复现，编译与复现步骤见对应 `report/`。
