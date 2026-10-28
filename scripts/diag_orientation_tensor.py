"""独立核验：结构张量法测局部条纹方向（8x8 分块），与相位斜率法对照。

用于判断 cd1 的"条纹方向随 y 转动 11°"是否真实（而非相位法伪迹）。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csmio, afmproc

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '20261026')

def block_orientation(z, bs=128):
    """结构张量法：返回每块的条纹法线角（相对 x 轴，度）。"""
    ny, nx = z.shape
    gy, gx = np.gradient(z)
    res = []
    for iy in range(0, ny - bs + 1, bs):
        row = []
        for ix in range(0, nx - bs + 1, bs):
            Jxx = np.mean(gx[iy:iy+bs, ix:ix+bs] ** 2)
            Jyy = np.mean(gy[iy:iy+bs, ix:ix+bs] ** 2)
            Jxy = np.mean(gx[iy:iy+bs, ix:ix+bs] * gy[iy:iy+bs, ix:ix+bs])
            # 主特征向量 = 梯度主方向 = 条纹法线
            theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
            row.append(np.degrees(theta))
        res.append(row)
    return np.array(res)

for tag in ('cd1', 'cd2', 'cd3'):
    img = csmio.CSMImage(f'{DATA}/{tag}-形貌000.csm')
    z, _ = img.physical()
    a = afmproc.flatten_rows(z.astype(np.float64), 1)
    O = block_orientation(a)
    print(f'=== {tag} 结构张量分块方向（度，相对 x 轴；条纹法线）===')
    for r in O:
        print('  ' + '  '.join(f'{v:6.2f}' for v in r))
    colmean = O.mean(axis=1)
    print(f'  逐行块均值: ' + ' '.join(f'{v:.2f}' for v in colmean))
    print(f'  变化范围: {colmean.min():.2f} ~ {colmean.max():.2f} 度（Δ={colmean.max()-colmean.min():.2f}）')
