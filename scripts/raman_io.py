"""拉曼光谱文件读取与通用处理原语（PECVD 石墨烯实验）。

数据格式：两列 ASCII，第一列拉曼位移 (cm^-1，降序)，第二列强度 (counts)。
文件由 LabSpec/仪器导出，行尾为 CRLF，首行可能带 UTF-8 BOM。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "raw"
FIG = REPO / "figures"
RES = REPO / "results"


def load_spectrum(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """读取两列光谱，按拉曼位移升序返回 (x, y)。"""
    arr = np.loadtxt(path, encoding="utf-8-sig")
    x, y = arr[:, 0].astype(float), arr[:, 1].astype(float)
    order = np.argsort(x)
    return x[order], y[order]


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 niter: int = 12) -> np.ndarray:
    """Eilers 非对称最小二乘基线。

    lam 控制平滑度，p 控制基线落在峰下方的程度（p 越小基线越低）。
    """
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve(W + D, w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def lorentz(x: np.ndarray, x0: float, fwhm: float, amp: float) -> np.ndarray:
    h = fwhm / 2.0
    return amp * h * h / ((x - x0) ** 2 + h * h)


def gauss(x: np.ndarray, x0: float, fwhm: float, amp: float) -> np.ndarray:
    s = fwhm / 2.3548200450309493
    return amp * np.exp(-0.5 * ((x - x0) / s) ** 2)


def fwhm_area(fwhm: float, amp: float, shape: str = "lorentz") -> float:
    """峰面积，用于比值的面积口径。"""
    if shape == "lorentz":
        return np.pi * amp * fwhm / 2.0
    return amp * fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0)))
