"""CSPM5500 (.csm) 图像解析器 — 本原 CSPM 系列 AFM 导出格式。

文件结构（实测，2026-10-26 CD-R 数据集）：
    [0:54]              BITMAPFILEHEADER + BITMAPINFOHEADER(40B)，24bpp，无压缩
    [54:54+W*H*3]       BGR 像素，行序 bottom-up（BMP 标准）
    [54+W*H*3:]         GBK 编码的文本参数块 key = value

像素编码：每像素 3 字节 = (B, G, R)，R 恒为 0；
          16 位原始量 = B + 256*G   （B 为低字节，G 为高字节）
"""
import struct
import numpy as np


class CSMImage:
    def __init__(self, path):
        self.path = str(path)
        raw = open(self.path, 'rb').read()
        self.raw_size = len(raw)
        if raw[:2] != b'BM':
            raise ValueError(f'{path}: not a BMP-wrapped CSM file')

        (self.bf_size,) = struct.unpack('<I', raw[2:6])
        (self.off_bits,) = struct.unpack('<I', raw[10:14])
        _, self.width, self.height, planes, self.bpp = struct.unpack('<IiiHH', raw[14:30])
        (self.compression,) = struct.unpack('<I', raw[30:34])
        (self.img_size,) = struct.unpack('<I', raw[34:38])
        if self.bpp != 24 or self.compression != 0:
            raise ValueError(f'{path}: unexpected BMP layout bpp={self.bpp} comp={self.compression}')

        pix = np.frombuffer(raw, dtype=np.uint8,
                            count=self.img_size, offset=self.off_bits)
        bgr = pix.reshape(self.height, self.width, 3)
        self.rgb_display = bgr[:, :, ::-1].copy()

        self.raw16 = (bgr[:, :, 0].astype(np.uint16)
                      + (bgr[:, :, 1].astype(np.uint32) << 8).astype(np.uint16))
        # BMP 行序为先存底行 → 翻正到常规"顶行在前"的数组
        self.raw16 = np.flipud(self.raw16)

        tail = raw[self.off_bits + self.img_size:]
        self.header_text = tail.decode('gbk', 'replace')
        self.meta = {}
        for line in self.header_text.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                self.meta[k.strip()] = v.strip()

    # ---------- 物理量换算 ----------
    @property
    def scan_size_nm(self):
        s = self.meta.get('ScanSize', '0nm')
        return float(s.replace('nm', '').replace('um', 'e3').strip())

    @property
    def px_size_nm(self):
        return self.scan_size_nm / self.width

    @property
    def z_scale_text(self):
        return self.meta.get('HeightScale', '')

    def physical(self):
        """把 16 位原始量按 HeightScale 映射为物理量（nm 或 V）。

        StartHeightScale ~ 0，HeightScale = 满量程；
        16 位上限 65535 对应 HeightScale。
        """
        hs = self.z_scale_text
        if hs.endswith('nm'):
            unit, full = 'nm', float(hs[:-2])
        elif hs.endswith('V'):
            unit, full = 'V', float(hs[:-1])
        else:
            unit, full = '', 1.0
        return self.raw16.astype(np.float64) / 65535.0 * full, unit

    def channel(self):
        return self.meta.get('Source', '?')


def load_dir(pattern='*.csm'):
    import glob
    return {p: CSMImage(p) for p in sorted(glob.glob(pattern))}


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        im = CSMImage(p)
        arr, unit = im.physical()
        print(f'{p}')
        print(f'  channel={im.channel()}  {im.width}x{im.height}  scan={im.scan_size_nm}nm'
              f'  px={im.px_size_nm:.3f}nm  z={im.z_scale_text}  unit={unit}')
        print(f'  raw16: min={im.raw16.min()} max={im.raw16.max()} mean={im.raw16.mean():.1f}')
        print(f'  phys : min={arr.min():.3f} max={arr.max():.3f} '
              f'mean={arr.mean():.3f} std={arr.std():.3f} {unit}')
        for k in ('Date', 'ScanSpeed', 'Ref', 'Signal Gain', 'Sensitivity', 'P.G.', 'I.G.', 'ZLimit'):
            print(f'    {k} = {im.meta.get(k)}')
