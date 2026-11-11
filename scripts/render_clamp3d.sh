#!/usr/bin/env bash
# 渲染报告图 1 的 (a) 面板：three.js 半剖三维模型 → PNG + 投影坐标 JSON
#
# 依赖：chromium（headless，需 --enable-unsafe-swiftshader 才能软渲染 WebGL）、
#       python3（临时起 http.server 供 ESM 模块加载）
# 用法：scripts/render_clamp3d.sh
#   环境变量：CHROME / PORT / PAGE_DIR / OUT_DIR
set -euo pipefail

BASE=$(cd "$(dirname "$0")/.." && pwd)                 # analysis/
PAGE_DIR=${PAGE_DIR:-$BASE/../report/3d}
OUT_DIR=${OUT_DIR:-$BASE/../report/figures}
CHROME=${CHROME:-$HOME/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome}
PORT=${PORT:-8731}

[[ -x $CHROME ]] || { echo "找不到 chromium：$CHROME" >&2; exit 1; }
mkdir -p "$OUT_DIR"
PNG="$OUT_DIR/clamp3d_render.png"
JSON="$OUT_DIR/clamp3d_render_proj.json"

cd "$PAGE_DIR"
python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
HTTP_PID=$!
trap 'kill $HTTP_PID 2>/dev/null || true' EXIT
sleep 1

COMMON=(--headless=new --no-sandbox --disable-dev-shm-usage
        --enable-unsafe-swiftshader --hide-scrollbars --virtual-time-budget=9000)

"$CHROME" "${COMMON[@]}" --window-size=1600,1200 \
    --screenshot="$PNG" "http://127.0.0.1:$PORT/clamp.html" 2>&1 \
    | grep -vi 'dbus\|GL Driver' || true

"$CHROME" "${COMMON[@]}" --dump-dom "http://127.0.0.1:$PORT/clamp.html" 2>/dev/null \
  | python3 -c '
import json, re, sys
html = sys.stdin.read()
m = re.search(r"<pre id=\"proj\">(.*?)</pre>", html, re.S)
if not m:
    sys.exit("渲染失败：页面未导出 proj 块（WebGL 未初始化？）")
d = json.loads(m.group(1))
json.dump(d, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("切点残差 A=%+.4f B=%+.4f（应为 0）" % (d["tangency"]["A"], d["tangency"]["B"]))
' "$JSON"

python3 - "$PNG" <<'PY'
import sys
import numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
m = a.mean(axis=2) < 248
if m.mean() < 0.02:
    sys.exit("渲染失败：非白像素仅 %.2f%%（WebGL 空画面）" % (m.mean() * 100))
ys, xs = np.nonzero(m)
print("OK %dx%d 非白 %.1f%% 内容 bbox x[%d,%d] y[%d,%d]"
      % (a.shape[1], a.shape[0], m.mean() * 100, xs.min(), xs.max(), ys.min(), ys.max()))
PY

echo "写出 $PNG / $JSON"
