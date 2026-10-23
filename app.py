# -*- coding: utf-8 -*-
"""
LDO/DCDC 反馈分压漂移工具 —— Web 服务版
==========================================
监听 0.0.0.0:80（HyperAI "部署"要求服务在容器内绑 0.0.0.0:80，否则被回收）。
纯 Python 标准库 http.server，不依赖 Flask，PyTorch CPU 镜像开箱即用。

行为：
  - 打开首页 -> 用当前(或默认)参数跑一遍完整分析，把文字报告 + 两张图渲染成网页
  - 页面顶部有参数表单，改完点"重新计算" -> 带 query 参数回来，实时重算
  - /healthz 返回 200，供探活

启动：
  python app.py                 # 默认端口 80
  PORT=8080 python app.py       # 本机测试用 8080（80 在 Windows 常需管理员）
"""
import os
import io
import sys
import base64
import contextlib
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import ldo_divider_drift as L

PORT = int(os.environ.get("PORT", "80"))
HOST = "0.0.0.0"  # 必须绑 0.0.0.0，不能 127.0.0.1

# query 参数名 -> (cfg 键, 类型, 换算)
NUM_FIELDS = {
    "Vref":  ("Vref_nom",   float, 1.0),
    "Vout":  ("Vout_target", float, 1.0),
    "R2k":   ("R2_nom",     float, 1e3),   # 输入 kΩ
    "Tmin":  ("Tmin",       float, 1.0),
    "Tmax":  ("Tmax",       float, 1.0),
    "tol":   ("tol1",       float, 0.01),   # 输入 % -> 小数，上下电阻同步
    "tcr":   ("tcr1_ppm",   float, 1.0),    # ppm/°C，上下同步
    "vrefTC": ("vref_tc_ppm", float, 1.0),
    "aging": ("aging",      float, 0.01),   # % -> 小数
    "iadj":  ("iadj_A",     float, 1e-9),   # nA -> A
    "spec":  ("spec_pct",   float, 1.0),    # ±%
}


def build_cfg(q):
    cfg = L.make_default_cfg()
    for name, (key, cast, scale) in NUM_FIELDS.items():
        raw = q.get(name, [None])[0]
        if raw is None or str(raw).strip() == "":
            continue
        try:
            cfg[key] = cast(raw) * scale
        except ValueError:
            pass
    # tol / tcr 上下电阻同步
    cfg["tol2"] = cfg["tol1"]
    cfg["tcr2_ppm"] = cfg["tcr1_ppm"]
    L.finalize_cfg(cfg)
    return cfg


def run_analysis(cfg):
    """跑完整分析，捕获 stdout 成文字报告，并读取生成的 PNG(base64)。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.report(cfg)

        wc = L.worst_case_corner(cfg)
        print("\n" + "=" * 78)
        print("[A. 最坏情况漂移分析]")
        print("=" * 78)
        print(f"  Corner 极值法:  Vout ∈ [{wc['Vmin']:.4f}, {wc['Vmax']:.4f}] V")
        print(f"    → 漂移范围 [{wc['drift_pct_min']:+.3f}%, {wc['drift_pct_max']:+.3f}%]  "
              f"最坏 |漂移| = {wc['worst_abs_drift_pct']:.3f}%")
        rss = L.rss_estimate(cfg)
        print(f"  RSS 统计估计:   ±{rss['rss_pct']:.3f}%")
        verdict = "[PASS] 满足" if wc["worst_abs_drift_pct"] <= cfg["spec_pct"] else "[FAIL] 不满足"
        print(f"  对照目标 ±{cfg['spec_pct']:.2f}%:  {verdict}")

        L.plot_results(cfg)
        L.recommend(cfg)
        cfg_loose = dict(cfg); cfg_loose["spec_pct"] = max(cfg["spec_pct"], 2.0)
        L.recommend(cfg_loose)
        L.build_ml_demo(cfg)

    text = buf.getvalue()
    imgs = {}
    for name in ("drift_analysis.png", "feature_importance.png"):
        p = L.out_path(name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                imgs[name] = base64.b64encode(f.read()).decode()
    return text, imgs


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_page(cfg, text, imgs, q):
    def val(name, dflt):
        return esc(str(q.get(name, [dflt])[0]))
    form = f"""
    <form method="get" action="/" class="grid">
      <label>Vref (V)<input name="Vref" value="{val('Vref', cfg['Vref_nom'])}"></label>
      <label>Vout 目标 (V)<input name="Vout" value="{val('Vout', cfg['Vout_target'])}"></label>
      <label>R2 下电阻 (kΩ)<input name="R2k" value="{val('R2k', cfg['R2_nom']/1e3)}"></label>
      <label>Tmin (°C)<input name="Tmin" value="{val('Tmin', cfg['Tmin'])}"></label>
      <label>Tmax (°C)<input name="Tmax" value="{val('Tmax', cfg['Tmax'])}"></label>
      <label>电阻容差 (±%)<input name="tol" value="{val('tol', cfg['tol1']*100)}"></label>
      <label>电阻 TCR (ppm/°C)<input name="tcr" value="{val('tcr', cfg['tcr1_ppm'])}"></label>
      <label>Vref 温漂 (ppm/°C)<input name="vrefTC" value="{val('vrefTC', cfg['vref_tc_ppm'])}"></label>
      <label>老化 (±%)<input name="aging" value="{val('aging', cfg['aging']*100)}"></label>
      <label>I_adj (nA)<input name="iadj" value="{val('iadj', cfg['iadj_A']*1e9)}"></label>
      <label>精度目标 (±%)<input name="spec" value="{val('spec', cfg['spec_pct'])}"></label>
      <div class="row"><button type="submit">重新计算</button>
      <a href="/" class="link">恢复默认</a></div>
    </form>"""

    img_html = ""
    for name in ("drift_analysis.png", "feature_importance.png"):
        if name in imgs:
            img_html += f'<div class="fig"><h3>{name}</h3><img src="data:image/png;base64,{imgs[name]}"></div>'

    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LDO/DCDC 反馈分压 · 输出漂移工具</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,"Helvetica Neue",Arial,sans-serif;
      margin:0;background:#f5f6f8;color:#1a202c}}
 header{{background:#2b6cb0;color:#fff;padding:18px 24px}}
 header h1{{margin:0;font-size:20px}} header p{{margin:4px 0 0;opacity:.85;font-size:13px}}
 .wrap{{max-width:1080px;margin:0 auto;padding:20px}}
 .card{{background:#fff;border-radius:10px;padding:18px 20px;margin:16px 0;
        box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}}
 label{{font-size:12px;color:#4a5568;display:flex;flex-direction:column;gap:4px}}
 input{{padding:7px 9px;border:1px solid #cbd5e0;border-radius:6px;font-size:14px}}
 .row{{display:flex;align-items:center;gap:14px;grid-column:1/-1}}
 button{{background:#2b6cb0;color:#fff;border:0;padding:9px 18px;border-radius:6px;
         font-size:14px;cursor:pointer}}
 .link{{color:#2b6cb0;font-size:13px}}
 pre{{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;overflow:auto;
      font-size:12.5px;line-height:1.5}}
 .fig img{{max-width:100%;border:1px solid #e2e8f0;border-radius:8px}}
 .fig h3{{font-size:13px;color:#718096;margin:6px 0}}
</style></head><body>
<header><h1>LDO/DCDC 反馈分压网络 · 输出电压漂移工具</h1>
<p>Vout = Vref·(1+R1/R2) + I_adj·R1 ｜ 容差 + TCR + Vref温漂/老化 + I_adj ｜ Corner · 蒙特卡洛 · 反推推荐 · sklearn</p></header>
<div class="wrap">
 <div class="card"><h2 style="margin-top:0;font-size:16px">参数</h2>{form}</div>
 <div class="card"><h2 style="margin-top:0;font-size:16px">分析报告</h2><pre>{esc(text)}</pre></div>
 <div class="card">{img_html}</div>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/healthz", "/health"):
            self._send(200, "ok", "text/plain")
            return
        if parsed.path not in ("/", "/index.html"):
            self._send(404, "not found", "text/plain")
            return
        q = parse_qs(parsed.query)
        try:
            cfg = build_cfg(q)
            text, imgs = run_analysis(cfg)
            self._send(200, render_page(cfg, text, imgs, q))
        except Exception:
            self._send(500, "<pre>error:\n" + esc(traceback.format_exc()) + "</pre>")


def main():
    print(f"LDO drift web tool listening on {HOST}:{PORT}")
    sys.stdout.flush()
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
