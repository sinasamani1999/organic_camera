"""Draw the two chapter-3 figures as HTML (Persian text shaped by Chromium) and screenshot them to PNG."""
import asyncio
from playwright.async_api import async_playwright

CSS = """
body{margin:0;background:#fff;font-family:'DejaVu Sans',sans-serif;}
.fig{position:relative;background:#fff;}
.box{position:absolute;box-sizing:border-box;border:1.6px solid #333;border-radius:8px;background:#f4f4f4;
     display:flex;align-items:center;justify-content:center;text-align:center;direction:rtl;
     font-size:15px;line-height:1.45;padding:6px 8px;color:#111}
.box>div{width:100%}
.box.k{background:#e8eef6}.box.u{background:#eef5ea}.box.o{background:#fff4e0}
.lane{position:absolute;border:1.4px dashed #888;border-radius:12px}
.lt{position:absolute;direction:rtl;font-size:15px;font-weight:bold;color:#333}
.ltr{direction:ltr;unicode-bidi:isolate;font-family:'DejaVu Sans Mono',monospace;font-size:13px}
svg{position:absolute;left:0;top:0}
"""

def page(w, h, boxes, arrows, extra=""):
    b = "".join(f'<div class="box {c}" style="left:{x}px;top:{y}px;width:{bw}px;height:{bh}px"><div>{t}</div></div>'
                for (x, y, bw, bh, t, c) in boxes)
    paths = "".join(f'<path d="{d}" fill="none" stroke="#333" stroke-width="1.8" marker-end="url(#a)"/>' for d in arrows)
    svg = (f'<svg width="{w}" height="{h}"><defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" '
           f'orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#333"/></marker></defs>{paths}</svg>')
    return f'<div class="fig" id="f" style="width:{w}px;height:{h}px">{extra}{svg}{b}</div>'

M = lambda s: f'<span class="ltr">{s}</span>'

# ---------------- Figure 3-1: overall system
W1, H1 = 1000, 560
bw, bh = 160, 78
xs = [810, 615, 420, 225, 30]
fig1_boxes = [
    (xs[0], 55, bw, bh, "ضبط ویدیو و ژیروسکوپ با گوشی همراه", ""),
    (xs[1], 55, bw, bh, "همگام‌سازی و محاسبهٔ چرخش هر فریم", ""),
    (xs[2], 55, bw, bh, "جدا کردن قصد و باقی‌مانده با فیلتر ۱ هرتز", ""),
    (xs[3], 55, bw, bh, "آموزش مدل " + M("CNN-GRU-Attention"), ""),
    (xs[4], 55, bw, bh, "وزن‌های مدل و پارامترهای نرمال‌سازی", "o"),
    (xs[0], 250, bw, bh, "آنریل: قصد بازیکن (ماوس) و سرعت حرکت", "u"),
    (xs[1], 250, bw, bh, "پیام " + M("/oc/state") + " پورت " + M("7001"), ""),
    (xs[2], 250, bw, bh, "پایتون: تجمیع در گام‌های ۳۰ هرتز و گرفتن تصویر بازی", "k"),
    (xs[3], 250, bw, bh, "پایتون: پیش‌بینی باقی‌مانده با مدل", "k"),
    (xs[4], 250, bw, bh, "پیام " + M("/oc/residual") + " پورت " + M("7000"), ""),
    (330, 410, 340, 72, "آنریل: جمع کردن باقی‌مانده (نشت ۵٪، سقف ۳ درجه) و اعمال روی دوربین", "u"),
]
fig1_arrows = [f"M{xs[i]},{55+bh/2} L{xs[i+1]+bw+2},{55+bh/2}" for i in range(4)]
fig1_arrows += [f"M{xs[i]},{250+bh/2} L{xs[i+1]+bw+2},{250+bh/2}" for i in range(4)]
fig1_arrows += [
    f"M{xs[4]+bw/2},{55+bh} L{xs[4]+bw/2},{190} L{xs[3]+bw/2},{190} L{xs[3]+bw/2},{248}",  # weights -> model
    f"M{xs[4]+bw/2},{250+bh} L{xs[4]+bw/2},{446} L{328},{446}",                            # residual -> apply
    f"M{672},{446} L{xs[0]+bw/2},{446} L{xs[0]+bw/2},{250+bh+2}",                          # apply -> next frame
]
fig1_extra = (
    '<div class="lane" style="left:12px;top:12px;width:976px;height:140px"></div>'
    '<div class="lt" style="right:24px;top:20px">مرحلهٔ آفلاین: آماده‌سازی داده و آموزش</div>'
    '<div class="lane" style="left:12px;top:208px;width:976px;height:300px"></div>'
    '<div class="lt" style="right:24px;top:216px">مرحلهٔ زنده: اجرا در کنار بازی (در هر فریم)</div>'
    '<div class="lt" style="left:900px;top:380px;font-weight:normal;font-size:13px">فریم بعد</div>'
)
FIG1 = page(W1, H1, fig1_boxes, fig1_arrows, fig1_extra)

# ---------------- Figure 3-2: model architecture
W2, H2 = 1000, 700
R, L_, C = 690, 40, 365   # column x (right: visual, left: motion, centre: fusion)
cw = 270
fig2_boxes = [
    (R, 20, cw, 70, "فریم بازی " + M("224×224×3"), "u"),
    (R, 120, cw, 70, M("ResNet-18") + " ازپیش‌آموزش‌دیده (وزن‌های ثابت)", ""),
    (R, 220, cw, 70, "نقشهٔ ویژگی " + M("7×7×512") + " ← ۴۹ نشانه", ""),
    (R, 320, cw, 80, "لایهٔ خطی " + M("512→128") + "، جاسازی مکانی و " + M("LayerNorm"), "k"),
    (L_, 20, cw, 70, "تاریخچهٔ حرکت " + M("10×5") + "<br>قصد، باقی‌مانده، سرعت", "u"),
    (L_, 120, cw, 70, M("GRU") + " با حالت پنهان ۶۴", "k"),
    (L_, 220, cw, 70, "لایهٔ خطی " + M("64→128") + " و " + M("LayerNorm"), "k"),
    (C, 430, cw, 90, "توجه متقاطع با ۴ سر<br>پرسش: حرکت / کلید و مقدار: ۴۹ نشانه<br>اتصال باقی‌مانده و " + M("LayerNorm"), "k"),
    (C, 550, cw, 50, "الحاق [زمینه، پرسش] ← ۲۵۶", ""),
    (C, 625, cw, 60, "پرسپترون دولایه " + M("256→128→2") + "<br>باقی‌ماندهٔ yaw و pitch در گام بعد", "o"),
]
cx = lambda x: x + cw / 2
fig2_arrows = [
    f"M{cx(R)},90 L{cx(R)},118", f"M{cx(R)},190 L{cx(R)},218", f"M{cx(R)},290 L{cx(R)},318",
    f"M{cx(R)},400 L{cx(R)},475 L{C+cw+2},475",                           # tokens -> attention (K,V)
    f"M{cx(L_)},90 L{cx(L_)},118", f"M{cx(L_)},190 L{cx(L_)},218",
    f"M{cx(L_)},290 L{cx(L_)},475 L{C-2},475",                             # query -> attention
    f"M{cx(L_)},475 L{cx(L_)},575 L{C-2},575",                             # query skip -> concat
    f"M{cx(C)},520 L{cx(C)},548", f"M{cx(C)},600 L{cx(C)},623",
]
fig2_extra = (
    '<div class="lt" style="left:{0}px;top:300px;font-weight:normal;font-size:13px;width:270px;text-align:center"></div>'
)
FIG2 = page(W2, H2, fig2_boxes, fig2_arrows)
LEGEND = ('<div style="direction:rtl;font-size:13px;margin:6px 0 0 0;color:#333">'
          '<span style="display:inline-block;width:14px;height:14px;background:#e8eef6;border:1px solid #333;vertical-align:middle"></span> لایه‌های آموزش‌پذیر &nbsp;&nbsp; '
          '<span style="display:inline-block;width:14px;height:14px;background:#f4f4f4;border:1px solid #333;vertical-align:middle"></span> بدون پارامتر آموزش‌پذیر</div>')


async def shot(html, out):
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = await b.new_page(device_scale_factor=2.5, viewport={"width": 1100, "height": 900})
        await pg.set_content(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>")
        await pg.locator("#f").screenshot(path=out)
        await b.close()

asyncio.run(shot(FIG1, "fig3_1.png"))
asyncio.run(shot(FIG2, "fig3_2.png"))
print("done")
