// Chapter 2 -> Word. Same look as chapter 1 (RTL, B Nazanin 13, headings, tables), plus Word equations.
const fs = require("fs");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, Footer, PageNumber, TabStopType,
  ImageRun, Math: OMath, MathRun, MathSquareBrackets, MathCurlyBrackets, MathSum, MathFraction, MathRadical, MathSubScript, MathSuperScript, MathSubSuperScript, MathIntegral,
} = D;

const FA = "B Nazanin", EN = "Times New Roman";
const BODY = 26, SMALL = 22;
const CH = "۳";
const md = fs.readFileSync(__dirname + "/ch3.md", "utf8").split("\n");
const toFa = n => String(n).replace(/\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[d]);

// ---------------- tiny LaTeX subset -> OMML
const SYM = { "\\Delta": "Δ", "\\theta": "θ", "\\omega": "ω", "\\sigma": "σ", "\\odot": "⊙", "\\cdot": "·",
  "\\tanh": "tanh", "\\,": " ", "\\{": "{", "\\}": "}", "\\ldots": "…", "\\ddot{x}": "ẍ", "\\dot{x}": "ẋ",
  "\\tilde{h}": "h̃", "\\phi": "φ", "\\tilde{m}": "m̃", "\\hat{\\mathbf{y}}": "ŷ", "\\hat{y}": "ŷ",
  "\\delta": "δ", "\\rho": "ρ", "\\lambda": "λ", "\\tau": "τ", "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\mu": "μ",
  "\\quad": "   ", "\\in": "∈", "\\|": "‖", "\\cos": "cos", "\\arg": "arg ", "\\max": "max", "\\times": "×" };
function parse(src) {
  let i = 0;
  function group() { // after '{'
    let depth = 1, j = i;
    while (depth) { if (src[j] === "{") depth++; else if (src[j] === "}") depth--; j++; }
    const inner = src.slice(i, j - 1); i = j; return inner;
  }
  function atom() { // one item for sub/sup
    if (src[i] === "{") { i++; return parse(group()); }
    if (src[i] === "\\") { const m = src.slice(i).match(/^\\[a-zA-Z]+/); i += m[0].length; return [new MathRun(SYM[m[0]] || m[0].slice(1))]; }
    return [new MathRun(src[i++])];
  }
  const out = [];
  let buf = "";
  const flush = () => { if (buf) { out.push(new MathRun(buf)); buf = ""; } };
  const scripts = base => {
    let sub = null, sup = null;
    while (src[i] === "_" || src[i] === "^") { const c = src[i++]; const a = atom(); if (c === "_") sub = a; else sup = a; }
    if (sub && sup) return new MathSubSuperScript({ children: base, subScript: sub, superScript: sup });
    if (sub) return new MathSubScript({ children: base, subScript: sub });
    if (sup) return new MathSuperScript({ children: base, superScript: sup });
    return null;
  };
  while (i < src.length) {
    const c = src[i];
    if (c === "|" || src.startsWith("\\|", i)) {
      const tok = c === "|" ? "|" : "\\|";
      const j = src.indexOf(tok, i + tok.length);
      const inner = parse(src.slice(i + tok.length, j)); i = j + tok.length; flush();
      const base = [tok === "|" ? new MathSquareBrackets({ children: inner }) : new MathCurlyBrackets({ children: inner })];
      const s2 = scripts(base); out.push(s2 || base[0]); continue;
    }
    if (c === "\\") {
      const long = Object.keys(SYM).filter(k => k.includes("{")).find(k => src.startsWith(k, i));
      let base;
      if (long) { i += long.length; base = [new MathRun(SYM[long])]; }
      else {
        const m = src.slice(i).match(/^\\([a-zA-Z]+|.)/); i += m[0].length; const cmd = m[1];
        if (cmd === "frac") { i++; const n = parse(group()); i++; const d = parse(group()); flush(); out.push(new MathFraction({ numerator: n, denominator: d })); continue; }
        if (cmd === "sqrt") { i++; const a = parse(group()); flush(); out.push(new MathRadical({ children: a })); continue; }
        if (cmd === "int") {
          let sub = [], sup = [];
          while (src[i] === "_" || src[i] === "^") { const k = src[i++]; const a = atom(); if (k === "_") sub = a; else sup = a; }
          flush(); out.push(new MathIntegral({ children: parse(src.slice(i)), subScript: sub, superScript: sup })); i = src.length; continue;
        }
        if (cmd === "mathrm") { i++; buf += group(); continue; }
        if (cmd === "sum") {
          let sub = [], sup = [];
          while (src[i] === "_" || src[i] === "^") { const k = src[i++]; const a = atom(); if (k === "_") sub = a; else sup = a; }
          flush(); out.push(new MathSum({ children: parse(src.slice(i)), subScript: sub, superScript: sup })); i = src.length; continue;
        }
        if (cmd === "mathbf" || cmd === "boldsymbol") {
          i++; const inner = parse(group()); flush();
          const s2 = scripts(inner); if (s2) out.push(s2); else out.push(...inner); continue;
        }
        base = [new MathRun(SYM["\\" + cmd] || cmd)];
      }
      flush();
      const s = scripts(base);
      out.push(s || base[0]);
      continue;
    }
    if (c === "_" || c === "^") { // scripts on the last character in buf
      const last = buf.slice(-1); buf = buf.slice(0, -1); flush();
      out.push(scripts([new MathRun(last)])); continue;
    }
    buf += c; i++;
  }
  flush();
  return out;
}
const mathInline = tex => new OMath({ children: parse(tex) });

const EQ = {
  delta: "\\delta^{*} = \\arg\\max_{\\delta} [\\rho(\\Delta\\theta^{yaw}(\\delta), f_x) + \\rho(\\Delta\\theta^{pitch}(\\delta), -f_y)]",
  rates: "\\omega_{yaw} = -\\boldsymbol{\\omega}\\cdot\\mathbf{u},\\quad \\omega_{pitch} = -\\omega_y",
  mvec: "\\mathbf{m}_t = [i^{yaw}_t, i^{pitch}_t, r^{yaw}_t, r^{pitch}_t, v_t]",
  sample: "X_t = (I_t, I_{t+1}, \\mathbf{M}_{t-T:t-1}, \\mathbf{M}_{t-T+1:t}),\\quad Y_t = (\\mathbf{r}_t, \\mathbf{r}_{t+1})",
  norm: "\\tilde{m} = \\frac{m - \\mu_{train}}{\\sigma_{train}}",
  ctx: "\\mathbf{c} = \\mathrm{LayerNorm}(\\mathbf{q} + \\mathrm{MultiHead}(\\mathbf{q}, K, V))",
  ltot: "L = \\alpha L_{pos} + \\beta L_{dir} + \\gamma L_{smooth}",
  lpos: "L_{pos} = \\frac{1}{4B}\\sum_{i=1}^{B}\\sum_{k=1}^{2}\\|\\hat{\\mathbf{y}}_{i,k} - \\mathbf{y}_{i,k}\\|^2",
  ldir: "L_{dir} = \\frac{1}{|S|}\\sum_{(i,k)\\in S}(1 - \\cos(\\hat{\\mathbf{y}}_{i,k}, \\mathbf{y}_{i,k}))",
  lsmooth: "L_{smooth} = \\frac{1}{2B}\\sum_{i=1}^{B}\\|(\\hat{\\mathbf{y}}_{i,2} - \\hat{\\mathbf{y}}_{i,1}) - (\\mathbf{y}_{i,2} - \\mathbf{y}_{i,1})\\|^2",
  perlin: "O^{yaw}(t) = A\\cdot P(f t),\\quad O^{pitch}(t) = A\\cdot P(f t + 57.3)",
  leak: "\\mathbf{O}_n = \\mathrm{clip}(e^{-\\lambda \\Delta t_n}(\\mathbf{O}_{n-1} + \\Delta\\mathbf{r}_n), -O_{max}, O_{max})",
  lat: "T_{total} = T_{wait} + T_{capture} + T_{pre} + T_{model} + T_{send}",
  rms: "A'_c = A_c\\cdot\\frac{\\mathrm{RMS}(O_{model})}{\\mathrm{RMS}(O_c)}",
};
let eqNo = 0;
function equation(key) {
  eqNo++;
  // LTR paragraph: equation centred, number "(۲-n)" at the left margin (right-to-left thesis convention)
  return new Paragraph({
    alignment: AlignmentType.LEFT, spacing: { before: 120, after: 160 },
    tabStops: [{ type: TabStopType.CENTER, position: 4513 }],
    children: [
      new TextRun({ text: `(${CH}‑${toFa(eqNo)})`, size: BODY, font: { ascii: EN, hAnsi: EN, cs: FA } }),
      new TextRun({ text: "\t" }),
      new OMath({ children: parse(EQ[key]) }),
    ],
  });
}

// ---------------- inline text: **bold**, *italic*, $math$
function runs(text, opts = {}) {
  if (!opts.ltr) text = text.replace(/([۰-۹])-(?=[۰-۹\s])/g, "$1‏-‏");
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\$[^$]+\$)/g;
  let last = 0, m;
  const push = (t, b, it) => t && out.push(new TextRun({
    text: t, bold: b || opts.bold, boldComplexScript: b || opts.bold, italics: it, italicsComplexScript: it,
    rightToLeft: !opts.ltr, size: opts.size || BODY, sizeComplexScript: opts.size || BODY,
    font: opts.ltr ? { ascii: EN, hAnsi: EN, cs: EN } : { ascii: EN, hAnsi: EN, cs: FA },
  }));
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const s = m[0];
    if (s.startsWith("$")) out.push(mathInline(s.slice(1, -1)));
    else if (s.startsWith("**")) push(s.slice(2, -2), true, false);
    else push(s.slice(1, -1), false, true);
    last = m.index + s.length;
  }
  push(text.slice(last));
  return out;
}

const para = (text, extra = {}) => new Paragraph({
  bidirectional: !extra.ltr, alignment: extra.ltr ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 360 }, children: runs(text, extra),
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
const TABLE_W = 9026;
let tableNo = 0;
function table(rows) {
  const cells = rows.map(r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const header = cells[0], body = cells.slice(2);
  const n = header.length;
  const widths = TABLE_WIDTHS[tableNo - 1] || (n === 3 ? [2800, 2600, 3626] : n === 4 ? [2256, 2256, 2257, 2257] : [2600, 6426]);
  const mk = (row, isHead) => new TableRow({
    tableHeader: isHead, cantSplit: true,
    children: row.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 60, bottom: 60, left: 90, right: 90 },
      shading: isHead ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
      children: c.split("؛ ").map(part => {
        const latin = !/[\u0600-\u06FF]/.test(part);
        const size = n === 5 ? SMALL - 2 : SMALL + 2;
        return new Paragraph({
          bidirectional: !latin, alignment: isHead ? AlignmentType.CENTER : (latin ? AlignmentType.RIGHT : AlignmentType.START),
          spacing: { after: 0, line: 280 }, children: runs(part, { bold: isHead, size, ltr: latin }),
        });
      }),
    })),
  });
  return new Table({
    visuallyRightToLeft: true, width: { size: TABLE_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(header, true), ...body.map(r => mk(r, false))],
  });
}
const caption = t => new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, keepNext: true,
  children: runs(t, { bold: true, size: SMALL + 2 }) });

// table captions (placed above tables, thesis convention)
const TABLE_CAPTIONS = [
  `جدول ${CH}-۱- تنظیمات ضبط در برنامهٔ OpenCamera Sensors`,
  `جدول ${CH}-۲- سناریوهای ضبط داده`,
  `جدول ${CH}-۳- ابعاد و تعداد پارامترهای بخش‌های مدل`,
  `جدول ${CH}-۴- تنظیمات آموزش مدل`,
  `جدول ${CH}-۵- پیام‌های OSC بین آنریل و پایتون`,
  `جدول ${CH}-۶- معیارهای ارزیابی عینی`,
  `جدول ${CH}-۷- شرط‌های آزمون کاربری`,
];
const TABLE_WIDTHS = [[2400, 2800, 3826], [2300, 3500, 3226], [3000, 2300, 1900, 1826], [3500, 5526],
  [1500, 2300, 3526, 1700], [2400, 3600, 3026], [2400, 6626]];

const children = [];
let tableBuf = [], inRefs = false, itemNo = 0;
const flushTable = () => {
  if (!tableBuf.length) return;
  children.push(caption(TABLE_CAPTIONS[tableNo++]));
  children.push(table(tableBuf)); children.push(para("")); tableBuf = [];
};

for (let i = 0; i < md.length; i++) {
  const line = md[i];
  if (line.startsWith("|")) { tableBuf.push(line); continue; }
  flushTable();
  if (!line.trim()) continue;
  let m;
  if ((m = line.match(/^\[\[EQ:(\w+)\]\]$/))) { children.push(equation(m[1])); continue; }
  if ((m = line.match(/^\[\[FIG:([^|]+)\|(.*)\]\]$/))) {
    const buf = fs.readFileSync(__dirname + "/" + m[1]);
    const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20), W = 590;
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: W, height: Math.round(W * h / w) },
        altText: { title: m[2], description: m[2], name: m[1] } })] }));
    children.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: runs(m[2], { bold: true, size: SMALL + 2 }) }));
    itemNo = 0; continue;
  }
  if ((m = line.match(/^(#{1,3}) (.*)/))) {
    const lvl = m[1].length;
    inRefs = m[2].includes("منابع"); itemNo = 0;
    children.push(new Paragraph({
      heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][lvl - 1],
      bidirectional: true, alignment: lvl === 1 ? AlignmentType.CENTER : AlignmentType.START,
      children: runs(m[2], { bold: true, size: [36, 30, 28][lvl - 1] }),
    }));
    continue;
  }
  if ((m = line.match(/^\d+\. (.*)/))) {
    children.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 360 },
      indent: { right: 425, hanging: 283 },
      children: [new TextRun({ text: toFa(++itemNo) + ".\t", rightToLeft: true, bold: true, boldComplexScript: true, size: BODY, sizeComplexScript: BODY, font: { cs: FA } }), ...runs(m[1])],
    }));
    continue;
  }
  if ((m = line.match(/^- (.*)/))) {
    if (inRefs) {
      children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 100, line: 276 },
        indent: { left: 567, hanging: 567 }, children: runs(m[1], { ltr: true, size: SMALL }) }));
    } else {
      children.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 360 },
        indent: { right: 425, hanging: 283 }, children: [new TextRun({ text: "•\t", rightToLeft: true, size: BODY }), ...runs(m[1])] }));
    }
    continue;
  }
  itemNo = 0;
  children.push(para(line));
}
flushTable();

const doc = new Document({
  styles: {
    default: { document: { run: { font: { ascii: EN, hAnsi: EN, cs: FA }, size: BODY, sizeComplexScript: BODY } } },
    paragraphStyles: [1, 2, 3].map(l => ({
      id: `Heading${l}`, name: `Heading ${l}`, basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { bold: true, boldComplexScript: true, color: "000000" },
      paragraph: { spacing: { before: l === 1 ? 0 : 280, after: l === 1 ? 360 : 140 }, outlineLevel: l - 1, keepNext: true },
    })),
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: SMALL })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(__dirname + "/fasl3.docx", b); console.log("ok", b.length, "equations", eqNo, "tables", tableNo); });
