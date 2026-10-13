// Chapter 2 -> Word. Same look as chapter 1 (RTL, B Nazanin 13, headings, tables), plus Word equations.
const fs = require("fs");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, Footer, PageNumber, TabStopType,
  Math: OMath, MathRun, MathSquareBrackets, MathCurlyBrackets, MathFraction, MathRadical, MathSubScript, MathSuperScript, MathSubSuperScript, MathIntegral,
} = D;

const FA = "B Nazanin", EN = "Times New Roman";
const BODY = 26, SMALL = 22;
const CH = "۲";
const md = fs.readFileSync(__dirname + "/ch2.md", "utf8").split("\n");
const toFa = n => String(n).replace(/\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[d]);

// ---------------- tiny LaTeX subset -> OMML
const SYM = { "\\Delta": "Δ", "\\theta": "θ", "\\omega": "ω", "\\sigma": "σ", "\\odot": "⊙", "\\cdot": "·",
  "\\tanh": "tanh", "\\,": " ", "\\{": "{", "\\}": "}", "\\ldots": "…", "\\ddot{x}": "ẍ", "\\dot{x}": "ẋ",
  "\\tilde{h}": "h̃", "\\phi": "φ" };
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
  gyro: "\\Delta\\theta_k = \\int_{t_k}^{t_{k+1}}\\omega(t)\\,dt",
  split1: "\\Delta\\theta^{int}_n = (h_{LP} * \\Delta\\theta)_n",
  split2: "r_n = \\Delta\\theta_n - \\Delta\\theta^{int}_n",
  butter: "|H(j\\omega)|^2 = \\frac{1}{1 + (\\omega/\\omega_c)^{2N}}",
  filtfilt: "H_{ff}(j\\omega) = |H(j\\omega)|^2",
  spring: "\\ddot{x} = k(x_{target} - x) - c\\dot{x}",
  shake: "\\theta(t) = A_{max}\\cdot T^2 \\cdot P(f t)",
  res1: "H(x) = F(x) + x",
  res2: "y = F(x, \\{W_i\\}) + W_s x",
  gru_z: "z_t = \\sigma(W_z x_t + U_z h_{t-1} + b_z)",
  gru_r: "r_t = \\sigma(W_r x_t + U_r h_{t-1} + b_r)",
  gru_n: "\\tilde{h}_t = \\tanh(W_h x_t + U_h(r_t \\odot h_{t-1}) + b_h)",
  gru_h: "h_t = (1 - z_t)\\odot h_{t-1} + z_t \\odot \\tilde{h}_t",
  attn: "\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
  mha1: "\\mathrm{MultiHead}(Q, K, V) = \\mathrm{Concat}(head_1, \\ldots, head_h)W^O",
  mha2: "head_i = \\mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)",
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
  const widths = n === 3 ? [2800, 2600, 3626] : n === 5 ? [1900, 1750, 1600, 1750, 2026] : [2600, 6426];
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
  `جدول ${CH}-۱- اجزای حرکت دوربین روی دست`,
  `جدول ${CH}-۲- مقایسهٔ پژوهش‌های مرتبط با پژوهش حاضر`,
];

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
  if ((m = line.match(/^(#{1,3}) (.*)/))) {
    const lvl = m[1].length;
    inRefs = m[2].includes("منابع");
    children.push(new Paragraph({
      heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][lvl - 1],
      bidirectional: true, alignment: lvl === 1 ? AlignmentType.CENTER : AlignmentType.START,
      children: runs(m[2], { bold: true, size: [36, 30, 28][lvl - 1] }),
    }));
    continue;
  }
  if ((m = line.match(/^\d+\. (.*)/))) {
    if (!/^\d+\. /.test(md[i - 1] || "")) itemNo = 0;
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
Packer.toBuffer(doc).then(b => { fs.writeFileSync(__dirname + "/fasl2.docx", b); console.log("ok", b.length, "equations", eqNo, "tables", tableNo); });
