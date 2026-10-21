// Ethics request letter and user-study forms (Persian, RTL, B Nazanin).
// node build_study.js  ->  ethics_letter.docx, study_forms.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Table, TableRow, TableCell, WidthType,
  BorderStyle, ShadingType, PageBreak, VerticalAlign,
} = require("docx");

const FA = "B Nazanin", EN = "Times New Roman";
const BODY = 24, SMALL = 21, W = 9026;
const T = JSON.parse(fs.readFileSync(__dirname + "/text.json", "utf8"));
const faNum = n => String(n).replace(/\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[d]);

function runs(text, o = {}) {
  const out = [];
  const re = /\*\*([^*]+)\*\*/g;
  let last = 0, m;
  const push = (t, b) => t && out.push(new TextRun({
    text: t, bold: b || o.bold, boldComplexScript: b || o.bold, rightToLeft: true,
    size: o.size || BODY, sizeComplexScript: o.size || BODY, font: { ascii: EN, hAnsi: EN, cs: FA },
  }));
  while ((m = re.exec(text))) { push(text.slice(last, m.index)); push(m[1], true); last = m.index + m[0].length; }
  push(text.slice(last));
  return out;
}
const P = (text, o = {}) => new Paragraph({
  bidirectional: true, alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { before: o.before || 0, after: o.after ?? 80, line: o.line || 300 },
  indent: o.bullet ? { start: 360, hanging: 240 } : undefined,
  children: runs(o.bullet ? "• " + text : text, o),
});
const H = (text, o = {}) => P(text, { bold: true, size: o.size || 30, align: o.align || AlignmentType.CENTER, after: 160, before: o.before || 0 });
const H2 = text => P(text, { bold: true, size: 27, align: AlignmentType.START, before: 160, after: 80 });
const BR = () => new Paragraph({ children: [new PageBreak()] });

const line = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: line, bottom: line, left: line, right: line };
function table(rows, widths, o = {}) {
  const mk = (row, ri) => new TableRow({
    tableHeader: ri === 0 && o.header !== false,
    height: o.rowHeight ? { value: o.rowHeight, rule: "atLeast" } : undefined,
    children: row.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
      margins: { top: 50, bottom: 50, left: 80, right: 80 },
      shading: ri === 0 && o.header !== false ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
      children: [new Paragraph({
        bidirectional: true, spacing: { after: 0, line: 280 },
        alignment: (o.center || []).includes(i) || (ri === 0 && o.header !== false) ? AlignmentType.CENTER : AlignmentType.START,
        children: runs(String(c), { size: o.size || SMALL, bold: ri === 0 && o.header !== false }),
      })],
    })),
  });
  return new Table({ visuallyRightToLeft: true, width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: rows.map(mk) });
}
const sign = labels => table([labels, labels.map(() => "\n\n")], labels.map(() => Math.floor(W / labels.length)), { rowHeight: 700 });
const field = items => P(items.map(s => `${s}: ........................`).join("     "), { after: 140 });

function doc(children) {
  return new Document({
    styles: { default: { document: { run: { font: { ascii: EN, hAnsi: EN, cs: FA }, size: BODY } } } },
    sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 850, bottom: 850, left: 1247, right: 1247 } } }, children }],
  });
}

// ------------------------------------------------------------------ 1. ethics letter
const L = T.letter;
const letter = [
  P(L.date, { align: AlignmentType.END, after: 0 }), P(L.attach, { align: AlignmentType.END, after: 240 }),
  P(L.to, { bold: true, after: 60 }), P(L.salam, { after: 160 }),
  ...L.body.map(t => P(t)),
  H2(L.summaryTitle), ...L.summary.map(t => P(t, { bullet: true })),
  H2(L.measuresTitle), ...L.measures.map(t => P(t, { bullet: true })),
  H2(L.attachTitle), ...L.attachments.map((t, i) => P(`${faNum(i + 1)}. ${t}`, { after: 40 })),
  P(L.closing, { before: 80, after: 120 }),
  sign(L.signs),
];

// ------------------------------------------------------------------ 2. study forms
const F = T.forms, out = [];
const consent = (C) => {
  out.push(H(C.title), P(C.sub, { align: AlignmentType.CENTER, size: SMALL, after: 200 }));
  C.sections.forEach(s => { out.push(H2(s.h)); s.p.forEach(t => out.push(P(t, { bullet: s.bullet }))); });
  out.push(H2(C.agreeTitle));
  C.agree.forEach(t => out.push(P("☐  " + t, { after: 60 })));
  out.push(P("", { after: 120 }), sign(C.signs));
};

// A. consent for the watching study
consent(F.consentStudy);
out.push(BR());
// B. consent for the phone recordings
consent(F.consentRec);
out.push(BR());

// C. screening + SSQ
const S = F.ssq;
out.push(H(S.title), field([F.code, F.date]));
out.push(P(S.phase, { after: 160 }));
out.push(P(S.intro, { after: 160 }));
const lv = S.levels;
out.push(table([["#", S.symptom, ...lv], ...S.items.map((it, i) => [faNum(i + 1), it, "☐", "☐", "☐", "☐"])],
  [600, 4426, 1000, 1000, 1000, 1000], { center: [0, 2, 3, 4, 5] }));
out.push(P(S.note, { size: SMALL, before: 120 }));
out.push(BR());

// D. rating sheet (blind: clip number only)
const R = F.rating;
out.push(H(R.title), field([F.code, F.date]));
R.intro.forEach(t => out.push(P(t, { after: 60 })));
out.push(table([R.legendHead, ...R.questions.map(q => [q.short, q.full, q.low, q.high])],
  [1400, 4626, 1500, 1500], { size: 20 }));
out.push(P("", { after: 80 }));
const nums = [1, 2, 3, 4, 5, 6, 7].map(faNum);
const box = nums.join("  ");
const head = [R.clip, ...R.questions.map(q => q.short)];
const rows = [head, [R.practice, ...R.questions.map(() => box)]];
for (let i = 1; i <= 12; i++) rows.push([faNum(i), ...R.questions.map(() => box)]);
out.push(table(rows, [1226, 2600, 2600, 2600], { center: [0, 1, 2, 3], rowHeight: 520 }));
out.push(P(R.foot, { size: SMALL, before: 100 }));
out.push(BR());

// E. final ranking
const K = F.ranking;
out.push(H(K.title), field([F.code]));
K.intro.forEach(t => out.push(P(t)));
out.push(table([[K.rank, K.clip], ...K.rows.map((r, i) => [faNum(i + 1) + " — " + r, ""])], [4513, 4513], { rowHeight: 600, center: [1] }));
out.push(P(K.researcher, { size: SMALL, before: 160 }));
out.push(table([K.keyHead, ...["الف", "ب", "ج", "د"].map(x => [x, ""])], [4513, 4513], { center: [0, 1], size: 20 }));
out.push(BR());

// F. session protocol (for the researcher)
const G = F.protocol;
out.push(H(G.title), P(G.sub, { align: AlignmentType.CENTER, size: SMALL, after: 160 }));
G.steps.forEach(s => {
  out.push(H2(s.h));
  s.p.forEach(t => out.push(P(t, { bullet: true, after: 50 })));
});
out.push(BR(), H(G.logTitle));
out.push(table([G.logHead, ...Array.from({ length: 6 }, () => G.logHead.map(() => ""))], [1300, 1300, 1300, 1300, 3826], { rowHeight: 420, size: 20 }));

Promise.all([
  Packer.toBuffer(doc(letter)).then(b => fs.writeFileSync(__dirname + "/ethics_letter.docx", b)),
  Packer.toBuffer(doc(out)).then(b => fs.writeFileSync(__dirname + "/study_forms.docx", b)),
]).then(() => console.log("ok"));
