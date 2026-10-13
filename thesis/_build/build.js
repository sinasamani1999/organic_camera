const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, LevelFormat, Footer, PageNumber,
} = require("docx");

const FA = "B Nazanin", EN = "Times New Roman";
const BODY = 26, SMALL = 22; // half-points: 13pt, 11pt
const md = fs.readFileSync(__dirname + "/ch1.md", "utf8").split("\n");

// inline **bold** and *italic*
function runs(text, opts = {}) {
  if (!opts.ltr) text = text.replace(/([\u06F0-\u06F9])-/g, "$1\u200F-\u200F");
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  const push = (t, b, i) => t && out.push(new TextRun({
    text: t, bold: b || opts.bold, boldComplexScript: b || opts.bold, italics: i, italicsComplexScript: i,
    rightToLeft: !opts.ltr, size: opts.size || BODY, sizeComplexScript: opts.size || BODY,
    font: opts.ltr ? { ascii: EN, hAnsi: EN, cs: EN } : { ascii: EN, hAnsi: EN, cs: FA },
  }));
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const s = m[0];
    s.startsWith("**") ? push(s.slice(2, -2), true, false) : push(s.slice(1, -1), false, true);
    last = m.index + s.length;
  }
  push(text.slice(last));
  return out;
}

const para = (text, extra = {}) => new Paragraph({
  bidirectional: !extra.ltr, alignment: extra.ltr ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 360 }, children: runs(text, extra), ...extra.p,
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
const TABLE_W = 9026; // A4 text width with 1" margins

function table(rows) {
  const cells = rows.map(r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const header = cells[0], body = cells.slice(2);
  const n = header.length;
  const widths = n === 3 ? [3300, 3300, 2426] : [2600, 6426];
  const mk = (row, isHead) => new TableRow({
    tableHeader: isHead,
    children: row.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      shading: isHead ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
      children: [new Paragraph({
        bidirectional: true, alignment: isHead ? AlignmentType.CENTER : AlignmentType.START,
        spacing: { after: 0, line: 300 }, children: runs(c, { bold: isHead, size: SMALL + 2 }),
      })],
    })),
  });
  return new Table({
    visuallyRightToLeft: true, width: { size: TABLE_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(header, true), ...body.map(r => mk(r, false))],
  });
}

const children = [];
let tableBuf = [], numList = 0, inRefs = false, itemNo = 0;
const toFa = n => String(n).replace(/\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[d]);
const flushTable = () => { if (tableBuf.length) { children.push(table(tableBuf)); children.push(para("")); tableBuf = []; } };

for (let i = 0; i < md.length; i++) {
  const line = md[i];
  if (line.startsWith("|")) { tableBuf.push(line); continue; }
  flushTable();
  if (!line.trim()) continue;
  if (/^[A-Z][a-z]{2} \d+, \d{4}/.test(line)) continue; // date · @mention line from the doc
  let m;
  if ((m = line.match(/^(#{1,3}) (.*)/))) {
    const lvl = m[1].length;
    inRefs = m[2].includes("منابع");
    if (lvl === 2) numList++; // restart numbering after each section
    children.push(new Paragraph({
      heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][lvl - 1],
      bidirectional: true, alignment: lvl === 1 ? AlignmentType.CENTER : AlignmentType.START,
      children: runs(m[2], { bold: true, size: [36, 30, 28][lvl - 1] }),
    }));
    continue;
  }
  if ((m = line.match(/^> (.*)/))) {
    children.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.JUSTIFIED, spacing: { before: 120, after: 200, line: 360 },
      indent: { left: 567, right: 567 },
      border: { right: { style: BorderStyle.SINGLE, size: 18, color: "808080", space: 8 } },
      children: runs(m[1], { bold: true }),
    }));
    continue;
  }
  if ((m = line.match(/^\d+\. (.*)/))) {
    if (!/^\d+\. /.test(md[i-1] || "")) itemNo = 0;
    children.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 360 },
      indent: { right: 425, hanging: 283 }, children: [new TextRun({ text: toFa(++itemNo) + ".\t", rightToLeft: true, bold: true, boldComplexScript: true, size: BODY, sizeComplexScript: BODY, font: { cs: FA } }), ...runs(m[1])],
    }));
    continue;
  }
  if ((m = line.match(/^- (.*)/))) {
    if (inRefs) {
      children.push(new Paragraph({
        alignment: AlignmentType.LEFT, spacing: { after: 100, line: 276 },
        indent: { left: 567, hanging: 567 }, children: runs(m[1], { ltr: true, size: SMALL }),
      }));
    } else {
      children.push(new Paragraph({
        bidirectional: true, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 360 },
        indent: { right: 425, hanging: 283 }, children: [new TextRun({ text: "\u2022\t", rightToLeft: true, size: BODY }), ...runs(m[1])],
      }));
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
  numbering: {
    config: [
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START,
        style: { paragraph: { indent: { right: 567, hanging: 360 } } } }] },
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.START,
        style: { paragraph: { indent: { right: 567, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: SMALL })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync(__dirname + "/fasl1.docx", b); console.log("ok", b.length); });
